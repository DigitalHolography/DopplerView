# -*- coding: utf-8 -*-
"""
UDVD for LDH videos

功能：
1. 读取：
   C:\\Users\\Novovorontsovka\\Downloads\\video_masqued

2. 每个 epoch 随机抽取 3000 个连续 5 帧 sequence。
3. 每个 sequence 随机裁剪一个 96×96 patch。
4. 使用 UDVD Blind-Spot 网络预测中间帧。
5. 每个 epoch 保存模型。
6. 从 epoch 5 开始，每 5 个 epoch 去噪一次文件夹中的全部视频。
7. 只保存：
   - epoch 模型
   - 去噪视频

输出：
C:\\Users\\Novovorontsovka\\Downloads\\udvd
├── epoch_models
│   ├── epoch_001.pth
│   ├── epoch_002.pth
│   └── ...
├── epoch_005
│   ├── video_1_denoised.avi
│   └── ...
└── epoch_006
    └── ...
"""

from __future__ import annotations

import gc
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ============================================================
# 配置
# ============================================================

@dataclass(frozen=True)
class Config:
    # 输入与输出
    video_folder: Path = Path(
        r"C:\Users\Novovorontsovka\Downloads\video_masqued"
    )
    output_folder: Path = Path(
        r"C:\Users\Novovorontsovka\Downloads\udvd"
    )

    # 支持的视频格式
    video_extensions: tuple[str, ...] = (
        ".avi",
        ".mp4",
        ".mov",
        ".mkv",
        ".mpg",
        ".mpeg",
    )

    # 训练
    num_epochs: int = 100
    sequences_per_epoch: int = 3000

    # 标准 UDVD 使用 5 帧
    sequence_length: int = 5

    # 训练 patch
    patch_size: int = 96

    batch_size: int = 8
    learning_rate: float = 1e-4

    # Windows 下建议先使用 0
    num_workers: int = 0

    # 从第几个 epoch 开始输出全部去噪视频
    denoise_start_epoch: int = 5

    # 每隔多少个 epoch 去噪一次
    denoise_every: int = 5

    # 推理时一次处理多少个 sequence
    # 512×512 + UDVD 比较占显存，建议 1
    inference_batch_size: int = 1

    # 视频编码
    output_codec: str = "MJPG"

    # 随机种子
    seed: int = 2026

    # 自动混合精度
    use_amp: bool = True

    # 梯度裁剪
    max_grad_norm: float = 5.0

    # 可选断点继续
    # 例如改成：
    # resume_model = Path(r"...\epoch_models\epoch_020.pth")
    resume_model: Optional[Path] = None

    # 如果加载 epoch_020，则填 21
    start_epoch: int = 1


CFG = Config()


# ============================================================
# 随机种子
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True


# ============================================================
# 视频信息
# ============================================================

@dataclass(frozen=True)
class VideoInfo:
    path: Path
    frame_count: int
    width: int
    height: int
    fps: float


def natural_sort_key(path: Path):
    """使 2.avi 排在 10.avi 前面。"""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def find_video_files(
    folder: Path,
    extensions: tuple[str, ...],
) -> list[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"输入文件夹不存在：{folder}")

    extensions_lower = {ext.lower() for ext in extensions}

    video_paths = [
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in extensions_lower
    ]

    video_paths.sort(key=natural_sort_key)

    if not video_paths:
        raise RuntimeError(f"文件夹中没有找到视频：{folder}")

    return video_paths


def inspect_video(path: Path) -> Optional[VideoInfo]:
    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        print(f"[跳过] 无法打开视频：{path.name}")
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))

    cap.release()

    if frame_count < CFG.sequence_length:
        print(
            f"[跳过] 帧数不足：{path.name}，"
            f"只有 {frame_count} 帧"
        )
        return None

    if width <= 0 or height <= 0:
        print(f"[跳过] 尺寸错误：{path.name}")
        return None

    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0

    return VideoInfo(
        path=path,
        frame_count=frame_count,
        width=width,
        height=height,
        fps=fps,
    )


def collect_video_information() -> list[VideoInfo]:
    paths = find_video_files(
        CFG.video_folder,
        CFG.video_extensions,
    )

    videos: list[VideoInfo] = []

    for path in paths:
        info = inspect_video(path)
        if info is not None:
            videos.append(info)

    if not videos:
        raise RuntimeError("没有可用于训练的视频。")

    return videos


# ============================================================
# 读取视频帧
# ============================================================

def read_gray_frame(
    cap: cv2.VideoCapture,
    frame_index: int,
) -> np.ndarray:
    """
    读取指定帧并转为灰度 uint8。
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    success, frame = cap.read()

    if not success or frame is None:
        raise RuntimeError(f"无法读取第 {frame_index} 帧")

    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    return frame


def read_five_consecutive_frames(
    video: VideoInfo,
    start_frame: int,
) -> np.ndarray:
    """
    返回形状：
    [5, H, W]
    """
    cap = cv2.VideoCapture(str(video.path))

    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video.path}")

    frames: list[np.ndarray] = []

    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))

        for offset in range(CFG.sequence_length):
            success, frame = cap.read()

            if not success or frame is None:
                # 某些编码器随机定位不稳定，重新精确读取
                frame = read_gray_frame(
                    cap,
                    start_frame + offset,
                )
            elif frame.ndim == 3:
                frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2GRAY,
                )

            frames.append(frame)
    finally:
        cap.release()

    if len(frames) != CFG.sequence_length:
        raise RuntimeError(
            f"视频读取失败：{video.path.name}"
        )

    return np.stack(frames, axis=0)


# ============================================================
# 随机训练数据集
# ============================================================

class RandomVideoSequenceDataset(Dataset):
    """
    每次 __getitem__ 都随机选择：
    1. 一个视频
    2. 一段连续 5 帧
    3. 一个空间 patch

    每个 epoch 长度固定为 3000。
    """

    def __init__(
        self,
        videos: list[VideoInfo],
        samples_per_epoch: int,
        patch_size: int,
    ):
        self.videos = [
            video
            for video in videos
            if video.height >= patch_size
            and video.width >= patch_size
            and video.frame_count >= CFG.sequence_length
        ]

        self.samples_per_epoch = samples_per_epoch
        self.patch_size = patch_size

        if not self.videos:
            raise RuntimeError(
                f"没有尺寸大于等于 {patch_size}×{patch_size} "
                "且帧数足够的视频。"
            )

        # 根据每个视频可用 sequence 数量进行加权
        self.sequence_weights = np.asarray(
            [
                video.frame_count
                - CFG.sequence_length
                + 1
                for video in self.videos
            ],
            dtype=np.float64,
        )

        self.sequence_weights /= self.sequence_weights.sum()

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _sample_video(self) -> VideoInfo:
        index = np.random.choice(
            len(self.videos),
            p=self.sequence_weights,
        )
        return self.videos[int(index)]

    def __getitem__(self, index: int):
        del index

        # 读取失败时允许重新抽样
        for _ in range(10):
            video = self._sample_video()

            max_start = (
                video.frame_count
                - CFG.sequence_length
            )

            start_frame = random.randint(
                0,
                max_start,
            )

            try:
                sequence = read_five_consecutive_frames(
                    video,
                    start_frame,
                )
            except RuntimeError:
                continue

            _, height, width = sequence.shape

            top = random.randint(
                0,
                height - self.patch_size,
            )
            left = random.randint(
                0,
                width - self.patch_size,
            )

            patch = sequence[
                :,
                top:top + self.patch_size,
                left:left + self.patch_size,
            ]

            # 数据增强：不会改变时间顺序
            if random.random() < 0.5:
                patch = patch[:, :, ::-1]

            if random.random() < 0.5:
                patch = patch[:, ::-1, :]

            rotation_k = random.randint(0, 3)
            patch = np.rot90(
                patch,
                k=rotation_k,
                axes=(1, 2),
            )

            # 防止负 stride
            patch = np.ascontiguousarray(patch)

            # [5,H,W]，范围 [0,1]
            input_tensor = torch.from_numpy(
                patch.astype(np.float32) / 255.0
            )

            # 中间帧作为含噪 target
            target_tensor = input_tensor[
                CFG.sequence_length // 2:
                CFG.sequence_length // 2 + 1
            ].clone()

            return input_tensor, target_tensor

        raise RuntimeError(
            "连续多次读取视频失败，请检查视频文件。"
        )


# ============================================================
# 官方 UDVD Blind-Spot 网络结构
# 针对灰度视频修改：
# channels_per_frame = 1
# out_channels = 1
# ============================================================

class Crop(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        return x[:, :, :height - 1, :width]


class Shift(nn.Module):
    def __init__(self):
        super().__init__()
        self.shift_down = nn.ZeroPad2d((0, 0, 1, 0))
        self.crop = Crop()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.crop(self.shift_down(x))


class BlindConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = False,
        blind: bool = True,
    ):
        super().__init__()

        self.blind = blind

        if blind:
            self.shift_down = nn.ZeroPad2d((0, 0, 1, 0))
            self.crop = Crop()

        self.replicate = nn.ReplicationPad2d(1)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            bias=bias,
        )
        self.activation = nn.LeakyReLU(
            negative_slope=0.1,
            inplace=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.blind:
            x = self.shift_down(x)

        x = self.replicate(x)
        x = self.conv(x)
        x = self.activation(x)

        if self.blind:
            x = self.crop(x)

        return x


class BlindPool(nn.Module):
    def __init__(self, blind: bool = True):
        super().__init__()

        self.blind = blind

        if blind:
            self.shift = Shift()

        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.blind:
            x = self.shift(x)

        return self.pool(x)


class Rotate(nn.Module):
    """
    将输入旋转为四个方向，然后在 batch 维拼接。
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_90 = x.transpose(2, 3).flip(3)
        x_180 = x.flip(2).flip(3)
        x_270 = x.transpose(2, 3).flip(2)

        return torch.cat(
            (x, x_90, x_180, x_270),
            dim=0,
        )


class Unrotate(nn.Module):
    """
    将四个方向旋转回来，在 channel 维拼接。
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_0, x_90, x_180, x_270 = torch.chunk(
            x,
            chunks=4,
            dim=0,
        )

        x_90 = x_90.transpose(2, 3).flip(2)
        x_180 = x_180.flip(2).flip(3)
        x_270 = x_270.transpose(2, 3).flip(3)

        return torch.cat(
            (x_0, x_90, x_180, x_270),
            dim=1,
        )


class EncoderConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        bias: bool = False,
        reduce: bool = True,
        blind: bool = True,
    ):
        super().__init__()

        self.reduce = reduce

        self.conv1 = BlindConv(
            in_channels,
            mid_channels,
            bias=bias,
            blind=blind,
        )
        self.conv2 = BlindConv(
            mid_channels,
            mid_channels,
            bias=bias,
            blind=blind,
        )
        self.conv3 = BlindConv(
            mid_channels,
            out_channels,
            bias=bias,
            blind=blind,
        )

        if reduce:
            self.pool = BlindPool(blind=blind)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        if self.reduce:
            x = self.pool(x)

        return x


class DecoderConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        bias: bool = False,
        blind: bool = True,
    ):
        super().__init__()

        self.upsample = nn.Upsample(
            scale_factor=2,
            mode="nearest",
        )

        self.conv1 = BlindConv(
            in_channels,
            mid_channels,
            bias=bias,
            blind=blind,
        )
        self.conv2 = BlindConv(
            mid_channels,
            mid_channels,
            bias=bias,
            blind=blind,
        )
        self.conv3 = BlindConv(
            mid_channels,
            mid_channels,
            bias=bias,
            blind=blind,
        )
        self.conv4 = BlindConv(
            mid_channels,
            out_channels,
            bias=bias,
            blind=blind,
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        x = self.upsample(x)

        difference_y = skip.size(2) - x.size(2)
        difference_x = skip.size(3) - x.size(3)

        x = F.pad(
            x,
            [
                difference_x // 2,
                difference_x - difference_x // 2,
                difference_y // 2,
                difference_y - difference_y // 2,
            ],
        )

        x = torch.cat((x, skip), dim=1)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        return x


class BlindUNet(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_output: int = 96,
        bias: bool = False,
        blind: bool = True,
    ):
        super().__init__()

        self.enc1 = EncoderConv(
            n_channels,
            48,
            48,
            bias=bias,
            blind=blind,
        )

        self.enc2 = EncoderConv(
            48,
            48,
            48,
            bias=bias,
            blind=blind,
        )

        self.enc3 = EncoderConv(
            48,
            96,
            48,
            bias=bias,
            reduce=False,
            blind=blind,
        )

        self.dec2 = DecoderConv(
            96,
            96,
            96,
            bias=bias,
            blind=blind,
        )

        self.dec1 = DecoderConv(
            96 + n_channels,
            96,
            n_output,
            bias=bias,
            blind=blind,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)

        x = self.dec2(x3, x1)
        x = self.dec1(x, x1.new_tensor([])) \
            if False else self.dec1(x, self._original_input)

        return x

    def run(self, x: torch.Tensor) -> torch.Tensor:
        """
        与官方 forward 相同，但显式保存原始输入，
        便于 decoder 最后一层做 skip connection。
        """
        original_input = x

        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)

        x = self.dec2(x3, x1)
        x = self.dec1(x, original_input)

        return x


class UDVD(nn.Module):
    """
    官方 blind-video-net-4 的灰度版本。

    输入：
        [B, 5, H, W]

    输出：
        [B, 1, H, W]
    """

    def __init__(
        self,
        channels_per_frame: int = 1,
        out_channels: int = 1,
        bias: bool = False,
        blind: bool = True,
    ):
        super().__init__()

        self.channels_per_frame = channels_per_frame
        self.blind = blind

        self.rotate = Rotate()

        # 三组相邻三帧：
        # [0,1,2], [1,2,3], [2,3,4]
        self.denoiser_1 = BlindUNet(
            n_channels=3 * channels_per_frame,
            n_output=32,
            bias=bias,
            blind=blind,
        )

        # 三组特征拼接：32 × 3 = 96
        self.denoiser_2 = BlindUNet(
            n_channels=96,
            n_output=96,
            bias=bias,
            blind=blind,
        )

        if blind:
            self.shift = Shift()

        self.unrotate = Unrotate()

        self.nin_a = nn.Conv2d(
            384,
            384,
            kernel_size=1,
            bias=bias,
        )
        self.nin_b = nn.Conv2d(
            384,
            96,
            kernel_size=1,
            bias=bias,
        )
        self.nin_c = nn.Conv2d(
            96,
            out_channels,
            kernel_size=1,
            bias=bias,
        )

    @staticmethod
    def _make_square(
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, int, int]:
        _, _, height, width = x.shape

        if height > width:
            difference = height - width
            x = F.pad(
                x,
                [
                    difference // 2,
                    difference - difference // 2,
                    0,
                    0,
                ],
                mode="reflect",
            )

        elif width > height:
            difference = width - height
            x = F.pad(
                x,
                [
                    0,
                    0,
                    difference // 2,
                    difference - difference // 2,
                ],
                mode="reflect",
            )

        return x, height, width

    @staticmethod
    def _restore_shape(
        x: torch.Tensor,
        original_height: int,
        original_width: int,
    ) -> torch.Tensor:
        if original_height > original_width:
            difference = original_height - original_width
            left = difference // 2

            x = x[
                :,
                :,
                :original_height,
                left:left + original_width,
            ]

        elif original_width > original_height:
            difference = original_width - original_height
            top = difference // 2

            x = x[
                :,
                :,
                top:top + original_height,
                :original_width,
            ]

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, original_height, original_width = (
            self._make_square(x)
        )

        channels = self.channels_per_frame

        # 5 帧中取三个重叠的三帧小组
        input_1 = self.rotate(
            x[:, 0:3 * channels]
        )
        input_2 = self.rotate(
            x[:, channels:4 * channels]
        )
        input_3 = self.rotate(
            x[:, 2 * channels:5 * channels]
        )

        feature_1 = self.denoiser_1.run(input_1)
        feature_2 = self.denoiser_1.run(input_2)
        feature_3 = self.denoiser_1.run(input_3)

        features = torch.cat(
            (feature_1, feature_2, feature_3),
            dim=1,
        )

        features = self.denoiser_2.run(features)

        if self.blind:
            features = self.shift(features)

        features = self.unrotate(features)

        output = F.leaky_relu(
            self.nin_a(features),
            negative_slope=0.1,
            inplace=True,
        )
        output = F.leaky_relu(
            self.nin_b(output),
            negative_slope=0.1,
            inplace=True,
        )
        output = self.nin_c(output)

        output = self._restore_shape(
            output,
            original_height,
            original_width,
        )

        return output


# ============================================================
# 初始化权重
# ============================================================

def initialize_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(
            module.weight,
            a=0.1,
            mode="fan_in",
            nonlinearity="leaky_relu",
        )

        if module.bias is not None:
            nn.init.zeros_(module.bias)


# ============================================================
# 推理辅助函数
# ============================================================

def load_full_gray_video(
    path: Path,
) -> tuple[np.ndarray, float]:
    """
    整段读取视频。

    返回：
        frames: [T,H,W], uint8
        fps
    """
    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))

    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0

    frames: list[np.ndarray] = []

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            if frame.ndim == 3:
                frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2GRAY,
                )

            frames.append(frame)
    finally:
        cap.release()

    if not frames:
        raise RuntimeError(f"视频没有有效帧：{path}")

    return np.stack(frames, axis=0), fps


def make_padded_sequence(
    frames: np.ndarray,
    center_index: int,
) -> np.ndarray:
    """
    构造 5 帧 sequence。

    边缘使用复制：
    center=0 -> [0,0,0,1,2]
    """
    frame_count = frames.shape[0]
    radius = CFG.sequence_length // 2

    indices = [
        min(
            max(center_index + offset, 0),
            frame_count - 1,
        )
        for offset in range(-radius, radius + 1)
    ]

    return frames[indices]


def create_video_writer(
    output_path: Path,
    fps: float,
    width: int,
    height: int,
) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(
        *CFG.output_codec
    )

    # 某些 OpenCV Windows 版本对单通道 AVI 支持不稳定，
    # 所以写成三通道灰度，视觉和像素值相同。
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height),
        True,
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"无法创建输出视频：{output_path}"
        )

    return writer


@torch.inference_mode()
def denoise_one_video(
    model: nn.Module,
    video_path: Path,
    output_path: Path,
    device: torch.device,
    amp_enabled: bool,
) -> None:
    frames, fps = load_full_gray_video(video_path)

    frame_count, height, width = frames.shape

    writer = create_video_writer(
        output_path,
        fps,
        width,
        height,
    )

    model.eval()

    try:
        for batch_start in tqdm(
            range(0, frame_count, CFG.inference_batch_size),
            desc=f"去噪 {video_path.name}",
            leave=False,
        ):
            batch_end = min(
                batch_start + CFG.inference_batch_size,
                frame_count,
            )

            sequences: list[np.ndarray] = []

            for center_index in range(
                batch_start,
                batch_end,
            ):
                sequence = make_padded_sequence(
                    frames,
                    center_index,
                )
                sequences.append(sequence)

            batch_array = np.stack(
                sequences,
                axis=0,
            ).astype(np.float32) / 255.0

            # [B,5,H,W]
            input_tensor = torch.from_numpy(
                batch_array
            ).to(
                device=device,
                non_blocking=True,
            )

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                output = model(input_tensor)

            output = output.float()
            output = torch.clamp(output, 0.0, 1.0)

            output_uint8 = (
                output[:, 0]
                .mul(255.0)
                .round()
                .byte()
                .cpu()
                .numpy()
            )

            for gray_frame in output_uint8:
                bgr_frame = cv2.cvtColor(
                    gray_frame,
                    cv2.COLOR_GRAY2BGR,
                )
                writer.write(bgr_frame)

    finally:
        writer.release()
        del frames
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@torch.inference_mode()
def denoise_all_videos(
    model: nn.Module,
    videos: list[VideoInfo],
    epoch: int,
    device: torch.device,
    amp_enabled: bool,
) -> None:
    epoch_output_folder = (
        CFG.output_folder
        / f"epoch_{epoch:03d}"
    )
    epoch_output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"\n开始使用 epoch {epoch} 模型"
        f"去噪全部 {len(videos)} 个视频"
    )

    for video_index, video in enumerate(
        videos,
        start=1,
    ):
        output_name = (
            f"{video.path.stem}_denoised.avi"
        )
        output_path = (
            epoch_output_folder / output_name
        )

        print(
            f"[{video_index}/{len(videos)}] "
            f"{video.path.name}"
        )

        try:
            denoise_one_video(
                model=model,
                video_path=video.path,
                output_path=output_path,
                device=device,
                amp_enabled=amp_enabled,
            )
        except Exception as error:
            print(
                f"[去噪失败] {video.path.name}: "
                f"{error}"
            )

    print(
        f"epoch {epoch} 全部视频去噪完成："
        f"{epoch_output_folder}\n"
    )


# ============================================================
# 单 epoch 训练
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
    amp_enabled: bool,
) -> float:
    model.train()

    total_loss = 0.0
    processed_samples = 0

    progress_bar = tqdm(
        loader,
        desc=f"Epoch {epoch:03d}/{CFG.num_epochs:03d}",
    )

    for input_tensor, target_tensor in progress_bar:
        input_tensor = input_tensor.to(
            device=device,
            non_blocking=True,
        )
        target_tensor = target_tensor.to(
            device=device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            prediction = model(input_tensor)

            # 含噪中间帧作为自监督 target
            loss = F.mse_loss(
                prediction,
                target_tensor,
            )

        if not torch.isfinite(loss):
            print(
                "\n[警告] 出现非有限 loss，"
                "跳过当前 batch。"
            )
            continue

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=CFG.max_grad_norm,
        )

        scaler.step(optimizer)
        scaler.update()

        batch_size = input_tensor.shape[0]

        total_loss += (
            float(loss.detach().item())
            * batch_size
        )
        processed_samples += batch_size

        average_loss = (
            total_loss / max(processed_samples, 1)
        )

        progress_bar.set_postfix(
            loss=f"{average_loss:.6f}"
        )

    if processed_samples == 0:
        raise RuntimeError(
            "本 epoch 没有成功训练任何样本。"
        )

    return total_loss / processed_samples


# ============================================================
# 主程序
# ============================================================

def main() -> None:
    set_seed(CFG.seed)

    if CFG.sequence_length != 5:
        raise ValueError(
            "这个 UDVD 网络固定使用 5 帧输入。"
        )

    CFG.output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_folder = (
        CFG.output_folder / "epoch_models"
    )
    model_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    amp_enabled = (
        CFG.use_amp
        and device.type == "cuda"
    )

    print("=" * 70)
    print("UDVD LDH Training")
    print("=" * 70)
    print(f"设备：{device}")

    if torch.cuda.is_available():
        print(
            "GPU："
            f"{torch.cuda.get_device_name(0)}"
        )

    print(f"输入文件夹：{CFG.video_folder}")
    print(f"输出文件夹：{CFG.output_folder}")
    print(
        f"每个 epoch 随机 sequence 数量："
        f"{CFG.sequences_per_epoch}"
    )
    print(f"输入帧数：{CFG.sequence_length}")
    print(f"Patch：{CFG.patch_size}×{CFG.patch_size}")
    print(f"Batch size：{CFG.batch_size}")
    print(
        f"从 epoch {CFG.denoise_start_epoch} 开始，"
        f"每 {CFG.denoise_every} 个 epoch 去噪一次全部视频"
    )
    print("=" * 70)

    videos = collect_video_information()

    print(f"有效视频数量：{len(videos)}")

    for index, video in enumerate(videos, start=1):
        print(
            f"{index:03d}. {video.path.name} | "
            f"{video.frame_count} 帧 | "
            f"{video.width}×{video.height} | "
            f"{video.fps:.3f} FPS"
        )

    dataset = RandomVideoSequenceDataset(
        videos=videos,
        samples_per_epoch=CFG.sequences_per_epoch,
        patch_size=CFG.patch_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        persistent_workers=(
            CFG.num_workers > 0
        ),
    )

    model = UDVD(
        channels_per_frame=1,
        out_channels=1,
        bias=False,
        blind=True,
    )

    model.apply(initialize_weights)
    model = model.to(device)

    if CFG.resume_model is not None:
        if not CFG.resume_model.exists():
            raise FileNotFoundError(
                f"断点模型不存在：{CFG.resume_model}"
            )

        state_dict = torch.load(
            CFG.resume_model,
            map_location=device,
            weights_only=True,
        )

        model.load_state_dict(state_dict)

        print(
            f"已加载断点模型："
            f"{CFG.resume_model}"
        )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CFG.learning_rate,
    )

    scaler = torch.amp.GradScaler(
        device=device.type,
        enabled=amp_enabled,
    )

    for epoch in range(
        CFG.start_epoch,
        CFG.num_epochs + 1,
    ):
        epoch_loss = train_one_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            amp_enabled=amp_enabled,
        )

        # 只保存模型参数
        model_path = (
            model_folder
            / f"epoch_{epoch:03d}.pth"
        )

        torch.save(
            model.state_dict(),
            model_path,
        )

        print(
            f"Epoch {epoch:03d} 完成 | "
            f"Loss = {epoch_loss:.8f}"
        )
        print(f"模型已保存：{model_path}")

        # 从 epoch 5 开始，每 5 个 epoch 去噪一次全部视频
        if (
            epoch >= CFG.denoise_start_epoch
            and (epoch - CFG.denoise_start_epoch) % CFG.denoise_every == 0
        ):
            denoise_all_videos(
                model=model,
                videos=videos,
                epoch=epoch,
                device=device,
                amp_enabled=amp_enabled,
            )

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("=" * 70)
    print("全部训练与视频去噪完成。")
    print(f"结果位置：{CFG.output_folder}")
    print("=" * 70)


if __name__ == "__main__":
    main()