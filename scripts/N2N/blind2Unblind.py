# -*- coding: utf-8 -*-
r"""
Pure Blind2Unblind baseline for LDH videos
Adapted from the official CVPR 2022 Blind2Unblind method.

外部训练条件与给定 UDVD 代码保持一致：
1. 输入：
   C:\Users\Novovorontsovka\Downloads\video_masqued
2. 训练 100 epochs。
3. 每个 epoch 随机抽取 3000 个单帧样本。
4. 每个样本随机裁剪一个 96×96 patch。
5. batch size = 8。
6. Adam，learning rate = 1e-4。
7. seed = 2026，AMP=True，max_grad_norm=5.0。
8. 每个 epoch 保存模型。
9. 只在 epoch 5、10、15、...、100 去噪全部视频。
10. 只保存 epoch 模型和去噪视频。

Blind2Unblind 方法本身：
- 单帧自监督，不使用连续帧、phase table、同相位帧或 ConvLSTM。
- 使用 width=4 的 global-aware mask mapper：
  4×4 cell 内的 16 个位置分别作为 blind spot。
- masked input 采用邻域插值替换。
- 使用官方 re-visible loss：
      diff = masked_prediction - noisy
      exp_diff = full_visible_prediction - noisy
      re_visible = diff + beta * exp_diff
      loss = Lambda1 * mean(diff^2) + mean(re_visible^2)
- 推理输出：
      output = (masked_prediction + beta * full_visible_prediction)
               / (1 + beta)
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
    video_folder: Path = Path(
        r"C:\Users\Novovorontsovka\Downloads\video_masqued"
    )
    output_folder: Path = Path(
        r"C:\Users\Novovorontsovka\Downloads\blind2unblind"
    )

    video_extensions: tuple[str, ...] = (
        ".avi", ".mp4", ".mov", ".mkv", ".mpg", ".mpeg"
    )

    # 与 UDVD 完全一致的外部训练条件
    num_epochs: int = 100
    samples_per_epoch: int = 3000
    patch_size: int = 96
    batch_size: int = 8
    learning_rate: float = 1e-4
    num_workers: int = 0
    denoise_start_epoch: int = 5
    denoise_interval: int = 5
    inference_batch_size: int = 1
    output_codec: str = "MJPG"
    seed: int = 2026
    use_amp: bool = True
    max_grad_norm: float = 5.0

    # Blind2Unblind 官方方法参数
    mask_width: int = 4
    lambda1: float = 1.0
    lambda2: float = 2.0
    increase_ratio: float = 20.0

    # 对 real/raw 数据，官方训练代码采用 0.4 -> 1.0 的 beta 增长区间
    beta_start_ratio: float = 0.4
    beta_end_ratio: float = 1.0

    # 推理时分批处理 16 个 mask，避免完整 512×512 图像显存过高
    inference_mask_chunk_size: int = 2

    # 网络：官方 UNet 默认 depth=5, wf=48
    network_depth: int = 5
    network_width: int = 48

    # 可选断点继续
    resume_model: Optional[Path] = None
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
# 视频信息与读取
# ============================================================

@dataclass(frozen=True)
class VideoInfo:
    path: Path
    frame_count: int
    width: int
    height: int
    fps: float


def natural_sort_key(path: Path):
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

    if frame_count < 1:
        print(f"[跳过] 没有有效帧：{path.name}")
        return None

    if width < CFG.patch_size or height < CFG.patch_size:
        print(
            f"[跳过] 尺寸小于 patch：{path.name}，"
            f"{width}×{height}"
        )
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


def read_gray_frame(
    video: VideoInfo,
    frame_index: int,
) -> np.ndarray:
    cap = cv2.VideoCapture(str(video.path))

    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video.path}")

    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        success, frame = cap.read()
    finally:
        cap.release()

    if not success or frame is None:
        raise RuntimeError(
            f"无法读取 {video.path.name} 的第 {frame_index} 帧"
        )

    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    return frame


# ============================================================
# 随机单帧 patch Dataset
# ============================================================

class RandomVideoFrameDataset(Dataset):
    """
    每次 __getitem__ 随机选择：
    1. 一个视频；
    2. 一张帧；
    3. 一个 96×96 patch。

    每个 epoch 固定 3000 个训练样本。
    """

    def __init__(
        self,
        videos: list[VideoInfo],
        samples_per_epoch: int,
        patch_size: int,
    ):
        self.videos = videos
        self.samples_per_epoch = samples_per_epoch
        self.patch_size = patch_size

        # 与 UDVD 类似：按可用帧数量加权选择视频
        self.frame_weights = np.asarray(
            [video.frame_count for video in videos],
            dtype=np.float64,
        )
        self.frame_weights /= self.frame_weights.sum()

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _sample_video(self) -> VideoInfo:
        index = np.random.choice(
            len(self.videos),
            p=self.frame_weights,
        )
        return self.videos[int(index)]

    def __getitem__(self, index: int) -> torch.Tensor:
        del index

        for _ in range(10):
            video = self._sample_video()
            frame_index = random.randint(
                0,
                video.frame_count - 1,
            )

            try:
                frame = read_gray_frame(video, frame_index)
            except RuntimeError:
                continue

            height, width = frame.shape

            top = random.randint(
                0,
                height - self.patch_size,
            )
            left = random.randint(
                0,
                width - self.patch_size,
            )

            patch = frame[
                top:top + self.patch_size,
                left:left + self.patch_size,
            ]

            # 与 UDVD 相同的数据增强
            if random.random() < 0.5:
                patch = patch[:, ::-1]

            if random.random() < 0.5:
                patch = patch[::-1, :]

            rotation_k = random.randint(0, 3)
            patch = np.rot90(
                patch,
                k=rotation_k,
                axes=(0, 1),
            )

            patch = np.ascontiguousarray(patch)

            return torch.from_numpy(
                patch.astype(np.float32) / 255.0
            ).unsqueeze(0)

        raise RuntimeError(
            "连续多次读取视频失败，请检查视频文件。"
        )


# ============================================================
# 官方风格 UNet（灰度版）
# ============================================================

class LR(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        slope: float = 0.1,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=True,
            ),
            nn.LeakyReLU(
                negative_slope=slope,
                inplace=True,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        slope: float = 0.1,
    ):
        super().__init__()

        self.conv_1 = LR(
            in_channels,
            out_channels,
            kernel_size=3,
            slope=slope,
        )
        self.conv_2 = LR(
            out_channels,
            out_channels,
            kernel_size=3,
            slope=slope,
        )

    @staticmethod
    def upsample(x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            x,
            scale_factor=2,
            mode="nearest",
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        x = self.upsample(x)

        # 正常情况下尺寸一致；保险起见进行中心对齐
        difference_y = skip.size(2) - x.size(2)
        difference_x = skip.size(3) - x.size(3)

        if difference_y != 0 or difference_x != 0:
            x = F.pad(
                x,
                [
                    difference_x // 2,
                    difference_x - difference_x // 2,
                    difference_y // 2,
                    difference_y - difference_y // 2,
                ],
            )

        x = torch.cat([x, skip], dim=1)
        x = self.conv_1(x)
        x = self.conv_2(x)

        return x


class Blind2UnblindUNet(nn.Module):
    """
    与官方 arch_unet.py 结构一致的单通道版本。
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        depth: int = 5,
        width: int = 48,
        slope: float = 0.1,
    ):
        super().__init__()

        self.depth = depth

        self.head = nn.Sequential(
            LR(in_channels, width, 3, slope),
            LR(width, width, 3, slope),
        )

        self.down_path = nn.ModuleList(
            [LR(width, width, 3, slope) for _ in range(depth)]
        )

        self.up_path = nn.ModuleList()

        for index in range(depth):
            if index != depth - 1:
                input_channels = (
                    width * 2
                    if index == 0
                    else width * 3
                )
            else:
                input_channels = width * 2 + in_channels

            self.up_path.append(
                UP(
                    in_channels=input_channels,
                    out_channels=width * 2,
                    slope=slope,
                )
            )

        self.last = nn.Sequential(
            LR(width * 2, width * 2, 1, slope),
            LR(width * 2, width * 2, 1, slope),
            nn.Conv2d(
                width * 2,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        blocks: list[torch.Tensor] = [x]

        x = self.head(x)

        for index, down in enumerate(self.down_path):
            x = F.max_pool2d(x, 2)

            if index != len(self.down_path) - 1:
                blocks.append(x)

            x = down(x)

        for index, up in enumerate(self.up_path):
            x = up(x, blocks[-index - 1])

        return self.last(x)


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
# Global-aware mask mapper
# ============================================================

def interpolate_mask(
    tensor: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    blind spot 使用周围 8 邻域加权插值替换。
    官方 kernel：
        0.5  1.0  0.5
        1.0  0.0  1.0
        0.5  1.0  0.5
    """

    batch_size, channels, height, width = tensor.shape

    kernel = torch.tensor(
        [
            [0.5, 1.0, 0.5],
            [1.0, 0.0, 1.0],
            [0.5, 1.0, 0.5],
        ],
        dtype=tensor.dtype,
        device=tensor.device,
    )
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, 3, 3)

    filtered = F.conv2d(
        tensor.reshape(
            batch_size * channels,
            1,
            height,
            width,
        ),
        kernel,
        stride=1,
        padding=1,
    )
    filtered = filtered.reshape_as(tensor)

    mask = mask.to(
        device=tensor.device,
        dtype=tensor.dtype,
    )

    return filtered * mask + tensor * (1.0 - mask)


def make_fixed_mask(
    batch_size: int,
    height: int,
    width: int,
    mask_width: int,
    position: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    每个 mask_width×mask_width cell 中固定选择一个位置。
    返回 [B,1,H,W]。
    """

    row_offset = position // mask_width
    column_offset = position % mask_width

    mask = torch.zeros(
        (batch_size, 1, height, width),
        device=device,
        dtype=dtype,
    )

    mask[
        :,
        :,
        row_offset::mask_width,
        column_offset::mask_width,
    ] = 1.0

    return mask


def make_all_masked_inputs(
    image: torch.Tensor,
    mask_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    生成全部 mask_width² 个 masked versions。

    返回：
        masked_inputs: [B*mask_width²,C,H,W]
        masks:         [B*mask_width²,1,H,W]
    """

    batch_size, _, height, width = image.shape
    masked_images: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []

    for position in range(mask_width ** 2):
        mask = make_fixed_mask(
            batch_size=batch_size,
            height=height,
            width=width,
            mask_width=mask_width,
            position=position,
            device=image.device,
            dtype=image.dtype,
        )

        masked = interpolate_mask(
            image,
            mask,
        )

        masked_images.append(masked)
        masks.append(mask)

    # 先按 mask position 拼接，再 reshape 回 [B*16,...]
    masked_stack = torch.stack(
        masked_images,
        dim=1,
    )
    mask_stack = torch.stack(
        masks,
        dim=1,
    )

    masked_stack = masked_stack.reshape(
        batch_size * mask_width ** 2,
        image.shape[1],
        height,
        width,
    )
    mask_stack = mask_stack.reshape(
        batch_size * mask_width ** 2,
        1,
        height,
        width,
    )

    return masked_stack, mask_stack


def aggregate_masked_predictions(
    predictions: torch.Tensor,
    masks: torch.Tensor,
    original_batch_size: int,
    mask_width: int,
) -> torch.Tensor:
    """
    将 16 个 masked prediction 在各自 blind spots 上取值，
    拼回完整图像。
    """

    _, channels, height, width = predictions.shape

    predictions = (
        predictions * masks
    ).reshape(
        original_batch_size,
        mask_width ** 2,
        channels,
        height,
        width,
    )

    return predictions.sum(dim=1)


# ============================================================
# Beta 调度与 Blind2Unblind loss
# ============================================================

def calculate_beta(epoch: int) -> float:
    progress = epoch / CFG.num_epochs

    if progress <= CFG.beta_start_ratio:
        return CFG.lambda2

    if progress <= CFG.beta_end_ratio:
        return (
            CFG.lambda2
            + (
                progress - CFG.beta_start_ratio
            )
            * (
                CFG.increase_ratio - CFG.lambda2
            )
            / (
                CFG.beta_end_ratio
                - CFG.beta_start_ratio
            )
        )

    return CFG.increase_ratio


def blind2unblind_loss(
    model: nn.Module,
    noisy_patch: torch.Tensor,
    beta: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    batch_size = noisy_patch.shape[0]

    masked_inputs, masks = make_all_masked_inputs(
        image=noisy_patch,
        mask_width=CFG.mask_width,
    )

    masked_raw_outputs = model(masked_inputs)

    masked_prediction = aggregate_masked_predictions(
        predictions=masked_raw_outputs,
        masks=masks,
        original_batch_size=batch_size,
        mask_width=CFG.mask_width,
    )

    diff = masked_prediction - noisy_patch

    # 官方实现中 full visible branch 不反向传播
    with torch.no_grad():
        full_visible_prediction = model(noisy_patch)
        exp_diff = full_visible_prediction - noisy_patch

    re_visible_residual = (
        diff + beta * exp_diff
    )

    loss_regularization = (
        CFG.lambda1 * torch.mean(diff ** 2)
    )
    loss_re_visible = torch.mean(
        re_visible_residual ** 2
    )
    total_loss = (
        loss_regularization
        + loss_re_visible
    )

    return (
        total_loss,
        loss_regularization,
        loss_re_visible,
        torch.mean(diff ** 2),
        torch.mean(exp_diff ** 2),
    )


# ============================================================
# 推理：内存安全的 16-mask aggregation
# ============================================================

def pad_to_multiple(
    image: torch.Tensor,
    multiple: int,
) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    _, _, height, width = image.shape

    pad_height = (
        multiple - height % multiple
    ) % multiple
    pad_width = (
        multiple - width % multiple
    ) % multiple

    top = pad_height // 2
    bottom = pad_height - top
    left = pad_width // 2
    right = pad_width - left

    if pad_height == 0 and pad_width == 0:
        return image, (0, 0, 0, 0)

    return (
        F.pad(
            image,
            (left, right, top, bottom),
            mode="reflect",
        ),
        (left, right, top, bottom),
    )


def remove_padding(
    image: torch.Tensor,
    padding: tuple[int, int, int, int],
) -> torch.Tensor:
    left, right, top, bottom = padding

    height_end = (
        image.shape[-2] - bottom
        if bottom > 0
        else image.shape[-2]
    )
    width_end = (
        image.shape[-1] - right
        if right > 0
        else image.shape[-1]
    )

    return image[
        :,
        :,
        top:height_end,
        left:width_end,
    ]


@torch.inference_mode()
def predict_masked_complete_image(
    model: nn.Module,
    image: torch.Tensor,
    amp_enabled: bool,
) -> torch.Tensor:
    """
    不一次性创建 16 张完整 512×512 图，
    而是分 chunk 推理，结果等价但更省显存。
    """

    batch_size, channels, height, width = image.shape

    aggregated = torch.zeros(
        (
            batch_size,
            channels,
            height,
            width,
        ),
        device=image.device,
        dtype=torch.float32,
    )

    total_positions = CFG.mask_width ** 2
    chunk_size = max(
        1,
        CFG.inference_mask_chunk_size,
    )

    for chunk_start in range(
        0,
        total_positions,
        chunk_size,
    ):
        chunk_end = min(
            chunk_start + chunk_size,
            total_positions,
        )

        chunk_inputs: list[torch.Tensor] = []
        chunk_masks: list[torch.Tensor] = []

        for position in range(
            chunk_start,
            chunk_end,
        ):
            mask = make_fixed_mask(
                batch_size=batch_size,
                height=height,
                width=width,
                mask_width=CFG.mask_width,
                position=position,
                device=image.device,
                dtype=image.dtype,
            )

            chunk_inputs.append(
                interpolate_mask(image, mask)
            )
            chunk_masks.append(mask)

        inputs = torch.cat(
            chunk_inputs,
            dim=0,
        )
        masks = torch.cat(
            chunk_masks,
            dim=0,
        )

        with torch.autocast(
            device_type=image.device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            outputs = model(inputs)

        outputs = outputs.float()
        masked_outputs = outputs * masks.float()

        positions_in_chunk = (
            chunk_end - chunk_start
        )

        masked_outputs = masked_outputs.reshape(
            positions_in_chunk,
            batch_size,
            channels,
            height,
            width,
        )

        aggregated += masked_outputs.sum(dim=0)

        del inputs, masks, outputs, masked_outputs

    return aggregated


@torch.inference_mode()
def predict_blind2unblind(
    model: nn.Module,
    image: torch.Tensor,
    beta: float,
    amp_enabled: bool,
) -> torch.Tensor:
    masked_prediction = predict_masked_complete_image(
        model=model,
        image=image,
        amp_enabled=amp_enabled,
    )

    with torch.autocast(
        device_type=image.device.type,
        dtype=torch.float16,
        enabled=amp_enabled,
    ):
        full_visible_prediction = model(image)

    output = (
        masked_prediction
        + beta * full_visible_prediction.float()
    ) / (
        1.0 + beta
    )

    return output


# ============================================================
# 视频推理
# ============================================================

def load_full_gray_video(
    path: Path,
) -> tuple[np.ndarray, float]:
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


def create_video_writer(
    output_path: Path,
    fps: float,
    width: int,
    height: int,
) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(
        *CFG.output_codec
    )

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
    beta: float,
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

    required_multiple = 2 ** CFG.network_depth

    try:
        for batch_start in tqdm(
            range(
                0,
                frame_count,
                CFG.inference_batch_size,
            ),
            desc=f"去噪 {video_path.name}",
            leave=False,
        ):
            batch_end = min(
                batch_start
                + CFG.inference_batch_size,
                frame_count,
            )

            batch_array = frames[
                batch_start:batch_end
            ].astype(np.float32) / 255.0

            input_tensor = torch.from_numpy(
                batch_array
            ).unsqueeze(1).to(
                device=device,
                non_blocking=True,
            )

            padded_input, padding = pad_to_multiple(
                input_tensor,
                multiple=required_multiple,
            )

            output = predict_blind2unblind(
                model=model,
                image=padded_input,
                beta=beta,
                amp_enabled=amp_enabled,
            )

            output = remove_padding(
                output,
                padding,
            )
            output = output.clamp(0.0, 1.0)

            output_uint8 = (
                output[:, 0]
                .mul(255.0)
                .round()
                .byte()
                .cpu()
                .numpy()
            )

            for gray_frame in output_uint8:
                writer.write(
                    cv2.cvtColor(
                        gray_frame,
                        cv2.COLOR_GRAY2BGR,
                    )
                )

            del input_tensor, padded_input, output

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
    beta: float,
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
        f"去噪全部 {len(videos)} 个视频 | "
        f"beta={beta:.4f}"
    )

    for video_index, video in enumerate(
        videos,
        start=1,
    ):
        output_path = epoch_output_folder / (
            f"{video.path.stem}_denoised.avi"
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
                beta=beta,
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
) -> tuple[float, float, float, float, float]:
    model.train()

    beta = calculate_beta(epoch)

    total_loss_sum = 0.0
    regularization_loss_sum = 0.0
    re_visible_loss_sum = 0.0
    diff_mse_sum = 0.0
    exp_diff_mse_sum = 0.0
    processed_samples = 0

    progress_bar = tqdm(
        loader,
        desc=f"Epoch {epoch:03d}/{CFG.num_epochs:03d}",
    )

    for noisy_patch in progress_bar:
        noisy_patch = noisy_patch.to(
            device=device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            (
                loss,
                loss_regularization,
                loss_re_visible,
                diff_mse,
                exp_diff_mse,
            ) = blind2unblind_loss(
                model=model,
                noisy_patch=noisy_patch,
                beta=beta,
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

        batch_size = noisy_patch.shape[0]

        total_loss_sum += float(
            loss.detach().item()
        ) * batch_size
        regularization_loss_sum += float(
            loss_regularization.detach().item()
        ) * batch_size
        re_visible_loss_sum += float(
            loss_re_visible.detach().item()
        ) * batch_size
        diff_mse_sum += float(
            diff_mse.detach().item()
        ) * batch_size
        exp_diff_mse_sum += float(
            exp_diff_mse.detach().item()
        ) * batch_size

        processed_samples += batch_size

        denominator = max(
            processed_samples,
            1,
        )

        progress_bar.set_postfix(
            total=(
                f"{total_loss_sum / denominator:.6f}"
            ),
            reg=(
                f"{regularization_loss_sum / denominator:.6f}"
            ),
            rev=(
                f"{re_visible_loss_sum / denominator:.6f}"
            ),
            beta=f"{beta:.3f}",
        )

    if processed_samples == 0:
        raise RuntimeError(
            "本 epoch 没有成功训练任何样本。"
        )

    return (
        total_loss_sum / processed_samples,
        regularization_loss_sum / processed_samples,
        re_visible_loss_sum / processed_samples,
        diff_mse_sum / processed_samples,
        exp_diff_mse_sum / processed_samples,
    )


# ============================================================
# 主程序
# ============================================================

def main() -> None:
    set_seed(CFG.seed)

    required_multiple = (
        CFG.mask_width
        * 2 ** CFG.network_depth
    )

    if CFG.patch_size % CFG.mask_width != 0:
        raise ValueError(
            "patch_size 必须能被 mask_width 整除。"
        )

    if CFG.patch_size % (2 ** CFG.network_depth) != 0:
        raise ValueError(
            "patch_size 必须能被网络下采样倍数整除。"
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

    print("=" * 76)
    print("Pure Blind2Unblind LDH Training")
    print("=" * 76)
    print(f"设备：{device}")

    if torch.cuda.is_available():
        print(
            "GPU："
            f"{torch.cuda.get_device_name(0)}"
        )

    print(f"输入文件夹：{CFG.video_folder}")
    print(f"输出文件夹：{CFG.output_folder}")
    print(f"Epoch：{CFG.num_epochs}")
    print(
        f"每个 epoch 随机 frame patch 数："
        f"{CFG.samples_per_epoch}"
    )
    print(
        f"Patch：{CFG.patch_size}×{CFG.patch_size}"
    )
    print(f"Batch size：{CFG.batch_size}")
    print(f"Learning rate：{CFG.learning_rate}")
    print(
        f"Mask：{CFG.mask_width}×{CFG.mask_width}，"
        f"共 {CFG.mask_width ** 2} 个 blind-spot positions"
    )
    print(
        f"Lambda1={CFG.lambda1} | "
        f"Lambda2={CFG.lambda2} | "
        f"increase_ratio={CFG.increase_ratio}"
    )
    print(
        f"从 epoch {CFG.denoise_start_epoch} 开始，"
        f"每隔 {CFG.denoise_interval} 个 epoch "
        "去噪全部视频"
    )
    print(
        f"网络要求输入尺寸可被 "
        f"{2 ** CFG.network_depth} 整除；"
        f"当前 patch 合法。"
    )
    print("=" * 76)

    videos = collect_video_information()

    print(f"有效视频数量：{len(videos)}")

    for index, video in enumerate(
        videos,
        start=1,
    ):
        print(
            f"{index:03d}. {video.path.name} | "
            f"{video.frame_count} 帧 | "
            f"{video.width}×{video.height} | "
            f"{video.fps:.3f} FPS"
        )

    dataset = RandomVideoFrameDataset(
        videos=videos,
        samples_per_epoch=CFG.samples_per_epoch,
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

    model = Blind2UnblindUNet(
        in_channels=1,
        out_channels=1,
        depth=CFG.network_depth,
        width=CFG.network_width,
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
        beta = calculate_beta(epoch)

        (
            epoch_loss,
            epoch_regularization_loss,
            epoch_re_visible_loss,
            epoch_diff_mse,
            epoch_exp_diff_mse,
        ) = train_one_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            amp_enabled=amp_enabled,
        )

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
            f"Total={epoch_loss:.8f} | "
            f"Reg={epoch_regularization_loss:.8f} | "
            f"ReVisible={epoch_re_visible_loss:.8f} | "
            f"diff={epoch_diff_mse:.8f} | "
            f"exp_diff={epoch_exp_diff_mse:.8f} | "
            f"beta={beta:.4f}"
        )
        print(f"模型已保存：{model_path}")

        if (
            epoch >= CFG.denoise_start_epoch
            and epoch % CFG.denoise_interval == 0
        ):
            denoise_all_videos(
                model=model,
                videos=videos,
                epoch=epoch,
                device=device,
                amp_enabled=amp_enabled,
                beta=beta,
            )

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("=" * 76)
    print("全部 Blind2Unblind 训练与视频去噪完成。")
    print(f"结果位置：{CFG.output_folder}")
    print("=" * 76)


if __name__ == "__main__":
    main()
  