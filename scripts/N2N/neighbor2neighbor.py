# -*- coding: utf-8 -*-
r"""
Pure Neighbor2Neighbor baseline for LDH videos

训练条件与给定 UDVD 代码保持一致：
1. 输入文件夹：
   C:\Users\Novovorontsovka\Downloads\video_masqued
2. 训练 100 epochs。
3. 每个 epoch 随机抽取 3000 个单帧样本。
4. 每个样本随机裁剪一个 96×96 patch。
5. batch size = 8。
6. Adam，learning rate = 1e-4。
7. seed = 2026，AMP=True，max_grad_norm=5.0。
8. 每个 epoch 保存模型。
9. 从 epoch 5 开始，每隔 5 个 epoch 去噪全部视频。
10. 只保存 epoch 模型和去噪视频。

Neighbor2Neighbor 方法本身：
- 不使用连续帧、不使用 phase table、不使用同相位帧。
- 从同一个 noisy patch 的每个 2×2 邻域随机选择一对相邻像素，
  生成两个 48×48 neighbor sub-images。
- 使用 reconstruction loss + Neighbor2Neighbor regularization loss。
- 推理时直接输入完整单帧，输出尺寸不变。
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
        r"C:\Users\Novovorontsovka\Downloads\neighbor2neighbor"
    )

    video_extensions: tuple[str, ...] = (
        ".avi", ".mp4", ".mov", ".mkv", ".mpg", ".mpeg"
    )

    # 与 UDVD 一致
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

    # Neighbor2Neighbor 方法参数
    gamma: float = 1.0

    # 断点继续
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


# ============================================================
# 读取单帧
# ============================================================

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

    每个 epoch 固定返回 3000 个 noisy patches。
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
                frame = read_gray_frame(
                    video,
                    frame_index,
                )
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

            # 与 UDVD 一样的数据增强
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
# Neighbor2Neighbor random neighbor sub-sampler
# ============================================================

def generate_neighbor_pair_indices(
    batch_size: int,
    half_height: int,
    half_width: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    每个 2×2 cell 内有 4 个位置：
        0: 左上
        1: 右上
        2: 左下
        3: 右下

    只从水平或垂直相邻的 8 个有向 pair 中随机选择：
        0->1, 1->0, 0->2, 2->0,
        1->3, 3->1, 2->3, 3->2
    """

    neighbor_pairs = torch.tensor(
        [
            [0, 1],
            [1, 0],
            [0, 2],
            [2, 0],
            [1, 3],
            [3, 1],
            [2, 3],
            [3, 2],
        ],
        dtype=torch.long,
        device=device,
    )

    random_pair_ids = torch.randint(
        low=0,
        high=len(neighbor_pairs),
        size=(batch_size, half_height, half_width),
        device=device,
    )

    selected_pairs = neighbor_pairs[random_pair_ids]

    return (
        selected_pairs[..., 0],
        selected_pairs[..., 1],
    )


def neighbor_subsample(
    image: torch.Tensor,
    index_map: torch.Tensor,
) -> torch.Tensor:
    """
    image:     [B,C,H,W]
    index_map: [B,H/2,W/2]，每个位置取 0~3 中的一个像素

    return:    [B,C,H/2,W/2]
    """

    if image.ndim != 4:
        raise ValueError(
            f"image 必须是 [B,C,H,W]，实际为 {image.shape}"
        )

    batch_size, channels, height, width = image.shape

    if height % 2 != 0 or width % 2 != 0:
        raise ValueError(
            "Neighbor2Neighbor 子采样要求高度和宽度为偶数。"
        )

    half_height = height // 2
    half_width = width // 2

    # [B,C,H/2,W/2,4]
    cells = image.reshape(
        batch_size,
        channels,
        half_height,
        2,
        half_width,
        2,
    )
    cells = cells.permute(
        0, 1, 2, 4, 3, 5
    ).reshape(
        batch_size,
        channels,
        half_height,
        half_width,
        4,
    )

    gather_indices = index_map.unsqueeze(1).unsqueeze(-1)
    gather_indices = gather_indices.expand(
        -1,
        channels,
        -1,
        -1,
        1,
    )

    return torch.gather(
        cells,
        dim=-1,
        index=gather_indices,
    ).squeeze(-1)


def make_neighbor_training_pair(
    noisy_image: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    batch_size, _, height, width = noisy_image.shape

    index_1, index_2 = generate_neighbor_pair_indices(
        batch_size=batch_size,
        half_height=height // 2,
        half_width=width // 2,
        device=noisy_image.device,
    )

    subimage_1 = neighbor_subsample(
        noisy_image,
        index_1,
    )
    subimage_2 = neighbor_subsample(
        noisy_image,
        index_2,
    )

    return (
        subimage_1,
        subimage_2,
        index_1,
        index_2,
    )


# ============================================================
# 2D U-Net denoiser
# ============================================================

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            nn.LeakyReLU(
                negative_slope=0.1,
                inplace=True,
            ),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            nn.LeakyReLU(
                negative_slope=0.1,
                inplace=True,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class NeighborUNet(nn.Module):
    """
    单帧灰度 U-Net。
    训练时输入 48×48 neighbor sub-image；
    推理时可直接输入完整 512×512 frame。
    """

    def __init__(self):
        super().__init__()

        self.enc1 = ConvBlock(1, 48)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(48, 96)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(96, 192)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(192, 384)

        self.up3 = nn.ConvTranspose2d(
            384, 192, kernel_size=2, stride=2
        )
        self.dec3 = ConvBlock(384, 192)

        self.up2 = nn.ConvTranspose2d(
            192, 96, kernel_size=2, stride=2
        )
        self.dec2 = ConvBlock(192, 96)

        self.up1 = nn.ConvTranspose2d(
            96, 48, kernel_size=2, stride=2
        )
        self.dec1 = ConvBlock(96, 48)

        # 原论文在网络末端增加 1×1 conv layers
        self.head = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 32, kernel_size=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    @staticmethod
    def _match_size(
        x: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        difference_y = reference.size(2) - x.size(2)
        difference_x = reference.size(3) - x.size(3)

        if difference_y == 0 and difference_x == 0:
            return x

        return F.pad(
            x,
            [
                difference_x // 2,
                difference_x - difference_x // 2,
                difference_y // 2,
                difference_y - difference_y // 2,
            ],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip1 = self.enc1(x)
        x = self.pool1(skip1)

        skip2 = self.enc2(x)
        x = self.pool2(skip2)

        skip3 = self.enc3(x)
        x = self.pool3(skip3)

        x = self.bottleneck(x)

        x = self.up3(x)
        x = self._match_size(x, skip3)
        x = self.dec3(
            torch.cat([x, skip3], dim=1)
        )

        x = self.up2(x)
        x = self._match_size(x, skip2)
        x = self.dec2(
            torch.cat([x, skip2], dim=1)
        )

        x = self.up1(x)
        x = self._match_size(x, skip1)
        x = self.dec1(
            torch.cat([x, skip1], dim=1)
        )

        return self.head(x)


def initialize_weights(module: nn.Module) -> None:
    if isinstance(
        module,
        (nn.Conv2d, nn.ConvTranspose2d),
    ):
        nn.init.kaiming_normal_(
            module.weight,
            a=0.1,
            mode="fan_in",
            nonlinearity="leaky_relu",
        )

        if module.bias is not None:
            nn.init.zeros_(module.bias)


# ============================================================
# Neighbor2Neighbor loss
# ============================================================

def neighbor2neighbor_loss(
    model: nn.Module,
    noisy_patch: torch.Tensor,
    gamma: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    L_rec = MSE(f(g1(y)), g2(y))

    L_reg = MSE(
        f(g1(y)) - g2(y),
        g1(f(y)) - g2(f(y))
    )

    f(y) 在 regularization target 分支中 stop-gradient。
    """

    (
        subimage_1,
        subimage_2,
        index_1,
        index_2,
    ) = make_neighbor_training_pair(noisy_patch)

    prediction_subimage = model(subimage_1)

    reconstruction_loss = F.mse_loss(
        prediction_subimage,
        subimage_2,
    )

    with torch.no_grad():
        full_prediction = model(noisy_patch)

        predicted_subimage_1 = neighbor_subsample(
            full_prediction,
            index_1,
        )
        predicted_subimage_2 = neighbor_subsample(
            full_prediction,
            index_2,
        )

        regularization_target = (
            predicted_subimage_1
            - predicted_subimage_2
        )

    regularization_residual = (
        prediction_subimage
        - subimage_2
    )

    regularization_loss = F.mse_loss(
        regularization_residual,
        regularization_target,
    )

    total_loss = (
        reconstruction_loss
        + gamma * regularization_loss
    )

    return (
        total_loss,
        reconstruction_loss,
        regularization_loss,
    )


# ============================================================
# 推理
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


def pad_to_multiple(
    image: torch.Tensor,
    multiple: int = 8,
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

    padded = F.pad(
        image,
        (left, right, top, bottom),
        mode="reflect",
    )

    return padded, (left, right, top, bottom)


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
                multiple=8,
            )

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                output = model(padded_input)

            output = remove_padding(
                output,
                padding,
            )
            output = output.float().clamp(0.0, 1.0)

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
) -> tuple[float, float, float]:
    model.train()

    total_loss_sum = 0.0
    reconstruction_loss_sum = 0.0
    regularization_loss_sum = 0.0
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
                reconstruction_loss,
                regularization_loss,
            ) = neighbor2neighbor_loss(
                model=model,
                noisy_patch=noisy_patch,
                gamma=CFG.gamma,
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

        total_loss_sum += (
            float(loss.detach().item())
            * batch_size
        )
        reconstruction_loss_sum += (
            float(reconstruction_loss.detach().item())
            * batch_size
        )
        regularization_loss_sum += (
            float(regularization_loss.detach().item())
            * batch_size
        )
        processed_samples += batch_size

        progress_bar.set_postfix(
            total=(
                f"{total_loss_sum / max(processed_samples, 1):.6f}"
            ),
            rec=(
                f"{reconstruction_loss_sum / max(processed_samples, 1):.6f}"
            ),
            reg=(
                f"{regularization_loss_sum / max(processed_samples, 1):.6f}"
            ),
        )

    if processed_samples == 0:
        raise RuntimeError(
            "本 epoch 没有成功训练任何样本。"
        )

    return (
        total_loss_sum / processed_samples,
        reconstruction_loss_sum / processed_samples,
        regularization_loss_sum / processed_samples,
    )


# ============================================================
# 主程序
# ============================================================

def main() -> None:
    set_seed(CFG.seed)

    if CFG.patch_size % 8 != 0:
        raise ValueError(
            "patch_size 必须能被 8 整除。"
        )

    if CFG.patch_size % 2 != 0:
        raise ValueError(
            "Neighbor2Neighbor patch_size 必须是偶数。"
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
    print("Pure Neighbor2Neighbor LDH Training")
    print("=" * 70)
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
    print(f"原始 Patch：{CFG.patch_size}×{CFG.patch_size}")
    print(
        "Neighbor 子图："
        f"{CFG.patch_size // 2}×{CFG.patch_size // 2}"
    )
    print(f"Batch size：{CFG.batch_size}")
    print(f"Learning rate：{CFG.learning_rate}")
    print(f"Gamma：{CFG.gamma}")
    print(
        f"从 epoch {CFG.denoise_start_epoch} 开始，"
        f"每隔 {CFG.denoise_interval} 个 epoch 去噪全部视频"
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

    model = NeighborUNet()
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
        (
            epoch_loss,
            epoch_reconstruction_loss,
            epoch_regularization_loss,
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
            f"Rec={epoch_reconstruction_loss:.8f} | "
            f"Reg={epoch_regularization_loss:.8f}"
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
            )

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("=" * 70)
    print("全部 Neighbor2Neighbor 训练与视频去噪完成。")
    print(f"结果位置：{CFG.output_folder}")
    print("=" * 70)


if __name__ == "__main__":
    main()
   