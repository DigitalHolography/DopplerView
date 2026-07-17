# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 14:51:49 2026

@author: solei

训练逻辑：
- 一个 dataset sample = 同一视频中连续 10 帧。
- 一个 batch = 2 条完整的 10 帧 sequence。
- 前 9 帧只用于建立 ConvLSTM 的时间记忆。
- 最后一帧才是当前 target frame。
- 每次预测最后一帧时，ConvLSTM 看：
  [t-9, t-8, ..., t-2, t-1, t]
- 只有最后第 t 帧会被遮挡 block。
- 只有最后第 t 帧遮挡的位置会计算 loss。
- 每个 sequence 都重新开始 state=None。
- 验证 sample 在训练开始前随机抽取一次，然后固定不变。
"""

from pathlib import Path
import cv2
import numpy as np
import winsound
from torch.utils.data import Dataset
import torch
import torch.nn as nn


# ============================================================
# 1. 放文件夹
# ============================================================

VIDEO_DIR = Path(
    r"C:\Users\Novovorontsovka\Downloads\LDH_pipeline_output\05_trimmed_videos"
)

BRIGHTNESS_DIR = Path(
    r"C:\Users\Novovorontsovka\Downloads\LDH_pipeline_output\04_brightness_tables"
)

VESSEL_MASK_DIR = Path(
    r"C:\Users\Novovorontsovka\Downloads\LDH_pipeline_output\03_vessel_masks"
)

# 只接受 512×512 视频；其他尺寸直接跳过。
EXPECTED_HEIGHT = 512
EXPECTED_WIDTH = 512

# 最多只使用 200 个通过检查的有效视频。
# 注意：程序会先跳过空视频、非 512×512 视频和帧数不足的视频，
# 然后累计到 200 个有效视频后停止继续加载。
MAX_VALID_VIDEOS = 200


# ============================================================
# 2. 从文件名中取出视频编号
# ============================================================

def get_video_number(video_path):

    parts = video_path.stem.split("_")

    for part in reversed(parts):

        if part.isdigit():

            return int(part)

    raise RuntimeError(
        f"文件名里找不到视频编号：\n{video_path.name}"
    )


# ============================================================
# 3. 找到文件夹里的全部 AVI 视频，并按编号排序
# ============================================================

def find_all_avi_videos(video_dir):

    video_dir = Path(video_dir)

    video_paths = list(
        video_dir.glob("*.avi")
    )

    video_paths += list(
        video_dir.glob("*.AVI")
    )

    # 按完整文件名排序。
    # 不再使用文件名里的 _7_p、_8_p 等局部编号排序，
    # 因为不同病人/不同采集的视频可能具有相同的局部编号。
    video_paths = sorted(
        video_paths,
        key=lambda path: path.name.lower()
    )

    return video_paths


# ============================================================
# 4. 读取一个视频，得到全部灰度帧
# ============================================================

def load_video_as_gray(video_path):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    frames_gray = []

    while True:

        ret, frame_bgr = cap.read()

        if not ret:
            break

        frame_gray = cv2.cvtColor(
            frame_bgr,
            cv2.COLOR_BGR2GRAY
        )

        frames_gray.append(
            frame_gray
        )

    cap.release()

    if len(frames_gray) == 0:

        return np.empty(
            (0, 0, 0),
            dtype=np.uint8
        )

    frames_gray = np.stack(
        frames_gray,
        axis=0
    )

    return frames_gray


# ============================================================
# 4.2 读取一个视频对应的平滑亮度和 phase table
# ============================================================

def create_circle_mask(H, W, cx=255, cy=255, r=260):

    valid_mask = np.zeros(
        (H, W),
        dtype=np.uint8
    )

    cv2.circle(
        valid_mask,
        (cx, cy),
        r,
        255,
        thickness=-1
    )

    return valid_mask


def get_smooth_brightness_path(
        video_path,
        brightness_dir
):

    video_path = Path(video_path)
    brightness_dir = Path(brightness_dir)

    training_video_stem = video_path.stem

    suffix = "_from_first_peak"

    if not training_video_stem.endswith(suffix):

        raise RuntimeError(
            "训练视频文件名不符合预期：\n"
            f"{video_path.name}\n\n"
            "训练视频必须以 "
            "_from_first_peak.avi 结尾。"
        )

    original_video_stem = training_video_stem[:-len(suffix)]

    smooth_brightness_path = brightness_dir / (
        original_video_stem + "_smooth_brightness.npy"
    )

    if not smooth_brightness_path.exists():

        raise FileNotFoundError(
            "找不到这个训练视频对应的 "
            "smooth brightness table：\n"
            f"{smooth_brightness_path}"
        )

    return smooth_brightness_path



# ============================================================
# 4.3 读取每个视频对应的固定血管 mask
# ============================================================

def _normalise_stem_for_mask(stem):

    suffixes_to_remove = [
        "_from_first_peak",
        "_circle_masked"
    ]

    changed = True

    while changed:

        changed = False

        for suffix in suffixes_to_remove:

            if stem.endswith(suffix):

                stem = stem[:-len(suffix)]
                changed = True

    return stem


def find_vessel_mask_path(
        video_path,
        vessel_mask_dir
):

    """
    为每个训练视频找到对应的固定血管 mask。

    支持 .npy / .png / .jpg / .tif / .tiff。
    文件名只要包含该视频的基础名字，并包含 mask、vessel 或 overlap
    之一即可。例如：
        R_..._9_p_vessel_mask.png
        R_..._9_p_final_overlap_matrix.npy

    若你的 mask 名称不同，程序会明确报出候选文件，方便只改这里。
    """

    video_path = Path(video_path)
    vessel_mask_dir = Path(vessel_mask_dir)

    if not vessel_mask_dir.exists():

        raise FileNotFoundError(
            "找不到 VESSEL_MASK_DIR：\n"
            f"{vessel_mask_dir}"
        )

    base_stem = _normalise_stem_for_mask(
        video_path.stem
    ).lower()

    allowed_suffixes = {
        ".npy",
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
        ".bmp"
    }

    candidates = []

    for candidate in vessel_mask_dir.iterdir():

        if not candidate.is_file():
            continue

        if candidate.suffix.lower() not in allowed_suffixes:
            continue

        candidate_name = candidate.stem.lower()

        if base_stem not in candidate_name:
            continue

        if (
                "vessel" not in candidate_name
                and "mask" not in candidate_name
                and "overlap" not in candidate_name
        ):
            continue

        score = 0

        if "final_overlap" in candidate_name:
            score += 30

        if "vessel_mask" in candidate_name:
            score += 25

        if "vessel" in candidate_name:
            score += 10

        if "mask" in candidate_name:
            score += 8

        if candidate.suffix.lower() == ".npy":
            score += 3

        candidates.append(
            (score, candidate)
        )

    if len(candidates) == 0:

        all_mask_like_files = [
            path.name
            for path in vessel_mask_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in allowed_suffixes
        ]

        raise FileNotFoundError(
            "找不到这个训练视频对应的 vessel mask。\n\n"
            f"视频：{video_path.name}\n"
            f"推导出的基础名字：{base_stem}\n"
            f"mask 文件夹：{vessel_mask_dir}\n\n"
            "请把对应 mask 命名为类似：\n"
            f"{base_stem}_vessel_mask.png\n"
            "或：\n"
            f"{base_stem}_final_overlap_matrix.npy\n\n"
            "当前文件夹内可见的 mask 类文件：\n"
            + "\n".join(all_mask_like_files[:30])
        )

    candidates.sort(
        key=lambda item: (-item[0], item[1].name)
    )

    return candidates[0][1]


def load_vessel_mask(
        vessel_mask_path,
        height,
        width
):

    vessel_mask_path = Path(vessel_mask_path)

    if vessel_mask_path.suffix.lower() == ".npy":

        vessel_mask = np.load(
            vessel_mask_path
        )

        # 兼容保存为 [1, H, W]、[H, W, 1] 或 boolean 的情况。
        vessel_mask = np.squeeze(
            vessel_mask
        )

    else:

        vessel_mask = cv2.imread(
            str(vessel_mask_path),
            cv2.IMREAD_GRAYSCALE
        )

        if vessel_mask is None:

            raise RuntimeError(
                "无法读取 vessel mask：\n"
                f"{vessel_mask_path}"
            )

    if vessel_mask.ndim != 2:

        raise RuntimeError(
            "vessel mask 必须是二维图像：\n"
            f"{vessel_mask_path}\n"
            f"实际 shape = {vessel_mask.shape}"
        )

    vessel_mask = vessel_mask.astype(
        np.float32
    )

    if vessel_mask.shape != (height, width):

        vessel_mask = cv2.resize(
            vessel_mask,
            (width, height),
            interpolation=cv2.INTER_NEAREST
        )

    # 支持 0/1、0/255 或 float probability mask。
    vessel_mask = (
        vessel_mask > 0
    ).astype(
        np.float32
    )

    return vessel_mask


# ============================================================
# 5. 多视频 Dataset
# ============================================================

class MultiVideoDataset(Dataset):

    def __init__(
            self,
            video_paths,
            sequence_length,
            brightness_dir,
            vessel_mask_dir
    ):

        super().__init__()

        self.video_paths = video_paths
        self.sequence_length = sequence_length
        self.brightness_dir = Path(brightness_dir)
        self.vessel_mask_dir = Path(vessel_mask_dir)

        self.all_frames = {}
        self.vessel_masks = {}
        self.video_paths_by_id = {}
        self.samples = []
        self.sample_indices_by_video = {}

        self.brightness_by_frame = {}
        self.phase_by_frame = {}
        self.phase_to_frames = {}

        # video_id 是按完整文件列表自动分配的唯一整数。
        # 即使多个文件名都包含 _7_p，它们也会得到不同的 video_id，
        # 因而不会在 self.all_frames / vessel_masks 等字典中互相覆盖。
        valid_video_count = 0

        for video_id, video_path in enumerate(self.video_paths):

            # 已经收集到 200 个有效视频后停止继续读取。
            if valid_video_count >= MAX_VALID_VIDEOS:
                print(
                    f"Reached MAX_VALID_VIDEOS={MAX_VALID_VIDEOS}. "
                    "Stop loading more videos."
                )
                break

            video_number = int(video_id)

            frames_gray = load_video_as_gray(video_path)

            # 空视频直接跳过。
            if frames_gray.ndim != 3 or len(frames_gray) == 0:

                print(
                    f"SKIP video_id={video_number}: empty or unreadable -> "
                    f"{video_path.name}"
                )
                continue

            frame_height = int(frames_gray.shape[1])
            frame_width = int(frames_gray.shape[2])

            # 不是 512×512 的视频不进入 dataset。
            if (
                    frame_height != EXPECTED_HEIGHT
                    or frame_width != EXPECTED_WIDTH
            ):

                print(
                    f"SKIP video_id={video_number}: "
                    f"size={frame_width}x{frame_height}, "
                    f"expected={EXPECTED_WIDTH}x{EXPECTED_HEIGHT} -> "
                    f"{video_path.name}"
                )
                continue

            total_frames = len(frames_gray)

            # 至少要能组成一条完整 sequence。
            if total_frames < self.sequence_length:

                print(
                    f"SKIP video_id={video_number}: "
                    f"only {total_frames} frames, "
                    f"need at least {self.sequence_length} -> "
                    f"{video_path.name}"
                )
                continue

            self.all_frames[video_number] = frames_gray
            self.video_paths_by_id[video_number] = video_path

            vessel_mask_path = find_vessel_mask_path(
                video_path=video_path,
                vessel_mask_dir=self.vessel_mask_dir
            )

            vessel_mask = load_vessel_mask(
                vessel_mask_path=vessel_mask_path,
                height=frames_gray.shape[1],
                width=frames_gray.shape[2]
            )

            self.vessel_masks[video_number] = vessel_mask

            print(
                f"video_id={video_number} | "
                f"file={video_path.name} | "
                f"vessel_mask={vessel_mask_path.name}"
            )

            smooth_brightness_path = get_smooth_brightness_path(
                video_path=video_path,
                brightness_dir=self.brightness_dir
            )

            smooth_table = np.load(
                smooth_brightness_path
            )

            frame_indices = smooth_table[0].astype(
                np.int32
            )

            smooth_brightness = smooth_table[1].astype(
                np.float32
            )

            phases = smooth_table[2].astype(
                np.int32
            )

            if len(frame_indices) != total_frames:

                raise RuntimeError(
                    f"video_id={video_number} 的视频帧数和 "
                    "brightness table 长度不一样：\n"
                    f"video frames = {total_frames}\n"
                    f"brightness table = {len(frame_indices)}"
                )

            brightness_dictionary = {}

            for frame_number, brightness_value in zip(
                    frame_indices,
                    smooth_brightness
            ):

                brightness_dictionary[
                    int(frame_number)
                ] = float(brightness_value)

            self.brightness_by_frame[
                video_number
            ] = brightness_dictionary

            phase_dictionary = {}

            for frame_number, phase_value in zip(
                    frame_indices,
                    phases
            ):

                phase_dictionary[
                    int(frame_number)
                ] = int(phase_value)

            self.phase_by_frame[
                video_number
            ] = phase_dictionary

            phase_to_frames_dictionary = {}

            unique_phases = np.unique(
                phases
            )

            for phase_value in unique_phases:

                same_phase_frames = frame_indices[
                    phases == phase_value
                ]

                phase_to_frames_dictionary[
                    int(phase_value)
                ] = same_phase_frames.astype(
                    np.int32
                )

            self.phase_to_frames[
                video_number
            ] = phase_to_frames_dictionary

            valid_start_count = (
                total_frames
                - self.sequence_length
                + 1
            )

            for start_frame_index in range(
                    valid_start_count
            ):

                dataset_sample_index = len(self.samples)

                self.samples.append(
                    (
                        video_number,
                        start_frame_index
                    )
                )

                self.sample_indices_by_video.setdefault(
                    video_number,
                    []
                ).append(
                    dataset_sample_index
                )

            valid_video_count += 1

        if len(self.samples) == 0:

            raise RuntimeError(
                "没有任何满足条件的训练视频。\n"
                f"要求视频尺寸为 {EXPECTED_WIDTH}x{EXPECTED_HEIGHT}，"
                f"且至少有 {self.sequence_length} 帧。"
            )

        print(
            f"Valid videos kept: {len(self.sample_indices_by_video)}"
            f"/{MAX_VALID_VIDEOS} | "
            f"Total sequence samples: {len(self.samples)}"
        )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        video_number, start_frame_index = self.samples[
            index
        ]

        frames_gray = self.all_frames[
            video_number
        ][
            start_frame_index:
            start_frame_index + self.sequence_length
        ]

        return (
            frames_gray,
            video_number,
            start_frame_index
        )


# ============================================================
# 一个 sample：
# 前 9 帧 history + 最后 1 帧 target
# ============================================================

HISTORY_FRAMES = 9

SEQUENCE_LENGTH = (
    HISTORY_FRAMES
    + 1
)

# 一个 batch 同时训练 2 条完整的 10 帧 sequence。
BATCH_SIZE = 2

video_paths = find_all_avi_videos(
    VIDEO_DIR
)

dataset = MultiVideoDataset(
    video_paths=video_paths,
    sequence_length=SEQUENCE_LENGTH,
    brightness_dir=BRIGHTNESS_DIR,
    vessel_mask_dir=VESSEL_MASK_DIR
)


# ============================================================
# 6. Hyperparameters : block hiding
# 每个 target 同时遮挡 32 个互不重叠的 32×32 block。
# ============================================================

BLOCK_SIZE = 32
LOSS_BLOCK_SIZE = 32
NUMBER_OF_BLOCKS = 32


# ============================================================
# 7. block hiding
# ============================================================

def replace_blocks_with_same_phase(
        frames_gray,
        video_numbers,
        start_frame_indices,
        dataset,
        target_time_index,
        block_size,
        number_of_blocks,
        max_block_position_tries,
        cx,
        cy,
        r
):

    original_sequence = frames_gray.float() / 255.0

    # 当前最后一帧 target 的原始图像。
    target_frame = original_sequence[
        :,
        target_time_index
    ].clone()

    # 前 9 帧保持原样。
    # 只替换最后 target frame 的 block。
    input_sequence = original_sequence.clone()

    block_mask = torch.zeros_like(
        target_frame
    )

    batch_size, _, height, width = (
        original_sequence.shape
    )

    circle_mask = create_circle_mask(
        height,
        width,
        cx,
        cy,
        r
    )

    loss_margin = (
        block_size
        - LOSS_BLOCK_SIZE
    ) // 2

    for batch_index in range(batch_size):

        video_number = int(
            video_numbers[batch_index]
        )

        start_frame_index = int(
            start_frame_indices[batch_index]
        )

        current_frame_number = (
            start_frame_index
            + target_time_index
        )

        current_phase = dataset.phase_by_frame[
            video_number
        ][
            current_frame_number
        ]

        same_phase_frames = dataset.phase_to_frames[
            video_number
        ][
            current_phase
        ]

        same_phase_frames = same_phase_frames[
            same_phase_frames != current_frame_number
        ]

        if len(same_phase_frames) == 0:
            continue

        current_brightness = dataset.brightness_by_frame[
            video_number
        ][
            current_frame_number
        ]

        for block_index in range(number_of_blocks):

            found_position = False

            for _ in range(
                    max_block_position_tries
            ):

                y0 = np.random.randint(
                    0,
                    height - block_size + 1
                )

                x0 = np.random.randint(
                    0,
                    width - block_size + 1
                )

                y1 = y0 + block_size
                x1 = x0 + block_size

                block_circle_mask = circle_mask[
                    y0:y1,
                    x0:x1
                ]

                if np.any(
                        block_circle_mask == 0
                ):
                    continue

                already_replaced = block_mask[
                    batch_index,
                    y0:y1,
                    x0:x1
                ].sum()

                if already_replaced > 0:
                    continue

                found_position = True

                break

            if not found_position:
                continue

            reference_frame_number = int(
                np.random.choice(
                    same_phase_frames
                )
            )

            reference_brightness = dataset.brightness_by_frame[
                video_number
            ][
                reference_frame_number
            ]

            if reference_brightness <= 1e-8:
                continue

            brightness_scale = (
                current_brightness
                / reference_brightness
            )

            reference_block = dataset.all_frames[
                video_number
            ][
                reference_frame_number,
                y0:y1,
                x0:x1
            ].astype(
                np.float32
            )

            reference_block = (
                reference_block
                * brightness_scale
            )

            reference_block = np.clip(
                reference_block,
                0,
                255
            )

            reference_block = torch.from_numpy(
                reference_block
            ).float() / 255.0

            # 只替换最后 target frame 的 block。
            input_sequence[
                batch_index,
                target_time_index,
                y0:y1,
                x0:x1
            ] = reference_block

            block_mask[
                batch_index,
                y0 + loss_margin:y1 - loss_margin,
                x0 + loss_margin:x1 - loss_margin
            ] = 1.0

    return (
        input_sequence,
        target_frame,
        block_mask
    )


# ============================================================
# 8. 网络
# ============================================================

class ConvBlock(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels
    ):

        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

        self.norm1 = nn.GroupNorm(
            8,
            out_channels
        )

        self.act1 = nn.SiLU()

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

        self.norm2 = nn.GroupNorm(
            8,
            out_channels
        )

        self.act2 = nn.SiLU()

    def forward(self, x):

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)

        return x


class Downsample(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels
    ):

        super().__init__()

        self.down = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1
        )

    def forward(self, x):

        return self.down(x)


class Upsample(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels
    ):

        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2
        )

    def forward(self, x):

        return self.up(x)


class ConvLSTMCell(nn.Module):

    def __init__(
            self,
            input_channels,
            hidden_channels
    ):

        super().__init__()

        self.hidden_channels = hidden_channels

        self.conv = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=3,
            padding=1
        )

    def forward(
            self,
            x,
            state=None
    ):

        batch_size, _, height, width = x.shape

        if state is None:

            h = torch.zeros(
                batch_size,
                self.hidden_channels,
                height,
                width,
                device=x.device,
                dtype=x.dtype
            )

            c = torch.zeros(
                batch_size,
                self.hidden_channels,
                height,
                width,
                device=x.device,
                dtype=x.dtype
            )

        else:

            h, c = state

        combined = torch.cat(
            [x, h],
            dim=1
        )

        gates = self.conv(
            combined
        )

        i, f, o, g = torch.chunk(
            gates,
            4,
            dim=1
        )

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c + i * g

        h_next = o * torch.tanh(
            c_next
        )

        return h_next, (
            h_next,
            c_next
        )


class UNetConvLSTM(nn.Module):

    def __init__(self):

        super().__init__()

        # 512 x 512
        self.enc1 = ConvBlock(
            1,
            32
        )

        # 256 x 256
        self.down1 = Downsample(
            32,
            64
        )

        self.enc2 = ConvBlock(
            64,
            64
        )

        # 128 x 128
        self.down2 = Downsample(
            64,
            128
        )

        self.enc3 = ConvBlock(
            128,
            128
        )

        # 64 x 64
        self.down3 = Downsample(
            128,
            256
        )

        self.enc4 = ConvBlock(
            256,
            256
        )

        # 32 x 32
        self.down4 = Downsample(
            256,
            512
        )

        self.enc5 = ConvBlock(
            512,
            512
        )

        # 16 x 16
        self.down5 = Downsample(
            512,
            512
        )

        self.enc6 = ConvBlock(
            512,
            512
        )

        self.bottleneck = ConvBlock(
            512,
            512
        )

        # ConvLSTM 只放在最深层 16 x 16 bottleneck。
        self.convlstm = ConvLSTMCell(
            input_channels=512,
            hidden_channels=512
        )

        self.dec6 = ConvBlock(
            1024,
            512
        )

        self.up5 = Upsample(
            512,
            512
        )

        self.dec5 = ConvBlock(
            1024,
            512
        )

        self.up4 = Upsample(
            512,
            256
        )

        self.dec4 = ConvBlock(
            512,
            256
        )

        self.up3 = Upsample(
            256,
            128
        )

        self.dec3 = ConvBlock(
            256,
            128
        )

        self.up2 = Upsample(
            128,
            64
        )

        self.dec2 = ConvBlock(
            128,
            64
        )

        self.up1 = Upsample(
            64,
            32
        )

        self.dec1 = ConvBlock(
            64,
            32
        )

        self.final_conv = nn.Conv2d(
            32,
            1,
            kernel_size=3,
            padding=1
        )

    def forward(
            self,
            x,
            state=None
    ):

        input_frame = x

        skip1 = self.enc1(x)

        x = self.down1(
            skip1
        )

        skip2 = self.enc2(x)

        x = self.down2(
            skip2
        )

        skip3 = self.enc3(x)

        x = self.down3(
            skip3
        )

        skip4 = self.enc4(x)

        x = self.down4(
            skip4
        )

        skip5 = self.enc5(x)

        x = self.down5(
            skip5
        )

        skip6 = self.enc6(x)

        x = self.bottleneck(
            skip6
        )

        # x 的 shape 是 [batch, 512, 16, 16]。
        x, state = self.convlstm(
            x,
            state
        )

        x = torch.cat(
            [x, skip6],
            dim=1
        )

        x = self.dec6(x)

        x = self.up5(x)

        x = torch.cat(
            [x, skip5],
            dim=1
        )

        x = self.dec5(x)

        x = self.up4(x)

        x = torch.cat(
            [x, skip4],
            dim=1
        )

        x = self.dec4(x)

        x = self.up3(x)

        x = torch.cat(
            [x, skip3],
            dim=1
        )

        x = self.dec3(x)

        x = self.up2(x)

        x = torch.cat(
            [x, skip2],
            dim=1
        )

        x = self.dec2(x)

        x = self.up1(x)

        x = torch.cat(
            [x, skip1],
            dim=1
        )

        x = self.dec1(x)

        predicted_noise = self.final_conv(
            x
        )

        denoised = (
            input_frame
            - predicted_noise
        )

        return denoised, state


# ============================================================
# 9. Loss
# ============================================================

def block_reconstruction_loss(
        prediction,
        target,
        block_mask,
        vessel_mask
):

    """
    真正的血管保护 loss。

    1. vessel-weighted reconstruction：
       血管和血管附近的像素比普通背景更重，细血管不会被背景数量淹没。

    2. vessel-gradient preservation：
       直接约束预测图和 target 在血管区域的横向/纵向梯度一致，
       抑制小血管被平滑、变宽或断裂。

    3. background Fair regularization：
       只在明确不是血管的 block 内背景做温和抑噪；
       不再对血管区域施加“变平滑”的压力。

    所有项都只在最后一帧被遮挡的 block 内计算。
    """

    EPSILON = 1e-8

    # --------------------------------------------------------
    # 可调超参数
    # --------------------------------------------------------
    VESSEL_DILATION_KERNEL = 3
    VESSEL_PIXEL_WEIGHT = 6.0
    LAMBDA_VESSEL_GRADIENT = 1.50
    LAMBDA_BACKGROUND_FAIR = 0.20
    FAIR_DELTA = 0.03

    # --------------------------------------------------------
    # 0. 细血管保护区域：原始 vessel mask 加一圈很窄的保护边界。
    #
    # 原始 mask 保护血管中心；
    # 3x3 膨胀同时保护细血管边缘，避免只学到“中间亮一条线”。
    # --------------------------------------------------------
    vessel_region = nn.functional.max_pool2d(
        vessel_mask,
        kernel_size=VESSEL_DILATION_KERNEL,
        stride=1,
        padding=VESSEL_DILATION_KERNEL // 2
    )

    vessel_region = torch.clamp(
        vessel_region,
        min=0.0,
        max=1.0
    )

    vessel_block_mask = (
        block_mask
        * vessel_region
    )

    background_block_mask = (
        block_mask
        * (
            1.0
            - vessel_region
        )
    )

    # --------------------------------------------------------
    # 1. 像素重建项：血管区域权重更高。
    # --------------------------------------------------------
    absolute_error = torch.abs(
        prediction
        - target
    )

    pixel_weight = (
        1.0
        + VESSEL_PIXEL_WEIGHT
        * vessel_region
    )

    weighted_data_loss_per_sample = (
        absolute_error
        * block_mask
        * pixel_weight
    ).sum(
        dim=(1, 2, 3)
    )

    # 单独记录血管/背景误差，便于观察是否真的在保护小血管。
    vessel_data_loss_per_sample = (
        absolute_error
        * vessel_block_mask
    ).sum(
        dim=(1, 2, 3)
    )

    background_data_loss_per_sample = (
        absolute_error
        * background_block_mask
    ).sum(
        dim=(1, 2, 3)
    )

    # --------------------------------------------------------
    # 2. 血管梯度保持项。
    #
    # 单纯像素 L1 容易允许“变成较平的一条灰线”；
    # 这里要求血管边缘的梯度也与 target 一致，
    # 对细小、弱血管尤其重要。
    # --------------------------------------------------------
    prediction_gradient_x = (
        prediction[:, :, :, 1:]
        - prediction[:, :, :, :-1]
    )

    target_gradient_x = (
        target[:, :, :, 1:]
        - target[:, :, :, :-1]
    )

    prediction_gradient_y = (
        prediction[:, :, 1:, :]
        - prediction[:, :, :-1, :]
    )

    target_gradient_y = (
        target[:, :, 1:, :]
        - target[:, :, :-1, :]
    )

    pair_block_mask_x = (
        block_mask[:, :, :, 1:]
        * block_mask[:, :, :, :-1]
    )

    pair_block_mask_y = (
        block_mask[:, :, 1:, :]
        * block_mask[:, :, :-1, :]
    )

    pair_vessel_region_x = torch.maximum(
        vessel_region[:, :, :, 1:],
        vessel_region[:, :, :, :-1]
    )

    pair_vessel_region_y = torch.maximum(
        vessel_region[:, :, 1:, :],
        vessel_region[:, :, :-1, :]
    )

    vessel_pair_mask_x = (
        pair_block_mask_x
        * pair_vessel_region_x
    )

    vessel_pair_mask_y = (
        pair_block_mask_y
        * pair_vessel_region_y
    )

    gradient_error_x = torch.abs(
        prediction_gradient_x
        - target_gradient_x
    )

    gradient_error_y = torch.abs(
        prediction_gradient_y
        - target_gradient_y
    )

    vessel_gradient_sum_per_sample = (
        gradient_error_x
        * vessel_pair_mask_x
    ).sum(
        dim=(1, 2, 3)
    ) + (
        gradient_error_y
        * vessel_pair_mask_y
    ).sum(
        dim=(1, 2, 3)
    )

    vessel_pair_count_per_sample = (
        vessel_pair_mask_x.sum(
            dim=(1, 2, 3)
        )
        + vessel_pair_mask_y.sum(
            dim=(1, 2, 3)
        )
    )

    # 先求 vessel gradient mean，再乘当前 sample 的 block 像素数。
    # 这样强度随 block 数量保持与原本 pixel-sum loss 同级，
    # 不会因为某条 sequence 的血管面积稍多/稍少而失控。
    masked_pixel_count_per_sample = block_mask.sum(
        dim=(1, 2, 3)
    )

    vessel_gradient_loss_per_sample = (
        vessel_gradient_sum_per_sample
        / (
            vessel_pair_count_per_sample
            + EPSILON
        )
        * masked_pixel_count_per_sample
    )

    # 若某个被遮挡区域恰好完全没有血管，梯度项自然为 0。
    vessel_gradient_loss_per_sample = torch.where(
        vessel_pair_count_per_sample > 0,
        vessel_gradient_loss_per_sample,
        torch.zeros_like(
            vessel_gradient_loss_per_sample
        )
    )

    # --------------------------------------------------------
    # 3. 背景 Fair regularization。
    #
    # 只在 background_block_mask 内平滑；
    # vessel_region 内完全没有 Fair 平滑惩罚。
    # --------------------------------------------------------
    background_pair_mask_x = (
        pair_block_mask_x
        * (
            1.0
            - pair_vessel_region_x
        )
    )

    background_pair_mask_y = (
        pair_block_mask_y
        * (
            1.0
            - pair_vessel_region_y
        )
    )

    def fair_penalty(values):

        absolute_values = torch.abs(
            values
        )

        return (
            FAIR_DELTA ** 2
            * (
                absolute_values
                / FAIR_DELTA
                - torch.log1p(
                    absolute_values
                    / FAIR_DELTA
                )
            )
        )

    background_fair_loss_per_sample = 0.5 * (
        (
            fair_penalty(
                prediction_gradient_x
            )
            * background_pair_mask_x
        ).sum(
            dim=(1, 2, 3)
        )
        + (
            fair_penalty(
                prediction_gradient_y
            )
            * background_pair_mask_y
        ).sum(
            dim=(1, 2, 3)
        )
    )

    # --------------------------------------------------------
    # 4. 总 loss
    # --------------------------------------------------------
    total_loss_per_sample = (
        weighted_data_loss_per_sample
        + LAMBDA_VESSEL_GRADIENT
        * vessel_gradient_loss_per_sample
        + LAMBDA_BACKGROUND_FAIR
        * background_fair_loss_per_sample
    )

    batch_total_loss = total_loss_per_sample.sum()

    return (
        batch_total_loss,
        weighted_data_loss_per_sample,
        vessel_data_loss_per_sample,
        background_data_loss_per_sample,
        vessel_gradient_loss_per_sample,
        background_fair_loss_per_sample,
        total_loss_per_sample,
        vessel_block_mask.sum(
            dim=(1, 2, 3)
        ),
        background_block_mask.sum(
            dim=(1, 2, 3)
        )
    )


# ============================================================
# 10. 训练配置与初始化
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "Using device:",
    DEVICE
)

if DEVICE.type == "cuda":

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

LEARNING_RATE = 5e-5
EPOCHS = 250

# 每个 epoch 抽取 8000 条 10-frame source sequence 来训练。
# 抽样时保证每个通过尺寸检查的视频至少贡献 1 条 sequence。
TRAIN_SAMPLES_PER_EPOCH = 8000

# 一开始随机选 20 条 source sequence 做 valid，之后不变。
VALID_SAMPLES = 20

EARLY_STOPPING_PATIENCE = 10
MAX_BLOCK_POSITION_TRIES = 100
SPLIT_RANDOM_SEED = 2026
VALIDATION_RANDOM_SEED = 10000
CX = 255
CY = 255
R = 260

model = UNetConvLSTM().to(
    DEVICE
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# 11. 手动取一个 batch
# ============================================================

def get_one_sequence_batch(
        dataset,
        sample_indices
):

    batch_frames_gray = []
    batch_video_numbers = []
    batch_start_frame_indices = []

    for sample_index in sample_indices:

        frames_gray, video_number, start_frame_index = dataset[
            int(sample_index)
        ]

        batch_frames_gray.append(
            frames_gray
        )

        batch_video_numbers.append(
            video_number
        )

        batch_start_frame_indices.append(
            start_frame_index
        )

    frames_gray = torch.from_numpy(
        np.stack(
            batch_frames_gray,
            axis=0
        )
    )

    video_numbers = torch.tensor(
        batch_video_numbers,
        dtype=torch.long
    )

    start_frame_indices = torch.tensor(
        batch_start_frame_indices,
        dtype=torch.long
    )

    return (
        frames_gray,
        video_numbers,
        start_frame_indices
    )



# ============================================================
# 11.5 取出一个 batch 对应的固定血管 mask
# ============================================================

def get_batch_vessel_masks(
        dataset,
        video_numbers
):

    vessel_masks = []

    for video_number in video_numbers.tolist():

        vessel_masks.append(
            dataset.vessel_masks[
                int(video_number)
            ]
        )

    vessel_masks = np.stack(
        vessel_masks,
        axis=0
    )

    vessel_masks = torch.from_numpy(
        vessel_masks
    ).float().unsqueeze(1)

    return vessel_masks


# ============================================================
# 12. 随机固定 train / valid split
# ============================================================

def create_train_valid_indices(
        dataset,
        valid_samples,
        split_random_seed
):

    all_sample_indices = np.arange(
        len(dataset),
        dtype=np.int64
    )

    random_generator = np.random.default_rng(
        split_random_seed
    )

    # 为每个有效视频固定保留至少 1 条 train sequence。
    reserved_train_indices = []

    for video_number in sorted(
            dataset.sample_indices_by_video.keys()
    ):

        video_sample_indices = np.asarray(
            dataset.sample_indices_by_video[video_number],
            dtype=np.int64
        )

        reserved_train_indices.append(
            int(random_generator.choice(video_sample_indices))
        )

    reserved_train_indices = np.asarray(
        reserved_train_indices,
        dtype=np.int64
    )

    valid_candidates = np.setdiff1d(
        all_sample_indices,
        reserved_train_indices
    )

    if len(valid_candidates) < valid_samples:

        raise RuntimeError(
            "为每个视频保留至少 1 条 train sample 后，"
            "剩余 sample 不足以建立 validation set。\n"
            f"valid candidates = {len(valid_candidates)}\n"
            f"requested valid samples = {valid_samples}"
        )

    valid_sample_indices = random_generator.choice(
        valid_candidates,
        size=valid_samples,
        replace=False
    )

    train_sample_indices = np.setdiff1d(
        all_sample_indices,
        valid_sample_indices
    )

    return (
        train_sample_indices,
        valid_sample_indices
    )


# ============================================================
# 13. 前 9 帧历史 + 当前最后一帧
# ============================================================

def run_history_window(
        model,
        input_sequence,
        target_time_index,
        history_frames
):

    state = None

    window_start_index = (
        target_time_index
        - history_frames
    )

    denoised_frame = None

    for time_index in range(
            window_start_index,
            target_time_index + 1
    ):

        current_input = input_sequence[
            :,
            time_index
        ].unsqueeze(1)

        denoised_frame, state = model(
            current_input,
            state
        )

    return denoised_frame


# ============================================================
# 14. 训练一个 batch：2 条完整的 10 帧 sequence
# ============================================================

def train_one_sequence(
        model,
        frames_gray,
        video_numbers,
        start_frame_indices,
        dataset,
        optimizer,
        sequence_length,
        history_frames,
        block_size,
        number_of_blocks,
        max_block_position_tries,
        cx,
        cy,
        r
):

    optimizer.zero_grad(
        set_to_none=True
    )

    # 只有最后一帧作为 target。
    target_time_index = (
        sequence_length
        - 1
    )

    input_sequence, target_frame, block_mask = (
        replace_blocks_with_same_phase(
            frames_gray=frames_gray,
            video_numbers=video_numbers,
            start_frame_indices=start_frame_indices,
            dataset=dataset,
            target_time_index=target_time_index,
            block_size=block_size,
            number_of_blocks=number_of_blocks,
            max_block_position_tries=max_block_position_tries,
            cx=cx,
            cy=cy,
            r=r
        )
    )

    if block_mask.sum().item() == 0:
        return None

    input_sequence = input_sequence.to(
        DEVICE
    )

    target_frame = target_frame.to(
        DEVICE
    ).unsqueeze(1)

    block_mask = block_mask.to(
        DEVICE
    ).unsqueeze(1)

    vessel_mask = get_batch_vessel_masks(
        dataset=dataset,
        video_numbers=video_numbers
    ).to(
        DEVICE
    )

    denoised_frame = run_history_window(
        model=model,
        input_sequence=input_sequence,
        target_time_index=target_time_index,
        history_frames=history_frames
    )

    (
        current_loss,
        weighted_data_loss_per_sample,
        vessel_data_loss_per_sample,
        background_data_loss_per_sample,
        vessel_gradient_loss_per_sample,
        background_fair_loss_per_sample,
        total_loss_per_sample,
        vessel_masked_pixels_per_sample,
        background_masked_pixels_per_sample
    ) = block_reconstruction_loss(
        prediction=denoised_frame,
        target=target_frame,
        block_mask=block_mask,
        vessel_mask=vessel_mask
    )

    current_loss.backward()

    optimizer.step()

    masked_pixels_per_sample = block_mask.sum(
        dim=(1, 2, 3)
    )

    return (
        current_loss.item(),
        weighted_data_loss_per_sample.detach().cpu().tolist(),
        vessel_data_loss_per_sample.detach().cpu().tolist(),
        background_data_loss_per_sample.detach().cpu().tolist(),
        vessel_gradient_loss_per_sample.detach().cpu().tolist(),
        background_fair_loss_per_sample.detach().cpu().tolist(),
        total_loss_per_sample.detach().cpu().tolist(),
        masked_pixels_per_sample.detach().cpu().tolist(),
        vessel_masked_pixels_per_sample.detach().cpu().tolist(),
        background_masked_pixels_per_sample.detach().cpu().tolist()
    )


# ============================================================
# 15. 验证一个 batch 的完整 sequence
# ============================================================

def validate_one_sequence(
        model,
        frames_gray,
        video_numbers,
        start_frame_indices,
        dataset,
        sequence_length,
        history_frames,
        block_size,
        number_of_blocks,
        max_block_position_tries,
        cx,
        cy,
        r
):

    # 只有最后一帧作为 target。
    target_time_index = (
        sequence_length
        - 1
    )

    with torch.no_grad():

        input_sequence, target_frame, block_mask = (
            replace_blocks_with_same_phase(
                frames_gray=frames_gray,
                video_numbers=video_numbers,
                start_frame_indices=start_frame_indices,
                dataset=dataset,
                target_time_index=target_time_index,
                block_size=block_size,
                number_of_blocks=number_of_blocks,
                max_block_position_tries=max_block_position_tries,
                cx=cx,
                cy=cy,
                r=r
            )
        )

        if block_mask.sum().item() == 0:
            return None

        input_sequence = input_sequence.to(
            DEVICE
        )

        target_frame = target_frame.to(
            DEVICE
        ).unsqueeze(1)

        block_mask = block_mask.to(
            DEVICE
        ).unsqueeze(1)

        vessel_mask = get_batch_vessel_masks(
            dataset=dataset,
            video_numbers=video_numbers
        ).to(
            DEVICE
        )

        denoised_frame = run_history_window(
            model=model,
            input_sequence=input_sequence,
            target_time_index=target_time_index,
            history_frames=history_frames
        )

        (
            current_loss,
            weighted_data_loss_per_sample,
            vessel_data_loss_per_sample,
            background_data_loss_per_sample,
            vessel_gradient_loss_per_sample,
            background_fair_loss_per_sample,
            total_loss_per_sample,
            vessel_masked_pixels_per_sample,
            background_masked_pixels_per_sample
        ) = block_reconstruction_loss(
            prediction=denoised_frame,
            target=target_frame,
            block_mask=block_mask,
            vessel_mask=vessel_mask
        )

        masked_pixels_per_sample = block_mask.sum(
            dim=(1, 2, 3)
        )

    return (
        current_loss.item(),
        weighted_data_loss_per_sample.detach().cpu().tolist(),
        vessel_data_loss_per_sample.detach().cpu().tolist(),
        background_data_loss_per_sample.detach().cpu().tolist(),
        vessel_gradient_loss_per_sample.detach().cpu().tolist(),
        background_fair_loss_per_sample.detach().cpu().tolist(),
        total_loss_per_sample.detach().cpu().tolist(),
        masked_pixels_per_sample.detach().cpu().tolist(),
        vessel_masked_pixels_per_sample.detach().cpu().tolist(),
        background_masked_pixels_per_sample.detach().cpu().tolist()
    )


# ============================================================
# 15.5 逐条 sample loss 输出
# ============================================================

LAMBDA_VESSEL_GRADIENT_FOR_PRINT = 1.50
LAMBDA_BACKGROUND_FAIR_FOR_PRINT = 0.20


def print_one_sample_loss(
        split_name,
        epoch_number,
        total_epochs,
        batch_number,
        total_batches,
        sample_position,
        samples_in_batch,
        dataset_sample_index,
        video_number,
        start_frame_index,
        target_time_index,
        masked_pixels,
        vessel_masked_pixels,
        background_masked_pixels,
        weighted_data_loss,
        vessel_data_loss,
        background_data_loss,
        vessel_gradient_loss,
        background_fair_loss,
        total_loss
):

    target_frame_number = int(start_frame_index) + int(target_time_index)

    print(
        f"{split_name} | epoch={epoch_number:03d}/{total_epochs:03d} | "
        f"batch={batch_number:04d}/{total_batches:04d} | "
        f"sample={sample_position:02d}/{samples_in_batch:02d} | "
        f"dataset_idx={int(dataset_sample_index)} | "
        f"video={int(video_number)} | "
        f"target_frame={target_frame_number} | "
        f"mask={int(masked_pixels)} | "
        f"vessel_mask={int(vessel_masked_pixels)} | "
        f"background_mask={int(background_masked_pixels)} | "
        f"weighted_data={float(weighted_data_loss):.6f} | "
        f"vessel_L1={float(vessel_data_loss):.6f} | "
        f"background_L1={float(background_data_loss):.6f} | "
        f"vessel_grad={float(vessel_gradient_loss):.6f} | "
        f"vessel_grad_x1.5={LAMBDA_VESSEL_GRADIENT_FOR_PRINT * float(vessel_gradient_loss):.6f} | "
        f"bg_fair={float(background_fair_loss):.6f} | "
        f"bg_fair_x0.2={LAMBDA_BACKGROUND_FAIR_FOR_PRINT * float(background_fair_loss):.6f} | "
        f"total={float(total_loss):.6f}",
        flush=True
    )


# ============================================================
# 15.8 分层抽样：每个有效视频至少取 1 条 sequence
# ============================================================

def select_train_samples_for_epoch(
        dataset,
        train_sample_indices,
        train_samples_per_epoch
):

    train_sample_indices = np.asarray(
        train_sample_indices,
        dtype=np.int64
    )

    train_index_set = set(
        train_sample_indices.tolist()
    )

    mandatory_indices = []

    # 每个有效视频先随机取 1 条仍属于 train split 的 sequence。
    for video_number in sorted(
            dataset.sample_indices_by_video.keys()
    ):

        candidates = [
            sample_index
            for sample_index in dataset.sample_indices_by_video[video_number]
            if sample_index in train_index_set
        ]

        if len(candidates) == 0:
            continue

        mandatory_indices.append(
            int(np.random.choice(candidates))
        )

    if len(mandatory_indices) > train_samples_per_epoch:

        raise RuntimeError(
            "有效视频数量大于每个 epoch 的 sample 数，"
            "无法保证每个视频至少出现一次。\n"
            f"valid videos with train samples = {len(mandatory_indices)}\n"
            f"requested samples = {train_samples_per_epoch}"
        )

    mandatory_set = set(mandatory_indices)

    remaining_candidates = np.asarray(
        [
            sample_index
            for sample_index in train_sample_indices
            if int(sample_index) not in mandatory_set
        ],
        dtype=np.int64
    )

    remaining_count = (
        train_samples_per_epoch
        - len(mandatory_indices)
    )

    if remaining_count > 0:

        # 可用 sequence 少于所需数量时允许重复抽样，
        # 这样每个 epoch 仍然固定得到 8000 个 sample。
        replace_remaining = (
            remaining_count > len(remaining_candidates)
        )

        sampling_pool = remaining_candidates

        if len(sampling_pool) == 0:
            sampling_pool = train_sample_indices
            replace_remaining = True

        remaining_indices = np.random.choice(
            sampling_pool,
            size=remaining_count,
            replace=replace_remaining
        )

        selected_indices = np.concatenate(
            [
                np.asarray(mandatory_indices, dtype=np.int64),
                remaining_indices.astype(np.int64)
            ]
        )

    else:

        selected_indices = np.asarray(
            mandatory_indices,
            dtype=np.int64
        )

    # 打乱顺序，避免每个 epoch 开头总是 mandatory samples。
    np.random.shuffle(selected_indices)

    return selected_indices


# ============================================================
# 16. 总训练循环
# ============================================================

def train_model(
        model,
        dataset,
        optimizer,
        epochs,
        train_samples_per_epoch,
        valid_samples,
        sequence_length,
        history_frames,
        block_size,
        number_of_blocks,
        max_block_position_tries,
        cx,
        cy,
        r
):

    train_sample_indices, valid_sample_indices = (
        create_train_valid_indices(
            dataset=dataset,
            valid_samples=valid_samples,
            split_random_seed=SPLIT_RANDOM_SEED
        )
    )

    # 只用于逐 epoch 保存模型；不会改变训练、loss 或验证逻辑。
    EPOCH_MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    best_valid_loss = float("inf")
    best_model_state = None
    epochs_without_improvement = 0

    for epoch_index in range(epochs):

        model.train()

        selected_train_indices = select_train_samples_for_epoch(
            dataset=dataset,
            train_sample_indices=train_sample_indices,
            train_samples_per_epoch=train_samples_per_epoch
        )

        train_loss_sum = 0.0
        train_loss_count = 0

        # BATCH_SIZE = 2。
        # 每次训练两条：
        # 前 9 帧 history + 最后 1 帧 target。
        for batch_start_index in range(
                0,
                len(selected_train_indices),
                BATCH_SIZE
        ):

            batch_sample_indices = selected_train_indices[
                batch_start_index:
                batch_start_index + BATCH_SIZE
            ]

            if len(batch_sample_indices) != BATCH_SIZE:
                continue

            frames_gray, video_numbers, start_frame_indices = (
                get_one_sequence_batch(
                    dataset,
                    batch_sample_indices
                )
            )

            batch_result = train_one_sequence(
                model=model,
                frames_gray=frames_gray,
                video_numbers=video_numbers,
                start_frame_indices=start_frame_indices,
                dataset=dataset,
                optimizer=optimizer,
                sequence_length=sequence_length,
                history_frames=history_frames,
                block_size=block_size,
                number_of_blocks=number_of_blocks,
                max_block_position_tries=max_block_position_tries,
                cx=cx,
                cy=cy,
                r=r
            )

            if batch_result is None:
                continue

            (
                batch_total_loss,
                weighted_data_losses,
                vessel_data_losses,
                background_data_losses,
                vessel_gradient_losses,
                background_fair_losses,
                total_losses,
                masked_pixel_counts,
                vessel_masked_pixel_counts,
                background_masked_pixel_counts
            ) = batch_result

            batch_number = (
                batch_start_index // BATCH_SIZE
                + 1
            )

            total_batches = (
                len(selected_train_indices)
                // BATCH_SIZE
            )

            for sample_position in range(
                    len(batch_sample_indices)
            ):

                print_one_sample_loss(
                    split_name="TRAIN",
                    epoch_number=epoch_index + 1,
                    total_epochs=epochs,
                    batch_number=batch_number,
                    total_batches=total_batches,
                    sample_position=sample_position + 1,
                    samples_in_batch=len(batch_sample_indices),
                    dataset_sample_index=batch_sample_indices[
                        sample_position
                    ],
                    video_number=video_numbers[
                        sample_position
                    ].item(),
                    start_frame_index=start_frame_indices[
                        sample_position
                    ].item(),
                    target_time_index=sequence_length - 1,
                    masked_pixels=masked_pixel_counts[
                        sample_position
                    ],
                    vessel_masked_pixels=vessel_masked_pixel_counts[
                        sample_position
                    ],
                    background_masked_pixels=background_masked_pixel_counts[
                        sample_position
                    ],
                    weighted_data_loss=weighted_data_losses[
                        sample_position
                    ],
                    vessel_data_loss=vessel_data_losses[
                        sample_position
                    ],
                    background_data_loss=background_data_losses[
                        sample_position
                    ],
                    vessel_gradient_loss=vessel_gradient_losses[
                        sample_position
                    ],
                    background_fair_loss=background_fair_losses[
                        sample_position
                    ],
                    total_loss=total_losses[
                        sample_position
                    ]
                )

            train_loss_sum += batch_total_loss
            train_loss_count += 1

        average_train_loss = train_loss_sum / max(
            train_loss_count,
            1
        )

        model.eval()

        valid_loss_sum = 0.0
        valid_loss_count = 0

        for valid_order, sample_index in enumerate(
                valid_sample_indices
        ):

            # validation 的 block 位置和 same-phase reference 固定。
            # 所以不同 epoch 的 valid loss 可以公平比较。
            random_state = np.random.get_state()

            np.random.seed(
                VALIDATION_RANDOM_SEED
                + valid_order
            )

            frames_gray, video_numbers, start_frame_indices = (
                get_one_sequence_batch(
                    dataset,
                    [sample_index]
                )
            )

            batch_result = validate_one_sequence(
                model=model,
                frames_gray=frames_gray,
                video_numbers=video_numbers,
                start_frame_indices=start_frame_indices,
                dataset=dataset,
                sequence_length=sequence_length,
                history_frames=history_frames,
                block_size=block_size,
                number_of_blocks=number_of_blocks,
                max_block_position_tries=max_block_position_tries,
                cx=cx,
                cy=cy,
                r=r
            )

            np.random.set_state(
                random_state
            )

            if batch_result is None:
                continue

            (
                batch_total_loss,
                weighted_data_losses,
                vessel_data_losses,
                background_data_losses,
                vessel_gradient_losses,
                background_fair_losses,
                total_losses,
                masked_pixel_counts,
                vessel_masked_pixel_counts,
                background_masked_pixel_counts
            ) = batch_result

            print_one_sample_loss(
                split_name="VALID",
                epoch_number=epoch_index + 1,
                total_epochs=epochs,
                batch_number=valid_order + 1,
                total_batches=len(valid_sample_indices),
                sample_position=1,
                samples_in_batch=1,
                dataset_sample_index=sample_index,
                video_number=video_numbers[0].item(),
                start_frame_index=start_frame_indices[0].item(),
                target_time_index=sequence_length - 1,
                masked_pixels=masked_pixel_counts[0],
                vessel_masked_pixels=vessel_masked_pixel_counts[0],
                background_masked_pixels=background_masked_pixel_counts[0],
                weighted_data_loss=weighted_data_losses[0],
                vessel_data_loss=vessel_data_losses[0],
                background_data_loss=background_data_losses[0],
                vessel_gradient_loss=vessel_gradient_losses[0],
                background_fair_loss=background_fair_losses[0],
                total_loss=total_losses[0]
            )

            valid_loss_sum += batch_total_loss
            valid_loss_count += 1

        average_valid_loss = valid_loss_sum / max(
            valid_loss_count,
            1
        )

        if average_valid_loss < best_valid_loss:

            best_valid_loss = average_valid_loss

            best_model_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        print(
            f"Epoch {epoch_index+1:03d}/{epochs:03d} | "
            f"train={average_train_loss:.6f} | "
            f"valid={average_valid_loss:.6f} | "
            f"best={best_valid_loss:.6f} | "
            f"no_improve={epochs_without_improvement:02d}/"
            f"{EARLY_STOPPING_PATIENCE:02d}"
        )

        # 保存当前 epoch 结束时的模型。
        # 这里保存的是 model.state_dict()，用于之后单独推理或比较任意 epoch。
        epoch_model_path = EPOCH_MODEL_DIR / (
            f"epoch_{epoch_index + 1:03d}.pth"
        )

        torch.save(
            model.state_dict(),
            epoch_model_path
        )

        print(
            "Epoch model saved:",
            epoch_model_path
        )

        winsound.Beep(
            1000,
            250
        )

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:

            print(
                "Early stopping: validation loss has not improved for",
                EARLY_STOPPING_PATIENCE,
                "epochs."
            )

            break

    if best_model_state is None:

        raise RuntimeError(
            "训练中没有得到有效的 validation loss，无法保存模型。"
        )

    model.load_state_dict(
        best_model_state
    )

    torch.save(
        model.state_dict(),
        MODEL_SAVE_PATH
    )

    print(
        "Best model saved:",
        MODEL_SAVE_PATH
    )

    return model


# ============================================================
# 17. 开始训练
# ============================================================

MODEL_SAVE_PATH = Path(
    r"C:\Users\Novovorontsovka\Downloads\model\best_block32_convlstm_vessel_protection_200videos_batch2_8000.pth"
)

# 每个 epoch 结束后保存当时模型的 state_dict。
# 之后可以直接拿任意一个 epoch_xxx.pth 做推理。
EPOCH_MODEL_DIR = Path(
    r"C:\Users\Novovorontsovka\Downloads\model\epoch_models_block32_vessel_protection_200videos_batch2_8000"
)

model = train_model(
    model=model,
    dataset=dataset,
    optimizer=optimizer,
    epochs=EPOCHS,
    train_samples_per_epoch=TRAIN_SAMPLES_PER_EPOCH,
    valid_samples=VALID_SAMPLES,
    sequence_length=SEQUENCE_LENGTH,
    history_frames=HISTORY_FRAMES,
    block_size=BLOCK_SIZE,
    number_of_blocks=NUMBER_OF_BLOCKS,
    max_block_position_tries=MAX_BLOCK_POSITION_TRIES,
    cx=CX,
    cy=CY,
    r=R
)   