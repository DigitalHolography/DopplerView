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
- loss 不使用任何 vessel mask。
- 总 loss = masked L1 reconstruction + gradient/Hessian/continuity constraint。
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
import torch.nn.functional as F


# ============================================================
# 1. 放文件夹
# ============================================================

VIDEO_DIR = Path(
    r"C:\Users\Novovorontsovka\Downloads\LDH_pipeline_output\05_trimmed_videos"
)

BRIGHTNESS_DIR = Path(
    r"C:\Users\Novovorontsovka\Downloads\LDH_pipeline_output\04_brightness_tables"
)


# 只将下面两个视频用于测试和去噪输出。
# 它们不会进入 training 或 validation。
TEST_VIDEO_NUMBERS = {760, 785}

# 每个 epoch 对 760 和 785 输出去噪视频。
TEST_DENOISED_OUTPUT_DIR = Path(
    r"C:\Users\Novovorontsovka\Downloads\denoised_test_videos_760_785"
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



def get_leading_video_number(video_path):
    """
    读取文件名开头的视频编号。

    例如：
        760_circle_masked_from_first_peak.avi -> 760
        785_xxx.avi -> 785
    """

    first_part = Path(
        video_path
    ).stem.split("_")[0]

    if not first_part.isdigit():

        return None

    return int(
        first_part
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
# 5. 多视频 Dataset
# ============================================================

class MultiVideoDataset(Dataset):

    def __init__(
            self,
            video_paths,
            sequence_length,
            brightness_dir
    ):

        super().__init__()

        self.video_paths = video_paths
        self.sequence_length = sequence_length
        self.brightness_dir = Path(brightness_dir)

        self.all_frames = {}
        self.video_paths_by_id = {}
        self.samples = []
        self.sample_indices_by_video = {}

        self.brightness_by_frame = {}
        self.phase_by_frame = {}
        self.phase_to_frames = {}

        # video_id 是按完整文件列表自动分配的唯一整数。
        # 即使多个文件名都包含 _7_p，它们也会得到不同的 video_id，
        # 因而不会在 self.all_frames 等字典中互相覆盖。
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

            print(
                f"video_id={video_number} | "
                f"file={video_path.name}"
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

all_video_paths = find_all_avi_videos(
    VIDEO_DIR
)

test_video_paths = [
    video_path
    for video_path in all_video_paths
    if get_leading_video_number(
        video_path
    ) in TEST_VIDEO_NUMBERS
]

video_paths = [
    video_path
    for video_path in all_video_paths
    if get_leading_video_number(
        video_path
    ) not in TEST_VIDEO_NUMBERS
]

found_test_numbers = {
    get_leading_video_number(
        video_path
    )
    for video_path in test_video_paths
}

missing_test_numbers = (
    TEST_VIDEO_NUMBERS
    - found_test_numbers
)

if missing_test_numbers:

    raise FileNotFoundError(
        "没有找到以下测试视频：\n"
        + ", ".join(
            str(number)
            for number in sorted(
                missing_test_numbers
            )
        )
        + "\n\n请检查 VIDEO_DIR 中是否存在以这些编号开头的 AVI 文件。"
    )

print(
    "Test videos excluded from training:"
)

for test_video_path in test_video_paths:

    print(
        "  ",
        test_video_path.name
    )

dataset = MultiVideoDataset(
    video_paths=video_paths,
    sequence_length=SEQUENCE_LENGTH,
    brightness_dir=BRIGHTNESS_DIR
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
# 9. Loss：masked L1 reconstruction + mask-free VFC
#
# 不使用任何人工 vessel mask。
# 所有图像损失只在当前帧被替换的 block 内计算。
# ============================================================

LAMBDA_VFC = 1.0
LAMBDA_VFC_GRADIENT = 1.0
LAMBDA_VFC_HESSIAN = 0.50
LAMBDA_VFC_CONTINUITY = 0.20

HESSIAN_BETA = 0.50
HESSIAN_C = 0.10
EPSILON = 1e-8


def masked_mean_per_sample(values, mask):

    numerator = (
        values * mask
    ).sum(
        dim=(1, 2, 3)
    )

    denominator = mask.sum(
        dim=(1, 2, 3)
    )

    return numerator / (
        denominator + EPSILON
    )


def first_order_gradients(image):

    gradient_x = (
        image[:, :, :, 1:]
        - image[:, :, :, :-1]
    )

    gradient_y = (
        image[:, :, 1:, :]
        - image[:, :, :-1, :]
    )

    return gradient_x, gradient_y


def second_order_hessian(image):
    """
    Compute Ixx, Iyy, and Ixy using fixed finite-difference kernels.
    The output size is identical to the input size.
    """

    kernel_xx = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, -2.0, 1.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=image.dtype,
        device=image.device,
    ).view(1, 1, 3, 3)

    kernel_yy = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [0.0, -2.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=image.dtype,
        device=image.device,
    ).view(1, 1, 3, 3)

    kernel_xy = torch.tensor(
        [
            [1.0, 0.0, -1.0],
            [0.0, 0.0, 0.0],
            [-1.0, 0.0, 1.0],
        ],
        dtype=image.dtype,
        device=image.device,
    ).view(1, 1, 3, 3) / 4.0

    padded = F.pad(
        image,
        (1, 1, 1, 1),
        mode="reflect"
    )

    hessian_xx = F.conv2d(
        padded,
        kernel_xx
    )

    hessian_yy = F.conv2d(
        padded,
        kernel_yy
    )

    hessian_xy = F.conv2d(
        padded,
        kernel_xy
    )

    return (
        hessian_xx,
        hessian_yy,
        hessian_xy
    )


def hessian_line_response(image):
    """
    Construct a polarity-independent soft line response from the
    eigenvalues of the 2x2 Hessian matrix.
    """

    hxx, hyy, hxy = second_order_hessian(
        image
    )

    trace = hxx + hyy

    discriminant = torch.sqrt(
        torch.clamp(
            (hxx - hyy) ** 2
            + 4.0 * hxy ** 2,
            min=EPSILON
        )
    )

    lambda_a = 0.5 * (
        trace + discriminant
    )

    lambda_b = 0.5 * (
        trace - discriminant
    )

    absolute_a = torch.abs(
        lambda_a
    )

    absolute_b = torch.abs(
        lambda_b
    )

    lambda_small = torch.where(
        absolute_a <= absolute_b,
        lambda_a,
        lambda_b
    )

    lambda_large = torch.where(
        absolute_a <= absolute_b,
        lambda_b,
        lambda_a
    )

    ratio_squared = (
        lambda_small ** 2
        / (
            lambda_large ** 2
            + EPSILON
        )
    )

    structure_strength_squared = (
        lambda_small ** 2
        + lambda_large ** 2
    )

    line_response = torch.exp(
        -ratio_squared
        / (
            2.0
            * HESSIAN_BETA ** 2
        )
    ) * (
        1.0
        - torch.exp(
            -structure_strength_squared
            / (
                2.0
                * HESSIAN_C ** 2
                + EPSILON
            )
        )
    )

    return line_response


def mask_free_vfc_loss(
        prediction,
        target,
        block_mask
):

    # 1. Masked L1 reconstruction
    l1_loss_per_sample = masked_mean_per_sample(
        torch.abs(
            prediction - target
        ),
        block_mask
    )

    # 2. First-order gradient consistency
    prediction_gradient_x, prediction_gradient_y = (
        first_order_gradients(
            prediction
        )
    )

    target_gradient_x, target_gradient_y = (
        first_order_gradients(
            target
        )
    )

    pair_mask_x = (
        block_mask[:, :, :, 1:]
        * block_mask[:, :, :, :-1]
    )

    pair_mask_y = (
        block_mask[:, :, 1:, :]
        * block_mask[:, :, :-1, :]
    )

    gradient_loss_per_sample = (
        masked_mean_per_sample(
            torch.abs(
                prediction_gradient_x
                - target_gradient_x
            ),
            pair_mask_x
        )
        +
        masked_mean_per_sample(
            torch.abs(
                prediction_gradient_y
                - target_gradient_y
            ),
            pair_mask_y
        )
    )

    # 3. Hessian line-structure consistency
    prediction_line_response = hessian_line_response(
        prediction
    )

    target_line_response = hessian_line_response(
        target
    )

    hessian_loss_per_sample = masked_mean_per_sample(
        torch.abs(
            prediction_line_response
            - target_line_response
        ),
        block_mask
    )

    # 4. Spatial continuity based on second differences
    prediction_gradient_xx = (
        prediction_gradient_x[:, :, :, 1:]
        - prediction_gradient_x[:, :, :, :-1]
    )

    target_gradient_xx = (
        target_gradient_x[:, :, :, 1:]
        - target_gradient_x[:, :, :, :-1]
    )

    prediction_gradient_yy = (
        prediction_gradient_y[:, :, 1:, :]
        - prediction_gradient_y[:, :, :-1, :]
    )

    target_gradient_yy = (
        target_gradient_y[:, :, 1:, :]
        - target_gradient_y[:, :, :-1, :]
    )

    continuity_mask_x = (
        pair_mask_x[:, :, :, 1:]
        * pair_mask_x[:, :, :, :-1]
    )

    continuity_mask_y = (
        pair_mask_y[:, :, 1:, :]
        * pair_mask_y[:, :, :-1, :]
    )

    continuity_loss_per_sample = (
        masked_mean_per_sample(
            torch.abs(
                prediction_gradient_xx
                - target_gradient_xx
            ),
            continuity_mask_x
        )
        +
        masked_mean_per_sample(
            torch.abs(
                prediction_gradient_yy
                - target_gradient_yy
            ),
            continuity_mask_y
        )
    )

    vfc_loss_per_sample = (
        LAMBDA_VFC_GRADIENT
        * gradient_loss_per_sample
        +
        LAMBDA_VFC_HESSIAN
        * hessian_loss_per_sample
        +
        LAMBDA_VFC_CONTINUITY
        * continuity_loss_per_sample
    )

    total_loss_per_sample = (
        l1_loss_per_sample
        +
        LAMBDA_VFC
        * vfc_loss_per_sample
    )

    batch_total_loss = total_loss_per_sample.sum()

    return (
        batch_total_loss,
        l1_loss_per_sample,
        gradient_loss_per_sample,
        hessian_loss_per_sample,
        continuity_loss_per_sample,
        vfc_loss_per_sample,
        total_loss_per_sample
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

    target_time_index = (
        sequence_length - 1
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

    denoised_frame = run_history_window(
        model=model,
        input_sequence=input_sequence,
        target_time_index=target_time_index,
        history_frames=history_frames
    )

    (
        current_loss,
        l1_loss_per_sample,
        gradient_loss_per_sample,
        hessian_loss_per_sample,
        continuity_loss_per_sample,
        vfc_loss_per_sample,
        total_loss_per_sample
    ) = mask_free_vfc_loss(
        prediction=denoised_frame,
        target=target_frame,
        block_mask=block_mask
    )

    current_loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0
    )

    optimizer.step()

    masked_pixels_per_sample = block_mask.sum(
        dim=(1, 2, 3)
    )

    return (
        current_loss.item(),
        l1_loss_per_sample.detach().cpu().tolist(),
        gradient_loss_per_sample.detach().cpu().tolist(),
        hessian_loss_per_sample.detach().cpu().tolist(),
        continuity_loss_per_sample.detach().cpu().tolist(),
        vfc_loss_per_sample.detach().cpu().tolist(),
        total_loss_per_sample.detach().cpu().tolist(),
        masked_pixels_per_sample.detach().cpu().tolist()
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

    target_time_index = (
        sequence_length - 1
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

        denoised_frame = run_history_window(
            model=model,
            input_sequence=input_sequence,
            target_time_index=target_time_index,
            history_frames=history_frames
        )

        (
            current_loss,
            l1_loss_per_sample,
            gradient_loss_per_sample,
            continuity_loss_per_sample,
            vfc_loss_per_sample,
            total_loss_per_sample
        ) = mask_free_vfc_loss(
            prediction=denoised_frame,
            target=target_frame,
            block_mask=block_mask
        )

        masked_pixels_per_sample = block_mask.sum(
            dim=(1, 2, 3)
        )

    return (
        current_loss.item(),
        l1_loss_per_sample.detach().cpu().tolist(),
        gradient_loss_per_sample.detach().cpu().tolist(),
        hessian_loss_per_sample.detach().cpu().tolist(),
        continuity_loss_per_sample.detach().cpu().tolist(),
        vfc_loss_per_sample.detach().cpu().tolist(),
        total_loss_per_sample.detach().cpu().tolist(),
        masked_pixels_per_sample.detach().cpu().tolist()
    )


# ============================================================
# 15.5 逐条 sample loss 输出
# ============================================================

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
        l1_loss,
        gradient_loss,
        hessian_loss,
        continuity_loss,
        vfc_loss,
        total_loss
):

    target_frame_number = (
        int(start_frame_index)
        + int(target_time_index)
    )

    print(
        f"{split_name} | "
        f"epoch={epoch_number:03d}/{total_epochs:03d} | "
        f"batch={batch_number:04d}/{total_batches:04d} | "
        f"sample={sample_position:02d}/{samples_in_batch:02d} | "
        f"dataset_idx={int(dataset_sample_index)} | "
        f"video={int(video_number)} | "
        f"target_frame={target_frame_number} | "
        f"mask={int(masked_pixels)} | "
        f"L1={float(l1_loss):.8f} | "
        f"grad={float(gradient_loss):.8f} | "
        f"hessian={float(hessian_loss):.8f} | "
        f"continuity={float(continuity_loss):.8f} | "
        f"VFC={float(vfc_loss):.8f} | "
        f"total={float(total_loss):.8f}",
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
# 15.9 只对视频 760 和 785 进行去噪
#
# 前 9 帧保持原图；
# 从第 10 帧开始，每一帧使用 [t-9, ..., t] 的 10 帧窗口。
# ============================================================

def denoise_one_test_video(
        model,
        source_video_path,
        output_path,
        history_frames
):

    frames_gray = load_video_as_gray(
        source_video_path
    )

    if (
            frames_gray.ndim != 3
            or len(frames_gray) == 0
    ):

        raise RuntimeError(
            "无法读取测试视频：\n"
            f"{source_video_path}"
        )

    total_frames, height, width = (
        frames_gray.shape
    )

    if (
            height != EXPECTED_HEIGHT
            or width != EXPECTED_WIDTH
    ):

        raise RuntimeError(
            "测试视频尺寸不符合要求：\n"
            f"{source_video_path}\n"
            f"actual={width}x{height}, "
            f"expected={EXPECTED_WIDTH}x{EXPECTED_HEIGHT}"
        )

    source_cap = cv2.VideoCapture(
        str(
            source_video_path
        )
    )

    source_fps = source_cap.get(
        cv2.CAP_PROP_FPS
    )

    source_cap.release()

    if source_fps <= 0:

        source_fps = 30.0

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    writer = cv2.VideoWriter(
        str(
            output_path
        ),
        cv2.VideoWriter_fourcc(
            *"MJPG"
        ),
        float(
            source_fps
        ),
        (
            width,
            height
        ),
        True
    )

    if not writer.isOpened():

        raise RuntimeError(
            "无法创建去噪视频：\n"
            f"{output_path}"
        )

    model.eval()

    with torch.no_grad():

        for frame_index in range(
                total_frames
        ):

            # 前 9 帧没有完整历史窗口，直接保存原图。
            if frame_index < history_frames:

                denoised_uint8 = frames_gray[
                    frame_index
                ]

            else:

                sequence_start = (
                    frame_index
                    - history_frames
                )

                input_sequence = torch.from_numpy(
                    frames_gray[
                        sequence_start:
                        frame_index + 1
                    ]
                ).float() / 255.0

                input_sequence = input_sequence.unsqueeze(
                    0
                ).to(
                    DEVICE
                )

                denoised_frame = run_history_window(
                    model=model,
                    input_sequence=input_sequence,
                    target_time_index=history_frames,
                    history_frames=history_frames
                )

                denoised_uint8 = (
                    denoised_frame[
                        0,
                        0
                    ]
                    .clamp(
                        0.0,
                        1.0
                    )
                    .mul(
                        255.0
                    )
                    .round()
                    .byte()
                    .cpu()
                    .numpy()
                )

            denoised_bgr = cv2.cvtColor(
                denoised_uint8,
                cv2.COLOR_GRAY2BGR
            )

            writer.write(
                denoised_bgr
            )

    writer.release()

    print(
        "Denoised test video saved:",
        output_path
    )


def denoise_selected_test_videos(
        model,
        test_video_paths,
        output_root,
        history_frames,
        output_subdirectory
):

    output_directory = (
        Path(
            output_root
        )
        / output_subdirectory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    for source_video_path in test_video_paths:

        video_number = get_leading_video_number(
            source_video_path
        )

        if video_number not in TEST_VIDEO_NUMBERS:

            continue

        output_path = output_directory / (
            f"{video_number}_denoised.avi"
        )

        denoise_one_test_video(
            model=model,
            source_video_path=source_video_path,
            output_path=output_path,
            history_frames=history_frames
        )


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
        r,
        test_video_paths,
        test_output_root
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
                l1_losses,
                gradient_losses,
                hessian_losses,
                continuity_losses,
                vfc_losses,
                total_losses,
                masked_pixel_counts
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
                    l1_loss=l1_losses[
                        sample_position
                    ],
                    gradient_loss=gradient_losses[
                        sample_position
                    ],
                    hessian_loss=hessian_losses[
                        sample_position
                    ],
                    continuity_loss=continuity_losses[
                        sample_position
                    ],
                    vfc_loss=vfc_losses[
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
                l1_losses,
                gradient_losses,
                hessian_losses,
                continuity_losses,
                vfc_losses,
                total_losses,
                masked_pixel_counts
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
                l1_loss=l1_losses[0],
                gradient_loss=gradient_losses[0],
                hessian_loss=hessian_losses[0],
                continuity_loss=continuity_losses[0],
                vfc_loss=vfc_losses[0],
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

        # 每个 epoch 只输出 760 和 785 的去噪视频。
        denoise_selected_test_videos(
            model=model,
            test_video_paths=test_video_paths,
            output_root=test_output_root,
            history_frames=history_frames,
            output_subdirectory=(
                f"epoch_{epoch_index + 1:03d}"
            )
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

    # 使用最佳模型再次只输出 760 和 785。
    denoise_selected_test_videos(
        model=model,
        test_video_paths=test_video_paths,
        output_root=test_output_root,
        history_frames=history_frames,
        output_subdirectory="best_model"
    )

    return model


# ============================================================
# 17. 开始训练
# ============================================================

MODEL_SAVE_PATH = Path(
    r"C:\Users\Novovorontsovka\Downloads\model\best_block32_convlstm_gradient_hessian_continuity_200videos_batch2_8000.pth"
)

# 每个 epoch 结束后保存当时模型的 state_dict。
# 之后可以直接拿任意一个 epoch_xxx.pth 做推理。
EPOCH_MODEL_DIR = Path(
    r"C:\Users\Novovorontsovka\Downloads\model\epoch_models_block32_gradient_hessian_continuity_200videos_batch2_8000"
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
    r=R,
    test_video_paths=test_video_paths,
    test_output_root=TEST_DENOISED_OUTPUT_DIR
)   