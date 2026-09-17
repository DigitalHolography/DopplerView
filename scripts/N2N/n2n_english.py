 # -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 14:51:49 2026

@author: solei

Training logic:
- One dataset sample = 10 consecutive frames from the same video.
- One batch = 2 complete 10-frame sequences.
- The first 9 frames are used only to build the temporal memory of the ConvLSTM.
- Only the last frame is the current target frame.
- When predicting the last frame, the ConvLSTM sees:
  [t-9, t-8, ..., t-2, t-1, t]
- Only the final frame t contains replaced blocks.
- The masked L2 loss is computed only at the replaced positions of the final frame t.
- Each sequence starts again with state=None.
- Validation samples are randomly selected once before training and then kept fixed.
"""

from pathlib import Path
import cv2
import numpy as np
import winsound
from torch.utils.data import Dataset
import torch
import torch.nn as nn


# ============================================================
# 1. Input folders
# ============================================================

VIDEO_DIR = Path(
    r"C:\Users\Novovorontsovka\Downloads\video_masqued\output\05_trimmed_videos"
)

BRIGHTNESS_DIR = Path(
    r"C:\Users\Novovorontsovka\Downloads\video_masqued\output\04_brightness_tables"
)


# Only 512x512 videos are accepted; videos with other sizes are skipped.
EXPECTED_HEIGHT = 512
EXPECTED_WIDTH = 512

# Use at most 200 valid videos that pass all checks.
# Note: the program first skips empty videos, non-512x512 videos, and videos with too few frames,
# then stops loading once 200 valid videos have been collected.
MAX_VALID_VIDEOS = 200


# ============================================================
# 2. Extract the video number from the filename
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
# 3. Find all AVI videos in the folder and sort them
# ============================================================

def find_all_avi_videos(video_dir):

    video_dir = Path(video_dir)

    video_paths = list(
        video_dir.glob("*.avi")
    )

    video_paths += list(
        video_dir.glob("*.AVI")
    )

    # Sort by the full filename.
    # Do not sort using local indices such as _7_p or _8_p in the filename,
    # because videos from different patients/acquisitions may share the same local index.
    video_paths = sorted(
        video_paths,
        key=lambda path: path.name.lower()
    )

    return video_paths


# ============================================================
# 4. Read one video and return all grayscale frames
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
# 4.2 Load the smoothed brightness and phase table corresponding to one video
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
# 5. Multi-video Dataset
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

        # video_id is a unique integer automatically assigned from the complete file list.
        # Even if several filenames contain _7_p, they will still receive different video_id values,
        # so they will not overwrite each other in dictionaries such as self.all_frames.
        valid_video_count = 0

        for video_id, video_path in enumerate(self.video_paths):

            # Stop reading once 200 valid videos have been collected.
            if valid_video_count >= MAX_VALID_VIDEOS:
                print(
                    f"Reached MAX_VALID_VIDEOS={MAX_VALID_VIDEOS}. "
                    "Stop loading more videos."
                )
                break

            video_number = int(video_id)

            frames_gray = load_video_as_gray(video_path)

            # Skip empty or unreadable videos.
            if frames_gray.ndim != 3 or len(frames_gray) == 0:

                print(
                    f"SKIP video_id={video_number}: empty or unreadable -> "
                    f"{video_path.name}"
                )
                continue

            frame_height = int(frames_gray.shape[1])
            frame_width = int(frames_gray.shape[2])

            # Videos that are not 512x512 are excluded from the dataset.
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

            # The video must contain enough frames to form at least one complete sequence.
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
# One sample:
# first 9 history frames + final 1 target frame
# ============================================================

HISTORY_FRAMES = 9

SEQUENCE_LENGTH = (
    HISTORY_FRAMES
    + 1
)

# One batch trains 2 complete 10-frame sequences at the same time.
BATCH_SIZE = 2

video_paths = find_all_avi_videos(
    VIDEO_DIR
)

dataset = MultiVideoDataset(
    video_paths=video_paths,
    sequence_length=SEQUENCE_LENGTH,
    brightness_dir=BRIGHTNESS_DIR
)


# ============================================================
# 6. Hyperparameters : block hiding
# For each target, replace 32 non-overlapping 32x32 blocks.
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

    # Original image of the current final target frame.
    target_frame = original_sequence[
        :,
        target_time_index
    ].clone()

    # Keep the first 9 frames unchanged.
    # Replace blocks only in the final target frame.
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

            # Replace blocks only in the final target frame.
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
# 8. Network
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

        # The ConvLSTM is placed only at the deepest 16x16 bottleneck.
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

        # x has shape [batch, 512, 16, 16].
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
# 9. Loss: compute L2 / MSE only inside the replaced blocks
# ============================================================

def block_reconstruction_loss(
        prediction,
        target,
        block_mask
):

    """
    Pure self-supervised masked L2 loss.

    Compute the loss only at positions in the final frame that were replaced by a same-phase reference:

        L = mean_{p in Omega} (prediction_p - target_p)^2

    No vessel mask is used, with no vessel/background distinction and no additional gradient or Fair term.
    """

    squared_error = (
        prediction
        - target
    ) ** 2

    masked_squared_error_sum_per_sample = (
        squared_error
        * block_mask
    ).sum(
        dim=(1, 2, 3)
    )

    masked_pixel_count_per_sample = block_mask.sum(
        dim=(1, 2, 3)
    )

    l2_loss_per_sample = (
        masked_squared_error_sum_per_sample
        / torch.clamp(
            masked_pixel_count_per_sample,
            min=1.0
        )
    )

    batch_loss = l2_loss_per_sample.mean()

    return (
        batch_loss,
        l2_loss_per_sample,
        masked_pixel_count_per_sample
    )
# ============================================================
# 10. Training configuration and initialization
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

# Sample 8000 ten-frame source sequences for training in each epoch.
# During sampling, ensure that every valid video contributes at least one sequence.
TRAIN_SAMPLES_PER_EPOCH = 8000

# Randomly select 20 source sequences for validation once at the beginning and keep them fixed.
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
# 11. Manually build one batch
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
# 12. Create a fixed random train / validation split
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

    # Reserve at least one training sequence for every valid video.
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
# 13. First 9 history frames + current final frame
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
# 14. Train one batch: 2 complete 10-frame sequences
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

    # Only the final frame is used as the target.
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

    denoised_frame = run_history_window(
        model=model,
        input_sequence=input_sequence,
        target_time_index=target_time_index,
        history_frames=history_frames
    )

    (
        current_loss,
        l2_loss_per_sample,
        masked_pixels_per_sample
    ) = block_reconstruction_loss(
        prediction=denoised_frame,
        target=target_frame,
        block_mask=block_mask
    )

    current_loss.backward()

    optimizer.step()

    return (
        current_loss.item(),
        l2_loss_per_sample.detach().cpu().tolist(),
        masked_pixels_per_sample.detach().cpu().tolist()
    )


# ============================================================
# 15. Validate one complete sequence batch
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

    # Only the final frame is used as the target.
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

        denoised_frame = run_history_window(
            model=model,
            input_sequence=input_sequence,
            target_time_index=target_time_index,
            history_frames=history_frames
        )

        (
            current_loss,
            l2_loss_per_sample,
            masked_pixels_per_sample
        ) = block_reconstruction_loss(
            prediction=denoised_frame,
            target=target_frame,
            block_mask=block_mask
        )

    return (
        current_loss.item(),
        l2_loss_per_sample.detach().cpu().tolist(),
        masked_pixels_per_sample.detach().cpu().tolist()
    )


# ============================================================
# 15.5 Print the loss of each individual sample
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
        l2_loss
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
        f"masked_L2={float(l2_loss):.8f}",
        flush=True
    )
# ============================================================
# 15.8 Stratified sampling: take at least one sequence from every valid video
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

    # First randomly select one sequence from each valid video that still belongs to the training split.
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

        # Allow sampling with replacement when fewer sequences are available than required,
        # so each epoch still contains exactly 8000 samples.
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

    # Shuffle the order so that mandatory samples do not always appear at the beginning of each epoch.
    np.random.shuffle(selected_indices)

    return selected_indices




# ============================================================
# 15.9 Denoise every training video frame by frame with the current model and save as AVI
# ============================================================

def denoise_one_full_video(
        model,
        frames_gray,
        history_frames
):

    """
    Denoise one complete video frame by frame.

    For frame t, use at most [t-history_frames, ..., t].
    Reset state=None for every target, matching the independent initialization used for each training sequence.
    For the first history_frames frames, use all currently available history because a full history window is not yet available.
    """

    model.eval()

    input_video = torch.from_numpy(
        frames_gray.astype(np.float32)
    ).to(
        DEVICE
    ) / 255.0

    denoised_frames = []

    with torch.no_grad():

        for target_index in range(len(frames_gray)):

            window_start = max(
                0,
                target_index - history_frames
            )

            state = None
            denoised_frame = None

            for time_index in range(
                    window_start,
                    target_index + 1
            ):

                current_input = input_video[
                    time_index
                ].unsqueeze(0).unsqueeze(0)

                denoised_frame, state = model(
                    current_input,
                    state
                )

            denoised_uint8 = (
                torch.clamp(
                    denoised_frame[0, 0],
                    0.0,
                    1.0
                )
                * 255.0
            ).round().byte().cpu().numpy()

            denoised_frames.append(
                denoised_uint8
            )

    return np.stack(
        denoised_frames,
        axis=0
    )


def save_gray_video(
        frames_gray,
        source_video_path,
        output_video_path
):

    source_video_path = Path(source_video_path)
    output_video_path = Path(output_video_path)

    output_video_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    cap = cv2.VideoCapture(
        str(source_video_path)
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    cap.release()

    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    height = int(frames_gray.shape[1])
    width = int(frames_gray.shape[2])

    writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        float(fps),
        (width, height),
        isColor=False
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"无法创建输出视频：{output_video_path}"
        )

    for frame in frames_gray:
        writer.write(frame)

    writer.release()


def denoise_all_training_videos(
        model,
        dataset,
        output_dir,
        history_frames
):

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    video_ids = sorted(
        dataset.all_frames.keys()
    )

    print(
        f"Denoising all {len(video_ids)} training videos -> {output_dir}",
        flush=True
    )

    for order, video_number in enumerate(
            video_ids,
            start=1
    ):

        source_video_path = dataset.video_paths_by_id[
            video_number
        ]

        frames_gray = dataset.all_frames[
            video_number
        ]

        denoised_frames = denoise_one_full_video(
            model=model,
            frames_gray=frames_gray,
            history_frames=history_frames
        )

        output_video_path = output_dir / (
            source_video_path.stem
            + "_denoised.avi"
        )

        save_gray_video(
            frames_gray=denoised_frames,
            source_video_path=source_video_path,
            output_video_path=output_video_path
        )

        print(
            f"DENOISE {order:03d}/{len(video_ids):03d} | "
            f"video_id={video_number} | "
            f"saved={output_video_path.name}",
            flush=True
        )


# ============================================================
# 16. Main training loop
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

    # Used only to save a model at each epoch; this does not change the training, loss, or validation logic.
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
        # Train two sequences at a time:
        # first 9 history frames + final 1 target frame.
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
                l2_losses,
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
                    l2_loss=l2_losses[
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

            # Keep the validation block positions and same-phase references fixed.
            # This makes validation losses comparable across epochs.
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
                l2_losses,
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
                l2_loss=l2_losses[0]
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

        # Save the model at the end of the current epoch.
        # Save model.state_dict() so any epoch can later be used independently for inference or comparison.
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

        # Starting from DENOISE_START_EPOCH:
        # use every epoch model to fully denoise all training videos.
        current_epoch_number = epoch_index + 1

        if current_epoch_number >= DENOISE_START_EPOCH:

            epoch_denoised_dir = DENOISED_EPOCH_ROOT / (
                f"epoch_{current_epoch_number:03d}"
            )

            denoise_all_training_videos(
                model=model,
                dataset=dataset,
                output_dir=epoch_denoised_dir,
                history_frames=history_frames
            )

        else:

            print(
                f"Skip full-video denoising at epoch {current_epoch_number:03d}; "
                f"it starts from epoch {DENOISE_START_EPOCH:03d}."
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

    denoise_all_training_videos(
        model=model,
        dataset=dataset,
        output_dir=BEST_DENOISED_DIR,
        history_frames=history_frames
    )

    return model


# ============================================================
# 17. Start training
# ============================================================

MODEL_SAVE_PATH = Path(
    r"C:\Users\Novovorontsovka\Downloads\model\best_block32_convlstm_L2_200videos_batch2_8000.pth"
)

# Save the model state_dict at the end of every epoch.
# Any epoch_xxx.pth can then be used directly for inference.
EPOCH_MODEL_DIR = Path(
    r"C:\Users\Novovorontsovka\Downloads\model\epoch_models_block32_L2_200videos_batch2_8000"
)

# Epoch from which full-video denoising starts for every epoch model.
# Example: if set to 5, epochs 1-4 only train and save models; from epoch 5 onward, all videos are denoised after every epoch.
DENOISE_START_EPOCH = 5

# Denoised outputs of all training videos for each epoch model.
DENOISED_EPOCH_ROOT = Path(
    r"C:\Users\Novovorontsovka\Downloads\model\denoised_all_videos_by_epoch_L2"
)

# After early stopping, reload the best model and additionally save denoised outputs for all videos.
BEST_DENOISED_DIR = Path(
    r"C:\Users\Novovorontsovka\Downloads\model\best_model_denoised_all_videos_L2"
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
