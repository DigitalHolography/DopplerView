from pathlib import Path
import random
import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 0. video_items
# ============================================================

video_items = [
    {
        "video_path": r"C:\Users\Novovorontsovka\Downloads\N2N\260310_AUZ0752_10_HD_M0.avi",
        "pair_path": r"C:\Users\Novovorontsovka\Downloads\LDH_traite\N2N_pair_results\260310_AUZ0752_10_HD_M0_pair_result.npz",
        "mask_path": r"C:\Users\Novovorontsovka\Downloads\LDH_traite\N2N_vessel_masks\260310_AUZ0752_10_HD_M0_vessel_mask.npy",
    },
    {
        "video_path": r"C:\Users\Novovorontsovka\Downloads\N2N\260310_AUZ0752_11_HD_M0.avi",
        "pair_path": r"C:\Users\Novovorontsovka\Downloads\LDH_traite\N2N_pair_results\260310_AUZ0752_11_HD_M0_pair_result.npz",
        "mask_path": r"C:\Users\Novovorontsovka\Downloads\LDH_traite\N2N_vessel_masks\260310_AUZ0752_11_HD_M0_vessel_mask.npy",
    },
    {
        "video_path": r"C:\Users\Novovorontsovka\Downloads\N2N\260310_AUZ0752_12_HD_M0.avi",
        "pair_path": r"C:\Users\Novovorontsovka\Downloads\LDH_traite\N2N_pair_results\260310_AUZ0752_12_HD_M0_pair_result.npz",
        "mask_path": r"C:\Users\Novovorontsovka\Downloads\LDH_traite\N2N_vessel_masks\260310_AUZ0752_12_HD_M0_vessel_mask.npy",
    },
    {
        "video_path": r"C:\Users\Novovorontsovka\Downloads\N2N\260310_AUZ0752_13_HD_M0.avi",
        "pair_path": r"C:\Users\Novovorontsovka\Downloads\LDH_traite\N2N_pair_results\260310_AUZ0752_13_HD_M0_pair_result.npz",
        "mask_path": r"C:\Users\Novovorontsovka\Downloads\LDH_traite\N2N_vessel_masks\260310_AUZ0752_13_HD_M0_vessel_mask.npy",
    },
    {
        "video_path": r"C:\Users\Novovorontsovka\Downloads\N2N\260310_AUZ0752_14_HD_M0.avi",
        "pair_path": r"C:\Users\Novovorontsovka\Downloads\LDH_traite\N2N_pair_results\260310_AUZ0752_14_HD_M0_pair_result.npz",
        "mask_path": r"C:\Users\Novovorontsovka\Downloads\LDH_traite\N2N_vessel_masks\260310_AUZ0752_14_HD_M0_vessel_mask.npy",
    },
    {
        "video_path": r"C:\Users\Novovorontsovka\Downloads\N2N\260310_AUZ0752_15_HD_M0.avi",
        "pair_path": r"C:\Users\Novovorontsovka\Downloads\LDH_traite\N2N_pair_results\260310_AUZ0752_15_HD_M0_pair_result.npz",
        "mask_path": r"C:\Users\Novovorontsovka\Downloads\LDH_traite\N2N_vessel_masks\260310_AUZ0752_15_HD_M0_vessel_mask.npy",
    },
    {
        "video_path": r"C:\Users\Novovorontsovka\Downloads\N2N\260310_AUZ0752_16_HD_M0.avi",
        "pair_path": r"C:\Users\Novovorontsovka\Downloads\LDH_traite\N2N_pair_results\260310_AUZ0752_16_HD_M0_pair_result.npz",
        "mask_path": r"C:\Users\Novovorontsovka\Downloads\LDH_traite\N2N_vessel_masks\260310_AUZ0752_16_HD_M0_vessel_mask.npy",
    },
    {
        "video_path": r"C:\Users\Novovorontsovka\Downloads\N2N\260310_AUZ0752_17_HD_M0.avi",
        "pair_path": r"C:\Users\Novovorontsovka\Downloads\LDH_traite\N2N_pair_results\260310_AUZ0752_17_HD_M0_pair_result.npz",
        "mask_path": r"C:\Users\Novovorontsovka\Downloads\LDH_traite\N2N_vessel_masks\260310_AUZ0752_17_HD_M0_vessel_mask.npy",
    },
    {
        "video_path": r"C:\Users\Novovorontsovka\Downloads\N2N\260310_AUZ0752_1_HD_M0.avi",
        "pair_path": r"C:\Users\Novovorontsovka\Downloads\LDH_traite\N2N_pair_results\260310_AUZ0752_1_HD_M0_pair_result.npz",
        "mask_path": r"C:\Users\Novovorontsovka\Downloads\LDH_traite\N2N_vessel_masks\260310_AUZ0752_1_HD_M0_vessel_mask.npy",
    },
    {
        "video_path": r"C:\Users\Novovorontsovka\Downloads\N2N\260310_AUZ0752_2_HD_M0.avi",
        "pair_path": r"C:\Users\Novovorontsovka\Downloads\LDH_traite\N2N_pair_results\260310_AUZ0752_2_HD_M0_pair_result.npz",
        "mask_path": r"C:\Users\Novovorontsovka\Downloads\LDH_traite\N2N_vessel_masks\260310_AUZ0752_2_HD_M0_vessel_mask.npy",
    },
    {
        "video_path": r"C:\Users\Novovorontsovka\Downloads\N2N\260310_AUZ0752_3_HD_M0.avi",
        "pair_path": r"C:\Users\Novovorontsovka\Downloads\LDH_traite\N2N_pair_results\260310_AUZ0752_3_HD_M0_pair_result.npz",
        "mask_path": r"C:\Users\Novovorontsovka\Downloads\LDH_traite\N2N_vessel_masks\260310_AUZ0752_3_HD_M0_vessel_mask.npy",
    },
    {
        "video_path": r"C:\Users\Novovorontsovka\Downloads\N2N\260310_AUZ0752_4_HD_M0.avi",
        "pair_path": r"C:\Users\Novovorontsovka\Downloads\LDH_traite\N2N_pair_results\260310_AUZ0752_4_HD_M0_pair_result.npz",
        "mask_path": r"C:\Users\Novovorontsovka\Downloads\LDH_traite\N2N_vessel_masks\260310_AUZ0752_4_HD_M0_vessel_mask.npy",
    },
    {
        "video_path": r"C:\Users\Novovorontsovka\Downloads\N2N\260310_AUZ0752_6_HD_M0.avi",
        "pair_path": r"C:\Users\Novovorontsovka\Downloads\LDH_traite\N2N_pair_results\260310_AUZ0752_6_HD_M0_pair_result.npz",
        "mask_path": r"C:\Users\Novovorontsovka\Downloads\LDH_traite\N2N_vessel_masks\260310_AUZ0752_6_HD_M0_vessel_mask.npy",
    },
    {
        "video_path": r"C:\Users\Novovorontsovka\Downloads\N2N\260310_AUZ0752_7_HD_M0.avi",
        "pair_path": r"C:\Users\Novovorontsovka\Downloads\LDH_traite\N2N_pair_results\260310_AUZ0752_7_HD_M0_pair_result.npz",
        "mask_path": r"C:\Users\Novovorontsovka\Downloads\LDH_traite\N2N_vessel_masks\260310_AUZ0752_7_HD_M0_vessel_mask.npy",
    },
    {
        "video_path": r"C:\Users\Novovorontsovka\Downloads\N2N\260310_AUZ0752_8_HD_M0.avi",
        "pair_path": r"C:\Users\Novovorontsovka\Downloads\LDH_traite\N2N_pair_results\260310_AUZ0752_8_HD_M0_pair_result.npz",
        "mask_path": r"C:\Users\Novovorontsovka\Downloads\LDH_traite\N2N_vessel_masks\260310_AUZ0752_8_HD_M0_vessel_mask.npy",
    },
    {
        "video_path": r"C:\Users\Novovorontsovka\Downloads\N2N\260310_AUZ0752_9_HD_M0.avi",
        "pair_path": r"C:\Users\Novovorontsovka\Downloads\LDH_traite\N2N_pair_results\260310_AUZ0752_9_HD_M0_pair_result.npz",
        "mask_path": r"C:\Users\Novovorontsovka\Downloads\LDH_traite\N2N_vessel_masks\260310_AUZ0752_9_HD_M0_vessel_mask.npy",
    },
]


# ============================================================
# 1. 基本设置
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

USE_5FRAMES = False
RADIUS = 0

# 因为每个 frame 的 pair 数量 K 可能不同，所以先用 batch_size=1
BATCH_SIZE = 1

EPOCHS = 10
LR = 1e-4
VAL_RATIO = 0.20

RESIDUAL_SCALE = 0.15

OUTPUT_DIR = Path(r"C:\Users\Novovorontsovka\Downloads\resultat\N2N_training_residual_1frame_all_pairs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = OUTPUT_DIR / "best_residual_unet_1frame_all_pairs.pth"
DENOISED_TEST_VIDEO_PATH = OUTPUT_DIR / "denoised_test_video_residual_1frame_all_pairs.avi"
RESIDUAL_TEST_VIDEO_PATH = OUTPUT_DIR / "predicted_residual_video_1frame_all_pairs.avi"


# ============================================================
# 2. train / validation / test split
# ============================================================

if len(video_items) < 3:
    raise RuntimeError("video_items 至少需要 3 个视频，才能分 train / val / test")

test_item = video_items[-1]
remaining_items = video_items[:-1]

random.shuffle(remaining_items)

num_val = max(1, int(len(remaining_items) * VAL_RATIO))

val_items = remaining_items[:num_val]
train_items = remaining_items[num_val:]

print("\n" + "=" * 80)
print("DATASET SPLIT")
print("=" * 80)
print("train videos:", len(train_items))
print("val videos:", len(val_items))
print("test video:", test_item["video_path"])


# ============================================================
# 3. 读取 video / pair
# ============================================================

def load_video_gray(video_path, normalize=True):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        if normalize:
            gray = gray / 255.0

        frames.append(gray)

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"视频为空: {video_path}")

    frames = np.stack(frames, axis=0)

    if fps <= 0:
        fps = 30.0

    return frames, fps


def load_pair_npz(pair_path, T):
    pair_path = Path(pair_path)

    if not pair_path.exists():
        raise FileNotFoundError(f"找不到 pair 文件: {pair_path}")

    data = np.load(pair_path, allow_pickle=True)
    files = list(data.files)

    print("\nPAIR FILE:", pair_path)
    print("keys:", files)

    if "pair_candidates" in files and "valid_train_frame" in files:
        pair_candidates = data["pair_candidates"]

        if isinstance(pair_candidates, np.ndarray):
            pair_candidates = pair_candidates.tolist()

        valid_train_frame = data["valid_train_frame"].astype(bool)

        return pair_candidates, valid_train_frame

    if "pair_indices" in files:
        pair_indices = data["pair_indices"].astype(np.int32)

        pair_candidates = [[] for _ in range(T)]
        valid_train_frame = np.zeros(T, dtype=bool)

        for t, j in pair_indices:
            t = int(t)
            j = int(j)

            if 0 <= t < T and 0 <= j < T:
                pair_candidates[t].append(j)
                valid_train_frame[t] = True

        return pair_candidates, valid_train_frame

    raise RuntimeError(
        f"pair 文件格式不认识: {pair_path}\n"
        f"里面 keys = {files}\n"
        f"需要 pair_candidates + valid_train_frame 或 pair_indices"
    )


# ============================================================
# 4. Dataset
# ============================================================

class MultiVideoN2NDataset(Dataset):
    def __init__(
        self,
        video_items,
        use_5frames=False,
        radius=0,
        max_samples_per_video=None,
    ):
        self.video_items = video_items
        self.use_5frames = use_5frames
        self.radius = radius
        self.max_samples_per_video = max_samples_per_video

        self.data = []
        self.samples = []

        for vid_idx, item in enumerate(video_items):
            video_path = item["video_path"]
            pair_path = item["pair_path"]
            mask_path = item["mask_path"]

            print("\n" + "-" * 80)
            print(f"Loading video {vid_idx + 1}/{len(video_items)}")
            print("video:", video_path)
            print("pair :", pair_path)
            print("mask :", mask_path)

            frames, fps = load_video_gray(video_path, normalize=True)
            T, H, W = frames.shape

            pair_candidates, valid_train_frame = load_pair_npz(pair_path, T=T)

            if len(pair_candidates) != T:
                raise RuntimeError(
                    f"pair_candidates 长度和视频帧数不一致:\n"
                    f"video={video_path}\n"
                    f"len(pair_candidates)={len(pair_candidates)}, T={T}"
                )

            if len(valid_train_frame) != T:
                raise RuntimeError(
                    f"valid_train_frame 长度和视频帧数不一致:\n"
                    f"video={video_path}\n"
                    f"len(valid_train_frame)={len(valid_train_frame)}, T={T}"
                )

            vessel_mask = np.load(mask_path).astype(np.float32)
            vessel_mask = (vessel_mask > 0.5).astype(np.float32)

            if vessel_mask.shape != (H, W):
                raise RuntimeError(
                    f"mask shape 和视频 shape 不一致:\n"
                    f"video={video_path}\n"
                    f"mask={vessel_mask.shape}, video={(H, W)}"
                )

            self.data.append({
                "frames": frames,
                "fps": fps,
                "pair_candidates": pair_candidates,
                "valid_train_frame": valid_train_frame,
                "vessel_mask": vessel_mask,
                "video_path": video_path,
            })

            video_samples = []

            for t in range(T):
                if not valid_train_frame[t]:
                    continue

                if len(pair_candidates[t]) == 0:
                    continue

                video_samples.append((vid_idx, t))

            if max_samples_per_video is not None:
                random.shuffle(video_samples)
                video_samples = video_samples[:max_samples_per_video]

            self.samples.extend(video_samples)

            print("frames:", T)
            print("usable samples:", len(video_samples))

        print("\nTotal usable samples:", len(self.samples))

        if len(self.samples) == 0:
            raise RuntimeError("没有任何可训练 sample，检查 pair_candidates 或 valid_train_frame")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vid_idx, t = self.samples[idx]
        item = self.data[vid_idx]

        frames = item["frames"]
        pair_candidates = item["pair_candidates"]
        vessel_mask = item["vessel_mask"]

        input_frame = frames[t:t + 1]
        input_center = frames[t:t + 1]

        js = pair_candidates[t]

        if len(js) == 0:
            raise RuntimeError(f"frame {t} 没有 pair")

        target_stack = []

        for j in js:
            j = int(j)

            if 0 <= j < frames.shape[0]:
                target_stack.append(frames[j:j + 1])

        if len(target_stack) == 0:
            raise RuntimeError(f"frame {t} 的 pair 都越界了")

        target_stack = np.stack(target_stack, axis=0)
        # shape = (K, 1, H, W)

        vessel_mask = vessel_mask[None, :, :]
        background_mask = 1.0 - vessel_mask

        input_tensor = torch.from_numpy(input_frame.astype(np.float32))
        target_tensor = torch.from_numpy(target_stack.astype(np.float32))
        input_center_tensor = torch.from_numpy(input_center.astype(np.float32))
        vessel_mask_tensor = torch.from_numpy(vessel_mask.astype(np.float32))
        background_mask_tensor = torch.from_numpy(background_mask.astype(np.float32))

        return (
            input_tensor,
            target_tensor,
            input_center_tensor,
            vessel_mask_tensor,
            background_mask_tensor,
        )


# ============================================================
# 5. Residual U-Net
# ============================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ResidualUNet(nn.Module):
    def __init__(
        self,
        in_ch=1,
        features=(32, 64, 128),
        residual_scale=0.15,
    ):
        super().__init__()

        self.residual_scale = residual_scale

        self.enc1 = ConvBlock(in_ch, features[0])
        self.enc2 = ConvBlock(features[0], features[1])
        self.enc3 = ConvBlock(features[1], features[2])

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(features[2], features[2] * 2)

        self.up3 = nn.ConvTranspose2d(
            features[2] * 2,
            features[2],
            kernel_size=2,
            stride=2
        )
        self.dec3 = ConvBlock(features[2] * 2, features[2])

        self.up2 = nn.ConvTranspose2d(
            features[2],
            features[1],
            kernel_size=2,
            stride=2
        )
        self.dec2 = ConvBlock(features[1] * 2, features[1])

        self.up1 = nn.ConvTranspose2d(
            features[1],
            features[0],
            kernel_size=2,
            stride=2
        )
        self.dec1 = ConvBlock(features[0] * 2, features[0])

        # 输出 raw residual，不用 sigmoid
        self.out = nn.Conv2d(features[0], 1, kernel_size=1)

    def forward(self, x):
        input_frame = x[:, 0:1, :, :]

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        raw_residual = self.out(d1)

        residual = self.residual_scale * torch.tanh(raw_residual)

        output = input_frame - residual
        output = torch.clamp(output, 0.0, 1.0)

        return output, residual


# ============================================================
# 6. Loss: output 和所有 pairs 对比后取平均
# ============================================================

def gradient_x(img):
    return img[:, :, :, 1:] - img[:, :, :, :-1]


def gradient_y(img):
    return img[:, :, 1:, :] - img[:, :, :-1, :]


def masked_smooth_l1(a, b, mask):
    eps = 1e-8
    loss_map = F.smooth_l1_loss(a, b, reduction="none")
    return (loss_map * mask).sum() / (mask.sum() + eps)


def residual_smoothness(residual):
    gx = gradient_x(residual)
    gy = gradient_y(residual)

    return torch.mean(torch.abs(gx)) + torch.mean(torch.abs(gy))


def n2n_vessel_background_residual_loss(
    output,
    residual,
    target,
    input_center,
    vessel_mask,
    background_mask,
    w_bg=0.50,
    w_vessel_n2n=0.10,
    w_vessel_brightness=0.20,
    w_vessel_edge=0.15,
    w_bg_smooth=0.05,
    w_residual_size=0.01,
    w_residual_smooth=0.005,
):
    """
    output:
        (B, 1, H, W)

    residual:
        (B, 1, H, W)

    target:
        (B, K, 1, H, W)
        K = 当前 frame t 的所有 pair frames
    """

    eps = 1e-8

    B, K, C, H, W = target.shape

    output_k = output[:, None, :, :, :].expand(-1, K, -1, -1, -1)

    vessel_mask_k = vessel_mask[:, None, :, :, :].expand(-1, K, -1, -1, -1)
    background_mask_k = background_mask[:, None, :, :, :].expand(-1, K, -1, -1, -1)

    n2n_loss_map = F.smooth_l1_loss(
        output_k,
        target,
        reduction="none"
    )

    bg_loss = (n2n_loss_map * background_mask_k).sum() / (
        background_mask_k.sum() + eps
    )

    vessel_n2n_loss = (n2n_loss_map * vessel_mask_k).sum() / (
        vessel_mask_k.sum() + eps
    )

    out_vessel_mean = (output * vessel_mask).sum(dim=(1, 2, 3)) / (
        vessel_mask.sum(dim=(1, 2, 3)) + eps
    )

    in_vessel_mean = (input_center * vessel_mask).sum(dim=(1, 2, 3)) / (
        vessel_mask.sum(dim=(1, 2, 3)) + eps
    )

    vessel_brightness_loss = F.smooth_l1_loss(
        out_vessel_mean,
        in_vessel_mean
    )

    gx_out = gradient_x(output)
    gx_in = gradient_x(input_center)
    gx_mask = vessel_mask[:, :, :, 1:]

    gy_out = gradient_y(output)
    gy_in = gradient_y(input_center)
    gy_mask = vessel_mask[:, :, 1:, :]

    vessel_edge_loss = (
        masked_smooth_l1(gx_out, gx_in, gx_mask)
        + masked_smooth_l1(gy_out, gy_in, gy_mask)
    )

    gx_bg = gradient_x(output)
    gy_bg = gradient_y(output)

    bg_mask_x = background_mask[:, :, :, 1:]
    bg_mask_y = background_mask[:, :, 1:, :]

    bg_smooth_loss = (
        torch.mean(torch.abs(gx_bg * bg_mask_x))
        + torch.mean(torch.abs(gy_bg * bg_mask_y))
    )

    residual_size_loss = torch.mean(torch.abs(residual))
    residual_smooth_loss = residual_smoothness(residual)

    total = (
        w_bg * bg_loss
        + w_vessel_n2n * vessel_n2n_loss
        + w_vessel_brightness * vessel_brightness_loss
        + w_vessel_edge * vessel_edge_loss
        + w_bg_smooth * bg_smooth_loss
        + w_residual_size * residual_size_loss
        + w_residual_smooth * residual_smooth_loss
    )

    return total


# ============================================================
# 7. Training
# ============================================================

def train_n2n_model(
    train_dataset,
    val_dataset,
    batch_size=1,
    epochs=40,
    lr=1e-4,
    save_path="best_residual_unet.pth",
    residual_scale=0.15,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nDevice:", device)

    model = ResidualUNet(
        in_ch=1,
        features=(32, 64, 128),
        residual_scale=residual_scale,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-5
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    best_val = float("inf")

    for epoch in range(1, epochs + 1):

        model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]")

        for input_tensor, target_tensor, input_center, vessel_mask, background_mask in pbar:
            input_tensor = input_tensor.to(device)
            target_tensor = target_tensor.to(device)
            input_center = input_center.to(device)
            vessel_mask = vessel_mask.to(device)
            background_mask = background_mask.to(device)

            output, residual = model(input_tensor)

            loss = n2n_vessel_background_residual_loss(
                output=output,
                residual=residual,
                target=target_tensor,
                input_center=input_center,
                vessel_mask=vessel_mask,
                background_mask=background_mask,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            pbar.set_postfix(loss=float(np.mean(train_losses)))

        mean_train = float(np.mean(train_losses))

        model.eval()
        val_losses = []

        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val]")

            for input_tensor, target_tensor, input_center, vessel_mask, background_mask in pbar_val:
                input_tensor = input_tensor.to(device)
                target_tensor = target_tensor.to(device)
                input_center = input_center.to(device)
                vessel_mask = vessel_mask.to(device)
                background_mask = background_mask.to(device)

                output, residual = model(input_tensor)

                loss = n2n_vessel_background_residual_loss(
                    output=output,
                    residual=residual,
                    target=target_tensor,
                    input_center=input_center,
                    vessel_mask=vessel_mask,
                    background_mask=background_mask,
                )

                val_losses.append(loss.item())
                pbar_val.set_postfix(loss=float(np.mean(val_losses)))

        mean_val = float(np.mean(val_losses))

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train loss = {mean_train:.6f} | "
            f"val loss = {mean_val:.6f}"
        )

        if mean_val < best_val:
            best_val = mean_val
            torch.save(model.state_dict(), save_path)
            print("Saved best model:", save_path)

    return model


# ============================================================
# 8. Inference
# ============================================================

def denoise_video_with_model(
    model_path,
    test_item,
    output_video_path,
    residual_video_path=None,
    residual_scale=0.15,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nInference device:", device)

    frames, fps = load_video_gray(test_item["video_path"], normalize=True)
    T, H, W = frames.shape

    model = ResidualUNet(
        in_ch=1,
        features=(32, 64, 128),
        residual_scale=residual_scale,
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    denoised_frames = []
    residual_frames = []

    with torch.no_grad():
        for t in tqdm(range(T), desc="Denoising test video"):
            input_frame = frames[t:t + 1]

            x = torch.from_numpy(input_frame.astype(np.float32))[None, :, :, :]
            x = x.to(device)

            y, residual = model(x)

            y_np = y[0, 0].detach().cpu().numpy()
            residual_np = residual[0, 0].detach().cpu().numpy()

            y_u8 = np.clip(y_np * 255.0, 0, 255).astype(np.uint8)
            denoised_frames.append(y_u8)

            residual_vis = residual_np / (residual_scale + 1e-8)
            residual_vis = np.clip(residual_vis, -1, 1)
            residual_vis = ((residual_vis + 1.0) * 127.5).astype(np.uint8)
            residual_frames.append(residual_vis)

    output_video_path = Path(output_video_path)
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")

    writer = cv2.VideoWriter(
        str(output_video_path),
        fourcc,
        fps,
        (W, H),
        isColor=False
    )

    for frame_u8 in denoised_frames:
        writer.write(frame_u8)

    writer.release()

    print("\nDenoised test video saved to:")
    print(output_video_path)

    if residual_video_path is not None:
        residual_video_path = Path(residual_video_path)
        residual_video_path.parent.mkdir(parents=True, exist_ok=True)

        writer_res = cv2.VideoWriter(
            str(residual_video_path),
            fourcc,
            fps,
            (W, H),
            isColor=False
        )

        for frame_u8 in residual_frames:
            writer_res.write(frame_u8)

        writer_res.release()

        print("\nPredicted residual video saved to:")
        print(residual_video_path)


# ============================================================
# 9. Build datasets + train + test
# ============================================================

train_dataset = MultiVideoN2NDataset(
    video_items=train_items,
    use_5frames=USE_5FRAMES,
    radius=RADIUS,
    max_samples_per_video=None,
)

val_dataset = MultiVideoN2NDataset(
    video_items=val_items,
    use_5frames=USE_5FRAMES,
    radius=RADIUS,
    max_samples_per_video=None,
)

model = train_n2n_model(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    lr=LR,
    save_path=str(BEST_MODEL_PATH),
    residual_scale=RESIDUAL_SCALE,
)

denoise_video_with_model(
    model_path=str(BEST_MODEL_PATH),
    test_item=test_item,
    output_video_path=str(DENOISED_TEST_VIDEO_PATH),
    residual_video_path=str(RESIDUAL_TEST_VIDEO_PATH),
    residual_scale=RESIDUAL_SCALE,
)