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
# 0. 路径设置
# ============================================================

VIDEO_DIR = Path(r"C:\Users\Novovorontsovka\Downloads\N2N")

PAIR_DIR = Path(r"C:\Users\Novovorontsovka\Downloads\LDH_traite\N2N_pair_results")
MASK_DIR = Path(r"C:\Users\Novovorontsovka\Downloads\LDH_traite\N2N_vessel_masks")
BRIGHTNESS_DIR = Path(r"C:\Users\Novovorontsovka\Downloads\LDH_traite\N2N_brightness_tables")

OUTPUT_DIR = Path(r"C:\Users\Novovorontsovka\Downloads\resultat\N2N_training_residual_unet_frame_t")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = OUTPUT_DIR / "best_residual_unet_frame_t.pth"
DENOISED_TEST_VIDEO_PATH = OUTPUT_DIR / "denoised_test_video_residual_unet_frame_t.avi"
RESIDUAL_TEST_VIDEO_PATH = OUTPUT_DIR / "predicted_residual_video.avi"


# ============================================================
# 1. 训练参数
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

BATCH_SIZE = 2
EPOCHS = 60
LR = 1e-4

VAL_RATIO = 0.20

EARLY_STOP_PATIENCE = 3
MIN_IMPROVE_RATIO = 0.005

RESIDUAL_SCALE = 0.15


# ============================================================
# 2. 构建 video_items
# ============================================================

video_ids = [10, 11, 12, 13, 14, 15, 16, 17, 1, 2, 3, 4, 6, 7, 8, 9]

video_items = []

for idx in video_ids:
    video_name = f"260310_AUZ0752_{idx}_HD_M0"

    video_path = VIDEO_DIR / f"{video_name}.avi"
    pair_path = PAIR_DIR / f"{video_name}_pair_result.npz"
    mask_path = MASK_DIR / f"{video_name}_vessel_mask.npy"
    brightness_table_path = BRIGHTNESS_DIR / f"{video_name}_brightness_table.npy"

    video_items.append({
        "id": idx,
        "video_name": video_name,
        "video_path": str(video_path),
        "pair_path": str(pair_path),
        "mask_path": str(mask_path),
        "brightness_table_path": str(brightness_table_path),
    })


# ============================================================
# 3. 路径检查
# ============================================================

print("\n" + "=" * 80)
print("CHECK video_items")
print("=" * 80)

for item in video_items:
    print("\nVideo ID:", item["id"])
    print("Video name:", item["video_name"])

    for key in ["video_path", "pair_path", "mask_path", "brightness_table_path"]:
        p = Path(item[key])
        if p.exists():
            print("OK  :", key, p)
        else:
            print("MISS:", key, p)

print("\nvideo_items 数量:", len(video_items))


# ============================================================
# 4. train / validation / test split
#    最后一个 video 留给 test
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

print("\nTrain videos:")
for item in train_items:
    print("  ", Path(item["video_path"]).name)

print("\nValidation videos:")
for item in val_items:
    print("  ", Path(item["video_path"]).name)

print("\nTest video:")
print("  ", Path(test_item["video_path"]).name)


# ============================================================
# 5. 读取 video
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


# ============================================================
# 6. 读取 pair
# ============================================================

def load_pair_npz(pair_path, T):
    """
    支持两种 pair 文件格式：

    格式 A:
        pair_candidates
        valid_train_frame

    格式 B:
        pair_indices, shape=(N, 2)
        第 0 列 = t
        第 1 列 = j
    """

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
        f"pair 文件里没有找到可用 key: {pair_path}\n"
        f"已有 keys = {files}\n"
        f"需要 pair_candidates + valid_train_frame 或 pair_indices"
    )


# ============================================================
# 7. 读取 smooth brightness curve
# ============================================================

def load_smooth_brightness_curve(brightness_table_path, T):
    """
    支持两种 brightness table:

    新格式:
        shape = (T, 2)
        [:, 0] = time_s
        [:, 1] = smooth_intensity

    旧格式:
        shape = (T, 5)
        [:, 0] = frame_index
        [:, 1] = time_s
        [:, 2] = raw_intensity
        [:, 3] = smooth_intensity
        [:, 4] = is_peak
    """

    brightness_table_path = Path(brightness_table_path)

    if not brightness_table_path.exists():
        raise FileNotFoundError(f"找不到 brightness table: {brightness_table_path}")

    table = np.load(brightness_table_path).astype(np.float32)

    if table.ndim != 2:
        raise RuntimeError(
            f"brightness table 维度不对: {brightness_table_path}, shape={table.shape}"
        )

    if table.shape[0] != T:
        raise RuntimeError(
            f"brightness table 长度和视频帧数不一致:\n"
            f"brightness_table={brightness_table_path}\n"
            f"table T={table.shape[0]}, video T={T}"
        )

    if table.shape[1] == 2:
        brightness_curve = table[:, 1]

    elif table.shape[1] >= 5:
        brightness_curve = table[:, 3]

    else:
        raise RuntimeError(
            f"brightness table 列数不对: {brightness_table_path}, shape={table.shape}"
        )

    if brightness_curve.max() > 1.5:
        brightness_curve = brightness_curve / 255.0

    brightness_curve = np.clip(brightness_curve, 0.0, 1.0).astype(np.float32)

    return brightness_curve


# ============================================================
# 8. Dataset
# ============================================================

class MultiVideoN2NDataset(Dataset):
    def __init__(
        self,
        video_items,
        max_samples_per_video=None,
    ):
        """
        每个样本:

            input = frame[t]

        shape = (1, H, W)

        target = pair_candidates[t] 里随机选一个 frame[j]

        vessel_mask 不进入 input，只在 loss 中用。
        brightness_curve 不进入 input，只在 temporal loss 中用。
        """

        self.video_items = video_items
        self.max_samples_per_video = max_samples_per_video

        self.data = []
        self.samples = []

        for vid_idx, item in enumerate(video_items):
            video_path = item["video_path"]
            pair_path = item["pair_path"]
            mask_path = item["mask_path"]
            brightness_table_path = item["brightness_table_path"]

            print("\n" + "-" * 80)
            print(f"Loading video {vid_idx + 1}/{len(video_items)}")
            print("video      :", video_path)
            print("pair       :", pair_path)
            print("mask       :", mask_path)
            print("brightness :", brightness_table_path)

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

            brightness_curve = load_smooth_brightness_curve(
                brightness_table_path=brightness_table_path,
                T=T
            )

            self.data.append({
                "frames": frames,
                "fps": fps,
                "pair_candidates": pair_candidates,
                "valid_train_frame": valid_train_frame,
                "vessel_mask": vessel_mask,
                "brightness_curve": brightness_curve,
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
            print(
                "brightness_curve min/max:",
                float(brightness_curve.min()),
                float(brightness_curve.max())
            )

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
        vessel_mask_2d = item["vessel_mask"]
        brightness_curve = item["brightness_curve"]

        j = random.choice(pair_candidates[t])

        input_frame = frames[t:t + 1].astype(np.float32)
        input_center = frames[t:t + 1].astype(np.float32)
        target = frames[j:j + 1].astype(np.float32)

        vessel_mask = vessel_mask_2d[None, :, :].astype(np.float32)
        background_mask = 1.0 - vessel_mask

        temporal_intensity = np.array(
            [brightness_curve[t]],
            dtype=np.float32
        )

        input_tensor = torch.from_numpy(input_frame)
        target_tensor = torch.from_numpy(target)
        input_center_tensor = torch.from_numpy(input_center)
        vessel_mask_tensor = torch.from_numpy(vessel_mask)
        background_mask_tensor = torch.from_numpy(background_mask)
        temporal_intensity_tensor = torch.from_numpy(temporal_intensity)

        return (
            input_tensor,
            target_tensor,
            input_center_tensor,
            vessel_mask_tensor,
            background_mask_tensor,
            temporal_intensity_tensor,
        )


# ============================================================
# 9. Residual U-Net
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
# 10. Loss
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


def n2n_vessel_temporal_residual_loss(
    output,
    residual,
    target,
    input_center,
    vessel_mask,
    background_mask,
    temporal_intensity,

    w_bg=0.45,
    w_vessel_n2n=0.03,
    w_vessel_brightness=0.10,
    w_vessel_edge=0.22,
    w_vessel_temporal=0.10,
    w_bg_smooth=0.08,

    w_residual_size=0.015,
    w_residual_smooth=0.005,
):
    eps = 1e-8

    bg_loss = masked_smooth_l1(
        output,
        target,
        background_mask
    )

    vessel_n2n_loss = masked_smooth_l1(
        output,
        target,
        vessel_mask
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

    temporal_intensity = temporal_intensity.view(-1)

    vessel_temporal_loss = F.smooth_l1_loss(
        out_vessel_mean,
        temporal_intensity
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
        + w_vessel_temporal * vessel_temporal_loss
        + w_bg_smooth * bg_smooth_loss
        + w_residual_size * residual_size_loss
        + w_residual_smooth * residual_smooth_loss
    )

    return total


# ============================================================
# 11. Training with early stopping
# ============================================================

def train_n2n_model(
    train_dataset,
    val_dataset,
    batch_size=2,
    epochs=40,
    lr=1e-4,
    save_path="best_model.pth",
    patience=3,
    min_improve_ratio=0.005,
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
    previous_val = None
    small_improve_count = 0

    train_loss_history = []
    val_loss_history = []
    improve_ratio_history = []

    for epoch in range(1, epochs + 1):

        model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]")

        for (
            input_tensor,
            target_tensor,
            input_center,
            vessel_mask,
            background_mask,
            temporal_intensity,
        ) in pbar:

            input_tensor = input_tensor.to(device)
            target_tensor = target_tensor.to(device)
            input_center = input_center.to(device)
            vessel_mask = vessel_mask.to(device)
            background_mask = background_mask.to(device)
            temporal_intensity = temporal_intensity.to(device)

            output, residual = model(input_tensor)

            loss = n2n_vessel_temporal_residual_loss(
                output=output,
                residual=residual,
                target=target_tensor,
                input_center=input_center,
                vessel_mask=vessel_mask,
                background_mask=background_mask,
                temporal_intensity=temporal_intensity,
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

            for (
                input_tensor,
                target_tensor,
                input_center,
                vessel_mask,
                background_mask,
                temporal_intensity,
            ) in pbar_val:

                input_tensor = input_tensor.to(device)
                target_tensor = target_tensor.to(device)
                input_center = input_center.to(device)
                vessel_mask = vessel_mask.to(device)
                background_mask = background_mask.to(device)
                temporal_intensity = temporal_intensity.to(device)

                output, residual = model(input_tensor)

                loss = n2n_vessel_temporal_residual_loss(
                    output=output,
                    residual=residual,
                    target=target_tensor,
                    input_center=input_center,
                    vessel_mask=vessel_mask,
                    background_mask=background_mask,
                    temporal_intensity=temporal_intensity,
                )

                val_losses.append(loss.item())
                pbar_val.set_postfix(loss=float(np.mean(val_losses)))

        mean_val = float(np.mean(val_losses))

        train_loss_history.append(mean_train)
        val_loss_history.append(mean_val)

        if previous_val is None:
            improve_ratio = None
            improve_ratio_history.append(np.nan)

            print(
                f"Epoch {epoch}/{epochs} | "
                f"train loss = {mean_train:.6f} | "
                f"val loss = {mean_val:.6f} | "
                f"improve = first epoch"
            )

        else:
            improve_ratio = (previous_val - mean_val) / (previous_val + 1e-8)
            improve_ratio_history.append(improve_ratio)

            print(
                f"Epoch {epoch}/{epochs} | "
                f"train loss = {mean_train:.6f} | "
                f"val loss = {mean_val:.6f} | "
                f"improve = {improve_ratio * 100:.4f}%"
            )

            if improve_ratio < min_improve_ratio:
                small_improve_count += 1

                print(
                    f"val loss 下降太小: "
                    f"{improve_ratio * 100:.4f}% < {min_improve_ratio * 100:.4f}%"
                )

                print(
                    f"small_improve_count = {small_improve_count}/{patience}"
                )

            else:
                small_improve_count = 0
                print("val loss 有明显下降，early stopping 计数清零")

        previous_val = mean_val

        if mean_val < best_val:
            best_val = mean_val
            torch.save(model.state_dict(), save_path)
            print("Saved best model:", save_path)
            print("best_val =", best_val)

        if small_improve_count >= patience:
            print("\n" + "=" * 80)
            print("Early stopping triggered")
            print("=" * 80)

            print(
                f"原因：val loss 连续 {patience} 轮下降比例 "
                f"小于 {min_improve_ratio * 100:.4f}%"
            )

            print("停止在 epoch:", epoch)
            print("best_val:", best_val)
            print("best model saved at:", save_path)
            break

    history_path = Path(save_path).parent / "loss_history_residual_unet_frame_t.npz"

    np.savez_compressed(
        history_path,
        train_loss=np.array(train_loss_history, dtype=np.float32),
        val_loss=np.array(val_loss_history, dtype=np.float32),
        improve_ratio=np.array(improve_ratio_history, dtype=np.float32),
        best_val=np.array(best_val, dtype=np.float32),
    )

    print("\nLoss history saved to:")
    print(history_path)

    return model


# ============================================================
# 12. Inference on test video
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

            input_frame = frames[t:t + 1].astype(np.float32)

            x = torch.from_numpy(input_frame)[None, :, :, :]
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
# 13. Build datasets + train + test
# ============================================================

if __name__ == "__main__":

    train_dataset = MultiVideoN2NDataset(
        video_items=train_items,
        max_samples_per_video=None,
    )

    val_dataset = MultiVideoN2NDataset(
        video_items=val_items,
        max_samples_per_video=None,
    )

    model = train_n2n_model(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        lr=LR,
        save_path=str(BEST_MODEL_PATH),
        patience=EARLY_STOP_PATIENCE,
        min_improve_ratio=MIN_IMPROVE_RATIO,
        residual_scale=RESIDUAL_SCALE,
    )

    denoise_video_with_model(
        model_path=str(BEST_MODEL_PATH),
        test_item=test_item,
        output_video_path=str(DENOISED_TEST_VIDEO_PATH),
        residual_video_path=str(RESIDUAL_TEST_VIDEO_PATH),
        residual_scale=RESIDUAL_SCALE,
    )