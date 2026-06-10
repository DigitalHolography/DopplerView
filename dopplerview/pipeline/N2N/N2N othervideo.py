from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn


# ============================================================
# 0. 路径
# ============================================================

VIDEO_DIR = Path(r"C:\Users\Novovorontsovka\Downloads\N2N")

MASK_DIR = Path(r"C:\Users\Novovorontsovka\Downloads\LDH_traite\N2N_vessel_masks")
BRIGHTNESS_DIR = Path(r"C:\Users\Novovorontsovka\Downloads\LDH_traite\N2N_brightness_tables")

OUTPUT_DIR = Path(r"C:\Users\Novovorontsovka\Downloads\resultat\N2N_training_1frame_all_pairs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = Path(
    r"C:\Users\Novovorontsovka\Downloads\resultat\N2N_training_1frame_all_pairs\best_n2n_model_1frame_all_pairs.pth"
)

# ============================================================
# 你想换哪个视频，就改这里
# ============================================================

TEST_VIDEO_NAME = "260310_AUZ0752_9_HD_M0"

VIDEO_PATH = VIDEO_DIR / f"{TEST_VIDEO_NAME}.avi"
MASK_PATH = MASK_DIR / f"{TEST_VIDEO_NAME}_vessel_mask.npy"
BRIGHTNESS_PATH = BRIGHTNESS_DIR / f"{TEST_VIDEO_NAME}_brightness_table.npy"

OUTPUT_VIDEO_PATH = OUTPUT_DIR / f"denoised_{TEST_VIDEO_NAME}.avi"
COMPARE_VIDEO_PATH = OUTPUT_DIR / f"comparison_raw_vs_denoised_{TEST_VIDEO_NAME}.avi"


# ============================================================
# 1. 读取 video
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
# 2. 读取 brightness curve
# ============================================================

def load_smooth_brightness_curve(brightness_table_path, T):
    table = np.load(brightness_table_path).astype(np.float32)

    if table.ndim != 2:
        raise RuntimeError(f"brightness table 维度不对: {table.shape}")

    if table.shape[0] != T:
        raise RuntimeError(
            f"brightness table 长度和视频帧数不一致: "
            f"table={table.shape[0]}, video={T}"
        )

    if table.shape[1] == 2:
        brightness_curve = table[:, 1]
    elif table.shape[1] >= 5:
        brightness_curve = table[:, 3]
    else:
        raise RuntimeError(f"brightness table 列数不对: {table.shape}")

    if brightness_curve.max() > 1.5:
        brightness_curve = brightness_curve / 255.0

    brightness_curve = np.clip(brightness_curve, 0.0, 1.0).astype(np.float32)

    return brightness_curve


# ============================================================
# 3. U-Net，必须和训练时一样
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


class SmallUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, features=(32, 64, 128)):
        super().__init__()

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

        self.out = nn.Conv2d(features[0], out_ch, kernel_size=1)

    def forward(self, x):
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

        return torch.sigmoid(self.out(d1))


# ============================================================
# 4. 用训练好的模型 denoise 一个视频
# ============================================================

def denoise_one_video():
    print("model:", MODEL_PATH)
    print("video:", VIDEO_PATH)
    print("mask:", MASK_PATH)
    print("brightness:", BRIGHTNESS_PATH)
    print("denoised output:", OUTPUT_VIDEO_PATH)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"找不到模型: {MODEL_PATH}")

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"找不到视频: {VIDEO_PATH}")

    if not MASK_PATH.exists():
        raise FileNotFoundError(f"找不到 vessel mask: {MASK_PATH}")

    if not BRIGHTNESS_PATH.exists():
        raise FileNotFoundError(f"找不到 brightness table: {BRIGHTNESS_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    frames, fps = load_video_gray(VIDEO_PATH, normalize=True)
    T, H, W = frames.shape

    vessel_mask = np.load(MASK_PATH).astype(np.float32)
    vessel_mask = (vessel_mask > 0.5).astype(np.float32)

    if vessel_mask.shape != (H, W):
        raise RuntimeError(
            f"mask shape 和 video shape 不一致: "
            f"mask={vessel_mask.shape}, video={(H, W)}"
        )

    brightness_curve = load_smooth_brightness_curve(
        BRIGHTNESS_PATH,
        T=T
    )

    model = SmallUNet(
        in_ch=3,
        out_ch=1,
        features=(32, 64, 128)
    ).to(device)

    state = torch.load(MODEL_PATH, map_location=device)

    # 兼容两种保存方式：
    # 1. torch.save(model.state_dict(), path)
    # 2. torch.save({"model_state_dict": model.state_dict()}, path)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)

    model.eval()

    vessel_mask_ch = vessel_mask[None, :, :].astype(np.float32)

    denoised_frames = []

    with torch.no_grad():
        for t in tqdm(range(T), desc="Denoising"):
            input_frame = frames[t:t + 1].astype(np.float32)

            brightness_value = float(brightness_curve[t])
            vessel_brightness_map = vessel_mask_ch * brightness_value

            input_with_condition = np.concatenate(
                [
                    input_frame,
                    vessel_mask_ch,
                    vessel_brightness_map,
                ],
                axis=0
            )

            x = torch.from_numpy(input_with_condition.astype(np.float32))[None, :, :, :]
            x = x.to(device)

            y = model(x)
            y = y[0, 0].detach().cpu().numpy()

            y_u8 = np.clip(y * 255.0, 0, 255).astype(np.uint8)
            denoised_frames.append(y_u8)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO_PATH),
        fourcc,
        fps,
        (W, H),
        isColor=False
    )

    for frame_u8 in denoised_frames:
        writer.write(frame_u8)

    writer.release()

    print("\n完成 denoised 视频:")
    print(OUTPUT_VIDEO_PATH)

    return frames, denoised_frames, fps


# ============================================================
# 5. 做原视频 vs 去噪视频左右对比
# ============================================================

def make_comparison_video(raw_frames, denoised_frames, fps):
    T, H, W = raw_frames.shape

    if len(denoised_frames) != T:
        raise RuntimeError(
            f"denoised frame 数量不对: denoised={len(denoised_frames)}, raw={T}"
        )

    print("comparison output:", COMPARE_VIDEO_PATH)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")

    writer = cv2.VideoWriter(
        str(COMPARE_VIDEO_PATH),
        fourcc,
        fps,
        (W * 2, H),
        isColor=True
    )

    for t in tqdm(range(T), desc="Making comparison"):
        raw_u8 = np.clip(raw_frames[t] * 255.0, 0, 255).astype(np.uint8)
        den_u8 = denoised_frames[t]

        raw_bgr = cv2.cvtColor(raw_u8, cv2.COLOR_GRAY2BGR)
        den_bgr = cv2.cvtColor(den_u8, cv2.COLOR_GRAY2BGR)

        cv2.putText(
            raw_bgr,
            "Raw video",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            den_bgr,
            "Denoised video",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        compare = np.concatenate([raw_bgr, den_bgr], axis=1)

        writer.write(compare)

    writer.release()

    print("\n完成 comparison 视频:")
    print(COMPARE_VIDEO_PATH)


# ============================================================
# 6. main
# ============================================================

if __name__ == "__main__":
    raw_frames, denoised_frames, fps = denoise_one_video()
    make_comparison_video(raw_frames, denoised_frames, fps)

    print("\n全部完成:")
    print("denoised video:")
    print(OUTPUT_VIDEO_PATH)
    print("comparison video:")
    print(COMPARE_VIDEO_PATH)