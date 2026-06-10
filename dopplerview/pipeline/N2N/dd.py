from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# ============================================================
# 1. 路径
# ============================================================

VIDEO_DIR = Path(r"C:\Users\Novovorontsovka\Downloads\ww")
OUTPUT_DIR = VIDEO_DIR / "comparison_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

video_files = sorted(
    list(VIDEO_DIR.glob("*.avi")) +
    list(VIDEO_DIR.glob("*.mp4")) +
    list(VIDEO_DIR.glob("*.mov")) +
    list(VIDEO_DIR.glob("*.mkv"))
)

if len(video_files) < 2:
    raise RuntimeError(f"这个文件夹里少于两个视频: {VIDEO_DIR}")

VIDEO_A = video_files[0]
VIDEO_B = video_files[1]

print("视频 A:", VIDEO_A)
print("视频 B:", VIDEO_B)


# ============================================================
# 2. 读取视频信息
# ============================================================

def get_video_info(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {path}")

    info = {
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }

    info["duration_sec"] = info["frame_count"] / info["fps"] if info["fps"] > 0 else None

    cap.release()
    return info


info_a = get_video_info(VIDEO_A)
info_b = get_video_info(VIDEO_B)

print("\n===== 视频信息对比 =====")
print("A:", info_a)
print("B:", info_b)


# ============================================================
# 3. 逐帧读取并对比
# ============================================================

def read_video_gray(path, max_frames=None):
    cap = cv2.VideoCapture(str(path))
    frames = []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames is not None:
        total = min(total, max_frames)

    for _ in tqdm(range(total), desc=f"读取 {path.name}"):
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray.astype(np.float32) / 255.0)

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"没有读到任何帧: {path}")

    return np.stack(frames, axis=0)


frames_a = read_video_gray(VIDEO_A)
frames_b = read_video_gray(VIDEO_B)

# 对齐帧数
n = min(len(frames_a), len(frames_b))
frames_a = frames_a[:n]
frames_b = frames_b[:n]

# 如果尺寸不同，把 B resize 到 A
h, w = frames_a.shape[1:]
if frames_b.shape[1:] != (h, w):
    print("两个视频尺寸不同，正在把 B resize 到 A 的尺寸")
    frames_b_resized = []
    for f in frames_b:
        frames_b_resized.append(cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR))
    frames_b = np.stack(frames_b_resized, axis=0)

print("\n用于对比的帧数:", n)
print("最终尺寸:", frames_a.shape)


# ============================================================
# 4. 计算差异
# ============================================================

diff = np.abs(frames_a - frames_b)

mse_per_frame = np.mean((frames_a - frames_b) ** 2, axis=(1, 2))
mae_per_frame = np.mean(diff, axis=(1, 2))

brightness_a = np.mean(frames_a, axis=(1, 2))
brightness_b = np.mean(frames_b, axis=(1, 2))

print("\n===== 数值对比 =====")
print("平均 MSE:", float(np.mean(mse_per_frame)))
print("平均 MAE:", float(np.mean(mae_per_frame)))
print("A 平均亮度:", float(np.mean(brightness_a)))
print("B 平均亮度:", float(np.mean(brightness_b)))


# ============================================================
# 5. 保存几张关键帧对比图
# ============================================================

sample_indices = [
    0,
    n // 4,
    n // 2,
    3 * n // 4,
    n - 1
]

for idx in sample_indices:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(frames_a[idx], cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"A - frame {idx}")
    axes[0].axis("off")

    axes[1].imshow(frames_b[idx], cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"B - frame {idx}")
    axes[1].axis("off")

    axes[2].imshow(diff[idx], cmap="hot")
    axes[2].set_title(f"Abs difference - frame {idx}")
    axes[2].axis("off")

    plt.tight_layout()
    save_path = OUTPUT_DIR / f"frame_compare_{idx:04d}.png"
    plt.savefig(save_path, dpi=150)
    plt.show()

    print("保存:", save_path)


# ============================================================
# 6. 画亮度曲线和误差曲线
# ============================================================

plt.figure(figsize=(12, 5))
plt.plot(brightness_a, label="Video A brightness")
plt.plot(brightness_b, label="Video B brightness")
plt.xlabel("Frame")
plt.ylabel("Mean intensity")
plt.title("Brightness curve comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "brightness_comparison.png", dpi=150)
plt.show()


plt.figure(figsize=(12, 5))
plt.plot(mse_per_frame, label="MSE per frame")
plt.plot(mae_per_frame, label="MAE per frame")
plt.xlabel("Frame")
plt.ylabel("Difference")
plt.title("Frame difference curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "difference_curve.png", dpi=150)
plt.show()


# ============================================================
# 7. 输出对比视频：A | B | difference
# ============================================================

out_path = OUTPUT_DIR / "side_by_side_comparison.avi"

fps = info_a["fps"] if info_a["fps"] > 0 else 30
fourcc = cv2.VideoWriter_fourcc(*"XVID")

# 输出宽度：三个图横向拼接
out_w = w * 3
out_h = h

writer = cv2.VideoWriter(str(out_path), fourcc, fps, (out_w, out_h))

for i in tqdm(range(n), desc="生成对比视频"):
    a = (frames_a[i] * 255).astype(np.uint8)
    b = (frames_b[i] * 255).astype(np.uint8)

    d = diff[i]
    d_norm = d / (d.max() + 1e-8)
    d_uint8 = (d_norm * 255).astype(np.uint8)
    d_color = cv2.applyColorMap(d_uint8, cv2.COLORMAP_HOT)

    a_color = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
    b_color = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)

    combined = np.concatenate([a_color, b_color, d_color], axis=1)

    cv2.putText(combined, "Video A", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(combined, "Video B", (w + 20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(combined, "Abs Difference", (2 * w + 20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    writer.write(combined)

writer.release()

print("\n完成！")
print("所有结果保存在:", OUTPUT_DIR)
print("对比视频:", out_path)