# -*- coding: utf-8 -*-
"""
LDH 完整预处理流水线

只需在代码顶部设置 INPUT_DIR 和 OUTPUT_DIR，程序会自动完成：
1. 原始视频 -> 圆形 ROI masked 视频
2. masked 视频 -> 血管增强视频
3. 增强视频 -> vessel mask (.npy)
4. masked 视频 + vessel mask -> 亮度表、亮度曲线、从第一个 peak 开始的视频

输出目录结构：
OUTPUT_DIR/
    01_masked_videos/
    02_enhanced_videos/
    03_vessel_masks/
    04_brightness_tables/
    05_trimmed_videos/
"""

import re
import traceback
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks


# ============================================================
# 0. 输入与输出路径：只需要修改这里
# ============================================================

# 原始 AVI 视频所在文件夹
INPUT_DIR = Path(
    r"C:\Users\Novovorontsovka\Downloads\video_masqued"
)

# 所有处理结果的总输出文件夹
OUTPUT_DIR = Path(
    r"C:\Users\Novovorontsovka\Downloads\video_masqued\output"
)


# ============================================================
# 1. 全局参数：通常不需要修改
# ============================================================

CX = 255
CY = 255
RADIUS = 260

# 增强参数
NORM_LOW = 0.5
NORM_HIGH = 99.7
CLAHE_CLIP = 2.0
CLAHE_GRID = (8, 8)
TOPHAT_SIZE = 17
BACKGROUND_SIGMA = 12
BACKGROUND_WEIGHT = 0.65
FINAL_LOW = 1.0
FINAL_HIGH = 99.5

# Vessel mask 参数
MIN_PEAK_DISTANCE = 15
PEAK_PROMINENCE_RATIO = 0.10
VOTE_RATIO = 0.40
MIN_COMPONENT_AREA = 15
SMOOTH_WINDOW = 5
HIGH_PERCENTILE = 88
LOW_PERCENTILE = 60


# ============================================================
# 1. 通用工具
# ============================================================

def natural_sort_key(path: Path):
    """按文件名中的数字自然排序；没有数字也不会报错。"""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def find_all_avi_videos(input_dir: Path):
    input_dir = Path(input_dir).expanduser().resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"找不到输入文件夹：\n{input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"这不是一个文件夹：\n{input_dir}")

    videos = list(input_dir.glob("*.avi")) + list(input_dir.glob("*.AVI"))
    videos = sorted(set(videos), key=natural_sort_key)

    if not videos:
        raise RuntimeError(f"这个文件夹里没有找到 AVI 视频：\n{input_dir}")

    return videos


def create_circle_mask(height, width, cx=CX, cy=CY, radius=RADIUS):
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), radius, 255, thickness=-1)
    return mask


def open_video(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"打不开视频：\n{video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print("警告：fps 读取异常，暂时设为 30")
        fps = 30.0
    return cap, fps


def create_gray_writer(output_path: Path, fps, width, height):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(
        str(output_path), fourcc, fps, (width, height), isColor=False
    )
    if not writer.isOpened():
        raise RuntimeError(f"无法创建输出视频：\n{output_path}")
    return writer


def load_video_as_gray_float(video_path: Path):
    cap, fps = open_video(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32))

    cap.release()

    if not frames:
        raise RuntimeError(f"视频没有读到任何帧：\n{video_path}")

    return np.stack(frames, axis=0), fps


def smooth_curve_edge(curve, win=5):
    if win <= 1:
        return curve.copy()

    pad_left = win // 2
    pad_right = win - 1 - pad_left
    curve_pad = np.pad(curve, (pad_left, pad_right), mode="edge")
    kernel = np.ones(win, dtype=np.float32) / win
    return np.convolve(curve_pad, kernel, mode="valid").astype(np.float32)


def calculate_peaks(curve, min_distance, prominence_ratio):
    centered = curve - np.median(curve)
    amplitude = np.percentile(centered, 95) - np.percentile(centered, 5)
    prominence = prominence_ratio * amplitude

    # 极端情况下曲线完全平坦，给出明确错误，而不是静默失败。
    if amplitude <= 0:
        raise RuntimeError("亮度曲线没有有效波动，无法寻找 peak。")

    peak_indices, _ = find_peaks(
        centered,
        distance=min_distance,
        prominence=prominence,
    )

    if len(peak_indices) == 0:
        raise RuntimeError(
            "没有找到 peak。可以尝试把 PEAK_PROMINENCE_RATIO "
            "从 0.10 降到 0.05。"
        )

    return peak_indices


# ============================================================
# 2. 第一步：圆形 ROI masked 视频
# ============================================================

def save_circle_masked_video(video_path, output_dir):
    video_path = Path(video_path)
    output_path = Path(output_dir) / f"{video_path.stem}_circle_masked.avi"

    cap, fps = open_video(video_path)
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError(f"这个视频没有读到任何帧：\n{video_path}")

    height, width = first_frame.shape[:2]
    circle_mask = create_circle_mask(height, width)
    writer = create_gray_writer(output_path, fps, width, height)

    try:
        frame = first_frame
        while True:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            masked = cv2.bitwise_and(gray, gray, mask=circle_mask)
            writer.write(masked)

            ret, frame = cap.read()
            if not ret:
                break
    finally:
        cap.release()
        writer.release()

    return output_path


# ============================================================
# 3. 第二步：血管增强视频
# ============================================================

def normalize_to_u8_inside_mask(img, valid_mask, p_low, p_high):
    inside_pixels = img[valid_mask > 0]
    if inside_pixels.size == 0:
        raise RuntimeError("valid_mask 圆内没有像素。")

    low_value, high_value = np.percentile(inside_pixels, [p_low, p_high])
    if high_value <= low_value:
        raise RuntimeError("归一化失败：high_value 小于或等于 low_value。")

    normalized = (img.astype(np.float32) - low_value) / (high_value - low_value)
    img_u8 = (np.clip(normalized, 0, 1) * 255).astype(np.uint8)
    return cv2.bitwise_and(img_u8, img_u8, mask=valid_mask)


def enhance_vessel_frame(gray_frame, valid_mask):
    normalized = normalize_to_u8_inside_mask(
        gray_frame, valid_mask, NORM_LOW, NORM_HIGH
    )

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP,
        tileGridSize=CLAHE_GRID,
    )
    clahe_frame = clahe.apply(normalized)
    clahe_frame = cv2.bitwise_and(clahe_frame, clahe_frame, mask=valid_mask)

    se = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (TOPHAT_SIZE, TOPHAT_SIZE),
    )
    tophat = cv2.morphologyEx(clahe_frame, cv2.MORPH_TOPHAT, se)
    tophat = cv2.bitwise_and(tophat, tophat, mask=valid_mask)

    background = cv2.GaussianBlur(
        clahe_frame.astype(np.float32),
        (0, 0),
        sigmaX=BACKGROUND_SIGMA,
        sigmaY=BACKGROUND_SIGMA,
    )
    bg_removed = clahe_frame.astype(np.float32) - BACKGROUND_WEIGHT * background
    bg_removed = np.clip(bg_removed, 0, 255).astype(np.uint8)
    bg_removed = cv2.bitwise_and(bg_removed, bg_removed, mask=valid_mask)

    enhanced = cv2.addWeighted(tophat, 0.65, bg_removed, 0.35, 0)
    enhanced = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=0.7, sigmaY=0.7)

    return normalize_to_u8_inside_mask(
        enhanced.astype(np.float32), valid_mask, FINAL_LOW, FINAL_HIGH
    )


def save_enhanced_video(masked_video_path, output_dir):
    masked_video_path = Path(masked_video_path)
    output_path = Path(output_dir) / f"{masked_video_path.stem}_enhanced.avi"

    cap, fps = open_video(masked_video_path)
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError(f"这个视频没有读到任何帧：\n{masked_video_path}")

    height, width = first_frame.shape[:2]
    valid_mask = create_circle_mask(height, width)
    writer = create_gray_writer(output_path, fps, width, height)

    try:
        frame = first_frame
        while True:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            writer.write(enhance_vessel_frame(gray, valid_mask))

            ret, frame = cap.read()
            if not ret:
                break
    finally:
        cap.release()
        writer.release()

    return output_path


# ============================================================
# 4. 第三步：生成 vessel mask
# ============================================================

def hysteresis_threshold_inside_mask(
    img_u8,
    valid_mask,
    high_percentile=HIGH_PERCENTILE,
    low_percentile=LOW_PERCENTILE,
    max_iter=200,
):
    inside_pixels = img_u8[valid_mask > 0]
    if inside_pixels.size == 0:
        raise RuntimeError("valid_mask 圆内没有像素。")

    high_value = np.percentile(inside_pixels, high_percentile)
    low_value = np.percentile(inside_pixels, low_percentile)

    strong = ((img_u8 >= high_value) & (valid_mask > 0)).astype(np.uint8) * 255
    weak = ((img_u8 >= low_value) & (valid_mask > 0)).astype(np.uint8) * 255
    current = strong.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    for _ in range(max_iter):
        grown = cv2.dilate(current, kernel, iterations=1)
        grown = cv2.bitwise_and(grown, weak)
        grown = cv2.bitwise_and(grown, grown, mask=valid_mask)
        if np.array_equal(grown, current):
            break
        current = grown

    return current


def remove_small_components(binary_img, min_area=MIN_COMPONENT_AREA):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_img, connectivity=8
    )
    output = np.zeros_like(binary_img)

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area >= min_area:
            output[labels == label_id] = 255

    return output


def get_candidate_mask_from_enhanced_frame(enhanced_frame, valid_mask):
    frame_u8 = np.clip(enhanced_frame, 0, 255).astype(np.uint8)
    frame_u8 = cv2.bitwise_and(frame_u8, frame_u8, mask=valid_mask)
    return hysteresis_threshold_inside_mask(frame_u8, valid_mask)


def create_vessel_mask(enhanced_video_path, output_dir):
    enhanced_video_path = Path(enhanced_video_path)
    frames_gray, _ = load_video_as_gray_float(enhanced_video_path)
    total_frames, height, width = frames_gray.shape
    valid_mask = create_circle_mask(height, width)

    first_candidate = get_candidate_mask_from_enhanced_frame(
        frames_gray[0], valid_mask
    )
    first_candidate_bool = first_candidate > 0
    if first_candidate_bool.sum() == 0:
        raise RuntimeError("第一帧的候选 vessel mask 是空的。")

    initial_curve = np.array(
        [frames_gray[i][first_candidate_bool].mean() for i in range(total_frames)],
        dtype=np.float32,
    )
    initial_curve_smooth = smooth_curve_edge(initial_curve, SMOOTH_WINDOW)
    peak_indices = calculate_peaks(
        initial_curve_smooth,
        MIN_PEAK_DISTANCE,
        PEAK_PROMINENCE_RATIO,
    )

    peak_masks = np.stack(
        [
            get_candidate_mask_from_enhanced_frame(frames_gray[i], valid_mask)
            for i in peak_indices
        ],
        axis=0,
    )

    number_of_peaks = peak_masks.shape[0]
    vote_threshold = max(int(np.ceil(VOTE_RATIO * number_of_peaks)), 1)
    votes = (peak_masks > 0).sum(axis=0)
    vessel_vote_mask = (votes >= vote_threshold).astype(np.uint8) * 255
    vessel_vote_mask = remove_small_components(vessel_vote_mask)
    vessel_vote_mask = cv2.bitwise_and(
        vessel_vote_mask, vessel_vote_mask, mask=valid_mask
    )
    vessel_mask = (vessel_vote_mask > 0).astype(np.uint8)

    if vessel_mask.sum() == 0:
        raise RuntimeError("最终 vessel mask 是空的。")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_path = output_dir / f"{enhanced_video_path.stem}_vessel_mask.npy"
    preview_path = output_dir / f"{enhanced_video_path.stem}_vessel_mask.png"
    np.save(mask_path, vessel_mask)
    cv2.imwrite(str(preview_path), vessel_mask * 255)

    return mask_path, preview_path, vessel_mask, peak_indices, vote_threshold


# ============================================================
# 5. 第四步：亮度表、亮度曲线、从第一个 peak 裁剪
# ============================================================

def build_phase_from_peaks(total_frames, peak_indices):
    phase = np.zeros(total_frames, dtype=np.int32)
    peak_indices = np.sort(peak_indices)

    for i, start in enumerate(peak_indices):
        end = peak_indices[i + 1] if i < len(peak_indices) - 1 else total_frames
        phase[start:end] = np.arange(end - start, dtype=np.int32)

    return phase


def save_trimmed_video(frames_gray, fps, output_path):
    total_frames, height, width = frames_gray.shape
    writer = create_gray_writer(Path(output_path), fps, width, height)
    try:
        for i in range(total_frames):
            writer.write(np.clip(frames_gray[i], 0, 255).astype(np.uint8))
    finally:
        writer.release()


def create_brightness_outputs(
    masked_video_path,
    vessel_mask,
    output_table_dir,
    output_video_dir,
):
    masked_video_path = Path(masked_video_path)
    frames_gray, fps = load_video_as_gray_float(masked_video_path)
    total_frames, height, width = frames_gray.shape

    if vessel_mask.shape != (height, width):
        raise RuntimeError(
            "视频尺寸和 vessel mask 尺寸不一致：\n"
            f"视频尺寸 = {(height, width)}\nmask 尺寸 = {vessel_mask.shape}"
        )

    vessel_mask_bool = vessel_mask == 1
    if vessel_mask_bool.sum() == 0:
        raise RuntimeError("vessel mask 里没有血管像素。")

    raw_full = np.array(
        [frames_gray[i][vessel_mask_bool].mean() for i in range(total_frames)],
        dtype=np.float32,
    )
    smooth_full = smooth_curve_edge(raw_full, SMOOTH_WINDOW)
    peak_indices_full = calculate_peaks(
        smooth_full,
        MIN_PEAK_DISTANCE,
        PEAK_PROMINENCE_RATIO,
    )

    first_peak = int(peak_indices_full[0])
    frames_trimmed = frames_gray[first_peak:]
    raw = raw_full[first_peak:]
    smooth = smooth_full[first_peak:]
    peak_indices = (peak_indices_full - first_peak).astype(np.int32)

    total_trimmed = len(raw)
    frame_index = np.arange(total_trimmed, dtype=np.int32)
    phase = build_phase_from_peaks(total_trimmed, peak_indices)

    raw_table = np.vstack((frame_index, raw, phase)).astype(np.float32)
    smooth_table = np.vstack((frame_index, smooth, phase)).astype(np.float32)

    output_table_dir = Path(output_table_dir)
    output_video_dir = Path(output_video_dir)
    output_table_dir.mkdir(parents=True, exist_ok=True)
    output_video_dir.mkdir(parents=True, exist_ok=True)

    raw_npy_path = output_table_dir / f"{masked_video_path.stem}_raw_brightness.npy"
    smooth_npy_path = output_table_dir / f"{masked_video_path.stem}_smooth_brightness.npy"
    raw_csv_path = output_table_dir / f"{masked_video_path.stem}_raw_brightness.csv"
    smooth_csv_path = output_table_dir / f"{masked_video_path.stem}_smooth_brightness.csv"
    figure_path = output_table_dir / f"{masked_video_path.stem}_brightness_curve.png"
    trimmed_video_path = output_video_dir / f"{masked_video_path.stem}_from_first_peak.avi"

    np.save(raw_npy_path, raw_table)
    np.save(smooth_npy_path, smooth_table)

    np.savetxt(
        raw_csv_path,
        raw_table.T,
        delimiter=",",
        header="frame_index,raw_brightness,phase",
        comments="",
    )
    np.savetxt(
        smooth_csv_path,
        smooth_table.T,
        delimiter=",",
        header="frame_index,smooth_brightness,phase",
        comments="",
    )

    save_trimmed_video(frames_trimmed, fps, trimmed_video_path)

    plt.figure(figsize=(12, 5))
    plt.plot(frame_index, raw, label="Raw brightness")
    plt.plot(frame_index, smooth, label=f"Smooth brightness (window={SMOOTH_WINDOW})")
    plt.scatter(peak_indices, smooth[peak_indices], label="Peaks", marker="o")
    for peak_frame in peak_indices:
        plt.axvline(x=peak_frame, linestyle="--", alpha=0.5)
    plt.xlabel("Frame index from first peak")
    plt.ylabel("Mean brightness inside vessel mask")
    plt.title(masked_video_path.stem)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=200)
    plt.close()

    return {
        "first_peak": first_peak,
        "peaks_after_trim": peak_indices,
        "raw_npy": raw_npy_path,
        "smooth_npy": smooth_npy_path,
        "raw_csv": raw_csv_path,
        "smooth_csv": smooth_csv_path,
        "figure": figure_path,
        "trimmed_video": trimmed_video_path,
    }


# ============================================================
# 6. 完整流水线
# ============================================================

def process_one_original_video(video_path, output_dirs):
    masked_path = save_circle_masked_video(
        video_path, output_dirs["masked"]
    )
    print(f"    1/4 masked 完成：{masked_path.name}")

    enhanced_path = save_enhanced_video(
        masked_path, output_dirs["enhanced"]
    )
    print(f"    2/4 enhanced 完成：{enhanced_path.name}")

    mask_path, mask_preview, vessel_mask, mask_peaks, vote_threshold = (
        create_vessel_mask(enhanced_path, output_dirs["masks"])
    )
    print(
        f"    3/4 vessel mask 完成：{mask_path.name} "
        f"(peaks={len(mask_peaks)}, vote={vote_threshold}/{len(mask_peaks)})"
    )

    brightness_outputs = create_brightness_outputs(
        masked_path,
        vessel_mask,
        output_dirs["tables"],
        output_dirs["trimmed"],
    )
    print(
        "    4/4 brightness/裁剪完成："
        f"first_peak={brightness_outputs['first_peak']}"
    )

    return {
        "original": Path(video_path),
        "masked": masked_path,
        "enhanced": enhanced_path,
        "mask": mask_path,
        "mask_preview": mask_preview,
        **brightness_outputs,
    }


def main():
    input_dir = INPUT_DIR.expanduser().resolve()
    output_root = OUTPUT_DIR.expanduser().resolve()

    video_paths = find_all_avi_videos(input_dir)
    output_dirs = {
        "masked": output_root / "01_masked_videos",
        "enhanced": output_root / "02_enhanced_videos",
        "masks": output_root / "03_vessel_masks",
        "tables": output_root / "04_brightness_tables",
        "trimmed": output_root / "05_trimmed_videos",
    }
    for directory in output_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print(f"输入文件夹：{input_dir}")
    print(f"输出根目录：{output_root}")
    print(f"找到 {len(video_paths)} 个 AVI 视频")
    print("=" * 88)

    success = []
    failed = []

    for index, video_path in enumerate(video_paths, start=1):
        print("\n" + "-" * 88)
        print(f"[{index}/{len(video_paths)}] 开始处理：{video_path.name}")

        try:
            result = process_one_original_video(video_path, output_dirs)
            success.append(result)
            print(f"处理成功：{video_path.name}")
        except Exception as error:
            failed.append((video_path.name, str(error)))
            print(f"处理失败：{video_path.name}")
            print(f"错误：{error}")
            traceback.print_exc()

    print("\n" + "=" * 88)
    print("全部处理结束")
    print(f"成功：{len(success)}")
    print(f"失败：{len(failed)}")
    print(f"输出位置：{output_root}")

    if failed:
        print("\n失败的视频：")
        for name, error in failed:
            print(f"- {name}: {error}")


if __name__ == "__main__":
    main()
