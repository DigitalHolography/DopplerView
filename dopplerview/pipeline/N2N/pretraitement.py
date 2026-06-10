import cv2
import numpy as np
from pathlib import Path
from scipy.signal import find_peaks


# ============================================================
# 0. 路径设置
# ============================================================

INPUT_DIR = Path(r"C:\Users\Novovorontsovka\Downloads\N2N")
OUTPUT_ROOT = Path(r"C:\Users\Novovorontsovka\Downloads\LDH_traite")

PAIR_RESULTS_DIR = OUTPUT_ROOT / "N2N_pair_results"
VESSEL_MASKS_DIR = OUTPUT_ROOT / "N2N_vessel_masks"
BRIGHTNESS_TABLES_DIR = OUTPUT_ROOT / "N2N_brightness_tables"

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
PAIR_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
VESSEL_MASKS_DIR.mkdir(parents=True, exist_ok=True)
BRIGHTNESS_TABLES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. 工具函数
# ============================================================

def normalize_to_u8_inside_mask(img, mask, p_low=0.5, p_high=99.7):
    inside = img[mask > 0]

    if inside.size == 0:
        raise RuntimeError("mask 内没有像素")

    lo, hi = np.percentile(inside, [p_low, p_high])

    x = (img - lo) / (hi - lo + 1e-8)
    x = np.clip(x, 0, 1)

    x_u8 = (x * 255).astype(np.uint8)
    x_u8 = cv2.bitwise_and(x_u8, x_u8, mask=mask)

    return x_u8


def hysteresis_threshold_inside_mask(
    img_u8,
    mask,
    high_percentile=88,
    low_percentile=60,
    max_iter=200
):
    inside = img_u8[mask > 0]

    if inside.size == 0:
        raise RuntimeError("mask 内没有像素，无法 threshold")

    high = np.percentile(inside, high_percentile)
    low = np.percentile(inside, low_percentile)

    strong = ((img_u8 >= high) & (mask > 0)).astype(np.uint8) * 255
    weak = ((img_u8 >= low) & (mask > 0)).astype(np.uint8) * 255

    current = strong.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    for _ in range(max_iter):
        grown = cv2.dilate(current, kernel, iterations=1)
        grown = cv2.bitwise_and(grown, weak)
        grown = cv2.bitwise_and(grown, grown, mask=mask)

        if np.array_equal(grown, current):
            break

        current = grown

    return current


def get_red_mask_from_gray(gray, circle_mask):
    perc_u8 = normalize_to_u8_inside_mask(
        gray,
        circle_mask,
        p_low=0.5,
        p_high=99.7
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    img = clahe.apply(perc_u8)
    img = cv2.bitwise_and(img, img, mask=circle_mask)

    se_size = 17
    se = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (se_size, se_size)
    )

    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, se)
    tophat = cv2.bitwise_and(tophat, tophat, mask=circle_mask)

    background = cv2.GaussianBlur(
        img.astype(np.float32),
        (0, 0),
        sigmaX=12,
        sigmaY=12
    )

    bg_removed = img.astype(np.float32) - 0.65 * background
    bg_removed = np.clip(bg_removed, 0, 255).astype(np.uint8)
    bg_removed = cv2.bitwise_and(bg_removed, bg_removed, mask=circle_mask)

    enhanced = cv2.addWeighted(
        tophat,
        0.65,
        bg_removed,
        0.35,
        0
    )

    enhanced = cv2.GaussianBlur(
        enhanced,
        (0, 0),
        sigmaX=0.7,
        sigmaY=0.7
    )

    enhanced = normalize_to_u8_inside_mask(
        enhanced.astype(np.float32),
        circle_mask,
        p_low=1,
        p_high=99.5
    )

    red_mask = hysteresis_threshold_inside_mask(
        enhanced,
        circle_mask,
        high_percentile=88,
        low_percentile=60
    )

    red_mask = cv2.bitwise_and(red_mask, red_mask, mask=circle_mask)

    return perc_u8, enhanced, red_mask


def smooth_curve(curve, win=5):
    if win <= 1:
        return curve.copy()

    kernel = np.ones(win, dtype=np.float32) / win
    return np.convolve(curve, kernel, mode="same")


def smooth_curve_edge(curve, win=5):
    if win <= 1:
        return curve.copy()

    pad_left = win // 2
    pad_right = win - 1 - pad_left

    curve_pad = np.pad(curve, (pad_left, pad_right), mode="edge")
    kernel = np.ones(win, dtype=np.float32) / win

    return np.convolve(curve_pad, kernel, mode="valid")


def remove_small_components(binary_img, min_area=15):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_img,
        connectivity=8
    )

    out = np.zeros_like(binary_img)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if area >= min_area:
            out[labels == i] = 255

    return out


def load_video_as_gray_float(video_path):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"打不开视频：{video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    frames_gray = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        frames_gray.append(gray)

    cap.release()

    if len(frames_gray) == 0:
        raise RuntimeError("视频没有读到任何帧")

    frames_gray = np.stack(frames_gray, axis=0)

    return frames_gray, fps


# ============================================================
# 2. pair 生成函数
# ============================================================

def build_pair_from_smooth_brightness(
    smooth_brightness_table,
    min_pair_distance=5
):
    """
    smooth_brightness_table:
        shape = (T, 2)
        第 0 列 = time_s
        第 1 列 = smooth_intensity

    pair 逻辑：
        对每一帧 i，找一个 j：
        1. j 和 i 不能太近，abs(j - i) >= min_pair_distance
        2. smooth_intensity[j] 和 smooth_intensity[i] 尽量接近

    返回：
        pair_indices:
            shape = (N, 2)
            第 0 列 = input_frame
            第 1 列 = target_frame

        pair_absdiff:
            每一对的 smooth intensity 差值
    """

    time_s = smooth_brightness_table[:, 0]
    smooth_intensity = smooth_brightness_table[:, 1]

    T = len(smooth_intensity)

    pair_indices = []
    pair_absdiff = []
    pair_time_s = []
    pair_intensity = []

    for i in range(T):
        valid = np.ones(T, dtype=bool)

        # 不和自己配，也不和太近的帧配
        left = max(0, i - min_pair_distance + 1)
        right = min(T, i + min_pair_distance)

        valid[left:right] = False

        if valid.sum() == 0:
            continue

        diff = np.abs(smooth_intensity - smooth_intensity[i])
        diff[~valid] = np.inf

        j = int(np.argmin(diff))

        pair_indices.append([i, j])
        pair_absdiff.append(diff[j])
        pair_time_s.append([time_s[i], time_s[j]])
        pair_intensity.append([smooth_intensity[i], smooth_intensity[j]])

    pair_indices = np.array(pair_indices, dtype=np.int32)
    pair_absdiff = np.array(pair_absdiff, dtype=np.float32)
    pair_time_s = np.array(pair_time_s, dtype=np.float32)
    pair_intensity = np.array(pair_intensity, dtype=np.float32)

    return pair_indices, pair_absdiff, pair_time_s, pair_intensity


# ============================================================
# 3. 单个视频处理函数
# ============================================================

def process_one_video(
    video_path,
    output_root,
    pair_results_dir,
    vessel_masks_dir,
    brightness_tables_dir,
    cx=255,
    cy=255,
    r=260,
    min_peak_distance=15,
    peak_prominence_ratio=0.10,
    vote_ratio=0.40,
    min_component_area=15,
    smooth_win_peak=5,
    smooth_win_final=5,
    min_pair_distance=5
):
    video_path = Path(video_path)
    video_name = video_path.stem

    print("\n" + "=" * 90)
    print("处理视频:", video_name)
    print("=" * 90)

    if not video_path.exists():
        raise FileNotFoundError(f"找不到视频文件：{video_path}")

    # ------------------------------------------------------------
    # 1. 读取视频
    # ------------------------------------------------------------

    frames_gray, fps = load_video_as_gray_float(video_path)

    T, H, W = frames_gray.shape

    print("fps:", fps)
    print("frames:", T)
    print("size:", W, H)

    if fps <= 0:
        print("警告：fps 读取异常，设置为 30")
        fps = 30.0

    time_s = np.arange(T, dtype=np.float32) / fps

    # ------------------------------------------------------------
    # 2. 圆形 mask
    # ------------------------------------------------------------

    circle_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(circle_mask, (int(cx), int(cy)), int(r), 255, thickness=-1)

    # ------------------------------------------------------------
    # 3. 第一帧提取初始血管 mask
    # ------------------------------------------------------------

    first_perc, first_enh, first_red = get_red_mask_from_gray(
        frames_gray[0],
        circle_mask
    )

    first_red_bool = first_red > 0

    if first_red_bool.sum() == 0:
        raise RuntimeError("第一帧 red mask 为空，调一下 threshold 参数")

    print("第一帧 red mask 像素数:", int(first_red_bool.sum()))

    # ------------------------------------------------------------
    # 4. 用第一帧 mask 计算初始亮度曲线，找 peak
    # ------------------------------------------------------------

    initial_curve = np.array([
        frames_gray[i][first_red_bool].mean()
        for i in range(T)
    ], dtype=np.float32)

    initial_curve_smooth = smooth_curve(
        initial_curve,
        win=smooth_win_peak
    )

    curve_d = initial_curve_smooth - np.median(initial_curve_smooth)

    prominence = peak_prominence_ratio * (
        np.percentile(curve_d, 95) - np.percentile(curve_d, 5)
    )

    peak_indices, properties = find_peaks(
        curve_d,
        distance=min_peak_distance,
        prominence=prominence
    )

    if len(peak_indices) == 0:
        raise RuntimeError(
            "没有找到 peak。可以把 peak_prominence_ratio 调低，比如 0.05"
        )

    selected_peaks = np.sort(peak_indices)

    print("找到 peak 帧:", selected_peaks.tolist())
    print("peak 总数:", len(selected_peaks))

    # ------------------------------------------------------------
    # 5. 对所有 peak 帧提 vessel mask
    # ------------------------------------------------------------

    peak_masks = []

    for n, idx in enumerate(selected_peaks):
        print(f"处理 peak {n + 1}/{len(selected_peaks)}: frame {idx}")

        _, _, red_mask = get_red_mask_from_gray(
            frames_gray[idx],
            circle_mask
        )

        peak_masks.append(red_mask)

    peak_masks = np.stack(peak_masks, axis=0)

    K = peak_masks.shape[0]

    print("用于 vote 的 peak 数 K =", K)

    # ------------------------------------------------------------
    # 6. vote 得到最终 vessel mask
    # ------------------------------------------------------------

    votes = (peak_masks > 0).sum(axis=0)

    vote_threshold = int(np.ceil(vote_ratio * K))
    vote_threshold = max(1, vote_threshold)

    vessel_vote_mask = (votes >= vote_threshold).astype(np.uint8) * 255

    vessel_vote_mask = remove_small_components(
        vessel_vote_mask,
        min_area=min_component_area
    )

    vessel_vote_mask = cv2.bitwise_and(
        vessel_vote_mask,
        vessel_vote_mask,
        mask=circle_mask
    )

    vessel_pixels = cv2.countNonZero(vessel_vote_mask)

    print("vote_ratio:", vote_ratio)
    print(f"vote_threshold: {vote_threshold}/{K}")
    print("vessel pixels:", vessel_pixels)

    if vessel_pixels == 0:
        raise RuntimeError(
            "vote 后 vessel mask 为空。可以降低 vote_ratio，比如 0.30"
        )

    final_overlap_matrix = (vessel_vote_mask > 0).astype(np.uint8)

    # ------------------------------------------------------------
    # 7. 保存 vessel mask 到单独文件夹
    # ------------------------------------------------------------

    vessel_mask_save_path = vessel_masks_dir / f"{video_name}_vessel_mask.npy"

    np.save(
        vessel_mask_save_path,
        final_overlap_matrix.astype(np.uint8)
    )

    print("vessel mask 保存到:")
    print(vessel_mask_save_path)

    # ------------------------------------------------------------
    # 8. 用最终 vessel mask 计算亮度曲线
    # ------------------------------------------------------------

    overlap_bool = final_overlap_matrix == 1

    brightness_curve_raw = np.array([
        frames_gray[i][overlap_bool].mean()
        for i in range(T)
    ], dtype=np.float32)

    brightness_curve_smooth = smooth_curve_edge(
        brightness_curve_raw,
        win=smooth_win_final
    ).astype(np.float32)

    curve_d2 = brightness_curve_smooth - np.median(brightness_curve_smooth)

    prominence2 = peak_prominence_ratio * (
        np.percentile(curve_d2, 95) - np.percentile(curve_d2, 5)
    )

    curve_peak_indices, _ = find_peaks(
        curve_d2,
        distance=min_peak_distance,
        prominence=prominence2
    )

    print("最终 smooth 亮度曲线 peak 数量:", len(curve_peak_indices))
    print("最终 smooth 亮度曲线 peak 帧:", curve_peak_indices.tolist())

    # ------------------------------------------------------------
    # 9. 保存 smooth brightness table 到单独文件夹
    #
    # 只保存：
    # 第 0 列 = time_s
    # 第 1 列 = smooth_intensity
    # ------------------------------------------------------------

    smooth_brightness_table = np.column_stack([
        time_s,
        brightness_curve_smooth
    ]).astype(np.float32)

    brightness_table_save_path = brightness_tables_dir / f"{video_name}_brightness_table.npy"

    np.save(
        brightness_table_save_path,
        smooth_brightness_table
    )

    print("smooth brightness table 保存到:")
    print(brightness_table_save_path)
    print("smooth_brightness_table.shape =", smooth_brightness_table.shape)

    # ------------------------------------------------------------
    # 10. 生成 pair result
    #
    # pair 逻辑：
    # 每一帧找一个 smooth brightness 最接近的帧
    # 但是不能是自己附近 min_pair_distance 范围内的帧
    # ------------------------------------------------------------

    pair_indices, pair_absdiff, pair_time_s, pair_intensity = build_pair_from_smooth_brightness(
        smooth_brightness_table=smooth_brightness_table,
        min_pair_distance=min_pair_distance
    )

    pair_save_path = pair_results_dir / f"{video_name}_pair_result.npz"

    np.savez_compressed(
        pair_save_path,

        video_name=np.array(video_name),
        fps=np.array(fps, dtype=np.float32),

        # 核心 pair
        pair_indices=pair_indices,
        pair_absdiff=pair_absdiff,
        pair_time_s=pair_time_s,
        pair_intensity=pair_intensity,

        # 方便后面检查
        smooth_brightness_table=smooth_brightness_table,
        selected_peaks=selected_peaks.astype(np.int32),
        curve_peak_indices=curve_peak_indices.astype(np.int32),

        # 对应 mask 的文件名
        vessel_mask_file=np.array(str(vessel_mask_save_path)),
        brightness_table_file=np.array(str(brightness_table_save_path))
    )

    print("pair result 保存到:")
    print(pair_save_path)
    print("pair_indices.shape =", pair_indices.shape)

    # ------------------------------------------------------------
    # 11. 返回结果
    # ------------------------------------------------------------

    results = {
        "video_name": video_name,

        "vessel_mask_path": vessel_mask_save_path,
        "brightness_table_path": brightness_table_save_path,
        "pair_result_path": pair_save_path,

        "final_overlap_matrix": final_overlap_matrix,
        "smooth_brightness_table": smooth_brightness_table,
        "pair_indices": pair_indices,

        "fps": fps,
        "selected_peaks": selected_peaks,
        "curve_peak_indices": curve_peak_indices
    }

    return results


# ============================================================
# 4. 主程序：批量处理所有视频
# ============================================================

if __name__ == "__main__":

    print("输入视频文件夹:")
    print(INPUT_DIR)

    print("\n总输出文件夹:")
    print(OUTPUT_ROOT)

    print("\npair 输出文件夹:")
    print(PAIR_RESULTS_DIR)

    print("\nvessel mask 输出文件夹:")
    print(VESSEL_MASKS_DIR)

    print("\nsmooth brightness table 输出文件夹:")
    print(BRIGHTNESS_TABLES_DIR)

    video_paths = sorted(INPUT_DIR.glob("*.avi"))

    if len(video_paths) == 0:
        raise RuntimeError(f"没有在这个文件夹找到 .avi 视频：{INPUT_DIR}")

    print("\n找到视频数量:", len(video_paths))

    for p in video_paths:
        print(" -", p.name)

    success_videos = []
    failed_videos = []

    for idx, video_path in enumerate(video_paths):

        print("\n" + "#" * 90)
        print(f"开始处理 {idx + 1}/{len(video_paths)}")
        print("#" * 90)

        try:
            results = process_one_video(
                video_path=video_path,
                output_root=OUTPUT_ROOT,
                pair_results_dir=PAIR_RESULTS_DIR,
                vessel_masks_dir=VESSEL_MASKS_DIR,
                brightness_tables_dir=BRIGHTNESS_TABLES_DIR,

                # 512 x 512 视频常用圆形区域
                cx=255,
                cy=255,
                r=260,

                # 找 peak 参数
                min_peak_distance=15,
                peak_prominence_ratio=0.10,

                # vessel mask vote 参数
                vote_ratio=0.40,
                min_component_area=15,

                # smooth 参数
                smooth_win_peak=5,
                smooth_win_final=5,

                # pair 参数
                # pair 不能和自己太近
                min_pair_distance=5
            )

            success_videos.append(results["video_name"])

        except Exception as e:
            video_name = video_path.stem
            print("\n处理失败:", video_name)
            print("错误:", e)
            failed_videos.append((video_name, str(e)))
            continue

    # ========================================================
    # 5. 保存失败列表
    # ========================================================

    failed_path = OUTPUT_ROOT / "failed_videos.txt"

    with open(failed_path, "w", encoding="utf-8") as f:
        for name, err in failed_videos:
            f.write(f"{name}\n")
            f.write(f"{err}\n\n")

    # ========================================================
    # 6. 总结
    # ========================================================

    print("\n" + "=" * 90)
    print("全部处理完成")
    print("=" * 90)

    print("成功视频数量:", len(success_videos))
    print("失败视频数量:", len(failed_videos))

    print("\n成功视频:")
    for name in success_videos:
        print(" -", name)

    print("\n失败视频:")
    for name, err in failed_videos:
        print(" -", name, ":", err)

    print("\n三个核心输出文件夹:")

    print("\n1. pair:")
    print(PAIR_RESULTS_DIR)

    print("\n2. vessel mask:")
    print(VESSEL_MASKS_DIR)

    print("\n3. smooth brightness table:")
    print(BRIGHTNESS_TABLES_DIR)

    print("\n失败列表:")
    print(failed_path)

    print("\n每个视频对应文件名格式:")

    print("\nxxx_pair_result.npz")
    print("  里面主要有:")
    print("  pair_indices      shape = (N, 2)")
    print("  pair_absdiff      shape = (N,)")
    print("  pair_time_s       shape = (N, 2)")
    print("  pair_intensity    shape = (N, 2)")

    print("\nxxx_vessel_mask.npy")
    print("  shape = (H, W)")
    print("  1 = 血管")
    print("  0 = 非血管")

    print("\nxxx_brightness_table.npy")
    print("  shape = (T, 2)")
    print("  第 0 列 = time_s")
    print("  第 1 列 = smooth_intensity")
    print("  没有 raw_intensity")