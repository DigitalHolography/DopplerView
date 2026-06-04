import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks


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


def overlay_green(gray_u8, green_mask):
    out = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
    out[green_mask > 0] = [0, 255, 0]
    return out


def overlay_red(gray_u8, red_mask):
    out = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
    out[red_mask > 0] = [0, 0, 255]
    return out


def overlay_vote_heatmap(gray_u8, votes_norm):
    heat = (votes_norm * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

    base = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)

    out = cv2.addWeighted(base, 0.55, heat_color, 0.45, 0)
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


def read_first_frame_u8(video_path):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"打不开视频：{video_path}")

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"无法读取第一帧：{video_path}")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

    gmin = gray.min()
    gmax = gray.max()

    gray_u8 = (
        (gray - gmin)
        / (gmax - gmin + 1e-8)
        * 255
    ).astype(np.uint8)

    return gray_u8


# ============================================================
# 2. 主函数：处理一个视频
# ============================================================

def process_video_temporal_vote(
    video_path,
    output_dir=None,
    cx=None,
    cy=None,
    r=None,
    min_peak_distance=15,
    peak_prominence_ratio=0.10,
    vote_ratio=0.40,
    min_component_area=15,
    smooth_win_peak=5,
    smooth_win_final=5,
    show_figures=False,
    save_peak_images=False,
    save_matrix_csv=False,
    save_brightness_csv=False
):
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"找不到视频文件：{video_path}")

    if output_dir is None:
        output_dir = video_path.parent / f"{video_path.stem}_temporal_vote_result"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("使用视频：", video_path)
    print("输出文件夹：", output_dir)

    # ========================================================
    # 1. 读取视频
    # ========================================================

    frames_gray, fps = load_video_as_gray_float(video_path)

    T, H, W = frames_gray.shape

    print("fps:", fps)
    print("frames:", T)
    print("size:", W, H)

    if fps <= 0:
        print("警告：fps 读取异常，设置为 30")
        fps = 30.0

    time = np.arange(T, dtype=np.float32) / fps

    # ========================================================
    # 2. 圆形 mask
    # ========================================================

    if cx is None:
        cx = W // 2

    if cy is None:
        cy = H // 2

    if r is None:
        r = min(H, W) // 2

    cx = int(cx)
    cy = int(cy)
    r = int(r)

    circle_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(circle_mask, (cx, cy), r, 255, thickness=-1)

    # ========================================================
    # 3. 第一帧提取初始血管 mask
    # ========================================================

    first_perc, first_enh, first_red = get_red_mask_from_gray(
        frames_gray[0],
        circle_mask
    )

    first_red_bool = first_red > 0

    if first_red_bool.sum() == 0:
        raise RuntimeError("第一帧 red mask 为空，调一下 threshold 参数")

    print("第一帧 red mask 像素数:", int(first_red_bool.sum()))

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

    print("initial peak prominence:", prominence)

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

    print("找到的全部 peak 帧:", selected_peaks.tolist())
    print("peak 总数:", len(selected_peaks))

    # ========================================================
    # 4. 对所有 peak 帧提血管 mask
    # ========================================================

    peak_masks = []
    peak_red_overlays = []
    peak_enhanced_images = []

    for n, idx in enumerate(selected_peaks):
        print(f"处理 peak {n + 1}/{len(selected_peaks)}: frame {idx}")

        perc_u8, enhanced, red_mask = get_red_mask_from_gray(
            frames_gray[idx],
            circle_mask
        )

        peak_masks.append(red_mask)

        if save_peak_images:
            peak_enhanced_images.append(enhanced)
            overlay_red_peak = overlay_red(perc_u8, red_mask)
            peak_red_overlays.append(overlay_red_peak)

    peak_masks = np.stack(peak_masks, axis=0)

    K = peak_masks.shape[0]

    print("用于 vote 的 peak 数 K =", K)

    # ========================================================
    # 5. vote 得到稳定血管区域
    # ========================================================

    votes = (peak_masks > 0).sum(axis=0)
    votes_norm = votes.astype(np.float32) / K

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

    print(f"vote ratio = {vote_ratio}")
    print(f"vote threshold = {vote_threshold}/{K}")
    print("vessel pixels:", vessel_pixels)

    if vessel_pixels == 0:
        raise RuntimeError(
            "vote 后的 vessel mask 是空的。可以降低 vote_ratio，比如 0.30"
        )

    # ========================================================
    # 6. final_overlap_matrix
    # 这个就是血管 0/1 图：
    # 1 = 血管
    # 0 = 非血管
    # ========================================================

    final_overlap_matrix = (vessel_vote_mask > 0).astype(np.uint8)

    final_matrix_npy = output_dir / "final_overlap_matrix.npy"
    final_matrix_csv = output_dir / "final_overlap_matrix.csv"

    np.save(final_matrix_npy, final_overlap_matrix)

    if save_matrix_csv:
        np.savetxt(
            final_matrix_csv,
            final_overlap_matrix,
            delimiter=",",
            fmt="%d"
        )

    print("final_overlap_matrix 保存到：")
    print(final_matrix_npy)

    # ========================================================
    # 7. brightness table
    # ========================================================

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

    print("最终亮度曲线 peak 数量:", len(curve_peak_indices))
    print("最终亮度曲线 peak 帧:", curve_peak_indices.tolist())

    is_peak_array = np.zeros(T, dtype=np.int32)
    is_peak_array[curve_peak_indices] = 1

    brightness_table = np.column_stack([
        np.arange(T, dtype=np.int32),
        time,
        brightness_curve_raw,
        brightness_curve_smooth,
        is_peak_array
    ]).astype(np.float32)

    brightness_table_npy = output_dir / "brightness_table.npy"
    brightness_table_csv = output_dir / "brightness_table.csv"

    np.save(brightness_table_npy, brightness_table)

    if save_brightness_csv:
        np.savetxt(
            brightness_table_csv,
            brightness_table,
            delimiter=",",
            header="frame_index,time_s,raw_intensity,smooth_intensity,is_peak",
            comments="",
            fmt=["%d", "%.6f", "%.6f", "%.6f", "%d"]
        )

    print("brightness_table 保存到：")
    print(brightness_table_npy)

    # ========================================================
    # 8. 保存 two_variables
    # ========================================================

    two_variables_npz = output_dir / "two_variables.npz"

    np.savez_compressed(
        two_variables_npz,
        brightness_table=brightness_table,
        final_overlap_matrix=final_overlap_matrix,
        selected_peaks=selected_peaks,
        curve_peak_indices=curve_peak_indices,
        brightness_curve_raw=brightness_curve_raw,
        brightness_curve_smooth=brightness_curve_smooth,
        fps=np.array(fps, dtype=np.float32),
        vote_threshold=np.array(vote_threshold, dtype=np.int32),
        vote_ratio=np.array(vote_ratio, dtype=np.float32)
    )

    print("两个核心变量保存到：")
    print(two_variables_npz)

    # ========================================================
    # 9. 可选保存图像
    # 这里默认不保存 PNG，因为你说你要数值
    # ========================================================

    if show_figures or save_peak_images:
        first_display = normalize_to_u8_inside_mask(
            frames_gray[0],
            circle_mask,
            p_low=0.5,
            p_high=99.7
        )

        vessel_overlay = overlay_green(first_display, vessel_vote_mask)
        vote_heatmap = overlay_vote_heatmap(first_display, votes_norm)
        overlap_overlay_red = overlay_red(first_display, vessel_vote_mask)

        vessel_mask_png = output_dir / "02_vessel_vote_mask.png"
        vessel_overlay_png = output_dir / "03_vessel_vote_overlay_green.png"
        vote_heatmap_png = output_dir / "04_vote_frequency_heatmap.png"
        overlap_overlay_red_png = output_dir / "06_overlap_region_overlay_red.png"

        cv2.imwrite(str(vessel_mask_png), vessel_vote_mask)
        cv2.imwrite(str(vessel_overlay_png), vessel_overlay)
        cv2.imwrite(str(vote_heatmap_png), vote_heatmap)
        cv2.imwrite(str(overlap_overlay_red_png), overlap_overlay_red)

        if save_peak_images:
            for n, idx in enumerate(selected_peaks):
                cv2.imwrite(
                    str(output_dir / f"peak_{n + 1}_frame_{idx}_red_overlay.png"),
                    peak_red_overlays[n]
                )

                cv2.imwrite(
                    str(output_dir / f"peak_{n + 1}_frame_{idx}_red_mask.png"),
                    peak_masks[n]
                )

                cv2.imwrite(
                    str(output_dir / f"peak_{n + 1}_frame_{idx}_enhanced.png"),
                    peak_enhanced_images[n]
                )

    results = {
        "video_path": video_path,
        "output_dir": output_dir,

        "brightness_table": brightness_table,
        "final_overlap_matrix": final_overlap_matrix,

        "selected_peaks": selected_peaks,
        "curve_peak_indices": curve_peak_indices,
        "brightness_curve_raw": brightness_curve_raw,
        "brightness_curve_smooth": brightness_curve_smooth,

        "fps": fps,
        "time": time,
        "circle_mask": circle_mask,

        "peak_masks": peak_masks,
        "votes": votes,
        "votes_norm": votes_norm,
        "vote_threshold": vote_threshold,

        "two_variables_npz": two_variables_npz,
        "brightness_table_npy": brightness_table_npy,
        "brightness_table_csv": brightness_table_csv if save_brightness_csv else None,
        "final_matrix_npy": final_matrix_npy,
        "final_matrix_csv": final_matrix_csv if save_matrix_csv else None
    }

    return results


# ============================================================
# 3. 主程序：批量处理所有视频
# ============================================================

if __name__ == "__main__":

    input_dir = Path(r"C:\Users\Novovorontsovka\Desktop\Choroidal_LDH")
    output_root = Path(r"C:\Users\Novovorontsovka\Downloads\LDH_traite")

    output_root.mkdir(parents=True, exist_ok=True)

    # True 会打印完整矩阵，非常长
    # 一开始建议 False
    PRINT_FULL_NUMBERS = False

    # ========================================================
    # 1. 找到所有视频
    # ========================================================

    video_paths = sorted(input_dir.glob("*.avi"))

    if len(video_paths) == 0:
        raise RuntimeError(f"没有在这个文件夹找到 .avi 视频：{input_dir}")

    print("找到视频数量:", len(video_paths))

    for p in video_paths:
        print(" -", p.name)

    # ========================================================
    # 2. 最终你要的两个变量
    # ========================================================

    variable_temps = {}
    variable_position = {}

    failed_videos = []

    # ========================================================
    # 3. 逐个视频处理
    # ========================================================

    for i, video_path in enumerate(video_paths):

        print("\n" + "=" * 90)
        print(f"处理视频 {i + 1}/{len(video_paths)}")
        print("视频:", video_path.name)
        print("=" * 90)

        video_name = video_path.stem
        output_dir = output_root / f"{video_name}_temporal_vote_result"

        try:
            results = process_video_temporal_vote(
                video_path=video_path,
                output_dir=output_dir,

                # 如果视频是 512 x 512，用这个
                cx=255,
                cy=255,
                r=260,

                # peak 参数
                min_peak_distance=15,
                peak_prominence_ratio=0.10,

                # vote 参数
                vote_ratio=0.40,

                # 不弹图，不保存 peak 图片
                show_figures=False,
                save_peak_images=False,

                # 你现在要数值，CSV 也可以关掉
                save_matrix_csv=False,
                save_brightness_csv=False
            )

            brightness_table = results["brightness_table"]
            final_overlap_matrix = results["final_overlap_matrix"]

            # ====================================================
            # A. variable_temps[video_name]
            #
            # brightness_table:
            # 0 = frame_index
            # 1 = time_s
            # 2 = raw_intensity
            # 3 = smooth_intensity，也就是 approximation
            # 4 = is_peak
            #
            # variable_temps[video_name].shape = (T, 2)
            # 第 0 列 = time_s
            # 第 1 列 = smooth_intensity
            # ====================================================

            time_s = brightness_table[:, 1]
            smooth_intensity = brightness_table[:, 3]

            variable_temps[video_name] = np.column_stack([
                time_s,
                smooth_intensity
            ]).astype(np.float32)

            # ====================================================
            # B. variable_position[video_name]
            #
            # 这就是你要的血管 0/1 图
            # 1 = 血管
            # 0 = 非血管
            # ====================================================

            variable_position[video_name] = final_overlap_matrix.astype(np.uint8)

            # ====================================================
            # C. 验证：
            # vessel_image = 第一帧 * 血管0/1图
            # background_image = 第一帧 * 非血管0/1图
            # ====================================================

            first_frame_u8 = read_first_frame_u8(video_path)

            vessel_image = first_frame_u8 * variable_position[video_name]
            background_image = first_frame_u8 * (1 - variable_position[video_name])

            # ====================================================
            # D. 打印当前视频结果
            # ====================================================

            print("\n成功处理:", video_name)

            print("\n[variable_temps]")
            print("variable_temps[video_name].shape =", variable_temps[video_name].shape)
            print("前 10 行:")
            print(variable_temps[video_name][:10])

            print("\n[variable_position]")
            print("variable_position[video_name].shape =", variable_position[video_name].shape)
            print("1 = 血管, 0 = 非血管")
            print("血管像素数量 =", int(variable_position[video_name].sum()))
            print("非血管像素数量 =", int((variable_position[video_name] == 0).sum()))

            print("\nvariable_position 前 10x10:")
            print(variable_position[video_name][:10, :10])

            print("\n验证 vessel_image = first_frame_u8 * variable_position[video_name]")
            print("vessel_image.shape =", vessel_image.shape)
            print("vessel_image 前 10x10:")
            print(vessel_image[:10, :10])

            print("\n验证 background_image = first_frame_u8 * (1 - variable_position[video_name])")
            print("background_image.shape =", background_image.shape)
            print("background_image 前 10x10:")
            print(background_image[:10, :10])

            if PRINT_FULL_NUMBERS:
                print("\n完整 variable_temps[video_name]:")
                print(variable_temps[video_name])

                print("\n完整 variable_position[video_name]:")
                print(variable_position[video_name])

                print("\n完整 vessel_image:")
                print(vessel_image)

                print("\n完整 background_image:")
                print(background_image)

        except Exception as e:
            print("\n处理失败:", video_path.name)
            print("错误:", e)
            failed_videos.append((video_path.name, str(e)))
            continue

    # ========================================================
    # 4. 全部视频处理完成后，打印总结
    # ========================================================

    print("\n" + "#" * 90)
    print("全部视频处理完成")
    print("#" * 90)

    print("\n成功视频数量:", len(variable_temps))
    print("失败视频数量:", len(failed_videos))

    print("\nvariable_temps 的 key:")
    print(list(variable_temps.keys()))

    print("\nvariable_position 的 key:")
    print(list(variable_position.keys()))

    for video_name in variable_temps.keys():

        temps = variable_temps[video_name]
        position = variable_position[video_name]

        print("\n" + "-" * 80)
        print("视频:", video_name)

        print("variable_temps[video_name].shape =", temps.shape)
        print("variable_position[video_name].shape =", position.shape)

        print("variable_temps 前 5 行:")
        print(temps[:5])

        print("variable_position 前 5x5:")
        print(position[:5, :5])

        print("血管像素数量 =", int(position.sum()))

    # ========================================================
    # 5. 保存两个变量
    # ========================================================
# ========================================================
# 5. 保存 variable_temps 和 variable_position 到单独文件夹
# ========================================================

    variables_dir = output_root / "variables"
    variables_dir.mkdir(parents=True, exist_ok=True)

    variable_temps_path = variables_dir / "variable_temps.npy"
    variable_position_path = variables_dir / "variable_position.npy"

    np.save(
        variable_temps_path,
     np.array(variable_temps, dtype=object),
        allow_pickle=True
    )

    np.save(
        variable_position_path,
        np.array(variable_position, dtype=object),
        allow_pickle=True
    )

    print("\nvariable_temps 已经保存到:")
    print(variable_temps_path)

    print("\nvariable_position 已经保存到:")
    print(variable_position_path)
    # ========================================================
    # 6. 保存失败视频列表
    # ========================================================

    failed_path = output_root / "failed_videos.txt"

    with open(failed_path, "w", encoding="utf-8") as f:
        for name, err in failed_videos:
            f.write(f"{name}\n")
            f.write(f"{err}\n\n")

    print("\n失败列表保存到:")
    print(failed_path)

    # ========================================================
    # 7. 最终说明
    # ========================================================

    print("\n最终变量说明:")

    print("\nvariable_temps[video_name]:")
    print("  shape = (T, 2)")
    print("  [:, 0] = time_s")
    print("  [:, 1] = smooth_intensity")

    print("\nvariable_position[video_name]:")
    print("  shape = (H, W)")
    print("  1 = 血管")
    print("  0 = 非血管")

    print("\n之后你要得到血管图：")
    print("  vessel_image = first_frame_u8 * variable_position[video_name]")

    print("\n之后你要得到背景图：")
    print("  background_image = first_frame_u8 * (1 - variable_position[video_name])")
# ========================================================
# 8. 用 variable_position 生成每个视频的血管图 / 背景图
# ========================================================

vessel_images_dir = output_root / "ALL_VIDEOS_vessel_images"
background_images_dir = output_root / "ALL_VIDEOS_background_images"

vessel_images_dir.mkdir(parents=True, exist_ok=True)
background_images_dir.mkdir(parents=True, exist_ok=True)

print("\n开始根据 variable_position 生成血管图和背景图...")

for video_path in video_paths:
    video_name = video_path.stem

    # 如果这个视频处理失败了，就跳过
    if video_name not in variable_position:
        print(f"跳过 {video_name}，因为它不在 variable_position 里")
        continue

    try:
        # ----------------------------------------------------
        # 1. 读取第一帧
        # ----------------------------------------------------
        first_frame_u8 = read_first_frame_u8(video_path)

        # ----------------------------------------------------
        # 2. 读取这个视频对应的血管 0/1 图
        # ----------------------------------------------------
        mask = variable_position[video_name].astype(np.uint8)

        # ----------------------------------------------------
        # 3. 得到血管图和背景图
        # ----------------------------------------------------
        vessel_image = (first_frame_u8 * mask).astype(np.uint8)
        background_image = (first_frame_u8 * (1 - mask)).astype(np.uint8)

        # ----------------------------------------------------
        # 4. 保存图片
        # ----------------------------------------------------
        vessel_save_path = vessel_images_dir / f"{video_name}_vessel.png"
        background_save_path = background_images_dir / f"{video_name}_background.png"

        cv2.imwrite(str(vessel_save_path), vessel_image)
        cv2.imwrite(str(background_save_path), background_image)

        # ----------------------------------------------------
        # 5. 打印验证
        # ----------------------------------------------------
        print("\n视频:", video_name)
        print("mask.shape =", mask.shape)
        print("vessel_image.shape =", vessel_image.shape)
        print("background_image.shape =", background_image.shape)
        print("血管像素数量 =", int(mask.sum()))
        print("血管图保存到:", vessel_save_path)
        print("背景图保存到:", background_save_path)

        print("mask 前 10x10:")
        print(mask[:10, :10])

        print("vessel_image 前 10x10:")
        print(vessel_image[:10, :10])

    except Exception as e:
        print(f"\n生成血管图失败: {video_name}")
        print("错误:", e)

print("\n全部血管图保存到:")
print(vessel_images_dir)

print("\n全部背景图保存到:")
print(background_images_dir)