# -*- coding: utf-8 -*-
"""
重新生成 Video 4、7、9 的空间亮度剖面对比图。

指定帧：
- Video 4, Frame 152
- Video 7, Frame 182
- Video 9, Frame 258

方法：
- Original
- Proposed
- UDVD
- Sliding Average
- Blind2Unblind
- Neighbor2Neighbor

每张 combined 图：
左侧：指定视频帧 + 红色竖直采样线
右侧：该竖线上像素亮度随 y 坐标的变化

注意：
- 不做平滑
- 不做归一化
- 不做插值
- 不做亮度增强
- 不自动缩放视频帧
- 所有方法对同一个视频使用相同的 line_x
- Frame 编号从 1 开始
"""

from pathlib import Path
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# 根目录
# ============================================================

DOWNLOADS = Path(
    r"C:\Users\Novovorontsovka\Downloads"
)


# ============================================================
# 视频目录
# ============================================================

ORIGINAL_FOLDER = (
    DOWNLOADS
    / "video_masqued"
)

UDVD_FOLDER = (
    DOWNLOADS
    / "udvd"
    / "epoch_035"
)

BLIND2UNBLIND_FOLDER = (
    DOWNLOADS
    / "blind2unblind"
    / "epoch_030"
)

NEIGHBOR2NEIGHBOR_FOLDER = (
    DOWNLOADS
    / "neighbor2neighbor"
    / "epoch_025"
)

SLIDING_AVERAGE_FOLDER = (
    DOWNLOADS
    / "video_masqued_sliding_average_12"
)

# Proposed：
#
# C:\Users\Novovorontsovka\Downloads\4\4_90.avi
# C:\Users\Novovorontsovka\Downloads\7\7_90.avi
# C:\Users\Novovorontsovka\Downloads\9\9_90.avi
PROPOSED_ROOT = DOWNLOADS


# ============================================================
# 输出文件夹
# ============================================================

OUTPUT_FOLDER = (
    DOWNLOADS
    / "spatial_profile_comparison_all_methods"
)

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 视频、帧和红线位置
#
# Frame 从 1 开始。
# line_x 是图像坐标，从左到右。
#
# 这里先统一使用 x=256。
# 如需恢复之前不同的红线位置，只改对应 line_x。
# ============================================================

VIDEO_SETTINGS = {
    4: {
        "frame": 152,
        "line_x": 256,
    },

    7: {
        "frame": 182,
        "line_x": 256,
    },

    9: {
        "frame": 258,
        "line_x": 256,
    },
}


# ============================================================
# 输出文件名
#
# 严格对应 Overleaf。
# ============================================================

OUTPUT_NAMES = {
    4: {
        "original":
            "4_frame_152_original_combined.png",

        "proposed":
            "4_frame_152_denoised_combined.png",

        "udvd":
            "4_frame_152_udvd_combined.png",

        "sliding_average":
            "4_frame_152_sliding_average_combined.png",

        "blind2unblind":
            "4_frame_152_blind2unblind_combined.png",

        "neighbor2neighbor":
            "4_frame_152_neighbor2neighbor_combined.png",
    },

    7: {
        "original":
            "7_frame_182_original_combined.png",

        "proposed":
            "7_frame_182_denoised_combined.png",

        "udvd":
            "7_frame_182_udvd_combined.png",

        "sliding_average":
            "7_frame_182_sliding_average_combined.png",

        "blind2unblind":
            "7_frame_182_blind2unblind_combined.png",

        "neighbor2neighbor":
            "7_frame_182_neighbor2neighbor_combined.png",
    },

    9: {
        "original":
            "9_frame_258_original_combined.png",

        "proposed":
            "9_frame_258_denoised_combined.png",

        "udvd":
            "9_frame_258_udvd_combined.png",

        "sliding_average":
            "9_frame_258_sliding_average_combined.png",

        "blind2unblind":
            "9_frame_258_blind2unblind_combined.png",

        "neighbor2neighbor":
            "9_frame_258_neighbor2neighbor_combined.png",
    },
}


# ============================================================
# 其他参数
# ============================================================

VIDEO_EXTENSIONS = {
    ".avi",
    ".mp4",
    ".mov",
    ".mkv",
    ".mpg",
    ".mpeg"
}

OUTPUT_DPI = 300

FIGURE_WIDTH = 6.2
FIGURE_HEIGHT = 3.1

RED_LINE_WIDTH = 1.2


# ============================================================
# 文件查找工具
# ============================================================

def natural_key(path: Path):
    """自然排序。"""

    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def contains_video_id(
    path: Path,
    video_id: int
):
    """
    匹配独立视频编号。

    Video 4 可以匹配：
    - 4_result.avi
    - video_4_denoised.avi
    - R_xxx_4_p.avi

    不会错误匹配：
    - 14.avi
    - 40.avi
    """

    pattern = (
        rf"(?<!\d)"
        rf"{re.escape(str(video_id))}"
        rf"(?!\d)"
    )

    return re.search(
        pattern,
        path.stem
    ) is not None


def find_video(
    folder: Path,
    video_id: int,
    method_name: str
):
    """
    在指定文件夹中递归查找对应编号的视频。
    """

    if not folder.exists():

        raise FileNotFoundError(
            f"{method_name} 文件夹不存在：\n"
            f"{folder}"
        )

    candidates = [
        path
        for path in folder.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower() in VIDEO_EXTENSIONS
            and contains_video_id(
                path,
                video_id
            )
        )
    ]

    candidates = sorted(
        candidates,
        key=natural_key
    )

    if len(candidates) == 0:

        raise FileNotFoundError(
            f"{method_name} 中找不到 "
            f"Video {video_id}：\n"
            f"{folder}"
        )

    def candidate_score(path: Path):

        begins_with_id = bool(
            re.match(
                rf"^{re.escape(str(video_id))}(?!\d)",
                path.stem
            )
        )

        return (
            0 if begins_with_id else 1,
            len(path.name),
            str(path).lower()
        )

    candidates = sorted(
        candidates,
        key=candidate_score
    )

    if len(candidates) > 1:

        print()
        print(
            f"警告：{method_name} 的 Video {video_id} "
            f"找到多个候选视频。"
        )

        print(
            f"最终使用：{candidates[0]}"
        )

        print("其他候选：")

        for candidate in candidates[1:]:

            print(
                f"  - {candidate}"
            )

    return candidates[0]


def find_proposed_video(
    video_id: int
):
    """
    Proposed 固定路径：

    Downloads\\编号\\编号_90.avi
    """

    video_path = (
        PROPOSED_ROOT
        / str(video_id)
        / f"{video_id}_90.avi"
    )

    if not video_path.exists():

        raise FileNotFoundError(
            f"找不到 Proposed 视频：\n"
            f"{video_path}"
        )

    return video_path


# ============================================================
# 读取指定帧
# ============================================================

def read_frame(
    video_path: Path,
    frame_number: int
):
    """
    frame_number 从 1 开始。

    Frame 152 对应 OpenCV 索引 151。
    """

    if frame_number < 1:

        raise ValueError(
            "frame_number 必须大于或等于 1。"
        )

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"无法打开视频：\n"
            f"{video_path}"
        )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if (
        total_frames > 0
        and frame_number > total_frames
    ):

        cap.release()

        raise IndexError(
            f"{video_path.name} 只有 "
            f"{total_frames} 帧，"
            f"无法读取第 {frame_number} 帧。"
        )

    frame_index = (
        frame_number - 1
    )

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        frame_index
    )

    ret, frame = cap.read()

    cap.release()

    if not ret:

        raise RuntimeError(
            f"无法读取 {video_path.name} "
            f"的第 {frame_number} 帧。"
        )

    if frame.ndim == 3:

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

    return frame.astype(
        np.uint8
    )


# ============================================================
# 尺寸检查
# ============================================================

def check_same_shape(
    original: np.ndarray,
    compared: np.ndarray,
    method_name: str,
    video_id: int
):
    """
    不允许自动缩放。
    """

    if original.shape != compared.shape:

        raise ValueError(
            f"Video {video_id} 的 {method_name} "
            f"尺寸与 Original 不同。\n"
            f"Original：{original.shape}\n"
            f"{method_name}：{compared.shape}\n"
            f"禁止自动缩放。"
        )


# ============================================================
# 提取竖直亮度剖面
# ============================================================

def extract_vertical_profile(
    frame: np.ndarray,
    line_x: int
):
    """
    提取固定 x 坐标上的全部像素值。

    横坐标为像素亮度；
    纵坐标为 y 位置。

    不做平滑或归一化。
    """

    height, width = frame.shape

    if not 0 <= line_x < width:

        raise ValueError(
            f"line_x={line_x} 超出图像范围。"
            f"有效范围为 0 到 {width - 1}。"
        )

    y_coordinates = np.arange(
        height
    )

    intensity_profile = (
        frame[:, line_x]
        .astype(np.float64)
    )

    return (
        y_coordinates,
        intensity_profile
    )


# ============================================================
# 生成一张 combined 图
# ============================================================

def create_combined_image(
    frame: np.ndarray,
    video_id: int,
    frame_number: int,
    line_x: int,
    method_display_name: str,
    output_filename: str
):
    """
    左侧：帧 + 红色竖线
    右侧：竖线位置的亮度剖面
    """

    height, width = frame.shape

    y_coordinates, intensity_profile = (
        extract_vertical_profile(
            frame,
            line_x
        )
    )

    figure = plt.figure(
        figsize=(
            FIGURE_WIDTH,
            FIGURE_HEIGHT
        )
    )

    grid = figure.add_gridspec(
        nrows=1,
        ncols=2,
        width_ratios=[
            1.0,
            1.15
        ],
        wspace=0.25
    )

    image_axis = figure.add_subplot(
        grid[0, 0]
    )

    profile_axis = figure.add_subplot(
        grid[0, 1]
    )

    # --------------------------------------------------------
    # 左侧图像
    # --------------------------------------------------------

    image_axis.imshow(
        frame,
        cmap="gray",
        vmin=0,
        vmax=255
    )

    image_axis.axvline(
        x=line_x,
        color="red",
        linewidth=RED_LINE_WIDTH
    )

    image_axis.set_title(
        (
            f"{method_display_name}\n"
            f"Video {video_id}, Frame {frame_number}"
        ),
        fontsize=10
    )

    image_axis.set_xlim(
        0,
        width - 1
    )

    image_axis.set_ylim(
        height - 1,
        0
    )

    image_axis.set_xticks([])
    image_axis.set_yticks([])

    # --------------------------------------------------------
    # 右侧亮度剖面
    # --------------------------------------------------------

    profile_axis.plot(
        intensity_profile,
        y_coordinates,
        linewidth=1.0
    )

    profile_axis.set_title(
        "Vertical intensity profile",
        fontsize=10
    )

    profile_axis.set_xlabel(
        "Pixel intensity",
        fontsize=9
    )

    profile_axis.set_ylabel(
        "Vertical position $y$",
        fontsize=9
    )

    # 保持与左侧图像方向一致
    profile_axis.set_ylim(
        height - 1,
        0
    )

    # 所有方法使用同一个亮度显示范围
    profile_axis.set_xlim(
        0,
        255
    )

    profile_axis.grid(
        True,
        linestyle="--",
        alpha=0.35
    )

    profile_axis.tick_params(
        axis="both",
        labelsize=8
    )

    output_path = (
        OUTPUT_FOLDER
        / output_filename
    )

    plt.savefig(
        output_path,
        dpi=OUTPUT_DPI,
        bbox_inches="tight",
        pad_inches=0.04
    )

    plt.close(
        figure
    )

    print(
        f"  已保存：{output_filename}"
    )


# ============================================================
# 处理一个视频
# ============================================================

def process_one_video(
    video_id: int,
    frame_number: int,
    line_x: int
):
    """
    为一个视频生成全部 6 种 combined 图。
    """

    print()
    print("=" * 90)
    print(
        f"Video {video_id}, "
        f"Frame {frame_number}, "
        f"line_x={line_x}"
    )
    print("=" * 90)

    # --------------------------------------------------------
    # 查找视频
    # --------------------------------------------------------

    original_path = find_video(
        ORIGINAL_FOLDER,
        video_id,
        "Original"
    )

    proposed_path = find_proposed_video(
        video_id
    )

    udvd_path = find_video(
        UDVD_FOLDER,
        video_id,
        "UDVD"
    )

    sliding_path = find_video(
        SLIDING_AVERAGE_FOLDER,
        video_id,
        "Sliding Average"
    )

    blind_path = find_video(
        BLIND2UNBLIND_FOLDER,
        video_id,
        "Blind2Unblind"
    )

    neighbor_path = find_video(
        NEIGHBOR2NEIGHBOR_FOLDER,
        video_id,
        "Neighbor2Neighbor"
    )

    paths = {
        "Original": original_path,
        "Proposed": proposed_path,
        "UDVD": udvd_path,
        "Sliding Average": sliding_path,
        "Blind2Unblind": blind_path,
        "Neighbor2Neighbor": neighbor_path,
    }

    print()
    print("使用的视频：")

    for method_name, path in paths.items():

        print(
            f"  {method_name}: {path}"
        )

    # --------------------------------------------------------
    # 读取同一帧
    # --------------------------------------------------------

    frames = {
        "original": read_frame(
            original_path,
            frame_number
        ),

        "proposed": read_frame(
            proposed_path,
            frame_number
        ),

        "udvd": read_frame(
            udvd_path,
            frame_number
        ),

        "sliding_average": read_frame(
            sliding_path,
            frame_number
        ),

        "blind2unblind": read_frame(
            blind_path,
            frame_number
        ),

        "neighbor2neighbor": read_frame(
            neighbor_path,
            frame_number
        ),
    }

    original_frame = frames[
        "original"
    ]

    for method_key, frame in frames.items():

        check_same_shape(
            original_frame,
            frame,
            method_key,
            video_id
        )

    names = OUTPUT_NAMES[
        video_id
    ]

    # --------------------------------------------------------
    # 生成全部图
    # --------------------------------------------------------

    create_combined_image(
        frame=frames["original"],
        video_id=video_id,
        frame_number=frame_number,
        line_x=line_x,
        method_display_name="Original",
        output_filename=names["original"]
    )

    create_combined_image(
        frame=frames["proposed"],
        video_id=video_id,
        frame_number=frame_number,
        line_x=line_x,
        method_display_name="Proposed",
        output_filename=names["proposed"]
    )

    create_combined_image(
        frame=frames["udvd"],
        video_id=video_id,
        frame_number=frame_number,
        line_x=line_x,
        method_display_name="UDVD",
        output_filename=names["udvd"]
    )

    create_combined_image(
        frame=frames["sliding_average"],
        video_id=video_id,
        frame_number=frame_number,
        line_x=line_x,
        method_display_name="Sliding Average",
        output_filename=names["sliding_average"]
    )

    create_combined_image(
        frame=frames["blind2unblind"],
        video_id=video_id,
        frame_number=frame_number,
        line_x=line_x,
        method_display_name="Blind2Unblind",
        output_filename=names["blind2unblind"]
    )

    create_combined_image(
        frame=frames["neighbor2neighbor"],
        video_id=video_id,
        frame_number=frame_number,
        line_x=line_x,
        method_display_name="Neighbor2Neighbor",
        output_filename=names["neighbor2neighbor"]
    )

    print()
    print(
        f"Video {video_id} 完成。"
    )


# ============================================================
# 主程序
# ============================================================

def main():

    required_folders = {
        "Original": ORIGINAL_FOLDER,
        "UDVD": UDVD_FOLDER,
        "Sliding Average": SLIDING_AVERAGE_FOLDER,
        "Blind2Unblind": BLIND2UNBLIND_FOLDER,
        "Neighbor2Neighbor": NEIGHBOR2NEIGHBOR_FOLDER,
    }

    for method_name, folder in (
        required_folders.items()
    ):

        if not folder.exists():

            raise FileNotFoundError(
                f"{method_name} 文件夹不存在：\n"
                f"{folder}"
            )

    print("=" * 90)
    print(
        "LDH 多方法垂直空间亮度剖面对比"
    )
    print("=" * 90)

    print(
        f"输出文件夹：{OUTPUT_FOLDER}"
    )

    print()
    print("指定帧：")
    print("  Video 4：Frame 152")
    print("  Video 7：Frame 182")
    print("  Video 9：Frame 258")

    print()
    print(
        "没有进行平滑、归一化、插值或增强。"
    )

    success_count = 0
    failures = []

    for video_id, settings in (
        VIDEO_SETTINGS.items()
    ):

        try:

            process_one_video(
                video_id=video_id,
                frame_number=settings["frame"],
                line_x=settings["line_x"]
            )

            success_count += 1

        except Exception as error:

            print()
            print(
                f"Video {video_id} 处理失败："
                f"{error}"
            )

            failures.append({
                "video_id": video_id,
                "error": str(error)
            })

    print()
    print("=" * 90)
    print("全部结束")
    print(
        f"成功：{success_count} 个视频"
    )
    print(
        f"失败：{len(failures)} 个视频"
    )
    print(
        f"输出位置：{OUTPUT_FOLDER}"
    )

    if failures:

        print()
        print("失败记录：")

        for failure in failures:

            print(
                f"  Video {failure['video_id']}："
                f"{failure['error']}"
            )

    print("=" * 90)


if __name__ == "__main__":
    main()