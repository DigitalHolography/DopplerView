from pathlib import Path

folder = Path(r"C:\Users\Novovorontsovka\Downloads\ldh_masked")

# 只读取常见视频文件
video_extensions = {".avi", ".mp4", ".mov", ".mkv", ".wmv"}

videos = sorted(
    [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in video_extensions
    ],
    key=lambda p: p.name.lower()
)

print(f"找到 {len(videos)} 个视频。")

# 先改成临时文件名，避免名称冲突
temp_videos = []

for index, old_path in enumerate(videos, start=1):
    temp_path = folder / f"__temp_video_{index:06d}{old_path.suffix.lower()}"
    old_path.rename(temp_path)
    temp_videos.append(temp_path)

# 再正式改名
for index, temp_path in enumerate(temp_videos, start=1):
    new_path = folder / f"{index}{temp_path.suffix.lower()}"
    temp_path.rename(new_path)
    print(f"{temp_path.name} -> {new_path.name}")

print(f"\n完成，共重命名 {len(temp_videos)} 个视频。")