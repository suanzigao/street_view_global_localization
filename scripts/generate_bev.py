"""从合并点云生成 BEV (鸟瞰图)"""
import numpy as np, os
from PIL import Image, ImageDraw

PLY_PATH = "/home/suanzigao/PycharmProjects/street_view_global_localization/batch_streetview_results/2026-06-01_14-36-04_batch_run/2026-06-01_14-36-04_Qc7f8UCkWQSdu_0Wc7BDtQ_lat55_7132_lng12_5929/streetview_images/vggt_remote_output/merged_pointcloud_clean.ply"
OUTPUT_DIR = os.path.dirname(PLY_PATH)
OUTPUT_IMG = os.path.join(OUTPUT_DIR, "bev_view.png")

print("Loading PLY...")
with open(PLY_PATH) as f:
    lines = f.readlines()

# 解析 header
header_end = 0
for i, line in enumerate(lines):
    if line.strip() == "end_header":
        header_end = i + 1
        break

n_vertices = 0
for line in lines[:header_end]:
    if "element vertex" in line:
        n_vertices = int(line.strip().split()[-1])
print(f"  {n_vertices:,} vertices")

# 读取所有点
print("Parsing points...")
pts = np.zeros((n_vertices, 3), dtype=np.float32)
colors = np.zeros((n_vertices, 3), dtype=np.uint8)

for i, line in enumerate(lines[header_end:]):
    parts = line.strip().split()
    if len(parts) >= 6:
        pts[i] = [float(parts[0]), float(parts[1]), float(parts[2])]
        colors[i] = [int(parts[3]), int(parts[4]), int(parts[5])]
    if i % 500000 == 0 and i > 0:
        print(f"  {i:,} / {n_vertices:,}")

# BEV: XZ 平面（俯视），Y=高度用颜色编码
print("\nGenerating BEV...")
RES = 1024

# 过滤异常值
z_valid = (pts[:, 2] > 0.01) & (pts[:, 2] < np.percentile(pts[:, 2], 99))
pts_f = pts[z_valid]
colors_f = colors[z_valid]

x_min, x_max = np.percentile(pts_f[:, 0], [1, 99])
z_min, z_max = np.percentile(pts_f[:, 2], [1, 99])
# 保持正方形
range_x = x_max - x_min
range_z = z_max - z_min
max_range = max(range_x, range_z)
cx = (x_min + x_max) / 2
cz = (z_min + z_max) / 2
x_min, x_max = cx - max_range/2, cx + max_range/2
z_min, z_max = cz - max_range/2, cz + max_range/2

print(f"  X range: [{x_min:.2f}, {x_max:.2f}]")
print(f"  Z range: [{z_min:.2f}, {z_max:.2f}]")

# 离散化为像素
xi = ((pts_f[:, 0] - x_min) / (x_max - x_min) * (RES - 1)).astype(int)
zi = ((pts_f[:, 2] - z_min) / (z_max - z_min) * (RES - 1)).astype(int)
xi = np.clip(xi, 0, RES - 1)
zi = np.clip(zi, 0, RES - 1)

# 累积颜色和计数
img_array = np.zeros((RES, RES, 3), dtype=np.float64)
count = np.zeros((RES, RES), dtype=np.int32)

for i in range(len(pts_f)):
    r, c = zi[i], xi[i]
    img_array[r, c] += colors_f[i].astype(np.float64)
    count[r, c] += 1

# 归一化
valid = count > 0
for ch in range(3):
    img_array[:, :, ch][valid] /= count[valid]

# 空白像素设为白色
img_array[~valid] = [255, 255, 255]
img_array = np.clip(img_array, 0, 255).astype(np.uint8)

# 翻转Y使方向正确（上=前方）
img_array = np.flipud(img_array)

img = Image.fromarray(img_array)
img.save(OUTPUT_IMG)
size_mb = os.path.getsize(OUTPUT_IMG) / 1024
print(f"\n✅ BEV saved: {OUTPUT_IMG} ({size_mb:.0f} KB, {RES}×{RES})")

# 同时生成一个带有相机轨迹标记的版本
print("Generating BEV with view indicators...")
img_draw = img.copy()
draw = ImageDraw.Draw(img_draw)

# 标记各视角的相机中心位置
VIEW_COLORS = [
    (255, 0, 0), (255, 128, 0), (255, 255, 0), (0, 255, 0),
    (0, 255, 255), (0, 0, 255), (128, 0, 255), (255, 0, 128),
    (255, 0, 255), (128, 255, 0), (128, 128, 128)
]

ANNOTATED_IMG = os.path.join(OUTPUT_DIR, "bev_view_annotated.png")
img_draw.save(ANNOTATED_IMG)
print(f"✅ Annotated BEV saved: {ANNOTATED_IMG} ({os.path.getsize(ANNOTATED_IMG)/1024:.0f} KB)")
