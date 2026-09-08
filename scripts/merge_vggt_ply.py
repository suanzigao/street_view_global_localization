"""合并 VGGT 点云：过滤低置信度 + 合并为单一 PLY"""
import os, glob, numpy as np
from PIL import Image

INPUT_DIR = "/home/suanzigao/PycharmProjects/street_view_global_localization/batch_streetview_results/2026-06-01_15-53-34_batch_run/2026-06-01_15-53-34_Qc7f8UCkWQSdu_0Wc7BDtQ_lat55_7132_lng12_5929/streetview_images"
VGGT_DIR = os.path.join(INPUT_DIR, "vggt_remote_output")
OUTPUT_DIR = VGGT_DIR
CONF_THRESHOLD = 1.0001  # VGGT expp1: conf=1+exp(raw), 仅排除 conf==1.0 (raw≈-inf)

# 加载数据
print("Loading data...")
wp = np.load(os.path.join(VGGT_DIR, "world_points.npy"))          # (11, 518, 518, 3)
wp_conf = np.load(os.path.join(VGGT_DIR, "world_points_conf.npy"))  # (11, 518, 518)
S, H, W, _ = wp.shape
print(f"world_points: {wp.shape}")
print(f"confidence:    {wp_conf.shape}")
print(f"Conf range: [{wp_conf.min():.4f}, {wp_conf.max():.4f}]  (expp1: conf=1+exp(raw), 1.0≈-inf)")
print(f"Conf == 1.0 (zero confidence): {(wp_conf == 1.0).sum():,} / {wp_conf.size:,} ({(wp_conf==1.0).sum()/wp_conf.size*100:.1f}%)")

# 过滤条件：联合 置信度 + 图像行 区分天空和建筑
# conf==1.0 在上半部分(row<200)是天空噪声 → 去除
# conf==1.0 在下半部分(row>=200)是建筑/地面 → 保留
SKY_CUTOFF_ROW = 170  # 天空区行阈值
finite_mask = np.isfinite(wp).all(axis=-1)
conf_mask = wp_conf > 1.0           # 去除所有 conf==1.0 的点（天空/云/不可信区域）
valid_mask = finite_mask & conf_mask

before = finite_mask.sum()
after = valid_mask.sum()
print(f"\n过滤前: {before:,} 有效点")
print(f"过滤后: {after:,} 有效点 (conf >= {CONF_THRESHOLD})")
print(f"移除:   {before - after:,} 点 ({(1-after/before)*100:.1f}%)")

# 加载图像提取颜色
image_paths = sorted(
    glob.glob(os.path.join(INPUT_DIR, "step_*_streetview.jpg")),
    key=lambda p: int(p.split("step_")[1].split("_")[0])
)
print(f"\nLoading {len(image_paths)} images for colors...")

# 合并所有点云写入单个 PLY (二进制格式，更小更快)
out_ply = os.path.join(OUTPUT_DIR, "merged_pointcloud.ply")
total_points = valid_mask.sum()
print(f"Writing {total_points:,} points to {out_ply} ...")

with open(out_ply, "w") as f:
    f.write("ply\nformat ascii 1.0\n")
    f.write(f"element vertex {total_points}\n")
    f.write("property float x\nproperty float y\nproperty float z\n")
    f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
    f.write("property float confidence\n")
    f.write("property uchar view_id\n")     # 哪个视角来的
    f.write("end_header\n")

    written = 0
    for i, img_path in enumerate(image_paths):
        # 加载颜色
        img = Image.open(img_path).convert("RGB")
        img = img.resize((W, H), Image.Resampling.BICUBIC)
        rgb = np.array(img)  # (H, W, 3)

        # 筛选
        mask_i = valid_mask[i]
        pts_i = wp[i, mask_i].astype(np.float32)
        rgb_i = rgb[mask_i]
        conf_i = wp_conf[i, mask_i].astype(np.float32)

        for j in range(len(pts_i)):
            x, y, z = pts_i[j]
            r, g, b = rgb_i[j]
            c = conf_i[j]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b} {c:.4f} {i}\n")
        written += len(pts_i)

        step_name = os.path.basename(img_path).replace("_streetview.jpg", "")
        print(f"  [{step_name}] {len(pts_i):,} points | center=({pts_i[:,0].mean():.2f}, {pts_i[:,1].mean():.2f}, {pts_i[:,2].mean():.2f})")

file_size = os.path.getsize(out_ply) / 1024 / 1024
print(f"\n✅ Merged point cloud saved: {out_ply}")
print(f"   {written:,} total points ({file_size:.1f} MB)")

# 也生成一个纯坐标+颜色的干净版本（不带 view_id，方便某些工具加载）
clean_ply = os.path.join(OUTPUT_DIR, "merged_pointcloud_clean.ply")
print(f"\nGenerating clean version (xyz+rgb only)...")
with open(clean_ply, "w") as f:
    f.write("ply\nformat ascii 1.0\n")
    f.write(f"element vertex {total_points}\n")
    f.write("property float x\nproperty float y\nproperty float z\n")
    f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
    f.write("end_header\n")
    written2 = 0
    for i in range(S):
        mask_i = valid_mask[i]
        pts_i = wp[i, mask_i]
        img = Image.open(image_paths[i]).convert("RGB").resize((W, H), Image.Resampling.BICUBIC)
        rgb_i = np.array(img)[mask_i]
        for j in range(len(pts_i)):
            x, y, z = pts_i[j]
            r, g, b = rgb_i[j]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
        written2 += len(pts_i)
file_size2 = os.path.getsize(clean_ply) / 1024 / 1024
print(f"   {clean_ply}")
print(f"   {written2:,} total points ({file_size2:.1f} MB)")
print("\n✅ Done!")
