"""测试远程 VGGT 服务，推理全部 11 张街景图并保存点云"""
import base64, glob, json, os, struct
import numpy as np
import requests

SERVER = "http://127.0.0.1:18888"
IMAGES_DIR = "/home/suanzigao/PycharmProjects/street_view_global_localization/batch_streetview_results/2026-06-01_15-53-34_batch_run/2026-06-01_15-53-34_Qc7f8UCkWQSdu_0Wc7BDtQ_lat55_7132_lng12_5929/streetview_images"
OUTPUT_DIR = os.path.join(IMAGES_DIR, "vggt_remote_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Health check
print("=== 1. Health Check ===")
r = requests.get(f"{SERVER}/health", timeout=10)
print(r.json())

# 2. Load all images
image_paths = sorted(
    glob.glob(os.path.join(IMAGES_DIR, "step_*_streetview.jpg")),
    key=lambda p: int(p.split("step_")[1].split("_")[0])
)
print(f"\n=== 2. Encoding {len(image_paths)} images ===")
imgs_b64 = []
for p in image_paths:
    with open(p, "rb") as f:
        imgs_b64.append(base64.b64encode(f.read()).decode())

# 3. Send request (all 11 images at once)
print(f"\n=== 3. Inference ({len(imgs_b64)} images, all outputs)... ===")
req = {"images": imgs_b64, "output_types": ["camera", "depth", "points"]}
r = requests.post(f"{SERVER}/vggt/infer", json=req, timeout=300)
result = r.json()

if not result.get("success"):
    print(f"FAILED: {json.dumps(result, indent=2)[:500]}")
    exit(1)

print("✅ Inference successful!")

# 4. Convert to numpy and save
pose_enc = np.array(result["pose_enc"])
depth = np.array(result["depth"])
depth_conf = np.array(result["depth_conf"])
world_points = np.array(result["world_points"])
world_points_conf = np.array(result["world_points_conf"])

print(f"\n=== Shapes ===")
print(f"  pose_enc:     {pose_enc.shape}")
print(f"  depth:        {depth.shape}")
print(f"  depth_conf:   {depth_conf.shape}")
print(f"  world_points: {world_points.shape}")
print(f"  wp_conf:      {world_points_conf.shape}")

# 5. Save numpy arrays
np.save(os.path.join(OUTPUT_DIR, "pose_enc.npy"), pose_enc)
np.save(os.path.join(OUTPUT_DIR, "depth.npy"), depth)
np.save(os.path.join(OUTPUT_DIR, "depth_conf.npy"), depth_conf)
np.save(os.path.join(OUTPUT_DIR, "world_points.npy"), world_points)
np.save(os.path.join(OUTPUT_DIR, "world_points_conf.npy"), world_points_conf)
print(f"\n=== 5. Saved .npy files to {OUTPUT_DIR} ===")


# 6. Export point cloud as PLY (每个视角单独一个文件)
def save_ply(filename, points, colors=None):
    """保存点云为 PLY 格式 (ASCII)"""
    pts = points.reshape(-1, 3)
    valid = np.isfinite(pts).all(axis=1)
    pts = pts[valid]
    n = len(pts)
    with open(filename, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if colors is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            x, y, z = pts[i]
            if colors is not None:
                r, g, b = colors[i % len(colors)]
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
            else:
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
    print(f"  -> {os.path.basename(filename)}: {n:,d} points")


print(f"\n=== 6. Exporting PLY files with color ===")
from PIL import Image
for i, p in enumerate(image_paths):
    step_name = os.path.basename(p).replace("_streetview.jpg", "")
    out_ply = os.path.join(OUTPUT_DIR, f"{step_name}_pointcloud.ply")

    img = Image.open(p).convert("RGB")
    img = img.resize((518, 518), Image.Resampling.BICUBIC)
    rgb = np.array(img).reshape(-1, 3)

    pts = world_points[i].reshape(-1, 3)
    valid = np.isfinite(pts).all(axis=1)
    colors = rgb[valid]

    save_ply(out_ply, world_points[i], colors)

# 7. Camera pose summary
print(f"\n=== 7. Camera Pose Summary ===")
print(f"{'View':>6} | {'Tx':>8} {'Ty':>8} {'Tz':>8} | {'FoV_h':>7} {'FoV_w':>7}")
print("-" * 55)
for i in range(pose_enc.shape[0]):
    t = pose_enc[i, :3]
    fh = np.rad2deg(pose_enc[i, 7])
    fw = np.rad2deg(pose_enc[i, 8])
    name = os.path.basename(image_paths[i]).replace("_streetview.jpg", "")
    print(f"{name:>6} | {t[0]:8.2f} {t[1]:8.2f} {t[2]:8.2f} | {fh:7.1f}° {fw:7.1f}°")

# 8. Point cloud summary
print(f"\n=== 8. Point Cloud Summary ===")
for i in range(world_points.shape[0]):
    pts = world_points[i].reshape(-1, 3)
    valid = np.isfinite(pts).all(axis=1)
    step_name = os.path.basename(image_paths[i]).replace("_streetview.jpg", "")
    print(f"  {step_name}: {valid.sum():,d} points | "
          f"center=({pts[valid, 0].mean():.2f}, {pts[valid, 1].mean():.2f}, {pts[valid, 2].mean():.2f})")

print(f"\n✅ Done! All files in: {OUTPUT_DIR}")
print(f"Files: {sorted(os.listdir(OUTPUT_DIR))}")
