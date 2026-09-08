"""
从正交视图重建点云
使用VGGT深度图和相机位姿进行精确重建
"""
import os
import cv2
import base64
import io
import json
import requests
import numpy as np
from PIL import Image

def save_ply(filename, points, colors=None):
    """保存点云为PLY格式"""
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

def main():
    INPUT_DIR = "/home/suanzigao/PycharmProjects/street_view_global_localization/test_image"
    OUTPUT_DIR = os.path.join(INPUT_DIR, "ortho_pointcloud")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    SERVER_URL = "http://127.0.0.1:18888"
    
    print("📡 连接VGGT远程服务...")
    try:
        r = requests.get(f"{SERVER_URL}/health", timeout=10)
        health = r.json()
        print(f"✅ VGGT服务状态: {health}")
    except Exception as e:
        print(f"❌ 无法连接VGGT服务: {e}")
        return
    
    image_paths = sorted([
        os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR)
        if f.endswith("_streetview.jpg")
    ], key=lambda p: int(os.path.basename(p).split("step_")[1].split("_")[0]))
    
    print(f"\n📷 找到 {len(image_paths)} 张图像")
    
    images = []
    for p in image_paths:
        img = cv2.imread(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
    
    print("\n🔄 正在进行VGGT推理...")
    imgs_b64 = []
    for img in images:
        pil_img = Image.fromarray(img)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=90)
        imgs_b64.append(base64.b64encode(buf.getvalue()).decode())
    
    req = {"images": imgs_b64, "output_types": ["camera", "depth", "points"]}
    r = requests.post(f"{SERVER_URL}/vggt/infer", json=req, timeout=300)
    result = r.json()
    
    if not result.get("success"):
        print(f"❌ VGGT推理失败: {result.get('error')}")
        return
    
    print("✅ VGGT推理成功!")
    
    world_points = np.array(result["world_points"])
    pose_encs = np.array(result["pose_enc"])
    
    print("\n🔄 正在从正交视角重建点云...")
    all_points = []
    all_colors = []
    
    for i, (image, points, pose_enc) in enumerate(zip(images, world_points, pose_encs)):
        pts = points.reshape(-1, 3)
        valid = np.isfinite(pts).all(axis=1)
        pts = pts[valid]
        
        img = cv2.resize(image, (518, 518))
        rgb = img.reshape(-1, 3)
        colors = rgb[valid]
        
        all_points.append(pts)
        all_colors.append(colors)
        
        base_name = os.path.basename(image_paths[i]).replace("_streetview.jpg", "")
        ply_path = os.path.join(OUTPUT_DIR, f"{base_name}_pointcloud.ply")
        save_ply(ply_path, pts, colors)
    
    print("\n🔄 合并所有视角的点云...")
    merged_points = np.vstack(all_points)
    merged_colors = np.vstack(all_colors)
    
    # 过滤离群点
    mean = np.mean(merged_points, axis=0)
    dist = np.linalg.norm(merged_points - mean, axis=1)
    threshold = np.percentile(dist, 95)
    mask = dist < threshold
    filtered_points = merged_points[mask]
    filtered_colors = merged_colors[mask]
    
    merged_ply = os.path.join(OUTPUT_DIR, "merged_pointcloud.ply")
    save_ply(merged_ply, filtered_points, filtered_colors)
    
    print(f"\n🎉 完成! 所有点云已保存到: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()