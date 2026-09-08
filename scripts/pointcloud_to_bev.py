"""
从点云生成BEV（鸟瞰）视角图像
改进版：去除云 + 修复上下翻转
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
    return n

def pointcloud_to_bev(points, colors, output_size=(512, 512), flip_vertical=True):
    """将点云转换为BEV视角图像"""
    x_coords = points[:, 0]
    z_coords = points[:, 2]
    
    x_min, x_max = np.percentile(x_coords, 2), np.percentile(x_coords, 98)
    z_min, z_max = np.percentile(z_coords, 2), np.percentile(z_coords, 98)
    
    x_range = x_max - x_min if x_max > x_min else 1.0
    z_range = z_max - z_min if z_max > z_min else 1.0
    
    bev_image = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
    
    img_x = ((x_coords - x_min) / x_range * output_size[0]).astype(np.int32)
    img_y = ((z_coords - z_min) / z_range * output_size[1]).astype(np.int32)
    
    mask = (img_x >= 0) & (img_x < output_size[0]) & (img_y >= 0) & (img_y < output_size[1])
    valid_x = img_x[mask]
    valid_y = img_y[mask]
    valid_colors = colors[mask]
    
    bev_image[valid_y, valid_x] = valid_colors
    
    if flip_vertical:
        bev_image = cv2.flip(bev_image, 0)
    
    return bev_image

def main():
    INPUT_DIR = "/home/suanzigao/PycharmProjects/street_view_global_localization/test_image"
    OUTPUT_DIR = os.path.join(INPUT_DIR, "ortho_pointcloud_optimized")
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
    world_points_conf = np.array(result["world_points_conf"])
    
    print("\n🔄 过滤低置信度点（去除云）...")
    all_points = []
    all_colors = []
    
    CONF_THRESHOLD = 1.0001  # 过滤置信度接近最小值的点（云）
    
    for i, (image, points, conf) in enumerate(zip(images, world_points, world_points_conf)):
        pts = points.reshape(-1, 3)
        conf_flat = conf.reshape(-1)
        
        mask = (np.isfinite(pts).all(axis=1)) & (conf_flat > CONF_THRESHOLD)
        filtered_pts = pts[mask]
        
        img = cv2.resize(image, (518, 518))
        rgb = img.reshape(-1, 3)
        filtered_colors = rgb[mask]
        
        all_points.append(filtered_pts)
        all_colors.append(filtered_colors)
        
        base_name = os.path.basename(image_paths[i]).replace("_streetview.jpg", "")
        ply_path = os.path.join(OUTPUT_DIR, f"{base_name}_pointcloud_clean.ply")
        n = save_ply(ply_path, filtered_pts, filtered_colors)
        print(f"  ✅ {os.path.basename(ply_path)}: {n:,d} points")
    
    print("\n🔄 合并所有视角的点云...")
    merged_points = np.vstack(all_points)
    merged_colors = np.vstack(all_colors)
    
    mean = np.mean(merged_points, axis=0)
    dist = np.linalg.norm(merged_points - mean, axis=1)
    threshold = np.percentile(dist, 95)
    mask = dist < threshold
    filtered_points = merged_points[mask]
    filtered_colors = merged_colors[mask]
    
    merged_ply = os.path.join(OUTPUT_DIR, "merged_pointcloud_clean.ply")
    n = save_ply(merged_ply, filtered_points, filtered_colors)
    print(f"  ✅ merged_pointcloud_clean.ply: {n:,d} points")
    
    print("\n🔄 生成优化后的BEV视角...")
    bev_image = pointcloud_to_bev(filtered_points, filtered_colors, flip_vertical=True)
    
    bev_path = os.path.join(OUTPUT_DIR, "merged_bev_clean.png")
    cv2.imwrite(bev_path, bev_image)
    print(f"  ✅ {bev_path}")
    
    print(f"\n🎉 完成! 所有结果已保存到: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()