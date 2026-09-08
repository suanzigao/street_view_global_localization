"""
测试脚本 - 不使用mask，验证投影公式正确性
"""

import os
import sys
import json
import math
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def read_metadata(metadata_path):
    with open(metadata_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_intrinsics(width, height, fov_deg):
    fov = math.radians(fov_deg)
    fx = width / (2 * math.tan(fov / 2))
    fy = height / (2 * math.tan(fov / 2))
    cx = width / 2
    cy = height / 2
    return fx, fy, cx, cy


def pixel_to_world(x, y, fx, fy, cx, cy, heading_deg, pitch_deg, cam_x, cam_y, cam_z):
    yaw = math.radians(heading_deg)
    pitch = math.radians(pitch_deg)
    
    u = (x - cx) / fx
    v = (y - cy) / fy
    
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    cos_pitch = math.cos(pitch)
    sin_pitch = math.sin(pitch)
    
    ray_world_x = u * cos_yaw * cos_pitch + sin_yaw * cos_pitch
    ray_world_y = -u * sin_yaw * cos_pitch + cos_yaw * cos_pitch
    ray_world_z = -v * cos_pitch - sin_pitch
    
    if ray_world_z >= -1e-6:
        return None
    
    t = -cam_z / ray_world_z
    
    if t < 0:
        return None
    
    world_x = cam_x + t * ray_world_x
    world_y = cam_y + t * ray_world_y
    
    return (world_x, world_y, 0.0)


def main():
    images_dir = 'test_relative_position_output/images'
    output_dir = 'reconstruction_output'
    
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
    
    all_points = []
    all_colors = []
    
    for img_file in image_files:
        base_name = img_file.replace('_streetview.jpg', '')
        metadata_file = f"{base_name}_metadata.json"
        
        image_path = os.path.join(images_dir, img_file)
        metadata_path = os.path.join(images_dir, metadata_file)
        
        if not os.path.exists(metadata_path):
            continue
        
        print(f"\n{'='*60}")
        print(f"=== 处理 {img_file} ===")
        
        image_pil = Image.open(image_path).convert('RGB')
        image = np.array(image_pil)
        metadata = read_metadata(metadata_path)
        
        h, w = image.shape[:2]
        
        heading = metadata.get('heading', 0.0)
        pitch = metadata.get('pitch', 0.0)
        fov = metadata.get('fov', 60)
        relative_x = metadata.get('relative_x', 0.0)
        relative_y = metadata.get('relative_y', 0.0)
        cam_z = 2.5
        
        fx, fy, cx, cy = compute_intrinsics(w, h, fov)
        
        print(f"\n📋 元数据:")
        print(f"  heading={heading}°, pitch={pitch}°, fov={fov}°")
        print(f"  相机位置: ({relative_x:.2f}, {relative_y:.2f}, {cam_z})")
        
        print(f"\n📐 内参:")
        print(f"  fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
        
        points_3d = []
        colors = []
        
        ground_mask = np.ones((h, w), dtype=np.uint8)
        
        for y in range(h):
            for x in range(w):
                world_pt = pixel_to_world(x, y, fx, fy, cx, cy, heading, pitch, relative_x, relative_y, cam_z)
                
                if world_pt is not None:
                    points_3d.append(world_pt)
                    colors.append(image[y, x])
        
        if points_3d:
            points_3d = np.array(points_3d)
            colors = np.array(colors)
            print(f"\n📊 投影结果:")
            print(f"  投影点数: {len(points_3d)}")
            print(f"  X范围: {points_3d[:, 0].min():.2f} ~ {points_3d[:, 0].max():.2f}")
            print(f"  Y范围: {points_3d[:, 1].min():.2f} ~ {points_3d[:, 1].max():.2f}")
            
            print(f"\n📍 关键点验证:")
            key_pixels = [
                (int(cx), int(cy), "中心"),
                (int(cx), h-1, "底部中心"),
                (0, h-1, "左下角"),
                (w-1, h-1, "右下角"),
                (int(cx), 0, "顶部中心"),
            ]
            for x, y, name in key_pixels:
                pt = pixel_to_world(x, y, fx, fy, cx, cy, heading, pitch, relative_x, relative_y, cam_z)
                if pt:
                    dx = pt[0] - relative_x
                    dy = pt[1] - relative_y
                    dist = math.sqrt(dx*dx + dy*dy)
                    print(f"  {name}({x},{y}): world=({pt[0]:.2f}, {pt[1]:.2f}), 相对相机=({dx:.2f}, {dy:.2f}), 距离={dist:.2f}m")
                else:
                    print(f"  {name}({x},{y}): 不指向地面")
            
            all_points.append(points_3d)
            all_colors.append(colors)
    
    if not all_points:
        print("\n❌ 没有成功处理的图像")
        return
    
    merged_points = np.vstack(all_points)
    merged_colors = np.vstack(all_colors)
    
    print(f"\n{'='*60}")
    print(f"\n📊 融合统计:")
    print(f"  总点数: {len(merged_points)}")
    
    x_min = merged_points[:, 0].min()
    x_max = merged_points[:, 0].max()
    y_min = merged_points[:, 1].min()
    y_max = merged_points[:, 1].max()
    
    print(f"  X范围: {x_min:.2f} ~ {x_max:.2f}")
    print(f"  Y范围: {y_min:.2f} ~ {y_max:.2f}")
    
    resolution = 10
    padding = 5
    
    bev_width = int((x_max - x_min + padding * 2) * resolution)
    bev_height = int((y_max - y_min + padding * 2) * resolution)
    
    bev_image = np.zeros((bev_height, bev_width, 3), dtype=np.uint8)
    
    for i in range(len(merged_points)):
        px = int((merged_points[i, 0] - x_min + padding) * resolution)
        py = int((merged_points[i, 1] - y_min + padding) * resolution)
        
        if 0 <= px < bev_width and 0 <= py < bev_height:
            bev_image[py, px] = merged_colors[i]
    
    bev_path = os.path.join(output_dir, 'test_no_mask_bev.png')
    cv2.imwrite(bev_path, cv2.cvtColor(bev_image, cv2.COLOR_RGB2BGR))
    print(f"\n✅ BEV图像已保存到: {bev_path}")


if __name__ == "__main__":
    main()
