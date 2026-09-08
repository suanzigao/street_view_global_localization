"""
简化投影脚本

使用直观的坐标系定义：
- 世界坐标系：X=东，Y=北，Z=上
- 相机在地面上方2.5m
- 地面是Z=0平面
"""

import os
import sys
import json
import math
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tools.sam3_tool import SAM3Tool


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
    """
    将像素坐标转换为世界坐标（投影到地面Z=0）
    
    坐标系定义：
    - 世界坐标系：X=东，Y=北，Z=上（右手系）
    - 相机坐标系：X=右，Y=下，Z=前（右手系）
    - 图像坐标系：x=右，y=下
    
    参数：
    - x, y: 图像像素坐标
    - fx, fy, cx, cy: 内参
    - heading_deg: 朝向（0=北，90=东）
    - pitch_deg: 俯仰角（0=水平，正数=向下看）
    - cam_x, cam_y, cam_z: 相机世界坐标
    """
    yaw = math.radians(heading_deg)
    pitch = math.radians(pitch_deg)
    
    u = (x - cx) / fx
    v = (y - cy) / fy
    
    ray_right = u
    ray_down = v
    ray_forward = 1.0
    
    R_yaw = np.array([
        [math.cos(yaw), 0, math.sin(yaw)],
        [0, 1, 0],
        [-math.sin(yaw), 0, math.cos(yaw)]
    ], dtype=np.float64)
    
    R_pitch = np.array([
        [1, 0, 0],
        [0, math.cos(pitch), math.sin(pitch)],
        [0, -math.sin(pitch), math.cos(pitch)]
    ], dtype=np.float64)
    
    R = R_yaw @ R_pitch
    
    ray_cam = np.array([ray_right, ray_down, ray_forward])
    ray_world = R @ ray_cam
    
    if ray_world[2] >= 0:
        return None
    
    t = -cam_z / ray_world[2]
    
    world_x = cam_x + t * ray_world[0]
    world_y = cam_y + t * ray_world[1]
    
    return (world_x, world_y, 0.0)


def main():
    images_dir = 'test_relative_position_output/images'
    output_dir = 'reconstruction_output'
    os.makedirs(output_dir, exist_ok=True)
    
    sam3_tool = SAM3Tool()
    sam3_tool.initialize()
    
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
        
        print(f"\n=== 处理 {img_file} ===")
        
        image_pil = Image.open(image_path).convert('RGB')
        image = np.array(image_pil)
        metadata = read_metadata(metadata_path)
        
        h, w = image.shape[:2]
        
        heading = metadata.get('heading', 0.0)
        pitch = metadata.get('pitch', 0.0)
        fov = metadata.get('fov', 60)
        relative_x = metadata.get('relative_x', 0.0)
        relative_y = metadata.get('relative_y', 0.0)
        
        print(f"  元数据: heading={heading}, pitch={pitch}, fov={fov}")
        print(f"  相对位置: ({relative_x:.2f}, {relative_y:.2f})")
        
        fx, fy, cx, cy = compute_intrinsics(w, h, fov)
        
        prompts = ["road", "street"]
        result = sam3_tool.get_segmented_images(image_pil, prompts)
        
        ground_mask = None
        for seg_result in result.get('results', []):
            if seg_result.get('success') and seg_result.get('masks'):
                for mask in seg_result.get('masks'):
                    if ground_mask is None:
                        ground_mask = mask.copy()
                    else:
                        ground_mask = np.logical_or(ground_mask, mask)
        
        if ground_mask is None:
            print(f"  ⚠️ 地面检测失败，使用整幅图像")
            ground_mask = np.ones((h, w), dtype=np.uint8)
        
        mask_path = os.path.join(output_dir, f"{base_name}_ground_mask.png")
        cv2.imwrite(mask_path, (ground_mask * 255).astype(np.uint8))
        print(f"  地面mask非零像素: {np.count_nonzero(ground_mask)}")
        
        points_3d = []
        colors = []
        
        for y in range(h):
            for x in range(w):
                if ground_mask[y, x] == 0:
                    continue
                
                world_pt = pixel_to_world(x, y, fx, fy, cx, cy, heading, pitch, relative_x, relative_y, 2.5)
                
                if world_pt is not None:
                    points_3d.append(world_pt)
                    colors.append(image[y, x])
        
        if points_3d:
            points_3d = np.array(points_3d)
            colors = np.array(colors)
            print(f"  投影点数: {len(points_3d)}")
            print(f"  X范围: {points_3d[:, 0].min():.2f} ~ {points_3d[:, 0].max():.2f}")
            print(f"  Y范围: {points_3d[:, 1].min():.2f} ~ {points_3d[:, 1].max():.2f}")
            
            all_points.append(points_3d)
            all_colors.append(colors)
    
    sam3_tool.close()
    
    if not all_points:
        print("\n❌ 没有成功处理的图像")
        return
    
    merged_points = np.vstack(all_points)
    merged_colors = np.vstack(all_colors)
    
    print(f"\n📊 融合统计:")
    print(f"  总点数: {len(merged_points)}")
    
    x_min = merged_points[:, 0].min()
    x_max = merged_points[:, 0].max()
    y_min = merged_points[:, 1].min()
    y_max = merged_points[:, 1].max()
    
    print(f"  X范围: {x_min:.2f} ~ {x_max:.2f} (宽度: {x_max-x_min:.2f}米)")
    print(f"  Y范围: {y_min:.2f} ~ {y_max:.2f} (高度: {y_max-y_min:.2f}米)")
    
    resolution = 10
    padding = 5
    
    bev_width = int((x_max - x_min + padding * 2) * resolution)
    bev_height = int((y_max - y_min + padding * 2) * resolution)
    
    print(f"  BEV尺寸: {bev_height} × {bev_width} (分辨率: {resolution}像素/米)")
    
    bev_image = np.zeros((bev_height, bev_width, 3), dtype=np.uint8)
    hit_count = np.zeros((bev_height, bev_width), dtype=np.int32)
    
    for i in range(len(merged_points)):
        px = int((merged_points[i, 0] - x_min + padding) * resolution)
        py = int((merged_points[i, 1] - y_min + padding) * resolution)
        
        if 0 <= px < bev_width and 0 <= py < bev_height:
            hit_count[py, px] += 1
            bev_image[py, px] = merged_colors[i]
    
    print(f"  有效像素数: {np.count_nonzero(hit_count)}")
    print(f"  平均每像素命中: {len(merged_points) / np.count_nonzero(hit_count) if np.count_nonzero(hit_count) > 0 else 0:.2f}")
    
    dilated_mask = (hit_count > 0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated_mask = cv2.dilate(dilated_mask, kernel, iterations=2)
    
    bev_with_fill = np.zeros((bev_height, bev_width, 3), dtype=np.uint8)
    
    for y in range(bev_height):
        for x in range(bev_width):
            if dilated_mask[y, x] > 0:
                if hit_count[y, x] > 0:
                    bev_with_fill[y, x] = bev_image[y, x]
                else:
                    dists = []
                    for dy in range(-2, 3):
                        for dx in range(-2, 3):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < bev_height and 0 <= nx < bev_width and hit_count[ny, nx] > 0:
                                dists.append((dy * dy + dx * dx, ny, nx))
                    if dists:
                        dists.sort()
                        bev_with_fill[y, x] = bev_image[dists[0][1], dists[0][2]]
    
    bev_path = os.path.join(output_dir, 'fused_bev.png')
    cv2.imwrite(bev_path, cv2.cvtColor(bev_image, cv2.COLOR_RGB2BGR))
    print(f"✅ BEV图像已保存到: {bev_path}")
    print(f"  BEV尺寸: {bev_image.shape[0]} × {bev_image.shape[1]}")
    print(f"  覆盖范围: X: {x_min:.2f}~{x_max:.2f}米, Y: {y_min:.2f}~{y_max:.2f}米")


if __name__ == "__main__":
    main()
