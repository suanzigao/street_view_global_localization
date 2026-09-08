"""
使用理论坐标计算相对位置的投影脚本

问题分析：
- Google Street View API返回的实际采样点与理论计算位置有偏差
- 使用实际坐标会导致相对位置不准确
- 解决方案：基于初始位置和动作序列计算理论相对位置

动作序列：
- step_1: 初始位置，heading=0°（朝北）
- step_2: 原地右转45°，heading=45°（朝东北），位置不变
- step_3: 向前移动15米，heading=45°（朝东北）
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


def calculate_theoretical_position(initial_x, initial_y, heading_deg, move_distance):
    """基于当前朝向和移动距离计算理论位置"""
    heading_rad = math.radians(heading_deg)
    delta_x = move_distance * math.sin(heading_rad)
    delta_y = move_distance * math.cos(heading_rad)
    return initial_x + delta_x, initial_y + delta_y


def pixel_to_world(x, y, fx, fy, cx, cy, heading_deg, pitch_deg, cam_x, cam_y, cam_z):
    yaw = -math.radians(heading_deg)
    pitch = math.radians(pitch_deg)
    
    u = (x - cx) / fx
    v = (y - cy) / fy
    
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    cos_pitch = math.cos(pitch)
    sin_pitch = math.sin(pitch)
    
    right_x = cos_yaw
    right_y = sin_yaw
    
    forward_x = -sin_yaw * cos_pitch
    forward_y = cos_yaw * cos_pitch
    forward_z = sin_pitch
    
    down_x = -sin_yaw * sin_pitch
    down_y = cos_yaw * sin_pitch
    down_z = -cos_pitch
    
    ray_world_x = u * right_x + v * down_x + forward_x
    ray_world_y = u * right_y + v * down_y + forward_y
    ray_world_z = u * 0 + v * down_z + forward_z
    
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
    os.makedirs(output_dir, exist_ok=True)
    
    sam3_tool = SAM3Tool()
    sam3_tool.initialize()
    
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
    
    all_points = []
    all_colors = []
    
    theoretical_x = 0.0
    theoretical_y = 0.0
    current_heading = 0.0
    move_step = 15.0
    
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
        action = metadata.get('action', '')
        
        actual_relative_x = metadata.get('relative_x', 0.0)
        actual_relative_y = metadata.get('relative_y', 0.0)
        
        if action == 'reset':
            theoretical_x = 0.0
            theoretical_y = 0.0
            current_heading = heading
        elif action == 'turn right':
            current_heading = (current_heading + 45) % 360
        elif action == 'turn left':
            current_heading = (current_heading - 45) % 360
        elif action == 'move forward':
            theoretical_x, theoretical_y = calculate_theoretical_position(
                theoretical_x, theoretical_y, current_heading, move_step
            )
        elif action == 'move backward':
            theoretical_x, theoretical_y = calculate_theoretical_position(
                theoretical_x, theoretical_y, current_heading, -move_step
            )
        elif action == 'move right':
            right_heading = (current_heading + 90) % 360
            theoretical_x, theoretical_y = calculate_theoretical_position(
                theoretical_x, theoretical_y, right_heading, move_step
            )
        elif action == 'move left':
            left_heading = (current_heading - 90) % 360
            theoretical_x, theoretical_y = calculate_theoretical_position(
                theoretical_x, theoretical_y, left_heading, move_step
            )
        
        cam_z = 2.5
        fx, fy, cx, cy = compute_intrinsics(w, h, fov)
        
        print(f"\n📋 元数据:")
        print(f"  heading={heading}°, pitch={pitch}°, fov={fov}°")
        print(f"  action={action}")
        print(f"  实际相对位置: ({actual_relative_x:.2f}, {actual_relative_y:.2f})")
        print(f"  理论相对位置: ({theoretical_x:.2f}, {theoretical_y:.2f})")
        
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
            ground_mask = np.ones((h, w), dtype=np.uint8)
        
        points_3d = []
        colors = []
        
        for y in range(h):
            for x in range(w):
                if ground_mask[y, x] == 0:
                    continue
                
                world_pt = pixel_to_world(x, y, fx, fy, cx, cy, heading, pitch, theoretical_x, theoretical_y, cam_z)
                
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
            
            all_points.append(points_3d)
            all_colors.append(colors)
    
    sam3_tool.close()
    
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
        py = bev_height - 1 - int((merged_points[i, 1] - y_min + padding) * resolution)
        
        if 0 <= px < bev_width and 0 <= py < bev_height:
            bev_image[py, px] = merged_colors[i]
    
    bev_path = os.path.join(output_dir, 'fused_bev_theoretical.png')
    cv2.imwrite(bev_path, cv2.cvtColor(bev_image, cv2.COLOR_RGB2BGR))
    print(f"\n✅ BEV图像已保存到: {bev_path}")


if __name__ == "__main__":
    main()
