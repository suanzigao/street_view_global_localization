"""
调试投影脚本

检查地面检测和投影效果
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


def compute_intrinsics(image_size, fov_deg):
    width, height = image_size
    fov = math.radians(fov_deg)
    
    fx = width / (2 * math.tan(fov / 2))
    fy = height / (2 * math.tan(fov / 2))
    cx = width / 2
    cy = height / 2
    
    return fx, fy, cx, cy


def compute_extrinsics(heading_deg, pitch_deg, relative_x, relative_y, camera_height=2.5, roll=0.0):
    yaw = math.radians(heading_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll)
    
    R_yaw = np.array([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1]
    ], dtype=np.float64)
    
    R_pitch = np.array([
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]
    ], dtype=np.float64)
    
    R_roll = np.array([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll), math.cos(roll)]
    ], dtype=np.float64)
    
    R = R_yaw @ R_pitch @ R_roll
    t = np.array([relative_x, relative_y, camera_height], dtype=np.float64)
    
    return R, t


def main():
    images_dir = 'test_relative_position_output/images'
    output_dir = 'debug_output'
    os.makedirs(output_dir, exist_ok=True)
    
    sam3_tool = SAM3Tool()
    sam3_tool.initialize()
    
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
    
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
        
        heading = metadata.get('heading', 0.0)
        pitch = metadata.get('pitch', 0.0)
        fov = metadata.get('fov', 60)
        relative_x = metadata.get('relative_x', 0.0)
        relative_y = metadata.get('relative_y', 0.0)
        
        print(f"  元数据: heading={heading}, pitch={pitch}, fov={fov}")
        print(f"  相对位置: ({relative_x:.2f}, {relative_y:.2f})")
        
        fx, fy, cx, cy = compute_intrinsics(image.shape[:2][::-1], fov)
        print(f"  内参: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
        
        R, t = compute_extrinsics(heading, pitch, relative_x, relative_y)
        C_w = -R.T @ t
        print(f"  相机位置: ({C_w[0]:.2f}, {C_w[1]:.2f}, {C_w[2]:.2f})")
        
        print(f"  SAM3可用: {sam3_tool.available}")
        
        if sam3_tool.available:
            prompts = ["road", "ground", "sidewalk", "street"]
            result = sam3_tool.get_segmented_images(image_pil, prompts)
            
            print(f"  分割结果: success={result.get('success')}, prompt_count={result.get('prompt_count')}")
            
            if 'error' in result:
                print(f"  错误信息: {result.get('error')}")
            
            ground_mask = None
            for seg_result in result.get('results', []):
                print(f"    prompt={seg_result.get('prompt')}, success={seg_result.get('success')}, count={seg_result.get('count')}")
                if seg_result.get('success') and seg_result.get('masks'):
                    for i, mask in enumerate(seg_result.get('masks')):
                        print(f"      mask[{i}]: shape={mask.shape if hasattr(mask, 'shape') else 'N/A'}")
                        if ground_mask is None:
                            ground_mask = mask.copy()
                        else:
                            ground_mask = np.logical_or(ground_mask, mask)
            
            if ground_mask is not None:
                mask_image = (ground_mask * 255).astype(np.uint8)
                mask_path = os.path.join(output_dir, f"{base_name}_ground_mask.png")
                cv2.imwrite(mask_path, mask_image)
                print(f"  地面mask已保存: {mask_path}")
                print(f"  mask形状: {ground_mask.shape}, 非零像素: {np.count_nonzero(ground_mask)}")
            else:
                print(f"  ⚠️ 地面检测失败")
                ground_mask = np.ones(image.shape[:2], dtype=np.uint8)
        else:
            print(f"  ⚠️ SAM3不可用，使用整幅图像")
            ground_mask = np.ones(image.shape[:2], dtype=np.uint8)
        
        h, w = image.shape[:2]
        
        C_w = np.array([relative_x, relative_y, 2.5], dtype=np.float64)
        
        points_3d = []
        for y in range(h):
            for x in range(w):
                if ground_mask[y, x] == 0:
                    continue
                
                X_c = (x - cx) / fx
                Y_c = (y - cy) / fy
                Z_c = 1.0
                
                ray_dir_c = np.array([X_c, Y_c, Z_c])
                ray_dir_w = R @ ray_dir_c
                
                if ray_dir_w[1] >= 0:
                    continue
                
                t_param = -C_w[1] / ray_dir_w[1]
                
                if t_param < 0:
                    continue
                
                P_w = C_w + t_param * ray_dir_w
                points_3d.append(P_w)
        
        points_3d = np.array(points_3d)
        print(f"  投影点数: {len(points_3d)}")
        if len(points_3d) > 0:
            print(f"  X范围: {points_3d[:, 0].min():.2f} ~ {points_3d[:, 0].max():.2f}")
            print(f"  Y范围: {points_3d[:, 1].min():.2f} ~ {points_3d[:, 1].max():.2f}")
            print(f"  Z范围: {points_3d[:, 2].min():.2f} ~ {points_3d[:, 2].max():.2f}")
    
    sam3_tool.close()
    print("\n调试完成！")


if __name__ == "__main__":
    main()
