"""
修复投影脚本

重新定义坐标系和旋转矩阵
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


def compute_extrinsics(heading_deg, pitch_deg, relative_x, relative_y, camera_height=2.5):
    """
    计算外参矩阵
    
    坐标系定义：
    - 世界坐标系：X=东，Y=北，Z=上（右手系）
    - 相机坐标系：X=右，Y=下，Z=前（右手系）
    
    heading: 相机朝向（0=北，90=东，180=南，270=西）
    pitch: 俯仰角（0=水平，正数=向上看，负数=向下看）
    
    R矩阵：将相机坐标系中的向量转换到世界坐标系
    """
    yaw = math.radians(heading_deg)
    pitch = math.radians(pitch_deg)
    
    forward_x = math.sin(yaw) * math.cos(pitch)
    forward_y = math.cos(yaw) * math.cos(pitch)
    forward_z = -math.sin(pitch)
    
    right_x = math.cos(yaw)
    right_y = -math.sin(yaw)
    right_z = 0
    
    down_x = math.sin(yaw) * math.sin(pitch)
    down_y = math.cos(yaw) * math.sin(pitch)
    down_z = -math.cos(pitch)
    
    R = np.array([
        [right_x, down_x, forward_x],
        [right_y, down_y, forward_y],
        [right_z, down_z, forward_z]
    ], dtype=np.float64)
    
    C_w = np.array([relative_x, relative_y, camera_height], dtype=np.float64)
    
    return R, C_w


def project_to_ground(image, fx, fy, cx, cy, R, C_w, ground_mask=None):
    """
    将图像像素投影到地面
    
    使用射线与地面平面（Z=0）求交
    
    坐标系：
    - 图像：y向下（像素值越大越靠下）
    - 相机：Y向上（标准相机模型）
    - 所以图像y需要取反得到相机Y
    """
    h, w = image.shape[:2]
    
    points_3d = []
    colors = []
    
    for y in range(h):
        for x in range(w):
            if ground_mask is not None and ground_mask[y, x] == 0:
                continue
            
            u = (x - cx) / fx
            v = -(y - cy) / fy
            
            ray_dir_c = np.array([u, v, 1.0])
            ray_dir_w = R @ ray_dir_c
            
            if ray_dir_w[2] >= -1e-6:
                continue
            
            t_param = -C_w[2] / ray_dir_w[2]
            
            if t_param < 0:
                continue
            
            P_w = C_w + t_param * ray_dir_w
            
            points_3d.append(P_w)
            colors.append(image[y, x])
    
    if not points_3d:
        return None, None
    
    return np.array(points_3d), np.array(colors)


def render_bev(points_3d, colors, resolution=5):
    if points_3d is None or len(points_3d) == 0:
        return None, None
    
    x_min = points_3d[:, 0].min()
    x_max = points_3d[:, 0].max()
    y_min = points_3d[:, 1].min()
    y_max = points_3d[:, 1].max()
    
    padding = 5
    
    bev_width = int((x_max - x_min + padding * 2) * resolution)
    bev_height = int((y_max - y_min + padding * 2) * resolution)
    
    bev_image = np.zeros((bev_height, bev_width, 3), dtype=np.uint8)
    
    for i in range(len(points_3d)):
        px = int((points_3d[i, 0] - x_min + padding) * resolution)
        py = int((points_3d[i, 1] - y_min + padding) * resolution)
        
        if 0 <= px < bev_width and 0 <= py < bev_height:
            bev_image[py, px] = colors[i]
    
    return bev_image, (x_min, x_max, y_min, y_max)


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
        
        heading = metadata.get('heading', 0.0)
        pitch = metadata.get('pitch', 0.0)
        fov = metadata.get('fov', 60)
        relative_x = metadata.get('relative_x', 0.0)
        relative_y = metadata.get('relative_y', 0.0)
        
        print(f"  元数据: heading={heading}, pitch={pitch}, fov={fov}")
        print(f"  相对位置: ({relative_x:.2f}, {relative_y:.2f})")
        
        fx, fy, cx, cy = compute_intrinsics(image.shape[:2][::-1], fov)
        print(f"  内参: fx={fx:.2f}, fy={fy:.2f}")
        
        R, C_w = compute_extrinsics(heading, pitch, relative_x, relative_y)
        print(f"  相机位置: ({C_w[0]:.2f}, {C_w[1]:.2f}, {C_w[2]:.2f})")
        
        prompts = ["road", "ground", "sidewalk", "street"]
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
            ground_mask = np.ones(image.shape[:2], dtype=np.uint8)
        
        mask_path = os.path.join(output_dir, f"{base_name}_ground_mask.png")
        cv2.imwrite(mask_path, (ground_mask * 255).astype(np.uint8))
        print(f"  地面mask非零像素: {np.count_nonzero(ground_mask)}")
        
        points_3d, colors = project_to_ground(image, fx, fy, cx, cy, R, C_w, ground_mask)
        
        if points_3d is not None:
            print(f"  投影点数: {len(points_3d)}")
            print(f"  X范围: {points_3d[:, 0].min():.2f} ~ {points_3d[:, 0].max():.2f}")
            print(f"  Y范围: {points_3d[:, 1].min():.2f} ~ {points_3d[:, 1].max():.2f}")
            print(f"  Z范围: {points_3d[:, 2].min():.2f} ~ {points_3d[:, 2].max():.2f}")
            
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
    
    bev_image, bounds = render_bev(merged_points, merged_colors)
    
    if bev_image is not None:
        bev_path = os.path.join(output_dir, 'fused_bev.png')
        cv2.imwrite(bev_path, cv2.cvtColor(bev_image, cv2.COLOR_RGB2BGR))
        print(f"✅ BEV图像已保存到: {bev_path}")
        print(f"  BEV尺寸: {bev_image.shape[0]} × {bev_image.shape[1]}")
        print(f"  覆盖范围: X: {bounds[0]:.2f}~{bounds[1]:.2f}米, Y: {bounds[2]:.2f}~{bounds[3]:.2f}米")


if __name__ == "__main__":
    main()
