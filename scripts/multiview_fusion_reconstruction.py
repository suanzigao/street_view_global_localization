"""
多视图融合重建模块

基于街景图像和相机参数进行投影重建，生成BEV图像

步骤：
1. 读取图像和元数据
2. 使用SAM3进行地面/道路检测
3. 根据相机参数计算内参和外参
4. 将地面区域投影到世界坐标系
5. 多视图融合生成完整BEV
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


class MultiviewReconstructor:
    """多视图融合重建器"""
    
    def __init__(self, camera_height=2.5, roll=0.0):
        self.camera_height = camera_height
        self.roll = roll
        self.sam3_tool = None
    
    def init_sam3(self):
        """初始化SAM3模型"""
        if self.sam3_tool is None:
            self.sam3_tool = SAM3Tool()
            self.sam3_tool.initialize()
    
    def close(self):
        """关闭SAM3模型"""
        if self.sam3_tool:
            self.sam3_tool.close()
            self.sam3_tool = None
    
    def read_metadata(self, metadata_path):
        """读取元数据文件"""
        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def compute_intrinsics(self, image_size, fov_deg):
        """计算内参矩阵K"""
        width, height = image_size
        fov = math.radians(fov_deg)
        
        fx = width / (2 * math.tan(fov / 2))
        fy = height / (2 * math.tan(fov / 2))
        cx = width / 2
        cy = height / 2
        
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        return K
    
    def compute_extrinsics(self, heading_deg, pitch_deg, relative_x, relative_y):
        """计算外参矩阵R和t"""
        yaw = math.radians(heading_deg)
        pitch = math.radians(pitch_deg)
        roll = math.radians(self.roll)
        
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
        
        t = np.array([relative_x, relative_y, self.camera_height], dtype=np.float64)
        
        return R, t
    
    def detect_ground(self, image):
        """使用SAM3检测地面和道路区域"""
        if not self.sam3_tool:
            self.init_sam3()
        
        prompts = ["road", "ground", "sidewalk", "street"]
        result = self.sam3_tool.get_segmented_images(image, prompts)
        
        if not result.get('success'):
            return None, None
        
        ground_mask = None
        for seg_result in result.get('results', []):
            if seg_result.get('success') and seg_result.get('masks'):
                for mask in seg_result.get('masks'):
                    if ground_mask is None:
                        ground_mask = mask.copy()
                    else:
                        ground_mask = np.logical_or(ground_mask, mask)
        
        return ground_mask, result
    
    def project_to_ground(self, image, K, R, t, ground_mask=None):
        """将图像投影到地面平面（使用射线与地面平面求交）"""
        h, w = image.shape[:2]
        
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        
        C_w = -R.T @ t
        
        points_3d = []
        colors = []
        
        for y in range(h):
            for x in range(w):
                if ground_mask is not None and ground_mask[y, x] == 0:
                    continue
                
                X_c = (x - cx) / fx
                Y_c = (y - cy) / fy
                Z_c = 1.0
                
                ray_dir_c = np.array([X_c, Y_c, Z_c])
                ray_dir_w = R @ ray_dir_c
                
                if abs(ray_dir_w[1]) < 1e-6:
                    continue
                
                t_param = -C_w[1] / ray_dir_w[1]
                
                if t_param < 0:
                    continue
                
                P_w = C_w + t_param * ray_dir_w
                
                points_3d.append(P_w)
                colors.append(image[y, x])
        
        if not points_3d:
            return None, None
        
        points_3d = np.array(points_3d)
        colors = np.array(colors)
        
        return points_3d, colors
    
    def render_bev(self, points_3d, colors, resolution=5):
        """从3D点渲染BEV图像"""
        if points_3d is None or len(points_3d) == 0:
            return None
        
        x_min, x_max = points_3d[:, 0].min(), points_3d[:, 0].max()
        z_min, z_max = points_3d[:, 2].min(), points_3d[:, 2].max()
        
        padding = 5
        
        bev_width = int((x_max - x_min + padding * 2) * resolution)
        bev_height = int((z_max - z_min + padding * 2) * resolution)
        
        bev_image = np.zeros((bev_height, bev_width, 3), dtype=np.uint8)
        
        for i in range(len(points_3d)):
            x = int((points_3d[i, 0] - x_min + padding) * resolution)
            z = int((points_3d[i, 2] - z_min + padding) * resolution)
            
            if 0 <= x < bev_width and 0 <= z < bev_height:
                bev_image[z, x] = colors[i]
        
        return bev_image, (x_min, x_max, z_min, z_max)
    
    def process_single_image(self, image_path, metadata_path):
        """处理单张图像"""
        image = np.array(Image.open(image_path).convert('RGB'))
        metadata = self.read_metadata(metadata_path)
        
        heading = metadata.get('heading', 0.0)
        pitch = metadata.get('pitch', 0.0)
        fov = metadata.get('fov', 60)
        relative_x = metadata.get('relative_x', 0.0)
        relative_y = metadata.get('relative_y', 0.0)
        
        K = self.compute_intrinsics(image.shape[:2][::-1], fov)
        R, t = self.compute_extrinsics(heading, pitch, relative_x, relative_y)
        
        ground_mask, _ = self.detect_ground(Image.fromarray(image))
        
        if ground_mask is None:
            print(f"⚠️ {os.path.basename(image_path)}: 地面检测失败")
            ground_mask = np.ones(image.shape[:2], dtype=np.uint8)
        
        points_3d, colors = self.project_to_ground(image, K, R, t, ground_mask)
        
        return points_3d, colors, relative_x, relative_y
    
    def fuse_multiview(self, images_dir, output_dir):
        """融合多视图图像"""
        os.makedirs(output_dir, exist_ok=True)
        
        image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
        
        all_points = []
        all_colors = []
        all_offsets = []
        
        for img_file in image_files:
            base_name = img_file.replace('_streetview.jpg', '')
            metadata_file = f"{base_name}_metadata.json"
            
            image_path = os.path.join(images_dir, img_file)
            metadata_path = os.path.join(images_dir, metadata_file)
            
            if not os.path.exists(metadata_path):
                print(f"⚠️ {metadata_file} 不存在，跳过")
                continue
            
            print(f"🔄 处理 {img_file}...")
            
            points_3d, colors, dx, dy = self.process_single_image(image_path, metadata_path)
            
            if points_3d is not None:
                all_points.append(points_3d)
                all_colors.append(colors)
                all_offsets.append((dx, dy))
        
        if not all_points:
            print("❌ 没有成功处理的图像")
            return
        
        merged_points = np.vstack(all_points)
        merged_colors = np.vstack(all_colors)
        
        print(f"\n📊 融合统计:")
        print(f"  图像数量: {len(image_files)}")
        print(f"  总点数: {len(merged_points)}")
        
        bev_image, bounds = self.render_bev(merged_points, merged_colors)
        
        if bev_image is not None:
            bev_path = os.path.join(output_dir, 'fused_bev.png')
            cv2.imwrite(bev_path, cv2.cvtColor(bev_image, cv2.COLOR_RGB2BGR))
            print(f"✅ BEV图像已保存到: {bev_path}")
            print(f"  BEV尺寸: {bev_image.shape[0]} × {bev_image.shape[1]}")
            print(f"  覆盖范围: X: {bounds[0]:.2f}~{bounds[1]:.2f}米, Z: {bounds[2]:.2f}~{bounds[3]:.2f}米")
        
        return bev_image, bounds


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='多视图融合重建')
    parser.add_argument('--input_dir', type=str, default='test_relative_position_output/images',
                        help='输入图像目录')
    parser.add_argument('--output_dir', type=str, default='reconstruction_output',
                        help='输出目录')
    parser.add_argument('--camera_height', type=float, default=2.5,
                        help='相机高度（米）')
    
    args = parser.parse_args()
    
    print("🚀 开始多视图融合重建")
    print(f"📁 输入目录: {args.input_dir}")
    print(f"📁 输出目录: {args.output_dir}")
    print(f"📐 相机高度: {args.camera_height}米")
    
    reconstructor = MultiviewReconstructor(camera_height=args.camera_height)
    
    try:
        reconstructor.init_sam3()
        
        bev_image, bounds = reconstructor.fuse_multiview(args.input_dir, args.output_dir)
        
        print("\n🎉 重建完成！")
        if bev_image is not None:
            print(f"📷 BEV图像尺寸: {bev_image.shape}")
            print(f"📐 覆盖范围: {bounds}")
    finally:
        reconstructor.close()


if __name__ == "__main__":
    main()
