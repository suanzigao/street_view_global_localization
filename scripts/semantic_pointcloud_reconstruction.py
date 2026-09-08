"""
语义点云重建完整流程
整合SAM3分割和VGGT推理，实现点云语义分类
"""
import os
import sys
import cv2
import base64
import io
import json
import requests
import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.sam3_tool import SAM3Tool


class SemanticPointcloudPipeline:
    def __init__(self, vggt_server="http://127.0.0.1:18888"):
        self.vggt_server = vggt_server
        self.sam3 = None
        self.vggt_size = 518
    
    def init_sam3(self):
        """初始化SAM3模型"""
        if self.sam3 is None:
            print("🔄 初始化SAM3模型...")
            self.sam3 = SAM3Tool()
            self.sam3.initialize()
            print("✅ SAM3模型初始化完成")
    
    def segment_streetview(self, image):
        """
        SAM3语义分割
        
        Args:
            image: 输入图像
        
        Returns:
            dict: 分割掩码
        """
        # 道路分割
        road_result = self.sam3.segment(
            Image.fromarray(image),
            prompt="road"
        )
        
        # 建筑分割
        building_result = self.sam3.segment(
            Image.fromarray(image),
            prompt="building"
        )
        
        # 天空分割（用于过滤）
        sky_result = self.sam3.segment(
            Image.fromarray(image),
            prompt="sky"
        )
        
        # 提取掩码（SAM3返回masks列表，合并多个掩码）
        road_masks = road_result.get("masks", [])
        building_masks = building_result.get("masks", [])
        sky_masks = sky_result.get("masks", [])
        
        # 转换为二值掩码（0和1）
        road_mask = np.any(road_masks, axis=0).astype(np.uint8) if len(road_masks) > 0 else np.zeros(image.shape[:2], dtype=np.uint8)
        building_mask = np.any(building_masks, axis=0).astype(np.uint8) if len(building_masks) > 0 else np.zeros(image.shape[:2], dtype=np.uint8)
        sky_mask = np.any(sky_masks, axis=0).astype(np.uint8) if len(sky_masks) > 0 else np.zeros(image.shape[:2], dtype=np.uint8)
        
        # 优化掩码：去除天空区域
        road_mask = road_mask & (1 - sky_mask)
        building_mask = building_mask & (1 - sky_mask)
        
        # 形态学优化
        kernel = np.ones((5, 5), np.uint8)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)
        building_mask = cv2.morphologyEx(building_mask, cv2.MORPH_CLOSE, kernel)
        
        # 调试：检查掩码
        print(f"    调试: road_masks数量={len(road_masks)}, road_mask非零={np.sum(road_mask > 0)}")
        print(f"    调试: building_masks数量={len(building_masks)}, building_mask非零={np.sum(building_mask > 0)}")
        print(f"    调试: sky_masks数量={len(sky_masks)}, sky_mask非零={np.sum(sky_mask > 0)}")
        
        return {
            "road": road_mask,
            "building": building_mask,
            "sky": sky_mask
        }
    
    def vggt_infer(self, images):
        """
        VGGT几何重建
        
        Args:
            images: 图像列表
        
        Returns:
            dict: VGGT输出结果
        """
        print("🔄 正在进行VGGT推理...")
        
        # 图像编码
        imgs_b64 = []
        for img in images:
            pil_img = Image.fromarray(img)
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=90)
            imgs_b64.append(base64.b64encode(buf.getvalue()).decode())
        
        # VGGT推理请求
        req = {
            "images": imgs_b64,
            "output_types": ["camera", "depth", "points"]
        }
        
        r = requests.post(
            f"{self.vggt_server}/vggt/infer",
            json=req,
            timeout=300
        )
        result = r.json()
        
        if not result.get("success"):
            raise RuntimeError(f"VGGT推理失败: {result.get('error')}")
        
        print("✅ VGGT推理完成")
        
        return {
            "pose_enc": np.array(result["pose_enc"]),
            "depth": np.array(result["depth"]),
            "world_points": np.array(result["world_points"]),
            "world_points_conf": np.array(result["world_points_conf"])
        }
    
    def align_masks(self, masks, original_size):
        """
        将分割掩码对齐到VGGT输出尺寸
        
        Args:
            masks: 原始尺寸的分割掩码
            original_size: 原始图像尺寸
        
        Returns:
            aligned_masks: 对齐后的掩码
        """
        aligned_masks = {}
        
        for key, mask in masks.items():
            print(f"    {key}掩码: 原始尺寸{mask.shape}, 非零像素{np.sum(mask > 0)}")
            aligned_mask = cv2.resize(
                mask.astype(np.uint8),
                (self.vggt_size, self.vggt_size),
                interpolation=cv2.INTER_NEAREST
            )
            aligned_mask = (aligned_mask > 0).astype(np.uint8)
            print(f"    {key}掩码: 对齐后尺寸{aligned_mask.shape}, 非零像素{np.sum(aligned_mask > 0)}")
            aligned_masks[key] = aligned_mask
        
        return aligned_masks
    
    def classify_pointcloud(self, masks, world_points, world_points_conf, colors):
        """
        点云语义分类
        
        Args:
            masks: 分割掩码
            world_points: 世界坐标系点云 [H, W, 3]
            world_points_conf: 点云置信度 [H, W]
            colors: 点云颜色 [H*W, 3]
        
        Returns:
            dict: 分类后的点云
        """
        # 展平数据
        points_flat = world_points.reshape(-1, 3)
        conf_flat = world_points_conf.flatten()
        
        # 过滤条件：有效点 + 置信度大于最小值（VGGT使用expp1激活，最小值为1）
        valid_mask = (
            (np.isfinite(points_flat).all(axis=1)) &
            (conf_flat > 1.0)  # 降低阈值，接受所有有效点
        )
        
        # 道路点云提取
        road_flat = masks["road"].flatten()
        road_valid = valid_mask & (road_flat > 0)
        road_points = points_flat[road_valid]
        road_colors = colors[road_valid]
        
        # 建筑点云提取
        building_flat = masks["building"].flatten()
        building_valid = valid_mask & (building_flat > 0)
        building_points = points_flat[building_valid]
        building_colors = colors[building_valid]
        
        # 道路点云优化：只保留地面附近的点
        if len(road_points) > 0:
            road_heights = road_points[:, 1]
            ground_threshold = 0.5
            road_ground_mask = np.abs(road_heights) < ground_threshold
            road_points = road_points[road_ground_mask]
            road_colors = road_colors[road_ground_mask]
        
        # 建筑点云优化：保留合理高度的点
        if len(building_points) > 0:
            building_heights = building_points[:, 1]
            building_mask = (
                (building_heights > -0.5) &
                (building_heights < 50.0)
            )
            building_points = building_points[building_mask]
            building_colors = building_colors[building_mask]
        
        return {
            "road": {
                "points": road_points,
                "colors": road_colors
            },
            "building": {
                "points": building_points,
                "colors": building_colors
            }
        }
    
    def generate_bev(self, points, colors, output_size=(2048, 2048)):
        """
        生成BEV视角
        
        Args:
            points: 点云 [N, 3]
            colors: 颜色 [N, 3]
            output_size: 输出尺寸（增大到2048x2048以覆盖更大范围）
        
        Returns:
            bev_image: BEV图像
        """
        if len(points) == 0:
            return np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
        
        x_coords = points[:, 0]
        z_coords = points[:, 2]
        
        # 计算边界（使用1%和99%来覆盖更多点云）
        x_min, x_max = np.percentile(x_coords, 1), np.percentile(x_coords, 99)
        z_min, z_max = np.percentile(z_coords, 1), np.percentile(z_coords, 99)
        
        x_range = x_max - x_min if x_max > x_min else 1.0
        z_range = z_max - z_min if z_max > z_min else 1.0
        
        # 创建BEV图像（都使用彩色）
        bev_image = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
        
        # 映射坐标
        img_x = ((x_coords - x_min) / x_range * output_size[0]).astype(np.int32)
        img_y = ((z_coords - z_min) / z_range * output_size[1]).astype(np.int32)
        
        # 过滤边界
        mask = (
            (img_x >= 0) & (img_x < output_size[0]) &
            (img_y >= 0) & (img_y < output_size[1])
        )
        
        valid_x = img_x[mask]
        valid_y = img_y[mask]
        valid_colors = colors[mask]
        
        # 填充BEV图像（都使用原始颜色）
        bev_image[valid_y, valid_x] = valid_colors
        
        # 形态学优化：填充空洞
        kernel = np.ones((5, 5), np.uint8)
        for c in range(3):  # 对每个通道分别处理
            bev_image[:, :, c] = cv2.morphologyEx(bev_image[:, :, c], cv2.MORPH_CLOSE, kernel)
        
        # 翻转图像（修复上下颠倒）
        bev_image = cv2.flip(bev_image, 0)
        
        return bev_image
    
    def save_ply(self, filename, points, colors=None):
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
    
    def run(self, image_paths, output_dir):
        """
        运行完整流程
        
        Args:
            image_paths: 图像路径列表
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 初始化SAM3
        self.init_sam3()
        
        # 2. 加载图像
        print("\n📷 加载图像...")
        images = []
        for p in image_paths:
            img = cv2.imread(p)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
        print(f"  ✅ 加载 {len(images)} 张图像")
        
        # 创建中间结果目录
        intermediate_dir = os.path.join(output_dir, "intermediate")
        masks_dir = os.path.join(intermediate_dir, "sam3_masks")
        vggt_dir = os.path.join(intermediate_dir, "vggt_output")
        aligned_dir = os.path.join(intermediate_dir, "aligned_masks")
        os.makedirs(masks_dir, exist_ok=True)
        os.makedirs(vggt_dir, exist_ok=True)
        os.makedirs(aligned_dir, exist_ok=True)
        
        # 3. SAM3分割
        print("\n🔄 SAM3语义分割...")
        all_masks = []
        for i, img in enumerate(images):
            masks = self.segment_streetview(img)
            all_masks.append(masks)
            
            # 保存分割掩码
            cv2.imwrite(os.path.join(masks_dir, f"step_{i}_road_mask.png"), (masks["road"] * 255).astype(np.uint8))
            cv2.imwrite(os.path.join(masks_dir, f"step_{i}_building_mask.png"), (masks["building"] * 255).astype(np.uint8))
            cv2.imwrite(os.path.join(masks_dir, f"step_{i}_sky_mask.png"), (masks["sky"] * 255).astype(np.uint8))
            
            # 保存叠加可视化
            overlay = img.copy()
            overlay[masks["road"] > 0] = [255, 0, 0]  # 道路红色
            overlay[masks["building"] > 0] = [0, 255, 0]  # 建筑绿色
            cv2.imwrite(os.path.join(masks_dir, f"step_{i}_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            
            print(f"  ✅ 图像 {i+1}/{len(images)} 分割完成")
        
        # 4. VGGT重建
        vggt_result = self.vggt_infer(images)
        
        # 保存VGGT输出
        depths = vggt_result["depth"]
        for i, depth in enumerate(depths):
            # 归一化深度图用于可视化
            depth_norm = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(vggt_dir, f"step_{i}_depth.png"), depth_norm)
        
        # 保存完整点云
        np.save(os.path.join(vggt_dir, "world_points.npy"), vggt_result["world_points"])
        np.save(os.path.join(vggt_dir, "world_points_conf.npy"), vggt_result["world_points_conf"])
        print(f"  ✅ VGGT结果已保存")
        
        # 5. 数据对齐
        print("\n🔄 数据对齐...")
        aligned_masks = []
        for i, masks in enumerate(all_masks):
            aligned = self.align_masks(masks, images[0].shape[:2])
            aligned_masks.append(aligned)
            
            # 保存对齐后的掩码
            cv2.imwrite(os.path.join(aligned_dir, f"step_{i}_road_aligned.png"), (aligned["road"] * 255).astype(np.uint8))
            cv2.imwrite(os.path.join(aligned_dir, f"step_{i}_building_aligned.png"), (aligned["building"] * 255).astype(np.uint8))
        
        print("  ✅ 数据对齐完成")
        
        # 保存中间变量供后续使用
        all_world_points = vggt_result["world_points"]
        all_world_points_conf = vggt_result["world_points_conf"]
        all_aligned_masks = aligned_masks
        all_images = images  # 保存原始图像列表
        
        # 6. 点云分类
        print("\n🔄 点云语义分类...")
        all_road_points = []
        all_road_colors = []
        all_building_points = []
        all_building_colors = []
        
        for i, (masks, points, conf) in enumerate(
            zip(aligned_masks, vggt_result["world_points"], vggt_result["world_points_conf"])
        ):
            # 提取颜色
            img = cv2.resize(images[i], (self.vggt_size, self.vggt_size))
            colors = img.reshape(-1, 3)
            
            # 分类点云
            classified = self.classify_pointcloud(masks, points, conf, colors)
            
            all_road_points.append(classified["road"]["points"])
            all_road_colors.append(classified["road"]["colors"])
            all_building_points.append(classified["building"]["points"])
            all_building_colors.append(classified["building"]["colors"])
            
            print(f"  ✅ 图像 {i+1}/{len(images)} 分类完成")
        
        # 7. 合并点云
        print("\n🔄 合并点云...")
        merged_road_points = np.vstack(all_road_points)
        merged_road_colors = np.vstack(all_road_colors)
        merged_building_points = np.vstack(all_building_points)
        merged_building_colors = np.vstack(all_building_colors)
        
        # 7.1 收集完整点云（只去除天空，不过滤低置信度）
        print("\n🔄 收集完整点云（去除天空，保留所有置信度）...")
        all_sky_removed_points = []
        all_sky_removed_colors = []
        
        for i, (masks, points, conf, img) in enumerate(
            zip(all_aligned_masks, all_world_points, all_world_points_conf, all_images)
        ):
            # 展平数据
            points_flat = points.reshape(-1, 3)
            colors = cv2.resize(img, (self.vggt_size, self.vggt_size)).reshape(-1, 3)
            
            # 有效点（只过滤NaN/Inf，不过滤置信度）
            valid_mask = np.isfinite(points_flat).all(axis=1)
            
            # 天空掩码（去除天空）
            sky_flat = masks["sky"].flatten()
            sky_removed = valid_mask & (sky_flat == 0)  # 去除天空区域
            
            sky_removed_pts = points_flat[sky_removed]
            sky_removed_colors = colors[sky_removed]
            
            all_sky_removed_points.append(sky_removed_pts)
            all_sky_removed_colors.append(sky_removed_colors)
        
        merged_sky_removed_points = np.vstack(all_sky_removed_points)
        merged_sky_removed_colors = np.vstack(all_sky_removed_colors)
        
        # 7.1.1 收集去除天空的点云（带语义染色）
        print("\n🔄 收集去除天空点云（带语义染色）...")
        all_sky_removed_colored_points = []
        all_sky_removed_colored_colors = []
        
        for i, (masks, points, img) in enumerate(zip(all_aligned_masks, all_world_points, all_images)):
            points_flat = points.reshape(-1, 3)
            colors = cv2.resize(img, (self.vggt_size, self.vggt_size)).reshape(-1, 3)
            
            sky_flat = masks["sky"].flatten()
            road_flat = masks["road"].flatten()
            building_flat = masks["building"].flatten()
            
            sky_removed = np.isfinite(points_flat).all(axis=1) & (sky_flat == 0)
            
            sky_removed_pts = points_flat[sky_removed]
            sky_removed_colors = colors[sky_removed]
            
            # 语义染色：道路红，建筑绿
            road_mask = road_flat[sky_removed] > 0
            building_mask = building_flat[sky_removed] > 0
            sky_removed_colors[road_mask] = [0, 0, 255]      # 道路标红（BGR格式）
            sky_removed_colors[building_mask] = [0, 255, 0]   # 建筑标绿（BGR格式）
            
            all_sky_removed_colored_points.append(sky_removed_pts)
            all_sky_removed_colored_colors.append(sky_removed_colors)
        
        merged_sky_removed_colored_points = np.vstack(all_sky_removed_colored_points)
        merged_sky_removed_colored_colors = np.vstack(all_sky_removed_colored_colors)
        
        # 7.2 收集完整点云（不做任何过滤）
        print("\n🔄 收集完整点云（不做任何过滤）...")
        all_full_points = []
        all_full_colors = []
        
        for i, (points, img) in enumerate(zip(all_world_points, all_images)):
            points_flat = points.reshape(-1, 3)
            colors = cv2.resize(img, (self.vggt_size, self.vggt_size)).reshape(-1, 3)
            valid_mask = np.isfinite(points_flat).all(axis=1)
            all_full_points.append(points_flat[valid_mask])
            all_full_colors.append(colors[valid_mask])
        
        merged_full_points = np.vstack(all_full_points)
        merged_full_colors = np.vstack(all_full_colors)
        
        # 7.3 收集置信度过滤点云（只过滤置信度，保留语义分割信息）
        print("\n🔄 收集置信度过滤点云（区分道路/建筑）...")
        all_conf_filtered_points = []
        all_conf_filtered_colors = []
        all_conf_filtered_labels = []  # 0=其他, 1=道路, 2=建筑
        
        CONF_THRESHOLD = 1.0001
        
        for i, (masks, points, conf, img) in enumerate(
            zip(all_aligned_masks, all_world_points, all_world_points_conf, all_images)
        ):
            points_flat = points.reshape(-1, 3)
            colors = cv2.resize(img, (self.vggt_size, self.vggt_size)).reshape(-1, 3)
            conf_flat = conf.reshape(-1)
            
            # 只过滤置信度（不过滤语义）
            valid_mask = np.isfinite(points_flat).all(axis=1) & (conf_flat > CONF_THRESHOLD)
            
            # 创建标签（0=其他, 1=道路, 2=建筑）
            labels = np.zeros(len(points_flat), dtype=np.int32)
            road_flat = masks["road"].flatten()
            building_flat = masks["building"].flatten()
            labels[road_flat > 0] = 1
            labels[building_flat > 0] = 2
            
            all_conf_filtered_points.append(points_flat[valid_mask])
            all_conf_filtered_colors.append(colors[valid_mask])
            all_conf_filtered_labels.append(labels[valid_mask])
        
        merged_conf_filtered_points = np.vstack(all_conf_filtered_points)
        merged_conf_filtered_colors = np.vstack(all_conf_filtered_colors)
        merged_conf_filtered_labels = np.hstack(all_conf_filtered_labels)
        
        # 创建带语义标签的颜色（用于BEV和点云保存）
        # 注意：OpenCV使用BGR格式，所以颜色顺序是[B, G, R]
        conf_filtered_colors = merged_conf_filtered_colors.copy()
        road_mask = merged_conf_filtered_labels == 1
        building_mask = merged_conf_filtered_labels == 2
        conf_filtered_colors[road_mask] = [0, 0, 255]      # 道路标红（BGR格式）
        conf_filtered_colors[building_mask] = [0, 255, 0]   # 建筑标绿（BGR格式）
        
        # 7.3.1 收集置信度过滤点云（只过滤置信度，保留原始颜色，不染色）
        print("\n🔄 收集置信度过滤点云（只过滤置信度，保留原始颜色）...")
        # 使用原始颜色，不进行语义染色
        merged_conf_filtered_raw_points = merged_conf_filtered_points.copy()
        merged_conf_filtered_raw_colors = merged_conf_filtered_colors.copy()  # 保留原始颜色
        
        # 7.4 收集道路点云（只通过分割，不过滤置信度）
        print("\n🔄 收集道路点云（只分割，不过滤置信度）...")
        all_road_no_conf_points = []
        all_road_no_conf_colors = []
        
        for i, (masks, points, img) in enumerate(zip(all_aligned_masks, all_world_points, all_images)):
            points_flat = points.reshape(-1, 3)
            colors = cv2.resize(img, (self.vggt_size, self.vggt_size)).reshape(-1, 3)
            
            # 只过滤NaN/Inf，不过滤置信度
            valid_mask = np.isfinite(points_flat).all(axis=1)
            
            # 道路掩码
            road_flat = masks["road"].flatten()
            road_valid = valid_mask & (road_flat > 0)
            
            all_road_no_conf_points.append(points_flat[road_valid])
            all_road_no_conf_colors.append(colors[road_valid])
        
        merged_road_no_conf_points = np.vstack(all_road_no_conf_points)
        merged_road_no_conf_colors = np.vstack(all_road_no_conf_colors)
        
        # 7.4 收集建筑点云（只通过分割，不过滤置信度）
        print("\n🔄 收集建筑点云（只分割，不过滤置信度）...")
        all_building_no_conf_points = []
        all_building_no_conf_colors = []
        
        for i, (masks, points, img) in enumerate(zip(all_aligned_masks, all_world_points, all_images)):
            points_flat = points.reshape(-1, 3)
            colors = cv2.resize(img, (self.vggt_size, self.vggt_size)).reshape(-1, 3)
            
            # 只过滤NaN/Inf，不过滤置信度
            valid_mask = np.isfinite(points_flat).all(axis=1)
            
            # 建筑掩码
            building_flat = masks["building"].flatten()
            building_valid = valid_mask & (building_flat > 0)
            
            all_building_no_conf_points.append(points_flat[building_valid])
            all_building_no_conf_colors.append(colors[building_valid])
        
        merged_building_no_conf_points = np.vstack(all_building_no_conf_points)
        merged_building_no_conf_colors = np.vstack(all_building_no_conf_colors)
        
        print(f"  完整点云（无过滤）: {len(merged_full_points):,d} 点")
        print(f"  完整点云（去除天空）: {len(merged_sky_removed_points):,d} 点")
        print(f"  置信度过滤点云: {len(merged_conf_filtered_points):,d} 点")
        print(f"    - 道路: {np.sum(merged_conf_filtered_labels == 1):,d} 点")
        print(f"    - 建筑: {np.sum(merged_conf_filtered_labels == 2):,d} 点")
        print(f"    - 其他: {np.sum(merged_conf_filtered_labels == 0):,d} 点")
        print(f"  道路点云: {len(merged_road_points):,d} 点")
        print(f"  建筑点云: {len(merged_building_points):,d} 点")
        
        # 8. 生成所有点云的BEV
        print("\n🔄 生成所有点云的BEV...")
        
        # 8.1 完整点云BEV
        print("  生成 full_bev...")
        full_bev = self.generate_bev(merged_full_points, merged_full_colors)
        
        # 8.2 去除天空点云BEV（原始颜色）
        print("  生成 sky_removed_bev...")
        sky_removed_bev = self.generate_bev(merged_sky_removed_points, merged_sky_removed_colors)
        
        # 8.2.1 去除天空点云BEV（带语义染色）
        print("  生成 sky_removed_colored_bev...")
        sky_removed_colored_bev = self.generate_bev(merged_sky_removed_colored_points, merged_sky_removed_colored_colors)
        
        # 8.3 道路点云BEV（置信度过滤）
        print("  生成 road_bev...")
        road_bev = self.generate_bev(merged_road_points, merged_road_colors)
        
        # 8.4 建筑点云BEV（置信度过滤）
        print("  生成 building_bev...")
        building_bev = self.generate_bev(merged_building_points, merged_building_colors)
        
        # 8.5 道路点云BEV（只分割）
        print("  生成 road_no_conf_bev...")
        road_no_conf_bev = self.generate_bev(merged_road_no_conf_points, merged_road_no_conf_colors)
        
        # 8.6 建筑点云BEV（只分割）
        print("  生成 building_no_conf_bev...")
        building_no_conf_bev = self.generate_bev(merged_building_no_conf_points, merged_building_no_conf_colors)
        
        # 8.7 置信度过滤点云BEV（带语义颜色）
        print("  生成 conf_filtered_bev...")
        conf_filtered_bev = self.generate_bev(merged_conf_filtered_points, conf_filtered_colors)
        
        # 8.7.1 置信度过滤点云BEV（保留原始颜色，不染色）
        print("  生成 conf_filtered_raw_bev...")
        conf_filtered_raw_bev = self.generate_bev(merged_conf_filtered_raw_points, merged_conf_filtered_raw_colors)
        
        # 8.8 融合BEV：道路叠加到建筑上
        print("  生成 fused_bev...")
        fused_bev = building_bev.copy()
        road_mask = np.any(road_bev > 0, axis=2)
        fused_bev[road_mask] = road_bev[road_mask]
        
        print("  ✅ BEV生成完成")
        
        # 9. 保存结果
        print("\n💾 保存结果...")
        
        # 保存所有BEV
        cv2.imwrite(os.path.join(output_dir, "full_bev.png"), full_bev)
        cv2.imwrite(os.path.join(output_dir, "sky_removed_bev.png"), sky_removed_bev)
        cv2.imwrite(os.path.join(output_dir, "sky_removed_colored_bev.png"), sky_removed_colored_bev)
        cv2.imwrite(os.path.join(output_dir, "road_bev.png"), road_bev)
        cv2.imwrite(os.path.join(output_dir, "building_bev.png"), building_bev)
        cv2.imwrite(os.path.join(output_dir, "road_no_conf_bev.png"), road_no_conf_bev)
        cv2.imwrite(os.path.join(output_dir, "building_no_conf_bev.png"), building_no_conf_bev)
        cv2.imwrite(os.path.join(output_dir, "conf_filtered_bev.png"), conf_filtered_bev)
        cv2.imwrite(os.path.join(output_dir, "conf_filtered_raw_bev.png"), conf_filtered_raw_bev)
        cv2.imwrite(os.path.join(output_dir, "fused_bev.png"), fused_bev)
        
        # 保存6种点云
        n_full = self.save_ply(
            os.path.join(output_dir, "full_pointcloud.ply"),
            merged_full_points,
            merged_full_colors
        )
        n_sky_removed = self.save_ply(
            os.path.join(output_dir, "sky_removed_pointcloud.ply"),
            merged_sky_removed_points,
            merged_sky_removed_colors
        )
        # 保存去除天空点云（带语义染色）
        n_sky_removed_colored = self.save_ply(
            os.path.join(output_dir, "sky_removed_colored_pointcloud.ply"),
            merged_sky_removed_colored_points,
            merged_sky_removed_colored_colors
        )
        n_road = self.save_ply(
            os.path.join(output_dir, "road_pointcloud.ply"),
            merged_road_points,
            merged_road_colors
        )
        n_building = self.save_ply(
            os.path.join(output_dir, "building_pointcloud.ply"),
            merged_building_points,
            merged_building_colors
        )
        n_road_no_conf = self.save_ply(
            os.path.join(output_dir, "road_no_conf_pointcloud.ply"),
            merged_road_no_conf_points,
            merged_road_no_conf_colors
        )
        n_building_no_conf = self.save_ply(
            os.path.join(output_dir, "building_no_conf_pointcloud.ply"),
            merged_building_no_conf_points,
            merged_building_no_conf_colors
        )
        
        # 保存置信度过滤点云（带语义标签）
        n_conf_filtered = self.save_ply(
            os.path.join(output_dir, "conf_filtered_pointcloud.ply"),
            merged_conf_filtered_points,
            conf_filtered_colors
        )
        
        # 保存置信度过滤点云（只过滤置信度，保留原始颜色）
        n_conf_filtered_raw = self.save_ply(
            os.path.join(output_dir, "conf_filtered_raw_pointcloud.ply"),
            merged_conf_filtered_raw_points,
            merged_conf_filtered_raw_colors
        )
        
        print(f"  ✅ full_pointcloud.ply: {n_full:,d} 点")
        print(f"  ✅ sky_removed_pointcloud.ply: {n_sky_removed:,d} 点")
        print(f"  ✅ sky_removed_colored_pointcloud.ply: {n_sky_removed_colored:,d} 点（去除天空，带语义染色）")
        print(f"  ✅ conf_filtered_pointcloud.ply: {n_conf_filtered:,d} 点")
        print(f"    - 道路（红色）: {np.sum(merged_conf_filtered_labels == 1):,d} 点")
        print(f"    - 建筑（绿色）: {np.sum(merged_conf_filtered_labels == 2):,d} 点")
        print(f"    - 其他（原色）: {np.sum(merged_conf_filtered_labels == 0):,d} 点")
        print(f"  ✅ conf_filtered_raw_pointcloud.ply: {n_conf_filtered_raw:,d} 点（只过滤置信度，保留原始颜色）")
        print(f"  ✅ road_pointcloud.ply: {n_road:,d} 点")
        print(f"  ✅ building_pointcloud.ply: {n_building:,d} 点")
        print(f"  ✅ road_no_conf_pointcloud.ply: {n_road_no_conf:,d} 点")
        print(f"  ✅ building_no_conf_pointcloud.ply: {n_building_no_conf:,d} 点")
        
        print(f"\n🎉 完成! 结果保存在: {output_dir}")
        
        return {
            "road_bev": road_bev,
            "building_bev": building_bev,
            "fused_bev": fused_bev,
            "full_points": merged_full_points,
            "sky_removed_points": merged_sky_removed_points,
            "road_points": merged_road_points,
            "building_points": merged_building_points,
            "road_no_conf_points": merged_road_no_conf_points,
            "building_no_conf_points": merged_building_no_conf_points
        }


def main():
    INPUT_DIR = "/home/suanzigao/PycharmProjects/street_view_global_localization/test_image"
    OUTPUT_DIR = os.path.join(INPUT_DIR, "semantic_pointcloud_output")
    
    # 加载图像路径
    image_paths = sorted([
        os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR)
        if f.endswith("_streetview.jpg")
    ], key=lambda p: int(os.path.basename(p).split("step_")[1].split("_")[0]))
    
    print(f"📷 找到 {len(image_paths)} 张图像")
    
    # 运行流程
    pipeline = SemanticPointcloudPipeline()
    pipeline.run(image_paths, OUTPUT_DIR)


if __name__ == "__main__":
    main()