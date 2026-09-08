"""
OSM路网匹配模块
将BEV道路图像与OpenStreetMap路网进行匹配
"""

import os
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt


class OSMRoadMatcher:
    def __init__(self):
        self.osnx_available = False
        try:
            import osmnx as ox
            self.ox = ox
            self.osnx_available = True
            print("✅ osmnx 已安装")
        except ImportError:
            print("⚠️ osmnx 未安装，将使用简化方案")

    def get_osm_road_network(self, lat: float, lng: float, radius: float = 100) -> Dict:
        """
        获取指定坐标周围的路网数据

        Args:
            lat: 中心纬度
            lng: 中心经度
            radius: 搜索半径（米）

        Returns:
            包含道路线段和节点信息的字典
        """
        if not self.osnx_available:
            print("⚠️ 使用模拟路网数据（请安装osmnx）")
            return self._get_mock_road_network(lat, lng, radius)

        try:
            # 获取路网图 (osmnx 2.x 使用 dist 参数)
            G = self.ox.graph_from_point((lat, lng), dist=radius, network_type='drive')

            # 提取道路节点和边
            nodes, edges = [], []
            for node_id, data in G.nodes(data=True):
                nodes.append({
                    'id': node_id,
                    'lat': data.get('y', lat),
                    'lng': data.get('x', lng)
                })

            for u, v, data in G.edges(data=True):
                if 'geometry' in data:
                    # 处理带几何形状的道路
                    coords = list(data['geometry'].coords)
                    edges.append({
                        'u': u, 'v': v,
                        'coords': [(c[1], c[0]) for c in coords],  # (lat, lng)
                        'type': data.get('highway', 'unknown')
                    })
                else:
                    # 简单直线路径
                    u_node = G.nodes[u]
                    v_node = G.nodes[v]
                    edges.append({
                        'u': u, 'v': v,
                        'coords': [(u_node['y'], u_node['x']), (v_node['y'], v_node['x'])],
                        'type': data.get('highway', 'unknown')
                    })

            return {'nodes': nodes, 'edges': edges, 'graph': G}

        except Exception as e:
            print(f"❌ OSM获取失败: {e}")
            return self._get_mock_road_network(lat, lng, radius)

    def _get_mock_road_network(self, lat: float, lng: float, radius: float) -> Dict:
        """生成模拟路网数据用于测试"""
        import random

        nodes, edges = [], []
        node_id = 0

        # 生成网格状路网
        for i in range(-2, 3):
            for j in range(-2, 3):
                nodes.append({
                    'id': node_id,
                    'lat': lat + i * 0.001,
                    'lng': lng + j * 0.001
                })
                node_id += 1

        # 水平道路
        for i in range(-2, 3):
            for j in range(4):
                edges.append({
                    'u': i * 5 + j,
                    'v': i * 5 + j + 1,
                    'coords': [(nodes[i * 5 + j]['lat'], nodes[i * 5 + j]['lng']),
                              (nodes[i * 5 + j + 1]['lat'], nodes[i * 5 + j + 1]['lng'])],
                    'type': 'residential'
                })

        # 垂直道路
        for j in range(-2, 3):
            for i in range(4):
                edges.append({
                    'u': i * 5 + j,
                    'v': (i + 1) * 5 + j,
                    'coords': [(nodes[i * 5 + j]['lat'], nodes[i * 5 + j]['lng']),
                              (nodes[(i + 1) * 5 + j]['lat'], nodes[(i + 1) * 5 + j]['lng'])],
                    'type': 'residential'
                })

        return {'nodes': nodes, 'edges': edges, 'graph': None}

    def extract_road_centerlines(self, bev_image: np.ndarray) -> List[np.ndarray]:
        """
        从BEV图像提取道路中心线

        Args:
            bev_image: BEV图像 (H, W, 3)

        Returns:
            道路中心线列表，每条线为 (N, 1, 2) 的坐标数组
        """
        # 1. 转换为灰度图
        if len(bev_image.shape) == 3:
            gray = cv2.cvtColor(bev_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = bev_image

        # 2. 自适应阈值分割
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        threshold = max(10, min(50, mean_val - std_val * 0.5))
        _, road_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

        # 3. 形态学处理（先膨胀后腐蚀填充空洞，再腐蚀后膨胀去除噪声）
        kernel_close = np.ones((10, 10), np.uint8)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel_close)
        
        kernel_open = np.ones((5, 5), np.uint8)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel_open)

        # 4. 提取道路骨架（使用改进的细化算法）
        skeleton = self._extract_skeleton(road_mask)

        # 如果骨架提取失败，直接从道路掩码提取线段
        if cv2.countNonZero(skeleton) == 0:
            lines = self._extract_lines_from_skeleton(road_mask)
        else:
            # 5. 从骨架提取线段
            lines = self._extract_lines_from_skeleton(skeleton)

        return lines

    def _extract_skeleton(self, binary_mask: np.ndarray) -> np.ndarray:
        """提取二值图像的骨架（使用OpenCV形态学细化）"""
        skeleton = np.zeros(binary_mask.shape, np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        
        temp = binary_mask.copy()
        while cv2.countNonZero(temp) > 0:
            eroded = cv2.erode(temp, element)
            opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, element)
            temp2 = cv2.subtract(eroded, opened)
            skeleton = cv2.bitwise_or(skeleton, temp2)
            temp = eroded.copy()
        
        return skeleton

    def _extract_lines_from_skeleton(self, skeleton: np.ndarray) -> List[np.ndarray]:
        """从骨架提取线段"""
        lines = []

        # 使用Hough变换提取直线
        detected_lines = cv2.HoughLinesP(
            skeleton,
            rho=1,
            theta=np.pi / 180,
            threshold=20,
            minLineLength=30,
            maxLineGap=10
        )

        if detected_lines is not None:
            for line in detected_lines:
                x1, y1, x2, y2 = line[0]
                lines.append(np.array([[[x1, y1]], [[x2, y2]]], dtype=np.float32))

        return lines

    def _match_with_range(
        self,
        bev_image: np.ndarray,
        bev_gps: Tuple[float, float],
        osm_data: Dict,
        bev_range_meters: float
    ) -> Dict:
        """使用指定的BEV覆盖范围进行匹配"""
        h, w = bev_image.shape[:2]
        bev_center_lat, bev_center_lng = bev_gps

        def pixel_to_gps(x: int, y: int) -> Tuple[float, float]:
            rel_y = (y - h / 2) / h
            rel_x = (x - w / 2) / w
            lat = bev_center_lat + rel_y * bev_range_meters / 111000
            lng = bev_center_lng + rel_x * bev_range_meters / (111000 * np.cos(np.radians(bev_center_lat)))
            return lat, lng

        bev_lines = self.extract_road_centerlines(bev_image)
        bev_lines_gps = []
        for line in bev_lines:
            gps_coords = []
            for point in line:
                x, y = point[0]
                lat, lng = pixel_to_gps(int(x), int(y))
                gps_coords.append((lat, lng))
            bev_lines_gps.append(gps_coords)

        match_score = self._calculate_match_score(bev_lines_gps, osm_data['edges'])
        best_offset, top3 = self._find_best_offset(bev_lines_gps, osm_data['edges'])

        return {
            'match_score': match_score,
            'best_offset': best_offset,
            'top3_offsets': top3,
            'bev_range_meters': bev_range_meters,
            'bev_lines_gps': bev_lines_gps,
            'estimated_gps': (
                bev_center_lat + best_offset[0] / 111000,
                bev_center_lng + best_offset[1] / (111000 * np.cos(np.radians(bev_center_lat)))
            )
        }

    def match_bev_to_osm(
        self,
        bev_image: np.ndarray,
        bev_gps: Tuple[float, float],
        osm_radius: float = 100,
        save_intermediate: bool = True,
        output_dir: str = "matching_results",
        try_multiple_ranges: bool = True
    ) -> Dict:
        """
        将BEV图像匹配到OSM路网

        Args:
            bev_image: BEV图像
            bev_gps: BEV对应的GPS坐标 (lat, lng)
            osm_radius: OSM搜索半径（米）
            save_intermediate: 是否保存中间结果
            output_dir: 中间结果保存目录
            try_multiple_ranges: 是否尝试多个BEV覆盖范围

        Returns:
            匹配结果字典
        """
        os.makedirs(output_dir, exist_ok=True)

        # 1. 获取OSM路网
        osm_data = self.get_osm_road_network(bev_gps[0], bev_gps[1], osm_radius)
        print(f"📡 获取OSM路网: {len(osm_data['edges'])}条边")

        # 保存中间结果1: BEV原始图像
        if save_intermediate:
            cv2.imwrite(os.path.join(output_dir, "1_bev_original.png"), bev_image)
            print(f"✅ 保存中间结果: 1_bev_original.png")

            gray = cv2.cvtColor(bev_image, cv2.COLOR_BGR2GRAY) if len(bev_image.shape) == 3 else bev_image
            cv2.imwrite(os.path.join(output_dir, "2_gray.png"), gray)

            _, road_mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
            kernel = np.ones((5, 5), np.uint8)
            road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)
            road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel)
            cv2.imwrite(os.path.join(output_dir, "3_road_mask.png"), road_mask)

            skeleton = self._extract_skeleton(road_mask)
            cv2.imwrite(os.path.join(output_dir, "4_skeleton.png"), skeleton * 255)

            # 提取道路线段并保存
            bev_lines = self.extract_road_centerlines(bev_image)
            bev_with_lines = bev_image.copy()
            for line in bev_lines:
                for i in range(len(line) - 1):
                    x1, y1 = int(line[i][0][0]), int(line[i][0][1])
                    x2, y2 = int(line[i+1][0][0]), int(line[i+1][0][1])
                    cv2.line(bev_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imwrite(os.path.join(output_dir, "4_road_lines.png"), bev_with_lines)
            print(f"✅ 保存中间结果: 4_road_lines.png")

        # 2. 使用固定50米范围进行匹配
        print("\n🔍 使用BEV覆盖范围: 50米")
        best_result = self._match_with_range(bev_image, bev_gps, osm_data, 50)

        # 3. 构建最终结果
        result = {
            'match_score': best_result['match_score'],
            'osm_data': osm_data,
            'bev_lines_gps': best_result['bev_lines_gps'],
            'best_offset': best_result['best_offset'],
            'bev_range_meters': best_result['bev_range_meters'],
            'estimated_gps': best_result['estimated_gps'],
            'original_gps': bev_gps,
            'top3_offsets': best_result.get('top3_offsets', [])
        }

        # 保存中间结果
        if save_intermediate:
            with open(os.path.join(output_dir, "8_final_result.txt"), 'w') as f:
                f.write("=" * 50 + "\n")
                f.write("OSM路网匹配结果汇总\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"输入GPS坐标: ({bev_gps[0]}, {bev_gps[1]})\n")
                f.write(f"OSM搜索半径: {osm_radius}米\n")
                f.write(f"OSM边数: {len(osm_data['edges'])}\n")
                f.write(f"最佳BEV覆盖范围: {best_result['bev_range_meters']}米\n")
                f.write(f"匹配分数: {best_result['match_score']:.4f}\n")
                f.write(f"偏移量: {best_result['best_offset']}\n")
                f.write(f"估计GPS坐标: {best_result['estimated_gps']}\n")
                f.write("\n🎯 Top3相似度偏移:\n")
                for i, (dx, dy, score) in enumerate(best_result.get('top3_offsets', []), 1):
                    est_lat = bev_gps[0] + dy / 111000
                    est_lng = bev_gps[1] + dx / (111000 * np.cos(np.radians(bev_gps[0])))
                    f.write(f"   Top{i}: 偏移=({dx:.1f}, {dy:.1f})米, 分数={score:.4f}, GPS=({est_lat:.8f}, {est_lng:.8f})\n")
            print(f"✅ 保存中间结果: 8_final_result.txt")

            # 在BEV图像中标出红点位置（对应top3偏移位置）
            self._mark_bev_with_red_points(bev_image, best_result, output_dir, bev_gps)

        return result

    def _mark_bev_with_red_points(
        self,
        bev_image: np.ndarray,
        match_result: Dict,
        output_dir: str,
        original_gps: Tuple[float, float]
    ):
        """在BEV图像中标出红点位置（对应top3偏移）"""
        h, w = bev_image.shape[:2]
        bev_range = match_result['bev_range_meters']
        bev_center_lat, bev_center_lng = original_gps

        # 将GPS坐标转换为BEV像素坐标
        def gps_to_pixel(lat, lng):
            rel_y = (lat - bev_center_lat) * 111000 / bev_range
            rel_x = (lng - bev_center_lng) * 111000 * np.cos(np.radians(bev_center_lat)) / bev_range
            x = int(w / 2 + rel_x * w)
            y = int(h / 2 + rel_y * h)
            return x, y

        # 创建BEV图像副本
        bev_marked = bev_image.copy()

        # 标出原始位置（绿色）
        orig_x, orig_y = gps_to_pixel(bev_center_lat, bev_center_lng)
        cv2.circle(bev_marked, (orig_x, orig_y), 15, (0, 255, 0), -1)
        cv2.circle(bev_marked, (orig_x, orig_y), 20, (0, 255, 0), 2)
        cv2.putText(bev_marked, "Original", (orig_x + 25, orig_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 标出top3位置（红色）
        colors = [(0, 0, 255), (0, 128, 255), (255, 0, 128)]
        labels = ["Top1", "Top2", "Top3"]
        for i, (dx, dy, score) in enumerate(match_result.get('top3_offsets', [])):
            est_lat = bev_center_lat + dy / 111000
            est_lng = bev_center_lng + dx / (111000 * np.cos(np.radians(bev_center_lat)))
            x, y = gps_to_pixel(est_lat, est_lng)
            
            cv2.circle(bev_marked, (x, y), 12, colors[i], -1)
            cv2.circle(bev_marked, (x, y), 17, colors[i], 2)
            cv2.putText(bev_marked, f"{labels[i]} ({score:.3f})", (x + 20, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2)

        cv2.imwrite(os.path.join(output_dir, "5_bev_with_points.png"), bev_marked)
        print(f"✅ 保存中间结果: 5_bev_with_points.png")

    def _calculate_match_score(
        self,
        bev_lines: List[List[Tuple[float, float]]],
        osm_edges: List[Dict]
    ) -> float:
        """计算BEV道路与OSM道路的匹配分数（方向+长度+位置）"""
        if not bev_lines or not osm_edges:
            return 0.0

        def haversine(lat1, lng1, lat2, lng2):
            R = 6371000
            dlat = np.radians(lat2 - lat1)
            dlng = np.radians(lng2 - lng1)
            a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlng/2)**2
            return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

        total_score = 0.0
        total_weight = 0.0

        for bev_line in bev_lines:
            if len(bev_line) < 2:
                continue

            bev_start_lat, bev_start_lng = bev_line[0]
            bev_end_lat, bev_end_lng = bev_line[-1]
            bev_dir = np.arctan2(bev_end_lat - bev_start_lat, bev_end_lng - bev_start_lng)
            bev_length = haversine(bev_start_lat, bev_start_lng, bev_end_lat, bev_end_lng)
            bev_center_lat = (bev_start_lat + bev_end_lat) / 2
            bev_center_lng = (bev_start_lng + bev_end_lng) / 2

            best_edge_score = 0.0
            for edge in osm_edges:
                if len(edge['coords']) < 2:
                    continue

                edge_start_lat, edge_start_lng = edge['coords'][0]
                edge_end_lat, edge_end_lng = edge['coords'][-1]
                osm_dir = np.arctan2(edge_end_lat - edge_start_lat, edge_end_lng - edge_start_lng)
                osm_length = haversine(edge_start_lat, edge_start_lng, edge_end_lat, edge_end_lng)
                edge_center_lat = (edge_start_lat + edge_end_lat) / 2
                edge_center_lng = (edge_start_lng + edge_end_lng) / 2

                angle_diff = abs(bev_dir - osm_dir)
                if angle_diff > np.pi:
                    angle_diff = 2 * np.pi - angle_diff

                dir_score = max(0, 1 - angle_diff / (np.pi / 3))

                if osm_length > 0:
                    length_ratio = min(bev_length, osm_length) / max(bev_length, osm_length)
                    length_score = 0.5 + 0.5 * length_ratio
                else:
                    length_score = 0.5

                dist = haversine(bev_center_lat, bev_center_lng, edge_center_lat, edge_center_lng)
                dist_score = max(0, 1 - dist / 200) if dist < 200 else 0

                edge_score = dir_score * 0.3 + length_score * 0.2 + dist_score * 0.5
                best_edge_score = max(best_edge_score, edge_score)

            total_score += best_edge_score
            total_weight += 1

        return total_score / max(total_weight, 1)

    def _find_best_offset(
        self,
        bev_lines: List[List[Tuple[float, float]]],
        osm_edges: List[Dict]
    ) -> Tuple[Tuple[float, float], List[Tuple[float, float, float]]]:
        """寻找最佳偏移量来对齐BEV和OSM（仅粗搜索筛选Top3），返回最佳偏移和top3偏移"""
        if not bev_lines or not osm_edges:
            return (0.0, 0.0), [(0.0, 0.0, 0.0)]

        def evaluate_offset(dx, dy):
            shifted_bev_lines = []
            for line in bev_lines:
                shifted_line = [(lat + dy / 111000, lng + dx / (111000 * np.cos(np.radians(lat)))) for lat, lng in line]
                shifted_bev_lines.append(shifted_line)
            return self._calculate_match_score(shifted_bev_lines, osm_edges)

        # 粗搜索（大范围，大步长）
        print("🔍 粗搜索（±300米，步长30米）")
        search_range, step = 300, 30
        all_scores = []
        for dx in range(-search_range, search_range + 1, step):
            for dy in range(-search_range, search_range + 1, step):
                score = evaluate_offset(dx, dy)
                all_scores.append((score, dx, dy))

        all_scores.sort(key=lambda x: -x[0])
        top3 = all_scores[:3]
        
        best_score, best_dx, best_dy = top3[0]
        best_offset = (best_dy, best_dx)
        
        print(f"🎯 粗搜索Top3:")
        for i, (score, dx, dy) in enumerate(top3, 1):
            print(f"   Top{i}: 偏移=({dx:.0f}, {dy:.0f})米, 分数={score:.4f}")

        top3_result = [(x[1], x[2], x[0]) for x in top3]
        print(f"🎯 最终Top3偏移: {[(f'({t[0]},{t[1]})米', f'{t[2]:.4f}') for t in top3_result]}")

        return best_offset, top3_result

    def visualize_match(
        self,
        bev_image: np.ndarray,
        osm_data: Dict,
        match_result: Dict,
        output_path: str
    ):
        """可视化匹配结果（完整路网 + BEV缩略图叠加）"""
        estimated_lat, estimated_lng = match_result['estimated_gps']
        bev_range_meters = match_result['bev_range_meters']
        
        all_lats = []
        all_lngs = []
        for edge in osm_data['edges']:
            coords = edge['coords']
            for lat, lng in coords:
                all_lats.append(lat)
                all_lngs.append(lng)
        
        if all_lats and all_lngs:
            lat_min, lat_max = min(all_lats), max(all_lats)
            lng_min, lng_max = min(all_lngs), max(all_lngs)
        else:
            margin = 500 / 111000
            lat_min = estimated_lat - margin
            lat_max = estimated_lat + margin
            lng_min = estimated_lng - margin / np.cos(np.radians(estimated_lat))
            lng_max = estimated_lng + margin / np.cos(np.radians(estimated_lat))
        
        padding = 0.1
        lat_range = lat_max - lat_min
        lng_range = lng_max - lng_min
        lat_min -= lat_range * padding
        lat_max += lat_range * padding
        lng_min -= lng_range * padding
        lng_max += lng_range * padding
        
        fig, ax = plt.subplots(figsize=(12, 12))
        
        for edge in osm_data['edges']:
            coords = edge['coords']
            lats = [c[0] for c in coords]
            lngs = [c[1] for c in coords]
            ax.plot(lngs, lats, 'b-', linewidth=2, alpha=0.8)
        
        ax.plot(estimated_lng, estimated_lat, 'ro', markersize=12, markeredgecolor='white', markeredgewidth=2, label='Top1')
        original_lat, original_lng = match_result.get('original_gps', (estimated_lat, estimated_lng))
        ax.plot(original_lng, original_lat, 'go', markersize=10, markeredgecolor='white', markeredgewidth=2, label='Original')
        
        # 显示Top2和Top3位置
        colors = ['#FF0000', '#0080FF', '#FF0080']
        labels = ['Top1', 'Top2', 'Top3']
        for i, (dx, dy, score) in enumerate(match_result.get('top3_offsets', [])):
            est_lat = original_lat + dy / 111000
            est_lng = original_lng + dx / (111000 * np.cos(np.radians(original_lat)))
            if i == 0:
                continue
            ax.plot(est_lng, est_lat, 'o', color=colors[i], markersize=10, 
                    markeredgecolor='white', markeredgewidth=2, label=f'{labels[i]} ({score:.3f})')
        
        bev_h, bev_w = bev_image.shape[:2]
        bev_rgb = cv2.cvtColor(bev_image, cv2.COLOR_BGR2RGB)
        bev_rgba = cv2.cvtColor(bev_rgb, cv2.COLOR_RGB2RGBA)
        black_mask = (bev_rgb[:, :, 0] < 20) & (bev_rgb[:, :, 1] < 20) & (bev_rgb[:, :, 2] < 20)
        bev_rgba[black_mask] = [0, 0, 0, 0]
        
        thumbnail_size = (200, 200)
        if bev_h > bev_w:
            scale = thumbnail_size[1] / bev_h
        else:
            scale = thumbnail_size[0] / bev_w
        bev_thumb = cv2.resize(bev_rgba, None, fx=scale, fy=scale)
        
        bev_thumb_lng = lng_max - (lng_max - lng_min) * 0.05 - (lng_max - lng_min) * 0.25
        bev_thumb_lat = lat_max - (lat_max - lat_min) * 0.05
        bev_thumb_width = (lng_max - lng_min) * 0.25
        bev_thumb_height = (lat_max - lat_min) * 0.25
        
        ax.imshow(
            bev_thumb,
            extent=[bev_thumb_lng, bev_thumb_lng + bev_thumb_width, bev_thumb_lat - bev_thumb_height, bev_thumb_lat],
            alpha=0.8,
            origin='upper'
        )
        
        ax.text(bev_thumb_lng, bev_thumb_lat - bev_thumb_height - (lat_max - lat_min) * 0.02, 
                f'BEV ({bev_range_meters}m)', fontsize=10, color='red', fontweight='bold')
        
        ax.set_xlim(lng_min, lng_max)
        ax.set_ylim(lat_min, lat_max)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'OSM Road Network + BEV Overlay (Score: {match_result["match_score"]:.4f})')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ 匹配可视化已保存: {output_path}")


def demo():
    """演示OSM路网匹配"""
    import requests

    # 测试坐标（柏林）
    test_lat, test_lng = 52.5200, 13.4050

    matcher = OSMRoadMatcher()

    # 创建测试BEV图像（模拟道路）
    bev_test = np.zeros((500, 500, 3), dtype=np.uint8)
    cv2.line(bev_test, (100, 250), (400, 250), (100, 100, 100), 10)  # 水平道路
    cv2.line(bev_test, (250, 100), (250, 400), (100, 100, 100), 10)  # 垂直道路

    # 执行匹配
    result = matcher.match_bev_to_osm(bev_test, (test_lat, test_lng), radius=100)

    print(f"\n📊 匹配结果:")
    print(f"   匹配分数: {result['match_score']:.2f}")
    print(f"   OSM边数: {len(result['osm_data']['edges'])}")
    print(f"   BEV线段数: {len(result['bev_lines_gps'])}")
    print(f"   估计GPS: {result['estimated_gps']}")

    # 可视化
    output_dir = "/home/suanzigao/PycharmProjects/street_view_global_localization/matching_results"
    os.makedirs(output_dir, exist_ok=True)
    matcher.visualize_match(
        bev_test,
        result['osm_data'],
        result,
        os.path.join(output_dir, "osm_match_demo.png")
    )


if __name__ == "__main__":
    demo()
