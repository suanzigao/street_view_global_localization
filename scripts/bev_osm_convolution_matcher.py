"""
BEV图像与OSM路网卷积匹配
"""

import os
import csv
import cv2
import numpy as np
import json
import math

try:
    import osmnx as ox
    OSMNX_AVAILABLE = True
except ImportError:
    OSMNX_AVAILABLE = False


class BEVOSMConvolutionMatcher:
    def __init__(self):
        self.ox = ox if OSMNX_AVAILABLE else None

    def read_points_from_csv(self, csv_path):
        points = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                points.append({
                    'name': row['name'],
                    'lat': float(row['lat']),
                    'lng': float(row['lng'])
                })
        return points

    def download_osm_road_network(self, lat, lng, radius=5000, output_dir=None):
        if not self.ox:
            print("❌ osmnx未安装")
            return None

        print(f"📡 下载OSM路网: ({lat:.6f}, {lng:.6f})")
        try:
            G = self.ox.graph_from_point((lat, lng), dist=radius, network_type='drive')
        except Exception as e:
            print(f"❌ OSM下载失败: {e}")
            return None

        print(f"✅ 路网下载完成: {len(G.nodes)}节点, {len(G.edges)}边")

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        return G

    def render_osm_to_image(self, G, center_lat, center_lng, radius=5000, resolution=2):
        meters_per_deg_lat = 111000.0
        meters_per_deg_lng = 111000.0 * np.cos(np.radians(center_lat))

        image_width = int(2 * radius * resolution)
        image_height = int(2 * radius * resolution)

        road_image = np.zeros((image_height, image_width), dtype=np.uint8)

        def lat_lng_to_pixel(lat, lng):
            dy_meters = (lat - center_lat) * meters_per_deg_lat
            dx_meters = (lng - center_lng) * meters_per_deg_lng

            x = int((dx_meters + radius) * resolution)
            y = int((radius - dy_meters) * resolution)

            return x, y

        for u, v, data in G.edges(data=True):
            highway_type = data.get('highway', 'residential')
            if isinstance(highway_type, list):
                highway_type = highway_type[0]

            width_map = {
                'motorway': 24,
                'motorway_link': 12,
                'trunk': 18,
                'trunk_link': 9,
                'primary': 12,
                'primary_link': 6,
                'secondary': 8,
                'secondary_link': 4,
                'tertiary': 6,
                'tertiary_link': 3,
                'residential': 5,
                'service': 3,
                'unclassified': 4,
            }
            width_meters = width_map.get(highway_type, 5)
            pixel_width = max(2, int(width_meters * resolution))

            if 'geometry' in data:
                coords = list(data['geometry'].coords)
                points = [lat_lng_to_pixel(c[1], c[0]) for c in coords]
            else:
                u_node = G.nodes[u]
                v_node = G.nodes[v]
                points = [
                    lat_lng_to_pixel(u_node['y'], u_node['x']),
                    lat_lng_to_pixel(v_node['y'], v_node['x'])
                ]

            def clip_line(x1, y1, x2, y2, x_min, y_min, x_max, y_max):
                dx = x2 - x1
                dy = y2 - y1
                
                p = [-dx, dx, -dy, dy]
                q = [x1 - x_min, x_max - x1, y1 - y_min, y_max - y1]
                
                t0, t1 = 0.0, 1.0
                
                for i in range(4):
                    if p[i] == 0:
                        if q[i] < 0:
                            return None, None, None, None
                    else:
                        t = q[i] / p[i]
                        if p[i] < 0:
                            t0 = max(t0, t)
                        else:
                            t1 = min(t1, t)
                        if t0 > t1:
                            return None, None, None, None
                
                x1_clip = x1 + t0 * dx
                y1_clip = y1 + t0 * dy
                x2_clip = x1 + t1 * dx
                y2_clip = y1 + t1 * dy
                
                return x1_clip, y1_clip, x2_clip, y2_clip

            for i in range(len(points) - 1):
                x1, y1 = points[i]
                x2, y2 = points[i + 1]

                cx1, cy1, cx2, cy2 = clip_line(x1, y1, x2, y2, 0, 0, image_width - 1, image_height - 1)
                
                if cx1 is not None:
                    cv2.line(road_image, (int(cx1), int(cy1)), (int(cx2), int(cy2)), 255, pixel_width)

        return road_image, image_width, image_height

    def load_bev_image(self, bev_path):
        if not os.path.exists(bev_path):
            print(f"❌ BEV图像不存在: {bev_path}")
            return None

        bev = cv2.imread(bev_path)
        if bev is None:
            print(f"❌ 无法读取BEV图像: {bev_path}")
            return None

        gray = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        return binary

    def compute_cross_correlation(self, osm_image, bev_image):
        osm_h, osm_w = osm_image.shape
        bev_h, bev_w = bev_image.shape

        if osm_h < bev_h or osm_w < bev_w:
            print("❌ OSM图像小于BEV图像")
            return None

        result = cv2.matchTemplate(osm_image, bev_image, cv2.TM_CCOEFF_NORMED)

        return result

    def find_best_match(self, result):
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        return max_loc, max_val

    def pixel_to_gps(self, pixel_x, pixel_y, center_lat, center_lng, radius=5000, resolution=2):
        meters_per_deg_lat = 111000.0
        meters_per_deg_lng = 111000.0 * np.cos(np.radians(center_lat))

        dx_meters = (pixel_x / resolution) - radius
        dy_meters = radius - (pixel_y / resolution)

        lat = center_lat + dy_meters / meters_per_deg_lat
        lng = center_lng + dx_meters / meters_per_deg_lng

        return lat, lng

    def match_point(self, point, batch_dir, osm_radius=5000, download_radius=None):
        name = point['name']
        center_lat = point['lat']
        center_lng = point['lng']
        
        if download_radius is None:
            download_radius = int(osm_radius * 1.5)

        print(f"\n{'='*70}")
        print(f"匹配点位: {name}")
        print(f"中心坐标: ({center_lat:.6f}, {center_lng:.6f})")
        print(f"{'='*70}")

        point_path = None
        for dir_name in os.listdir(batch_dir):
            if name in dir_name and os.path.isdir(os.path.join(batch_dir, dir_name)):
                point_path = os.path.join(batch_dir, dir_name)
                break
        
        if point_path is None:
            print(f"❌ 未找到点位目录: {name}")
            return None

        bev_path = os.path.join(point_path, 'projection_output', 'fused_bev_lowres_cropped.png')
        metadata_path = os.path.join(point_path, 'projection_output', 'bev_lowres_cropped_metadata.json')
        
        if not os.path.exists(bev_path) or not os.path.exists(metadata_path):
            bev_path = os.path.join(point_path, 'projection_output', 'fused_bev_lowres.png')
            metadata_path = os.path.join(point_path, 'projection_output', 'bev_lowres_metadata.json')
        
        if not os.path.exists(bev_path):
            print(f"❌ 未找到BEV图像")
            return None

        bev_binary = self.load_bev_image(bev_path)
        if bev_binary is None:
            print(f"❌ 无法加载BEV图像")
            return None

        print(f"📐 BEV图像尺寸: {bev_binary.shape[1]}×{bev_binary.shape[0]}")
        
        initial_px = bev_binary.shape[1] // 2
        initial_py = bev_binary.shape[0] // 2
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                bev_metadata = json.load(f)
                initial_px = bev_metadata.get('initial_pixel_x', initial_px)
                initial_py = bev_metadata.get('initial_pixel_y', initial_py)
            print(f"📍 BEV中初始点位置: ({initial_px}, {initial_py})")
        else:
            print(f"⚠️ 未找到BEV元数据，使用中心点作为初始点")

        print(f"📡 下载OSM路网（半径: {download_radius}米）...")
        G = self.download_osm_road_network(center_lat, center_lng, radius=download_radius)
        if G is None:
            return None

        print(f"🖼️ 渲染OSM路网（显示范围: {osm_radius}米）...")
        osm_image, osm_w, osm_h = self.render_osm_to_image(G, center_lat, center_lng, radius=osm_radius)
        print(f"📐 OSM图像尺寸: {osm_w}×{osm_h}")

        result = self.compute_cross_correlation(osm_image, bev_binary)
        if result is None:
            return None

        max_loc, max_val = self.find_best_match(result)
        print(f"🎯 BEV左上角匹配位置: ({max_loc[0]}, {max_loc[1]}), 相似度: {max_val:.4f}")

        correct_x = max_loc[0] + initial_px
        correct_y = max_loc[1] + initial_py
        print(f"📍 修正后初始点位置: ({correct_x}, {correct_y})")

        match_lat, match_lng = self.pixel_to_gps(correct_x, correct_y, center_lat, center_lng, radius=osm_radius)
        print(f"📍 匹配GPS坐标: ({match_lat:.8f}, {match_lng:.8f})")

        actual_lat, actual_lng = self._get_actual_position(point_path)
        if actual_lat is not None:
            print(f"📍 实际GPS坐标: ({actual_lat:.8f}, {actual_lng:.8f})")

            distance = self._calculate_distance(match_lat, match_lng, actual_lat, actual_lng)
            print(f"📏 匹配误差: {distance:.2f}米")

        output_dir = os.path.join(point_path, 'matching_results')
        os.makedirs(output_dir, exist_ok=True)

        cv2.imwrite(os.path.join(output_dir, 'osm_road_image.png'), osm_image)
        cv2.imwrite(os.path.join(output_dir, 'bev_template.png'), bev_binary)
        
        heatmap = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        bev_h, bev_w = bev_binary.shape
        heatmap_h, heatmap_w = heatmap_color.shape[:2]
        osm_h, osm_w = osm_image.shape
        
        shifted_heatmap = np.zeros((osm_h, osm_w, 3), dtype=np.uint8)
        shifted_heatmap[:heatmap_h, :heatmap_w] = heatmap_color
        
        shifted_heatmap_initial = np.zeros((osm_h, osm_w, 3), dtype=np.uint8)
        
        src_y_start = max(0, -initial_py)
        src_y_end = min(heatmap_h, osm_h - initial_py)
        src_x_start = max(0, -initial_px)
        src_x_end = min(heatmap_w, osm_w - initial_px)
        
        dst_y_start = max(0, initial_py)
        dst_y_end = dst_y_start + (src_y_end - src_y_start)
        dst_x_start = max(0, initial_px)
        dst_x_end = dst_x_start + (src_x_end - src_x_start)
        
        shifted_heatmap_initial[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
            heatmap_color[src_y_start:src_y_end, src_x_start:src_x_end]
        
        # BEV几何正中心点对应的热力图
        center_px = bev_w // 2
        center_py = bev_h // 2
        
        shifted_heatmap_center = np.zeros((osm_h, osm_w, 3), dtype=np.uint8)
        
        src_y_start_c = max(0, -center_py)
        src_y_end_c = min(heatmap_h, osm_h - center_py)
        src_x_start_c = max(0, -center_px)
        src_x_end_c = min(heatmap_w, osm_w - center_px)
        
        dst_y_start_c = max(0, center_py)
        dst_y_end_c = dst_y_start_c + (src_y_end_c - src_y_start_c)
        dst_x_start_c = max(0, center_px)
        dst_x_end_c = dst_x_start_c + (src_x_end_c - src_x_start_c)
        
        shifted_heatmap_center[dst_y_start_c:dst_y_end_c, dst_x_start_c:dst_x_end_c] = \
            heatmap_color[src_y_start_c:src_y_end_c, src_x_start_c:src_x_end_c]
        
        colorbar_w = 60
        colorbar_h = osm_h
        colorbar = np.zeros((colorbar_h, colorbar_w, 3), dtype=np.uint8)
        for i in range(colorbar_h):
            val = 255 - int(i / colorbar_h * 255)
            colorbar[i, :] = cv2.applyColorMap(np.array([[val]], dtype=np.uint8), cv2.COLORMAP_JET)[0, 0]
        
        heatmap_with_colorbar = np.hstack([shifted_heatmap, colorbar])
        
        osm_color = cv2.cvtColor(osm_image, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(osm_color, 0.4, shifted_heatmap, 0.6, 0)
        overlay_with_colorbar = np.hstack([overlay, colorbar])
        
        overlay_initial = cv2.addWeighted(osm_color, 0.4, shifted_heatmap_initial, 0.6, 0)
        overlay_initial_with_colorbar = np.hstack([overlay_initial, colorbar])
        
        overlay_center = cv2.addWeighted(osm_color, 0.4, shifted_heatmap_center, 0.6, 0)
        overlay_center_with_colorbar = np.hstack([overlay_center, colorbar])
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(overlay_with_colorbar, 'High', (osm_w + 10, 30), font, 0.8, (255, 255, 255), 2)
        cv2.putText(overlay_with_colorbar, 'Low', (osm_w + 10, overlay_with_colorbar.shape[0] - 10), font, 0.8, (255, 255, 255), 2)
        
        cv2.putText(overlay_initial_with_colorbar, 'High', (osm_w + 10, 30), font, 0.8, (255, 255, 255), 2)
        cv2.putText(overlay_initial_with_colorbar, 'Low', (osm_w + 10, overlay_initial_with_colorbar.shape[0] - 10), font, 0.8, (255, 255, 255), 2)
        
        # 在热力图上标注关键点
        marker_size = max(25, osm_w // 600)
        
        heatmap_marked = shifted_heatmap_initial.copy()
        heatmap_overlay_marked = overlay_initial.copy()
        
        # 黄色十字：原始卷积最高分位置（BEV左上角对应位置）
        raw_match_x = max_loc[0] + initial_px
        raw_match_y = max_loc[1] + initial_py
        cv2.drawMarker(heatmap_marked, (int(raw_match_x), int(raw_match_y)), (0, 255, 255), 
                       cv2.MARKER_CROSS, marker_size * 2, 3)
        cv2.drawMarker(heatmap_overlay_marked, (int(raw_match_x), int(raw_match_y)), (0, 255, 255), 
                       cv2.MARKER_CROSS, marker_size * 2, 3)
        
        # 绿色圆点：修正后BEV中心点/初始点位置
        cv2.circle(heatmap_marked, (int(correct_x), int(correct_y)), marker_size, (0, 255, 0), -1)
        cv2.circle(heatmap_marked, (int(correct_x), int(correct_y)), marker_size + 5, (0, 255, 0), 2)
        cv2.circle(heatmap_overlay_marked, (int(correct_x), int(correct_y)), marker_size, (0, 255, 0), -1)
        cv2.circle(heatmap_overlay_marked, (int(correct_x), int(correct_y)), marker_size + 5, (0, 255, 0), 2)
        
        # 红色圆点：真实位置
        if actual_lat is not None:
            actual_x, actual_y = self._gps_to_pixel(actual_lat, actual_lng, center_lat, center_lng, radius=osm_radius)
            if 0 <= actual_x < osm_w and 0 <= actual_y < osm_h:
                cv2.circle(heatmap_marked, (actual_x, actual_y), marker_size, (0, 0, 255), -1)
                cv2.circle(heatmap_marked, (actual_x, actual_y), marker_size + 5, (0, 0, 255), 2)
                cv2.circle(heatmap_overlay_marked, (actual_x, actual_y), marker_size, (0, 0, 255), -1)
                cv2.circle(heatmap_overlay_marked, (actual_x, actual_y), marker_size + 5, (0, 0, 255), 2)
                cv2.line(heatmap_overlay_marked, (actual_x, actual_y), (int(correct_x), int(correct_y)), (255, 255, 255), 2)
        
        # 添加图例文字
        legend_y = 30
        cv2.putText(heatmap_overlay_marked, 'YELLOW CROSS: raw best match (BEV top-left)', (10, legend_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(heatmap_overlay_marked, 'GREEN CIRCLE: BEV center / initial point match', (10, legend_y + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(heatmap_overlay_marked, 'RED CIRCLE: actual GPS position', (10, legend_y + 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 几何正中心点标记
        correct_center_x = max_loc[0] + center_px
        correct_center_y = max_loc[1] + center_py
        
        heatmap_center_marked = shifted_heatmap_center.copy()
        heatmap_overlay_center_marked = overlay_center.copy()
        
        # 黄色十字：原始卷积最高分位置（BEV左上角对应位置）
        raw_match_center_x = max_loc[0] + center_px
        raw_match_center_y = max_loc[1] + center_py
        cv2.drawMarker(heatmap_center_marked, (int(raw_match_center_x), int(raw_match_center_y)), (0, 255, 255),
                       cv2.MARKER_CROSS, marker_size * 2, 3)
        cv2.drawMarker(heatmap_overlay_center_marked, (int(raw_match_center_x), int(raw_match_center_y)), (0, 255, 255),
                       cv2.MARKER_CROSS, marker_size * 2, 3)
        
        # 绿色圆点：BEV几何正中心点对应的匹配位置
        cv2.circle(heatmap_center_marked, (int(correct_center_x), int(correct_center_y)), marker_size, (0, 255, 0), -1)
        cv2.circle(heatmap_center_marked, (int(correct_center_x), int(correct_center_y)), marker_size + 5, (0, 255, 0), 2)
        cv2.circle(heatmap_overlay_center_marked, (int(correct_center_x), int(correct_center_y)), marker_size, (0, 255, 0), -1)
        cv2.circle(heatmap_overlay_center_marked, (int(correct_center_x), int(correct_center_y)), marker_size + 5, (0, 255, 0), 2)
        
        # 红色圆点：真实位置
        if actual_lat is not None:
            actual_x, actual_y = self._gps_to_pixel(actual_lat, actual_lng, center_lat, center_lng, radius=osm_radius)
            if 0 <= actual_x < osm_w and 0 <= actual_y < osm_h:
                cv2.circle(heatmap_center_marked, (actual_x, actual_y), marker_size, (0, 0, 255), -1)
                cv2.circle(heatmap_center_marked, (actual_x, actual_y), marker_size + 5, (0, 0, 255), 2)
                cv2.circle(heatmap_overlay_center_marked, (actual_x, actual_y), marker_size, (0, 0, 255), -1)
                cv2.circle(heatmap_overlay_center_marked, (actual_x, actual_y), marker_size + 5, (0, 0, 255), 2)
                cv2.line(heatmap_overlay_center_marked, (actual_x, actual_y), (int(correct_center_x), int(correct_center_y)), (255, 255, 255), 2)
        
        # 添加图例文字
        legend_y = 30
        cv2.putText(heatmap_overlay_center_marked, 'YELLOW CROSS: raw best match (BEV top-left)', (10, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(heatmap_overlay_center_marked, 'GREEN CIRCLE: BEV geometric center match', (10, legend_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(heatmap_overlay_center_marked, 'RED CIRCLE: actual GPS position', (10, legend_y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        cv2.imwrite(os.path.join(output_dir, 'match_heatmap.png'), heatmap_with_colorbar)
        cv2.imwrite(os.path.join(output_dir, 'match_heatmap_overlay.png'), overlay_with_colorbar)
        cv2.imwrite(os.path.join(output_dir, 'match_heatmap_initial.png'), np.hstack([shifted_heatmap_initial, colorbar]))
        cv2.imwrite(os.path.join(output_dir, 'match_heatmap_initial_overlay.png'), overlay_initial_with_colorbar)
        cv2.imwrite(os.path.join(output_dir, 'match_heatmap_initial_marked.png'), np.hstack([heatmap_marked, colorbar]))
        cv2.imwrite(os.path.join(output_dir, 'match_heatmap_initial_overlay_marked.png'), np.hstack([heatmap_overlay_marked, colorbar]))
        cv2.imwrite(os.path.join(output_dir, 'match_heatmap_center.png'), np.hstack([shifted_heatmap_center, colorbar]))
        cv2.imwrite(os.path.join(output_dir, 'match_heatmap_center_overlay.png'), overlay_center_with_colorbar)
        cv2.imwrite(os.path.join(output_dir, 'match_heatmap_center_marked.png'), np.hstack([heatmap_center_marked, colorbar]))
        cv2.imwrite(os.path.join(output_dir, 'match_heatmap_center_overlay_marked.png'), np.hstack([heatmap_overlay_center_marked, colorbar]))
        print(f"📊 卷积热力图已保存（基于左上角、初始点和几何中心，含标记版本）")
        
        bev_overlay = cv2.imread(bev_path)
        if bev_overlay is not None:
            osm_center_x = osm_w // 2
            osm_center_y = osm_h // 2
            
            bev_overlay_h, bev_overlay_w = bev_overlay.shape[:2]
            
            overlay_x_start = osm_center_x - initial_px
            overlay_y_start = osm_center_y - initial_py
            
            overlay_x_end = overlay_x_start + bev_overlay_w
            overlay_y_end = overlay_y_start + bev_overlay_h
            
            bev_on_osm = osm_color.copy()
            
            src_x_start = max(0, -overlay_x_start)
            src_y_start = max(0, -overlay_y_start)
            src_x_end = min(bev_overlay_w, osm_w - overlay_x_start)
            src_y_end = min(bev_overlay_h, osm_h - overlay_y_start)
            
            dst_x_start = max(0, overlay_x_start)
            dst_y_start = max(0, overlay_y_start)
            dst_x_end = dst_x_start + (src_x_end - src_x_start)
            dst_y_end = dst_y_start + (src_y_end - src_y_start)
            
            if src_x_end > src_x_start and src_y_end > src_y_start:
                bev_patch = bev_overlay[src_y_start:src_y_end, src_x_start:src_x_end]
                bev_patch_bgr = cv2.cvtColor(bev_patch, cv2.COLOR_BGR2BGRA) if len(bev_patch.shape) == 3 else bev_patch
                
                osm_patch = bev_on_osm[dst_y_start:dst_y_end, dst_x_start:dst_x_end]
                
                overlay_alpha = 0.5
                blended = cv2.addWeighted(osm_patch, 1 - overlay_alpha, bev_patch, overlay_alpha, 0)
                
                bev_on_osm[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = blended
            
            cv2.circle(bev_on_osm, (osm_center_x, osm_center_y), 5, (0, 0, 255), -1)
            cv2.circle(bev_on_osm, (osm_center_x, osm_center_y), 8, (0, 0, 255), 2)
            cv2.putText(bev_on_osm, 'OSM CENTER', (osm_center_x + 15, osm_center_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            bev_on_osm_path = os.path.join(output_dir, 'bev_on_osm.png')
            cv2.imwrite(bev_on_osm_path, bev_on_osm)
            print(f"✅ BEV叠加OSM图像已保存: {bev_on_osm_path}")
            
            bev_on_osm_match = osm_color.copy()
            
            match_center_x = max_loc[0] + initial_px
            match_center_y = max_loc[1] + initial_py
            
            overlay_x_start_match = match_center_x - initial_px
            overlay_y_start_match = match_center_y - initial_py
            
            overlay_x_end_match = overlay_x_start_match + bev_overlay_w
            overlay_y_end_match = overlay_y_start_match + bev_overlay_h
            
            src_x_start_match = max(0, -overlay_x_start_match)
            src_y_start_match = max(0, -overlay_y_start_match)
            src_x_end_match = min(bev_overlay_w, osm_w - overlay_x_start_match)
            src_y_end_match = min(bev_overlay_h, osm_h - overlay_y_start_match)
            
            dst_x_start_match = max(0, overlay_x_start_match)
            dst_y_start_match = max(0, overlay_y_start_match)
            dst_x_end_match = dst_x_start_match + (src_x_end_match - src_x_start_match)
            dst_y_end_match = dst_y_start_match + (src_y_end_match - src_y_start_match)
            
            if src_x_end_match > src_x_start_match and src_y_end_match > src_y_start_match:
                bev_patch_match = bev_overlay[src_y_start_match:src_y_end_match, src_x_start_match:src_x_end_match]
                osm_patch_match = bev_on_osm_match[dst_y_start_match:dst_y_end_match, dst_x_start_match:dst_x_end_match]
                blended_match = cv2.addWeighted(osm_patch_match, 1 - overlay_alpha, bev_patch_match, overlay_alpha, 0)
                bev_on_osm_match[dst_y_start_match:dst_y_end_match, dst_x_start_match:dst_x_end_match] = blended_match
            
            cv2.circle(bev_on_osm_match, (match_center_x, match_center_y), 5, (0, 255, 0), -1)
            cv2.circle(bev_on_osm_match, (match_center_x, match_center_y), 8, (0, 255, 0), 2)
            cv2.putText(bev_on_osm_match, f'MATCH ({max_val:.3f})', (match_center_x + 15, match_center_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            bev_on_osm_match_path = os.path.join(output_dir, 'bev_on_osm_at_match.png')
            cv2.imwrite(bev_on_osm_match_path, bev_on_osm_match)
            print(f"✅ BEV叠加OSM图像（匹配位置对齐）已保存: {bev_on_osm_match_path}")
            
            result_flat = result.flatten()
        top_indices = np.argsort(result_flat)[::-1]
        top_locs = []
        
        min_distance_meters = 5
        min_distance_pixels = int(min_distance_meters * 2)

        for idx in top_indices:
            y = int(idx // result.shape[1])
            x = int(idx % result.shape[1])
            
            valid = True
            for loc in top_locs:
                dist = math.sqrt((x - loc['pixel_x'])**2 + (y - loc['pixel_y'])**2)
                if dist < min_distance_pixels:
                    valid = False
                    break
            
            if valid:
                correct_x = x + initial_px
                correct_y = y + initial_py
                top_lat, top_lng = self.pixel_to_gps(correct_x, correct_y, center_lat, center_lng, radius=osm_radius)
                top_locs.append({
                    'pixel_x': correct_x,
                    'pixel_y': correct_y,
                    'lat': top_lat,
                    'lng': top_lng,
                    'confidence': float(result_flat[idx])
                })
            
            if len(top_locs) >= 5:
                break

        osm_marked = cv2.cvtColor(osm_image, cv2.COLOR_GRAY2BGR)

        marker_size = max(30, osm_w // 500)
        small_marker_size = max(20, osm_w // 800)

        if actual_lat is not None:
            actual_x, actual_y = self._gps_to_pixel(actual_lat, actual_lng, center_lat, center_lng, radius=osm_radius)
            if 0 <= actual_x < osm_w and 0 <= actual_y < osm_h:
                cv2.circle(osm_marked, (actual_x, actual_y), marker_size, (0, 0, 255), -1)
                cv2.circle(osm_marked, (actual_x, actual_y), marker_size + 5, (0, 0, 255), 2)
                cv2.putText(osm_marked, 'ACTUAL', (actual_x + marker_size + 10, actual_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        for i, loc in enumerate(top_locs):
            px, py = loc['pixel_x'], loc['pixel_y']
            conf = loc['confidence']
            
            if px < 0 or px >= osm_w or py < 0 or py >= osm_h:
                print(f"  ⚠️ TOP{i+1}匹配位置({px},{py})超出图像边界")
                continue
            
            if i == 0:
                cv2.circle(osm_marked, (px, py), marker_size, (0, 255, 0), -1)
                cv2.circle(osm_marked, (px, py), marker_size + 5, (0, 255, 0), 2)
                cv2.putText(osm_marked, f'TOP1 ({conf:.3f})', (px + marker_size + 10, py), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                print(f"  ✅ TOP1匹配位置: ({px},{py}), 置信度: {conf:.4f}")
            else:
                cv2.circle(osm_marked, (px, py), small_marker_size, (255, 0, 0), -1)
                cv2.circle(osm_marked, (px, py), small_marker_size + 3, (255, 0, 0), 2)
                cv2.putText(osm_marked, f'TOP{i+1}', (px + small_marker_size + 10, py), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                print(f"  ✅ TOP{i+1}匹配位置: ({px},{py}), 置信度: {conf:.4f}")

        cv2.imwrite(os.path.join(output_dir, 'osm_marked.png'), osm_marked)

        match_result = {
            'name': name,
            'center_lat': center_lat,
            'center_lng': center_lng,
            'match_lat': match_lat,
            'match_lng': match_lng,
            'match_confidence': float(max_val),
            'match_pixel_x': int(max_loc[0]),
            'match_pixel_y': int(max_loc[1]),
            'bev_size': (int(bev_binary.shape[1]), int(bev_binary.shape[0])),
            'osm_size': (int(osm_w), int(osm_h)),
            'actual_lat': actual_lat,
            'actual_lng': actual_lng,
            'distance_error': distance if 'distance' in locals() else None,
            'top5_matches': top_locs
        }

        with open(os.path.join(output_dir, 'match_result.json'), 'w') as f:
            json.dump(match_result, f, indent=2)

        print(f"✅ 匹配结果已保存")

        return match_result

    def _gps_to_pixel(self, lat, lng, center_lat, center_lng, radius=5000, resolution=2):
        meters_per_deg_lat = 111000.0
        meters_per_deg_lng = 111000.0 * np.cos(np.radians(center_lat))

        dy_meters = (lat - center_lat) * meters_per_deg_lat
        dx_meters = (lng - center_lng) * meters_per_deg_lng

        x = int((dx_meters + radius) * resolution)
        y = int((radius - dy_meters) * resolution)

        return x, y

    def _get_actual_position(self, point_path):
        metadata_path = os.path.join(point_path, 'streetview_images', 'step_0_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                data = json.load(f)
                return data.get('actual_lat'), data.get('actual_lng')
        return None, None

    def _calculate_distance(self, lat1, lng1, lat2, lng2):
        R = 6371000
        dlat = math.radians(lat2 - lat1)
        dlng = math.radians(lng2 - lng1)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlng/2)**2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    def run_batch_matching(self, csv_path, batch_dir, osm_radius=1000):
        points = self.read_points_from_csv(csv_path)
        print(f"📊 共读取 {len(points)} 个点位")

        all_results = []

        for point in points:
            result = self.match_point(point, batch_dir, osm_radius=osm_radius)
            if result:
                all_results.append(result)

        print(f"\n{'='*70}")
        print(f"批量匹配完成")
        print(f"{'='*70}")

        for result in all_results:
            print(f"\n点位: {result['name']}")
            print(f"  中心坐标: ({result['center_lat']:.6f}, {result['center_lng']:.6f})")
            print(f"  匹配坐标: ({result['match_lat']:.8f}, {result['match_lng']:.8f})")
            print(f"  匹配置信度: {result['match_confidence']:.4f}")
            if result['distance_error']:
                print(f"  匹配误差: {result['distance_error']:.2f}米")

        summary_path = os.path.join(batch_dir, 'matching_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n✅ 匹配汇总已保存: {summary_path}")

        return all_results

    def match_point_large_scale(self, point, batch_dir, large_radius=25000):
        name = point['name']
        center_lat = point['lat']
        center_lng = point['lng']

        print(f"\n{'='*70}")
        print(f"大规模匹配点位（{large_radius}米）: {name}")
        print(f"中心坐标: ({center_lat:.6f}, {center_lng:.6f})")
        print(f"{'='*70}")

        point_path = None
        for dir_name in os.listdir(batch_dir):
            if name in dir_name and os.path.isdir(os.path.join(batch_dir, dir_name)):
                point_path = os.path.join(batch_dir, dir_name)
                break
        
        if point_path is None:
            print(f"❌ 未找到点位目录: {name}")
            return None

        bev_path = os.path.join(point_path, 'projection_output', 'fused_bev_lowres_cropped.png')
        metadata_path = os.path.join(point_path, 'projection_output', 'bev_lowres_cropped_metadata.json')
        
        if not os.path.exists(bev_path) or not os.path.exists(metadata_path):
            bev_path = os.path.join(point_path, 'projection_output', 'fused_bev_lowres.png')
            metadata_path = os.path.join(point_path, 'projection_output', 'bev_lowres_metadata.json')
        
        if not os.path.exists(bev_path):
            print(f"❌ 未找到BEV图像")
            return None

        bev_binary = self.load_bev_image(bev_path)
        if bev_binary is None:
            print(f"❌ 无法加载BEV图像")
            return None

        print(f"📐 原始BEV图像尺寸: {bev_binary.shape[1]}×{bev_binary.shape[0]}")
        
        initial_px_orig = bev_binary.shape[1] // 2
        initial_py_orig = bev_binary.shape[0] // 2
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                bev_metadata = json.load(f)
                initial_px_orig = bev_metadata.get('initial_pixel_x', initial_px_orig)
                initial_py_orig = bev_metadata.get('initial_pixel_y', initial_py_orig)
                bev_resolution = bev_metadata.get('resolution', 2)
            print(f"📍 BEV中初始点位置: ({initial_px_orig}, {initial_py_orig}), 分辨率: {bev_resolution}像素/米")
        else:
            bev_resolution = 2
            print(f"⚠️ 未找到BEV元数据，使用默认分辨率: {bev_resolution}像素/米")

        large_resolution = 0.5
        scale_factor = bev_resolution / large_resolution
        
        print(f"📏 缩放因子: {scale_factor} (BEV: {bev_resolution}像素/米 → OSM: {large_resolution}像素/米)")
        
        new_width = int(bev_binary.shape[1] / scale_factor)
        new_height = int(bev_binary.shape[0] / scale_factor)
        bev_scaled = cv2.resize(bev_binary, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        initial_px = int(initial_px_orig / scale_factor)
        initial_py = int(initial_py_orig / scale_factor)
        
        print(f"📐 缩放后BEV图像尺寸: {new_width}×{new_height}")
        print(f"📍 缩放后初始点位置: ({initial_px}, {initial_py})")

        download_radius = int(large_radius * 1.2)
        
        print(f"📡 下载OSM路网（半径: {download_radius}米）...")
        G = self.download_osm_road_network(center_lat, center_lng, radius=download_radius)
        if G is None:
            return None

        print(f"🖼️ 渲染OSM路网（显示范围: {large_radius}米, 分辨率: {large_resolution}像素/米）...")
        osm_image, osm_w, osm_h = self.render_osm_to_image(G, center_lat, center_lng, radius=large_radius, resolution=large_resolution)
        print(f"📐 OSM图像尺寸: {osm_w}×{osm_h}")

        result = self.compute_cross_correlation(osm_image, bev_scaled)
        if result is None:
            return None

        max_loc, max_val = self.find_best_match(result)
        print(f"🎯 BEV左上角匹配位置: ({max_loc[0]}, {max_loc[1]}), 相似度: {max_val:.4f}")

        correct_x = max_loc[0] + initial_px
        correct_y = max_loc[1] + initial_py
        print(f"📍 修正后初始点位置: ({correct_x}, {correct_y})")

        match_lat, match_lng = self.pixel_to_gps(correct_x, correct_y, center_lat, center_lng, radius=large_radius, resolution=large_resolution)
        print(f"📍 匹配GPS坐标: ({match_lat:.8f}, {match_lng:.8f})")

        actual_lat, actual_lng = self._get_actual_position(point_path)
        distance = None
        if actual_lat is not None:
            print(f"📍 实际GPS坐标: ({actual_lat:.8f}, {actual_lng:.8f})")
            distance = self._calculate_distance(match_lat, match_lng, actual_lat, actual_lng)
            print(f"📏 匹配误差: {distance:.2f}米")

        output_dir = os.path.join(point_path, f'matching_results_{large_radius}m')
        os.makedirs(output_dir, exist_ok=True)

        cv2.imwrite(os.path.join(output_dir, 'osm_road_image.png'), osm_image)
        cv2.imwrite(os.path.join(output_dir, 'bev_template.png'), bev_scaled)
        
        heatmap = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        osm_h, osm_w = osm_image.shape
        
        shifted_heatmap_initial = np.zeros((osm_h, osm_w, 3), dtype=np.uint8)
        
        src_y_start = max(0, -initial_py)
        src_y_end = min(heatmap_color.shape[0], osm_h - initial_py)
        src_x_start = max(0, -initial_px)
        src_x_end = min(heatmap_color.shape[1], osm_w - initial_px)
        
        dst_y_start = max(0, initial_py)
        dst_y_end = dst_y_start + (src_y_end - src_y_start)
        dst_x_start = max(0, initial_px)
        dst_x_end = dst_x_start + (src_x_end - src_x_start)
        
        if src_x_end > src_x_start and src_y_end > src_y_start:
            shifted_heatmap_initial[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
                heatmap_color[src_y_start:src_y_end, src_x_start:src_x_end]
        
        colorbar_w = 60
        colorbar = np.zeros((osm_h, colorbar_w, 3), dtype=np.uint8)
        for i in range(osm_h):
            val = 255 - int(i / osm_h * 255)
            colorbar[i, :] = cv2.applyColorMap(np.array([[val]], dtype=np.uint8), cv2.COLORMAP_JET)[0, 0]
        
        osm_color = cv2.cvtColor(osm_image, cv2.COLOR_GRAY2BGR)
        overlay_initial = cv2.addWeighted(osm_color, 0.4, shifted_heatmap_initial, 0.6, 0)
        overlay_initial_with_colorbar = np.hstack([overlay_initial, colorbar])
        
        # 在热力图上标注关键点
        marker_size = max(25, osm_w // 600)
        
        heatmap_marked = shifted_heatmap_initial.copy()
        heatmap_overlay_marked = overlay_initial.copy()
        
        # 黄色十字：原始卷积最高分位置（BEV左上角对应位置）
        raw_match_x = max_loc[0] + initial_px
        raw_match_y = max_loc[1] + initial_py
        cv2.drawMarker(heatmap_marked, (int(raw_match_x), int(raw_match_y)), (0, 255, 255), 
                       cv2.MARKER_CROSS, marker_size * 2, 3)
        cv2.drawMarker(heatmap_overlay_marked, (int(raw_match_x), int(raw_match_y)), (0, 255, 255), 
                       cv2.MARKER_CROSS, marker_size * 2, 3)
        
        # 绿色圆点：修正后BEV中心点/初始点位置
        cv2.circle(heatmap_marked, (int(correct_x), int(correct_y)), marker_size, (0, 255, 0), -1)
        cv2.circle(heatmap_marked, (int(correct_x), int(correct_y)), marker_size + 5, (0, 255, 0), 2)
        cv2.circle(heatmap_overlay_marked, (int(correct_x), int(correct_y)), marker_size, (0, 255, 0), -1)
        cv2.circle(heatmap_overlay_marked, (int(correct_x), int(correct_y)), marker_size + 5, (0, 255, 0), 2)
        
        # 红色圆点：真实位置
        if actual_lat is not None:
            actual_x, actual_y = self._gps_to_pixel(actual_lat, actual_lng, center_lat, center_lng, radius=large_radius, resolution=large_resolution)
            if 0 <= actual_x < osm_w and 0 <= actual_y < osm_h:
                cv2.circle(heatmap_marked, (actual_x, actual_y), marker_size, (0, 0, 255), -1)
                cv2.circle(heatmap_marked, (actual_x, actual_y), marker_size + 5, (0, 0, 255), 2)
                cv2.circle(heatmap_overlay_marked, (actual_x, actual_y), marker_size, (0, 0, 255), -1)
                cv2.circle(heatmap_overlay_marked, (actual_x, actual_y), marker_size + 5, (0, 0, 255), 2)
                cv2.line(heatmap_overlay_marked, (actual_x, actual_y), (int(correct_x), int(correct_y)), (255, 255, 255), 2)
        
        # 添加图例文字
        legend_y = 30
        cv2.putText(heatmap_overlay_marked, 'YELLOW CROSS: raw best match (BEV top-left)', (10, legend_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(heatmap_overlay_marked, 'GREEN CIRCLE: BEV center / initial point match', (10, legend_y + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(heatmap_overlay_marked, 'RED CIRCLE: actual GPS position', (10, legend_y + 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        cv2.imwrite(os.path.join(output_dir, 'match_heatmap_initial.png'), np.hstack([shifted_heatmap_initial, colorbar]))
        cv2.imwrite(os.path.join(output_dir, 'match_heatmap_initial_overlay.png'), overlay_initial_with_colorbar)
        cv2.imwrite(os.path.join(output_dir, 'match_heatmap_initial_marked.png'), np.hstack([heatmap_marked, colorbar]))
        cv2.imwrite(os.path.join(output_dir, 'match_heatmap_initial_overlay_marked.png'), np.hstack([heatmap_overlay_marked, colorbar]))
        print(f"📊 卷积热力图已保存（含标记版本）")
        
        osm_marked = cv2.cvtColor(osm_image, cv2.COLOR_GRAY2BGR)
        marker_size = max(30, osm_w // 500)
        
        if actual_lat is not None:
            actual_x, actual_y = self._gps_to_pixel(actual_lat, actual_lng, center_lat, center_lng, radius=large_radius, resolution=large_resolution)
            if 0 <= actual_x < osm_w and 0 <= actual_y < osm_h:
                cv2.circle(osm_marked, (actual_x, actual_y), marker_size, (0, 0, 255), -1)
                cv2.circle(osm_marked, (actual_x, actual_y), marker_size + 5, (0, 0, 255), 2)
                cv2.putText(osm_marked, 'ACTUAL', (actual_x + marker_size + 10, actual_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.circle(osm_marked, (correct_x, correct_y), marker_size, (0, 255, 0), -1)
        cv2.circle(osm_marked, (correct_x, correct_y), marker_size + 5, (0, 255, 0), 2)
        cv2.putText(osm_marked, f'MATCH ({max_val:.3f})', (correct_x + marker_size + 10, correct_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imwrite(os.path.join(output_dir, 'osm_marked.png'), osm_marked)

        match_result = {
            'name': name,
            'radius_meters': large_radius,
            'resolution': large_resolution,
            'center_lat': center_lat,
            'center_lng': center_lng,
            'match_lat': match_lat,
            'match_lng': match_lng,
            'match_confidence': float(max_val),
            'match_pixel_x': int(correct_x),
            'match_pixel_y': int(correct_y),
            'bev_size': (int(bev_scaled.shape[1]), int(bev_scaled.shape[0])),
            'bev_original_size': (int(bev_binary.shape[1]), int(bev_binary.shape[0])),
            'scale_factor': scale_factor,
            'osm_size': (int(osm_w), int(osm_h)),
            'actual_lat': actual_lat,
            'actual_lng': actual_lng,
            'distance_error': distance
        }

        with open(os.path.join(output_dir, 'match_result.json'), 'w') as f:
            json.dump(match_result, f, indent=2)

        print(f"✅ 大规模匹配结果已保存")

        return match_result


def main():
    matcher = BEVOSMConvolutionMatcher()

    batch_dir = '/home/suanzigao/PycharmProjects/street_view_global_localization/manual_streetview_results'
    
    point_dirs = sorted([d for d in os.listdir(batch_dir) if os.path.isdir(os.path.join(batch_dir, d)) and not d.startswith('.')])
    
    points = []
    for point_dir in point_dirs:
        images_dir = os.path.join(batch_dir, point_dir, 'streetview_images')
        metadata_path = os.path.join(images_dir, 'step_0_metadata.json')
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                meta = json.load(f)
                actual_lat = meta.get('actual_lat')
                actual_lng = meta.get('actual_lng')
                if actual_lat and actual_lng:
                    points.append({
                        'name': point_dir,
                        'lat': actual_lat,
                        'lng': actual_lng
                    })
                    print(f"📍 找到点位: {point_dir}, 坐标: ({actual_lat:.6f}, {actual_lng:.6f})")
    
    print(f"\n📊 共读取 {len(points)} 个点位")
    
    print(f"\n{'='*70}")
    print(f"第一阶段：1km范围匹配")
    print(f"{'='*70}")
    
    all_results = []
    for point in points:
        result = matcher.match_point(point, batch_dir, osm_radius=1000)
        if result:
            all_results.append(result)
    
    print(f"\n{'='*70}")
    print(f"第二阶段：25km范围匹配")
    print(f"{'='*70}")
    
    all_results_25km = []
    for point in points:
        result = matcher.match_point_large_scale(point, batch_dir, large_radius=25000)
        if result:
            all_results_25km.append(result)
    
    print(f"\n{'='*70}")
    print(f"批量匹配完成")
    print(f"{'='*70}")
    
    print(f"\n📊 1km范围匹配结果:")
    for result in all_results:
        print(f"  {result['name']}: 误差 {result['distance_error']:.2f}米" if result['distance_error'] else f"  {result['name']}: 无误差")
    
    print(f"\n📊 25km范围匹配结果:")
    for result in all_results_25km:
        print(f"  {result['name']}: 误差 {result['distance_error']:.2f}米" if result['distance_error'] else f"  {result['name']}: 无误差")
    
    if all_results:
        errors = [r['distance_error'] for r in all_results if r['distance_error']]
        if errors:
            print(f"\n📊 1km统计: 平均{sum(errors)/len(errors):.2f}米, 最小{min(errors):.2f}米, 最大{max(errors):.2f}米")
    
    if all_results_25km:
        errors_25km = [r['distance_error'] for r in all_results_25km if r['distance_error']]
        if errors_25km:
            print(f"📊 25km统计: 平均{sum(errors_25km)/len(errors_25km):.2f}米, 最小{min(errors_25km):.2f}米, 最大{max(errors_25km):.2f}米")
    
    summary_path = os.path.join(batch_dir, 'matching_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ 1km匹配汇总已保存: {summary_path}")
    
    summary_path_25km = os.path.join(batch_dir, 'matching_summary_25km.json')
    with open(summary_path_25km, 'w') as f:
        json.dump(all_results_25km, f, indent=2)
    print(f"✅ 25km匹配汇总已保存: {summary_path_25km}")


if __name__ == "__main__":
    main()