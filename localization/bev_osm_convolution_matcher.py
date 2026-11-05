"""
BEV图像与OSM路网卷积匹配
"""

import os
import csv
import cv2
import numpy as np
import json
import math
import time

try:
    import osmnx as ox
    OSMNX_AVAILABLE = True
except ImportError:
    OSMNX_AVAILABLE = False

# 官方推荐的公共 Overpass 镜像（base URL，osmnx 会拼接 /interpreter 与 /status）。
# 当某个镜像失败/限流时自动轮换到下一个。
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api",
    "https://overpass.kumi.systems/api",
    "https://overpass.private.coffee/api",
    "https://overpass.osm.jp/api",
]


class BEVOSMConvolutionMatcher:
    def __init__(self, min_request_interval=2.0):
        self.ox = ox if OSMNX_AVAILABLE else None
        self._min_request_interval = min_request_interval
        self._last_request_time = 0.0
        self._endpoint_idx = 0
        if OSMNX_AVAILABLE:
            # 确保 osmnx 内置节流与本地缓存开启（避免重复联网请求）
            ox.settings.overpass_rate_limit = True
            ox.settings.use_cache = True
            ox.settings.overpass_url = OVERPASS_ENDPOINTS[0]

    def _throttle(self):
        """全局请求节流：保证相邻两次下载至少间隔 min_request_interval 秒。"""
        if self._min_request_interval <= 0:
            return
        elapsed = time.time() - self._last_request_time
        wait = self._min_request_interval - elapsed
        if wait > 0:
            print(f"⏳ 请求节流：等待 {wait:.1f} 秒...")
            time.sleep(wait)
        self._last_request_time = time.time()

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

        # 官方 Overpass 端点优先，并对 502（服务器过载）做慢速指数退避重试。
        # 公共服务器过载时，短间隔重试无意义，用 30s 起步、最长 300s 的慢退避。
        official = "https://overpass-api.de/api"
        self.ox.settings.overpass_url = official
        last_err = None
        official_retries = 5
        for attempt in range(official_retries):
            try:
                self._throttle()
                G = self.ox.graph_from_point((lat, lng), dist=radius, network_type='drive')
                print(f"✅ 路网下载完成: {len(G.nodes)}节点, {len(G.edges)}边 (via {official})")
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                return G
            except Exception as e:
                last_err = e
                wait = min(30 * (2 ** attempt), 300)
                print(f"⚠️ 官方镜像下载失败(尝试{attempt + 1}/{official_retries}): {e}")
                if attempt < official_retries - 1:
                    print(f"   ⏳ 退避等待 {wait} 秒后重试...")
                    time.sleep(wait)

        # 官方多次失败后，轮换其余镜像兜底
        errors = [f"{official}: {last_err}"]
        for endpoint in OVERPASS_ENDPOINTS:
            if endpoint == official:
                continue
            self.ox.settings.overpass_url = endpoint
            try:
                self._throttle()
                G = self.ox.graph_from_point((lat, lng), dist=radius, network_type='drive')
                print(f"✅ 路网下载完成: {len(G.nodes)}节点, {len(G.edges)}边 (via {endpoint})")
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                return G
            except Exception as e:
                errors.append(f"{endpoint}: {e}")
                print(f"⚠️ 镜像 {endpoint} 下载失败: {e}")

        print(f"❌ 所有镜像下载均失败: {errors}")
        return None

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

    # =====================================================================
    # 分层（Coarse-to-Fine）匹配：低分辨率覆盖大范围 + 高分辨率局部精修
    # =====================================================================
    def _downsample_bev(self, bev, scale):
        """把 BEV 模板等比降采样 scale 倍（scale < 1 时缩小）。"""
        if abs(scale - 1.0) < 1e-6:
            return bev
        new_w = max(1, int(round(bev.shape[1] * scale)))
        new_h = max(1, int(round(bev.shape[0] * scale)))
        return cv2.resize(bev, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _extract_top_peaks(self, result, top_k=5, min_dist_px=10.0):
        """在相关图 result 上提取 top-k 个峰值，带最小间隔抑制。

        返回 list of {'pixel_x', 'pixel_y', 'confidence'}（像素坐标为模板左上角位置）。
        """
        h, w = result.shape
        flat_idx = np.argsort(result.ravel())[::-1]
        peaks = []
        for idx in flat_idx:
            py = int(idx // w)
            px = int(idx % w)
            val = float(result.ravel()[idx])
            ok = True
            for p in peaks:
                if math.hypot(px - p['pixel_x'], py - p['pixel_y']) < min_dist_px:
                    ok = False
                    break
            if ok:
                peaks.append({'pixel_x': px, 'pixel_y': py, 'confidence': val})
            if len(peaks) >= top_k:
                break
        return peaks

    def _load_bev_template(self, point_path):
        """加载 BEV 模板 + 元数据（分辨率、初始点像素坐标）。"""
        bev_path = os.path.join(point_path, 'projection_output', 'fused_bev_lowres_cropped.png')
        metadata_path = os.path.join(point_path, 'projection_output', 'bev_lowres_cropped_metadata.json')
        if not (os.path.exists(bev_path) and os.path.exists(metadata_path)):
            bev_path = os.path.join(point_path, 'projection_output', 'fused_bev_lowres.png')
            metadata_path = os.path.join(point_path, 'projection_output', 'bev_lowres_metadata.json')

        if not os.path.exists(bev_path):
            print(f"❌ 未找到BEV图像")
            return None, None, None, None, None

        bev_binary = self.load_bev_image(bev_path)
        if bev_binary is None:
            return None, None, None, None, None

        resolution = 2.0
        initial_px = bev_binary.shape[1] // 2
        initial_py = bev_binary.shape[0] // 2
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                meta = json.load(f)
            resolution = float(meta.get('resolution', resolution))
            initial_px = int(meta.get('initial_pixel_x', initial_px))
            initial_py = int(meta.get('initial_pixel_y', initial_py))

        return bev_binary, resolution, initial_px, initial_py, metadata_path

    def match_point_multiscale(self, point, batch_dir, levels, top_k=5, dedup_m=300.0):
        """通用多尺度级联（Coarse-to-Fine）匹配。

        levels: list of (radius_m, resolution_px_per_m)，从粗到细排列；
                最后一级为精修层。resolution 传 None 表示使用 BEV 原生分辨率。
                例：[(15000, 0.1), (5000, 0.5), (1000, None)] 为 15km 三层。
        每层保留 top_k 个候选（束搜索），下一层围绕这些候选做局部精修。
        返回结构与 match_point 一致（match_lat/lng、match_confidence、top5_matches）。
        """
        name = point['name']
        center_lat = point['lat']
        center_lng = point['lng']

        print(f"\n{'='*70}")
        print(f"多尺度级联匹配点位: {name}")
        print(f"中心坐标: ({center_lat:.6f}, {center_lng:.6f})")
        print(f"层级: {levels} | top-k: {top_k} | 去重距离: {dedup_m}m")
        print(f"{'='*70}")

        point_path = None
        for dir_name in os.listdir(batch_dir):
            if name in dir_name and os.path.isdir(os.path.join(batch_dir, dir_name)):
                point_path = os.path.join(batch_dir, dir_name)
                break
        if point_path is None:
            print(f"❌ 未找到点位目录: {name}")
            return None

        bev_binary, bev_resolution, initial_px, initial_py, _ = self._load_bev_template(point_path)
        if bev_binary is None:
            return None

        print(f"📐 BEV模板尺寸: {bev_binary.shape[1]}×{bev_binary.shape[0]}, "
              f"分辨率 {bev_resolution}px/m, 初始点 ({initial_px},{initial_py})")

        # 下载一次最大范围 OSM，各层共用
        max_radius = levels[0][0]
        download_radius = int(max_radius * 1.2)
        print(f"📡 下载OSM路网（半径: {download_radius}米）...")
        G = self.download_osm_road_network(center_lat, center_lng, radius=download_radius)
        if G is None:
            return None

        # 束搜索：初始候选只有搜索中心，逐层收敛
        centers = [(center_lat, center_lng)]
        final_candidates = []

        for lvl_idx, (radius, resolution) in enumerate(levels):
            if resolution is None:
                resolution = bev_resolution
            is_final = (lvl_idx == len(levels) - 1)

            scale = resolution / bev_resolution
            bev_lvl = self._downsample_bev(bev_binary, scale)
            ipx = int(initial_px * scale)
            ipy = int(initial_py * scale)

            print(f"\n--- Layer {lvl_idx}: {radius}m @ {resolution}px/m "
                  f"(BEV {bev_lvl.shape[1]}×{bev_lvl.shape[0]}, 候选数 {len(centers)}) ---")

            all_peaks = []
            for (clat, clng) in centers:
                img, w, h = self.render_osm_to_image(
                    G, clat, clng, radius=radius, resolution=resolution)
                if bev_lvl.shape[0] > img.shape[0] or bev_lvl.shape[1] > img.shape[1]:
                    continue
                result = self.compute_cross_correlation(img, bev_lvl)
                if result is None:
                    continue
                peaks = self._extract_top_peaks(
                    result, top_k=top_k, min_dist_px=dedup_m * resolution)
                for pk in peaks:
                    lat, lng = self.pixel_to_gps(
                        pk['pixel_x'] + ipx, pk['pixel_y'] + ipy,
                        clat, clng, radius=radius, resolution=resolution)
                    all_peaks.append({
                        'lat': lat, 'lng': lng, 'confidence': pk['confidence'],
                        'level': lvl_idx,
                    })

            if not all_peaks:
                print(f"❌ Layer {lvl_idx} 无候选")
                return None

            # 按置信度降序 + 去重
            all_peaks.sort(key=lambda c: c['confidence'], reverse=True)
            dedup = []
            for c in all_peaks:
                if any(self._calculate_distance(c['lat'], c['lng'], q['lat'], q['lng']) < dedup_m
                       for q in dedup):
                    continue
                dedup.append(c)

            if is_final:
                final_candidates = dedup
            else:
                centers = [(c['lat'], c['lng']) for c in dedup[:top_k]]
                print(f"  → 保留 {len(centers)} 个候选进入下一层")

        best = final_candidates[0]
        top5 = [{
            'pixel_x': None, 'pixel_y': None,
            'lat': c['lat'], 'lng': c['lng'], 'confidence': c['confidence'],
        } for c in final_candidates[:5]]

        actual_lat, actual_lng = self._get_actual_position(point_path)
        distance = None
        if actual_lat is not None:
            distance = self._calculate_distance(best['lat'], best['lng'], actual_lat, actual_lng)
            print(f"📏 多尺度匹配误差: {distance:.2f}米")

        output_dir = os.path.join(point_path, 'matching_results_multiscale')
        os.makedirs(output_dir, exist_ok=True)
        first_radius, first_res = levels[0]
        if first_res is None:
            first_res = bev_resolution
        first_img, first_w, first_h = self.render_osm_to_image(
            G, center_lat, center_lng, radius=first_radius, resolution=first_res)
        cv2.imwrite(os.path.join(output_dir, 'osm_road_level0.png'), first_img)
        cv2.imwrite(os.path.join(output_dir, 'bev_template.png'), bev_binary)

        match_result = {
            'name': name,
            'center_lat': center_lat,
            'center_lng': center_lng,
            'match_lat': best['lat'],
            'match_lng': best['lng'],
            'match_confidence': best['confidence'],
            'match_pixel_x': None,
            'match_pixel_y': None,
            'bev_size': (int(bev_binary.shape[1]), int(bev_binary.shape[0])),
            'osm_size': (first_w, first_h),
            'actual_lat': actual_lat,
            'actual_lng': actual_lng,
            'distance_error': distance,
            'top5_matches': top5,
            'levels': levels,
        }

        with open(os.path.join(output_dir, 'match_result.json'), 'w') as f:
            json.dump(match_result, f, indent=2)
        print(f"✅ 多尺度匹配结果已保存")

        return match_result

    def match_point_hierarchical(self, point, batch_dir,
                                 coarse_radius=5000, coarse_resolution=0.5,
                                 fine_window=500, top_k=5):
        """两层分层匹配（粗层全局 + 细层精修），match_point_multiscale 的便捷封装。"""
        levels = [
            (coarse_radius, coarse_resolution),
            (fine_window, None),  # 精修层用 BEV 原生分辨率
        ]
        return self.match_point_multiscale(point, batch_dir, levels, top_k=top_k)

    def match_point_15km(self, point, batch_dir, top_k=5):
        """15km 三层分层匹配：15km@0.1 → 5km@0.5 → 1km@BEV分辨率。"""
        levels = [
            (15000, 0.1),
            (5000, 0.5),
            (1000, None),  # 精修层用 BEV 原生分辨率
        ]
        return self.match_point_multiscale(point, batch_dir, levels, top_k=top_k)


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