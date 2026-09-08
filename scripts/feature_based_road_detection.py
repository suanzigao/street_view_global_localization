"""基于特征提取的鲁棒道路检测系统"""
import numpy as np
import cv2
import os

OUTPUT_DIR = "/home/suanzigao/PycharmProjects/street_view_global_localization/batch_streetview_results/2026-06-01_14-36-04_batch_run/2026-06-01_14-36-04_Qc7f8UCkWQSdu_0Wc7BDtQ_lat55_7132_lng12_5929/streetview_images/vggt_remote_output"

def load_pointcloud_with_color(ply_path):
    """加载带颜色的PLY点云"""
    print(f"Loading PLY with color: {ply_path}")
    with open(ply_path) as f:
        lines = f.readlines()

    header_end = 0
    n_vertices = 0
    has_color = False

    for i, line in enumerate(lines):
        if line.strip() == "end_header":
            header_end = i + 1
            break
        if "element vertex" in line:
            n_vertices = int(line.strip().split()[-1])
        if "property uchar red" in line:
            has_color = True

    print(f"  {n_vertices:,} vertices, has_color={has_color}")

    pts = np.zeros((n_vertices, 3), dtype=np.float32)
    colors = np.zeros((n_vertices, 3), dtype=np.uint8) if has_color else None

    for i, line in enumerate(lines[header_end:header_end + n_vertices]):
        parts = line.strip().split()
        if len(parts) >= 3:
            pts[i] = [float(parts[0]), float(parts[1]), float(parts[2])]
            if has_color and len(parts) >= 6:
                colors[i] = [int(parts[3]), int(parts[4]), int(parts[5])]

    return pts, colors

def generate_bev_with_color(pts, colors, resolution=512):
    """生成带颜色的BEV"""
    ground_mask = (pts[:, 1] > -0.5) & (pts[:, 1] < 2.0)
    ground_pts = pts[ground_mask]
    ground_colors = colors[ground_mask] if colors is not None else None
    print(f"  地面点数量: {len(ground_pts):,}")

    x_min, x_max = np.percentile(ground_pts[:, 0], [1, 99])
    z_min, z_max = np.percentile(ground_pts[:, 2], [1, 99])

    max_range = max(x_max - x_min, z_max - z_min)
    cx, cz = (x_min + x_max) / 2, (z_min + z_max) / 2
    x_min, x_max = cx - max_range/2, cx + max_range/2
    z_min, z_max = cz - max_range/2, cz + max_range/2

    xi = ((ground_pts[:, 0] - x_min) / (x_max - x_min) * (resolution - 1)).astype(int)
    zi = ((ground_pts[:, 2] - z_min) / (z_max - z_min) * (resolution - 1)).astype(int)
    xi = np.clip(xi, 0, resolution - 1)
    zi = np.clip(zi, 0, resolution - 1)

    bev_density = np.zeros((resolution, resolution), dtype=np.float32)
    np.add.at(bev_density, (zi, xi), 1)

    bev_color = np.zeros((resolution, resolution, 3), dtype=np.uint8)
    if ground_colors is not None:
        for i in range(len(ground_pts)):
            bev_color[zi[i], xi[i]] = ground_colors[i]

    return bev_density, bev_color, (x_min, x_max, z_min, z_max)

def extract_color_feature(bev_color):
    """提取颜色特征：道路通常是低饱和度的灰色区域"""
    if bev_color is None or bev_color.size == 0:
        return None

    hsv = cv2.cvtColor(bev_color, cv2.COLOR_BGR2HSV)
    saturation = hsv[..., 1]
    value = hsv[..., 2]

    low_saturation = (saturation < 40).astype(np.uint8) * 255
    medium_value = ((value > 10) & (value < 240)).astype(np.uint8) * 255

    color_mask = cv2.bitwise_and(low_saturation, medium_value)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "feature_color.png"), color_mask)
    print("✅ 保存: feature_color.png (颜色特征)")

    return color_mask

def extract_texture_feature(bev_density):
    """提取纹理特征：道路纹理相对均匀"""
    blur = cv2.GaussianBlur(bev_density, (15, 15), 0)
    texture_diff = cv2.absdiff(bev_density, blur)
    texture_mask = (texture_diff < np.percentile(texture_diff, 50)).astype(np.uint8) * 255
    
    cv2.imwrite(os.path.join(OUTPUT_DIR, "feature_texture.png"), texture_mask)
    print("✅ 保存: feature_texture.png (纹理特征)")

    return texture_mask

def extract_elongated_feature(bev_density):
    """提取长条形结构特征"""
    binary = (bev_density > np.percentile(bev_density, 50)).astype(np.uint8) * 255
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 30))
    opened2 = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    elongated_mask = cv2.bitwise_or(opened, opened2)
    
    cv2.imwrite(os.path.join(OUTPUT_DIR, "feature_elongated.png"), elongated_mask)
    print("✅ 保存: feature_elongated.png (长条形特征)")

    return elongated_mask

def suppress_edge_regions(mask, margin=15):
    """抑制图像边缘区域"""
    result = mask.copy()
    h, w = mask.shape

    result[:margin, :] = 0
    result[-margin:, :] = 0
    result[:, :margin] = 0
    result[:, -margin:] = 0

    cv2.imwrite(os.path.join(OUTPUT_DIR, "feature_edge_suppressed.png"), result)
    print("✅ 保存: feature_edge_suppressed.png (边缘抑制)")

    return result

def fuse_features(features):
    """多特征融合"""
    if len(features) == 0:
        return None

    result = features[0].copy()
    for feat in features[1:]:
        result = cv2.bitwise_and(result, feat)

    # 形态学后处理
    kernel = np.ones((5, 5), np.uint8)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)

    cv2.imwrite(os.path.join(OUTPUT_DIR, "feature_fused.png"), result)
    print("✅ 保存: feature_fused.png (特征融合)")

    return result

def extract_road_segments_from_mask(mask, min_length=25):
    """从道路掩码提取线段"""
    edges = cv2.Canny(mask, 30, 100)
    
    cv2.imwrite(os.path.join(OUTPUT_DIR, "road_edges.png"), edges)
    print("✅ 保存: road_edges.png (道路边缘)")

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=15,
        minLineLength=min_length,
        maxLineGap=15
    )

    line_segments = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1)
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            line_segments.append({
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'angle_deg': np.rad2deg(angle) % 180,
                'length': length
            })

    return line_segments, edges

def cluster_by_angle(line_segments, bin_size=15):
    """角度分箱聚类"""
    if not line_segments:
        return []

    angle_bins = {}
    for seg in line_segments:
        angle = seg['angle_deg']
        bin_id = int(angle / bin_size) * bin_size
        if bin_id not in angle_bins:
            angle_bins[bin_id] = []
        angle_bins[bin_id].append(angle)

    directions = []
    for bin_id, angles in sorted(angle_bins.items()):
        if len(angles) >= 2:
            rads = np.deg2rad(angles)
            x_mean = np.cos(rads).mean()
            y_mean = np.sin(rads).mean()
            avg_angle = np.rad2deg(np.arctan2(y_mean, x_mean)) % 180
            directions.append(avg_angle)

    return sorted(set(directions))

def detect_intersection_type(directions):
    """检测路口类型"""
    n_dirs = len(directions)

    if n_dirs == 1:
        return 'straight'
    elif n_dirs == 2:
        diff = abs(directions[1] - directions[0])
        diff = min(diff, 180 - diff)
        return 'straight' if diff < 30 else 'unknown'
    elif n_dirs == 3:
        return '3way'
    elif n_dirs >= 4:
        return '4way'
    return 'unknown'

def visualize_results(bev_color, road_mask, line_segments, directions):
    """可视化所有结果"""
    # 基础BEV（上下翻转使方向正确）
    if bev_color is not None:
        bev_color_flipped = np.flipud(bev_color)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "bev_color.png"), bev_color_flipped)
        print("✅ 保存: bev_color.png (彩色BEV)")

    # 道路区域叠加
    vis_road = cv2.cvtColor(road_mask, cv2.COLOR_GRAY2BGR)
    vis_road[road_mask > 0] = [0, 255, 0]
    cv2.imwrite(os.path.join(OUTPUT_DIR, "road_region.png"), vis_road)
    print("✅ 保存: road_region.png (道路区域)")

    # 线段可视化
    vis_segments = cv2.cvtColor(road_mask, cv2.COLOR_GRAY2BGR)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
              (255, 255, 0), (255, 0, 255)]
    for i, seg in enumerate(line_segments):
        color = colors[i % len(colors)]
        cv2.line(vis_segments, (seg['x1'], seg['y1']),
                 (seg['x2'], seg['y2']), color, 2)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "detected_segments.png"), vis_segments)
    print("✅ 保存: detected_segments.png (检测线段)")

    # 方向箭头
    vis_directions = vis_segments.copy()
    center = (vis_directions.shape[0] // 2, vis_directions.shape[1] // 2)
    length = 100
    for i, angle in enumerate(directions):
        rad = np.deg2rad(angle)
        x_end = int(center[0] + length * np.cos(rad))
        y_end = int(center[1] + length * np.sin(rad))
        cv2.arrowedLine(vis_directions, center, (x_end, y_end), colors[i % len(colors)], 3)
        cv2.putText(vis_directions, f"{angle:.1f}°", (x_end + 5, y_end),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i % len(colors)], 1)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "final_directions.png"), vis_directions)
    print("✅ 保存: final_directions.png (最终方向)")

def main():
    ply_path = os.path.join(OUTPUT_DIR, "merged_pointcloud_clean.ply")

    print("=" * 60)
    print("基于特征提取的鲁棒道路检测")
    print("=" * 60)

    # 1. 加载点云
    print("\n[1/5] 加载点云...")
    pts, colors = load_pointcloud_with_color(ply_path)

    # 2. 生成BEV
    print("\n[2/5] 生成BEV...")
    bev_density, bev_color, bounds = generate_bev_with_color(pts, colors)
    print(f"  BEV范围: X[{bounds[0]:.2f}, {bounds[1]:.2f}], Z[{bounds[2]:.2f}, {bounds[3]:.2f}]")

    # 3. 特征提取
    print("\n[3/5] 提取特征...")
    features = []

    color_feat = extract_color_feature(bev_color)
    if color_feat is not None:
        features.append(color_feat)

    texture_feat = extract_texture_feature(bev_density)
    features.append(texture_feat)

    elongated_feat = extract_elongated_feature(bev_density)
    features.append(elongated_feat)

    # 4. 特征融合
    print("\n[4/5] 特征融合...")
    road_mask = fuse_features(features)
    road_mask = suppress_edge_regions(road_mask)

    # 5. 提取线段和方向
    print("\n[5/5] 提取道路线段和方向...")
    line_segments, edges = extract_road_segments_from_mask(road_mask)
    print(f"  检测到 {len(line_segments)} 条线段")

    directions = cluster_by_angle(line_segments)
    print(f"  检测到 {len(directions)} 个方向: {[f'{d:.1f}°' for d in directions]}")

    intersection_type = detect_intersection_type(directions)
    print(f"  路口类型: {intersection_type}")

    # 6. 可视化
    visualize_results(bev_color, road_mask, line_segments, directions)

    # 7. 保存结果
    with open(os.path.join(OUTPUT_DIR, "road_detection_report.txt"), 'w') as f:
        f.write("基于特征提取的道路检测报告\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"点云顶点数: {len(pts):,}\n")
        f.write(f"检测到的方向数: {len(directions)}\n")
        f.write(f"检测到的线段数: {len(line_segments)}\n")
        f.write(f"路口类型: {intersection_type}\n\n")
        f.write("道路方向:\n")
        for d in directions:
            f.write(f"  - {d:.1f}°\n")
    print("\n✅ 检测报告已保存: road_detection_report.txt")

    print("\n" + "=" * 60)
    print("检测完成!")
    print("=" * 60)

if __name__ == "__main__":
    main()