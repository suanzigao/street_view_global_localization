"""路口类型完整检测系统"""
import numpy as np
import cv2
import os

OUTPUT_DIR = "/home/suanzigao/PycharmProjects/street_view_global_localization/batch_streetview_results/2026-06-01_14-36-04_batch_run/2026-06-01_14-36-04_Qc7f8UCkWQSdu_0Wc7BDtQ_lat55_7132_lng12_5929/streetview_images/vggt_remote_output"

# 路口类型枚举
INTERSECTION_TYPES = {
    'straight': '直线路段',
    't_junction': 'T字路口',
    '3way': '三岔路口',
    '4way': '十字路口',
    'unknown': '未知类型'
}

def load_pointcloud(ply_path):
    """加载PLY点云文件"""
    print(f"Loading PLY: {ply_path}")
    with open(ply_path) as f:
        lines = f.readlines()

    header_end = 0
    n_vertices = 0
    for i, line in enumerate(lines):
        if line.strip() == "end_header":
            header_end = i + 1
            break
        if "element vertex" in line:
            n_vertices = int(line.strip().split()[-1])

    print(f"  {n_vertices:,} vertices")
    pts = np.zeros((n_vertices, 3), dtype=np.float32)
    for i, line in enumerate(lines[header_end:]):
        parts = line.strip().split()
        if len(parts) >= 3:
            pts[i] = [float(parts[0]), float(parts[1]), float(parts[2])]

    return pts

def generate_bev_points(pts, resolution=512):
    """生成BEV点云"""
    ground_mask = (pts[:, 1] > -0.3) & (pts[:, 1] < 1.5)
    ground_pts = pts[ground_mask]
    print(f"  地面点数量: {len(ground_pts):,} / {len(pts):,}")

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

    bev = np.zeros((resolution, resolution), dtype=np.float32)
    np.add.at(bev, (zi, xi), 1)

    return bev, (x_min, x_max, z_min, z_max)

def extract_multiple_road_directions(bev):
    """从BEV提取多个道路方向"""
    # 二值化
    threshold = np.percentile(bev, 70)
    binary = (bev > threshold).astype(np.uint8) * 255
    cv2.imwrite(os.path.join(OUTPUT_DIR, "intersection_binary.png"), binary)

    # 形态学操作
    kernel = np.ones((7, 7), np.uint8)
    morph = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "intersection_morph.png"), morph)

    # Canny边缘检测
    edges = cv2.Canny(morph, 30, 100)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "intersection_edges.png"), edges)

    # 概率霍夫变换
    lines_p = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=30,
                               minLineLength=30, maxLineGap=10)

    line_segments = []
    if lines_p is not None:
        for line in lines_p:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1)
            line_segments.append({
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'angle_deg': np.rad2deg(angle) % 180
            })

    # 可视化线段
    bev_color = cv2.applyColorMap((bev / bev.max() * 255).astype(np.uint8), cv2.COLORMAP_JET)
    hough_vis = bev_color.copy()
    for seg in line_segments:
        cv2.line(hough_vis, (seg['x1'], seg['y1']), (seg['x2'], seg['y2']), (0, 255, 0), 2)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "intersection_segments.png"), hough_vis)

    # 角度分箱聚类
    if line_segments:
        directions = cluster_by_angle(line_segments)
    else:
        directions = [estimate_direction_by_pca(bev) % 180]

    return directions, line_segments, bev_color

def cluster_by_angle(line_segments, bin_size=15):
    """角度分箱聚类"""
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

def estimate_direction_by_pca(bev):
    """PCA估计方向"""
    y_coords, x_coords = np.where(bev > 0)
    points = np.column_stack((x_coords, y_coords)).astype(np.float32)
    
    if len(points) < 2:
        return 0.0
    
    points -= points.mean(axis=0)
    cov_matrix = np.cov(points.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    main_dir = eigenvectors[:, np.argmax(eigenvalues)]
    angle = np.arctan2(main_dir[1], main_dir[0])
    return np.rad2deg(angle) % 360

def count_segments_per_direction(directions, line_segments, tolerance=15):
    """统计各方向线段数量"""
    counts = {d: 0 for d in directions}
    for seg in line_segments:
        seg_angle = seg['angle_deg']
        for dir_angle in directions:
            diff = abs(seg_angle - dir_angle)
            diff = min(diff, 180 - diff)
            if diff < tolerance:
                counts[dir_angle] += 1
                break
    return counts

def is_t_junction(directions, line_segments):
    """检测T字路口"""
    if len(directions) != 3:
        return False, None, None

    dirs = sorted(directions)
    diffs = []
    for i in range(3):
        j = (i + 1) % 3
        diff = abs(dirs[j] - dirs[i])
        diff = min(diff, 180 - diff)
        diffs.append(diff)

    max_diff = max(diffs)
    max_idx = diffs.index(max_diff)
    condition1 = max_diff > 150

    other_diffs = [d for i, d in enumerate(diffs) if i != max_idx]
    sum_other = sum(other_diffs)
    condition2 = sum_other > 60 and sum_other < 120

    if condition1 and condition2:
        main_dir = dirs[(max_idx + 2) % 3]
        branch_dirs = [dirs[i] for i in range(3) if i != (max_idx + 2) % 3]
        branch_dir = np.mean(branch_dirs) % 180
        return True, main_dir, branch_dir

    return False, None, None

def check_3way_intersection(directions):
    """检测三岔路口"""
    if len(directions) != 3:
        return False

    dirs = sorted(directions)
    diffs = []
    for i in range(3):
        j = (i + 1) % 3
        diff = abs(dirs[j] - dirs[i])
        diff = min(diff, 180 - diff)
        diffs.append(diff)

    max_diff = max(diffs)
    min_diff = min(diffs)

    condition1 = max_diff >= 60 and max_diff <= 170
    condition2 = min_diff >= 20

    return condition1 and condition2

def detect_intersection_type(directions, line_segments):
    """检测路口类型"""
    n_dirs = len(directions)

    if n_dirs == 1:
        return 'straight', {}

    elif n_dirs == 2:
        diff = abs(directions[1] - directions[0])
        diff = min(diff, 180 - diff)
        if diff < 30:
            return 'straight', {}
        return 'unknown', {}

    elif n_dirs == 3:
        # 先检查T字路口
        is_t, main_dir, branch_dir = is_t_junction(directions, line_segments)
        if is_t:
            return 't_junction', {
                'main_direction': main_dir,
                'branch_direction': branch_dir
            }

        # 再检查普通三岔路口
        if check_3way_intersection(directions):
            return '3way', {
                'directions': directions
            }

        return 'unknown', {}

    elif n_dirs >= 4:
        return '4way', {
            'directions': directions
        }

    return 'unknown', {}

def analyze_intersection(directions, line_segments, bev_color):
    """详细分析路口"""
    print("\n" + "=" * 60)
    print("路口类型分析")
    print("=" * 60)

    # 检测路口类型
    intersection_type, details = detect_intersection_type(directions, line_segments)

    print(f"\n📋 检测结果: {INTERSECTION_TYPES[intersection_type]}")
    print(f"📍 检测到 {len(directions)} 个道路方向")

    # 详细分析
    if intersection_type == 'straight':
        print("\n🛤️ 直线路段特征:")
        print(f"   - 主方向: {directions[0]:.1f}° - {get_direction_description(directions[0])}")

    elif intersection_type == 't_junction':
        print("\n🚦 T字路口特征:")
        print(f"   - 主路方向: {details['main_direction']:.1f}° - {get_direction_description(details['main_direction'])}")
        print(f"   - 支路方向: {details['branch_direction']:.1f}° - {get_direction_description(details['branch_direction'])}")
        
        angle_diff = abs(details['main_direction'] - details['branch_direction'])
        angle_diff = min(angle_diff, 180 - angle_diff)
        print(f"   - 支路夹角: {angle_diff:.1f}°")
        
        if angle_diff < 60:
            print("   - 类型: 锐角T字路口")
        elif angle_diff < 120:
            print("   - 类型: 标准T字路口")
        else:
            print("   - 类型: 钝角T字路口")

    elif intersection_type == '3way':
        print("\n🚧 三岔路口特征:")
        dirs = sorted(directions)
        diffs = []
        for i in range(3):
            j = (i + 1) % 3
            diff = abs(dirs[j] - dirs[i])
            diff = min(diff, 180 - diff)
            diffs.append(diff)
        
        max_idx = diffs.index(max(diffs))
        branch1, branch2 = dirs[max_idx], dirs[(max_idx + 1) % 3]
        main_dir = dirs[(max_idx + 2) % 3]
        
        print(f"   - 主方向: {main_dir:.1f}° - {get_direction_description(main_dir)}")
        print(f"   - 分支1: {branch1:.1f}° - {get_direction_description(branch1)}")
        print(f"   - 分支2: {branch2:.1f}° - {get_direction_description(branch2)}")
        print(f"   - 分支夹角: {max(diffs):.1f}°")

    elif intersection_type == '4way':
        print("\n🚥 十字路口特征:")
        for i, d in enumerate(sorted(directions)):
            print(f"   - 方向 {i+1}: {d:.1f}° - {get_direction_description(d)}")

    # 统计线段数量
    dir_counts = count_segments_per_direction(directions, line_segments)
    print("\n📊 各方向线段数量:")
    for d in sorted(directions):
        print(f"   {d:.1f}°: {dir_counts[d]} 条")

    # 可视化最终结果
    visualize_final_result(bev_color, directions, intersection_type)

    # 保存结果
    save_result(intersection_type, directions, details)

    return intersection_type

def visualize_final_result(bev_color, directions, intersection_type):
    """可视化最终结果"""
    vis = bev_color.copy()
    center = (vis.shape[0] // 2, vis.shape[1] // 2)
    length = 120

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
              (255, 255, 0), (255, 0, 255), (0, 255, 255)]

    for i, angle in enumerate(directions):
        rad = np.deg2rad(angle)
        x_end = int(center[0] + length * np.cos(rad))
        y_end = int(center[1] + length * np.sin(rad))
        color = colors[i % len(colors)]
        cv2.arrowedLine(vis, center, (x_end, y_end), color, 3)
        cv2.putText(vis, f"{angle:.1f}°", (x_end + 10, y_end),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.putText(vis, f"Type: {INTERSECTION_TYPES[intersection_type]}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imwrite(os.path.join(OUTPUT_DIR, "intersection_final.png"), vis)
    print(f"\n✅ 可视化结果已保存: intersection_final.png")

def save_result(intersection_type, directions, details):
    """保存分析结果"""
    result_path = os.path.join(OUTPUT_DIR, "intersection_analysis.txt")
    with open(result_path, 'w') as f:
        f.write("路口类型分析报告\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"路口类型: {INTERSECTION_TYPES[intersection_type]}\n")
        f.write(f"检测到的方向数量: {len(directions)}\n\n")
        f.write("道路方向:\n")
        for d in sorted(directions):
            f.write(f"  - {d:.1f}° ({get_direction_description(d)})\n")
        f.write("\n详细信息:\n")
        for k, v in details.items():
            if isinstance(v, float):
                f.write(f"  {k}: {v:.1f}°\n")
            else:
                f.write(f"  {k}: {v}\n")
    print(f"✅ 分析报告已保存: intersection_analysis.txt")

def get_direction_description(angle):
    """角度转方向描述"""
    angle = angle % 360
    if angle < 22.5 or angle >= 337.5:
        return "正北 (N)"
    elif angle < 67.5:
        return "东北 (NE)"
    elif angle < 112.5:
        return "正东 (E)"
    elif angle < 157.5:
        return "东南 (SE)"
    elif angle < 202.5:
        return "正南 (S)"
    elif angle < 247.5:
        return "西南 (SW)"
    elif angle < 292.5:
        return "正西 (W)"
    else:
        return "西北 (NW)"

def main():
    ply_path = os.path.join(OUTPUT_DIR, "merged_pointcloud_clean.ply")

    print("=" * 60)
    print("路口类型完整检测系统")
    print("=" * 60)

    # 1. 加载点云
    print("\n[1/4] 加载点云...")
    pts = load_pointcloud(ply_path)

    # 2. 生成BEV
    print("\n[2/4] 生成BEV...")
    bev, bounds = generate_bev_points(pts)
    print(f"  BEV范围: X[{bounds[0]:.2f}, {bounds[1]:.2f}], Z[{bounds[2]:.2f}, {bounds[3]:.2f}]")

    # 3. 提取道路方向
    print("\n[3/4] 提取道路方向...")
    directions, line_segments, bev_color = extract_multiple_road_directions(bev)
    print(f"  检测到 {len(directions)} 个方向: {[f'{d:.1f}°' for d in directions]}")

    # 4. 分析路口类型
    print("\n[4/4] 分析路口类型...")
    analyze_intersection(directions, line_segments, bev_color)

    print("\n" + "=" * 60)
    print("分析完成!")
    print("=" * 60)

if __name__ == "__main__":
    main()