"""路口场景的多道路方向提取 - 完整中间结果输出"""
import numpy as np
import cv2
from PIL import Image
import os

OUTPUT_DIR = "/home/suanzigao/PycharmProjects/street_view_global_localization/batch_streetview_results/2026-06-01_14-36-04_batch_run/2026-06-01_14-36-04_Qc7f8UCkWQSdu_0Wc7BDtQ_lat55_7132_lng12_5929/streetview_images/vggt_remote_output"

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

def extract_multiple_road_directions(bev, save_intermediate=True):
    """
    从BEV图像中提取多个道路方向（适用于路口）
    """
    # ============ 中间结果1: 原始BEV ============
    bev_normalized = (bev / bev.max() * 255).astype(np.uint8)
    bev_color = cv2.applyColorMap(bev_normalized, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "step1_bev_density.png"), bev_color)
    print("✅ 保存: step1_bev_density.png (BEV密度图)")

    # ============ 中间结果2: 二值化 ============
    threshold = np.percentile(bev, 70)
    binary = (bev > threshold).astype(np.uint8) * 255
    cv2.imwrite(os.path.join(OUTPUT_DIR, "step2_binary.png"), binary)
    print(f"✅ 保存: step2_binary.png (二值化, threshold={threshold:.1f})")

    # ============ 中间结果3: 形态学操作 ============
    kernel = np.ones((7, 7), np.uint8)
    morph = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "step3_morphology.png"), morph)
    print("✅ 保存: step3_morphology.png (形态学开闭运算)")

    # ============ 中间结果4: Canny边缘 ============
    edges = cv2.Canny(morph, 30, 100)
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "step4_canny_edges.png"), edges_color)
    print("✅ 保存: step4_canny_edges.png (Canny边缘检测)")

    # ============ 中间结果5: 标准霍夫直线检测 ============
    lines_standard = cv2.HoughLines(edges, 1, np.pi/180, threshold=50)
    hough_vis = bev_color.copy()

    standard_angles = []
    if lines_standard is not None:
        for line in lines_standard:
            rho, theta = line[0]
            standard_angles.append(theta)
            a, b = np.cos(theta), np.sin(theta)
            x0, y0 = a * rho, b * rho
            x1 = int(x0 + 2000 * (-b))
            y1 = int(y0 + 2000 * (a))
            x2 = int(x0 - 2000 * (-b))
            y2 = int(y0 - 2000 * (a))
            cv2.line(hough_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imwrite(os.path.join(OUTPUT_DIR, "step5_hough_standard.png"), hough_vis)
    print(f"✅ 保存: step5_hough_standard.png (标准霍夫变换, 检测到 {len(standard_angles) if standard_angles else 0} 条线)")

    # ============ 中间结果6: 概率霍夫变换（检测线段） ============
    lines_p = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=30,
                               minLineLength=30, maxLineGap=10)

    houghp_vis = bev_color.copy()
    line_segments = []

    if lines_p is not None:
        for line in lines_p:
            x1, y1, x2, y2 = line[0]
            # 计算线段角度
            angle = np.arctan2(y2 - y1, x2 - x1)
            line_segments.append({
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'angle_deg': np.rad2deg(angle) % 180
            })
            cv2.line(houghp_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imwrite(os.path.join(OUTPUT_DIR, "step6_houghp_segments.png"), houghp_vis)
    print(f"✅ 保存: step6_houghp_segments.png (概率霍夫变换, 检测到 {len(line_segments) if line_segments else 0} 条线段)")

    # ============ 中间结果7: 角度分箱聚类 ============
    # 使用简单的角度分箱代替DBSCAN
    angle_bins = {}
    bin_size = 15  # 15度为一个bin

    for seg in line_segments:
        angle = seg['angle_deg']
        bin_id = int(angle / bin_size) * bin_size
        if bin_id not in angle_bins:
            angle_bins[bin_id] = []
        angle_bins[bin_id].append(angle)

    # 计算每个bin的平均角度
    directions = []
    bin_centers = []

    for bin_id, angles in sorted(angle_bins.items()):
        if len(angles) >= 2:  # 至少2条线段才认为是有效方向
            rads = np.deg2rad(angles)
            x_mean = np.cos(rads).mean()
            y_mean = np.sin(rads).mean()
            avg_angle = np.rad2deg(np.arctan2(y_mean, x_mean)) % 180
            directions.append(avg_angle)
            bin_centers.append(bin_id)

    # 可视化聚类结果
    cluster_vis = bev_color.copy()
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
              (255, 255, 0), (255, 0, 255), (0, 255, 255),
              (128, 0, 128), (0, 128, 128), (128, 128, 0)]

    for i, seg in enumerate(line_segments):
        angle = seg['angle_deg']
        bin_id = int(angle / bin_size) * bin_size
        color_idx = bin_centers.index(bin_id) if bin_id in bin_centers else -1
        if color_idx >= 0:
            color = colors[color_idx % len(colors)]
        else:
            color = (128, 128, 128)  # 噪声点用灰色
        cv2.line(cluster_vis, (seg['x1'], seg['y1']),
                 (seg['x2'], seg['y2']), color, 2)

    cv2.imwrite(os.path.join(OUTPUT_DIR, "step7_clustering.png"), cluster_vis)

    print(f"✅ 保存: step7_clustering.png (角度分箱聚类, {len(directions)} 个方向类别)")
    for i, (bin_id, angles) in enumerate(sorted(angle_bins.items())):
        if len(angles) >= 2:
            print(f"   方向 {i+1} (bin {bin_id}°): {len(angles)} 条线段")

    directions = sorted(set(directions))

    # ============ 中间结果8: 最终方向指示 ============
    final_vis = bev_color.copy()
    center = (bev.shape[0] // 2, bev.shape[1] // 2)
    length = 150

    for i, angle in enumerate(directions):
        rad = np.deg2rad(angle)
        x_end = int(center[0] + length * np.cos(rad))
        y_end = int(center[1] + length * np.sin(rad))
        color = colors[i % len(colors)]
        cv2.arrowedLine(final_vis, center, (x_end, y_end), color, 3)
        cv2.putText(final_vis, f"{angle:.1f}°", (x_end + 10, y_end),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imwrite(os.path.join(OUTPUT_DIR, "step8_final_directions.png"), final_vis)
    print(f"✅ 保存: step8_final_directions.png (最终道路方向)")

    return directions, line_segments

def estimate_direction_by_pca(bev):
    """PCA备选方案"""
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

def main():
    ply_path = os.path.join(OUTPUT_DIR, "merged_pointcloud_clean.ply")

    print("=" * 60)
    print("路口多道路方向检测 - 完整流程")
    print("=" * 60)

    # 1. 加载点云
    pts = load_pointcloud(ply_path)

    # 2. 生成BEV
    print("\n[Step 1-2] 生成BEV...")
    bev, bounds = generate_bev_points(pts)
    print(f"  BEV范围: X[{bounds[0]:.2f}, {bounds[1]:.2f}], Z[{bounds[2]:.2f}, {bounds[3]:.2f}]")

    # 3. 多方向检测
    print("\n[Step 3-8] 多道路方向检测...")
    directions, line_segments = extract_multiple_road_directions(bev)

    # 4. 输出结果
    print("\n" + "=" * 60)
    print("最终结果")
    print("=" * 60)
    print(f"检测到的道路方向 ({len(directions)} 个):")
    for i, angle in enumerate(directions):
        desc = get_direction_description(angle)
        print(f"   方向 {i+1}: {angle:.1f}° - {desc}")

    print(f"\n共检测到 {len(line_segments) if line_segments else 0} 条线段")

    # 5. 保存结果文本
    result_path = os.path.join(OUTPUT_DIR, "road_directions.txt")
    with open(result_path, 'w') as f:
        f.write("路口多道路方向检测结果\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"检测到的道路方向:\n")
        for i, angle in enumerate(directions):
            desc = get_direction_description(angle)
            f.write(f"  方向 {i+1}: {angle:.1f}° - {desc}\n")
        f.write(f"\n线段总数: {len(line_segments) if line_segments else 0}\n")
    print(f"\n✅ 结果已保存: {result_path}")

def get_direction_description(angle):
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

if __name__ == "__main__":
    main()
