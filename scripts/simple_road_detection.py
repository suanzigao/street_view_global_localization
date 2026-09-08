"""简化版道路检测：直接从密度图提取"""
import numpy as np
import cv2
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

    pts = np.zeros((n_vertices, 3), dtype=np.float32)
    for i, line in enumerate(lines[header_end:header_end + n_vertices]):
        parts = line.strip().split()
        if len(parts) >= 3:
            pts[i] = [float(parts[0]), float(parts[1]), float(parts[2])]

    return pts

def generate_bev(pts, resolution=512):
    """生成BEV密度图"""
    ground_mask = (pts[:, 1] > -0.5) & (pts[:, 1] < 2.0)
    ground_pts = pts[ground_mask]
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

    bev = np.zeros((resolution, resolution), dtype=np.float32)
    np.add.at(bev, (zi, xi), 1)

    return bev

def extract_road_segments(bev):
    """从BEV密度图提取道路线段"""
    # 1. 二值化
    threshold = np.percentile(bev, 60)
    binary = (bev > threshold).astype(np.uint8) * 255
    cv2.imwrite(os.path.join(OUTPUT_DIR, "simple_binary.png"), binary)

    # 2. 形态学操作
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "simple_morph.png"), binary)

    # 3. 边缘抑制（去掉图像边缘的点）
    margin = 20
    binary[:margin, :] = 0
    binary[-margin:, :] = 0
    binary[:, :margin] = 0
    binary[:, -margin:] = 0
    cv2.imwrite(os.path.join(OUTPUT_DIR, "simple_edge_suppressed.png"), binary)

    # 4. 边缘检测
    edges = cv2.Canny(binary, 50, 150)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "simple_edges.png"), edges)

    # 5. 霍夫线段检测
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=20,
        minLineLength=30,
        maxLineGap=10
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

    return line_segments, binary

def cluster_by_angle(line_segments, bin_size=20):
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

def visualize(bev, binary, line_segments, directions):
    """可视化结果"""
    # 原始BEV
    bev_vis = (bev / bev.max() * 255).astype(np.uint8)
    bev_vis = cv2.applyColorMap(bev_vis, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "simple_bev.png"), bev_vis)

    # 线段可视化
    seg_vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
              (255, 255, 0), (255, 0, 255)]
    for i, seg in enumerate(line_segments):
        color = colors[i % len(colors)]
        cv2.line(seg_vis, (seg['x1'], seg['y1']),
                 (seg['x2'], seg['y2']), color, 2)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "simple_segments.png"), seg_vis)

    # 方向箭头
    dir_vis = seg_vis.copy()
    center = (dir_vis.shape[0] // 2, dir_vis.shape[1] // 2)
    length = 100
    for i, angle in enumerate(directions):
        rad = np.deg2rad(angle)
        x_end = int(center[0] + length * np.cos(rad))
        y_end = int(center[1] + length * np.sin(rad))
        cv2.arrowedLine(dir_vis, center, (x_end, y_end), colors[i % len(colors)], 3)
        cv2.putText(dir_vis, f"{angle:.1f}°", (x_end + 5, y_end),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i % len(colors)], 1)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "simple_directions.png"), dir_vis)

def main():
    ply_path = os.path.join(OUTPUT_DIR, "merged_pointcloud_clean.ply")

    print("=" * 60)
    print("简化版道路检测")
    print("=" * 60)

    # 1. 加载点云
    print("\n[1/3] 加载点云...")
    pts = load_pointcloud(ply_path)
    print(f"  点云规模: {len(pts):,} 个顶点")

    # 2. 生成BEV
    print("\n[2/3] 生成BEV...")
    bev = generate_bev(pts)

    # 3. 提取线段
    print("\n[3/3] 提取道路线段...")
    line_segments, binary = extract_road_segments(bev)
    print(f"  检测到 {len(line_segments)} 条线段")

    # 4. 角度聚类
    directions = cluster_by_angle(line_segments)
    print(f"  检测到 {len(directions)} 个方向: {[f'{d:.1f}°' for d in directions]}")

    # 5. 可视化
    visualize(bev, binary, line_segments, directions)

    # 6. 保存报告
    with open(os.path.join(OUTPUT_DIR, "simple_road_report.txt"), 'w') as f:
        f.write("简化版道路检测报告\n")
        f.write("=" * 30 + "\n\n")
        f.write(f"线段数: {len(line_segments)}\n")
        f.write(f"方向数: {len(directions)}\n")
        f.write("方向列表:\n")
        for d in directions:
            f.write(f"  - {d:.1f}°\n")
    print("\n✅ 报告已保存: simple_road_report.txt")

    print("\n" + "=" * 60)
    print("检测完成!")
    print("=" * 60)

if __name__ == "__main__":
    main()