"""从点云BEV视角提取道路方向"""
import numpy as np
import cv2
from PIL import Image
import os

def load_pointcloud(ply_path):
    """加载PLY点云文件"""
    print(f"Loading PLY: {ply_path}")
    with open(ply_path) as f:
        lines = f.readlines()
    
    # 解析header
    header_end = 0
    n_vertices = 0
    for i, line in enumerate(lines):
        if line.strip() == "end_header":
            header_end = i + 1
            break
        if "element vertex" in line:
            n_vertices = int(line.strip().split()[-1])
    
    print(f"  {n_vertices:,} vertices")
    
    # 读取点
    pts = np.zeros((n_vertices, 3), dtype=np.float32)
    for i, line in enumerate(lines[header_end:]):
        parts = line.strip().split()
        if len(parts) >= 3:
            pts[i] = [float(parts[0]), float(parts[1]), float(parts[2])]
    
    return pts

def generate_bev_points(pts, resolution=512):
    """生成BEV点云（仅保留地面附近的点）"""
    # 过滤地面点（Y坐标接近0的点）
    ground_mask = (pts[:, 1] > -0.3) & (pts[:, 1] < 1.5)
    ground_pts = pts[ground_mask]
    
    # 计算范围（使用99%分位数去除异常值）
    x_min, x_max = np.percentile(ground_pts[:, 0], [1, 99])
    z_min, z_max = np.percentile(ground_pts[:, 2], [1, 99])
    
    range_x = x_max - x_min
    range_z = z_max - z_min
    max_range = max(range_x, range_z)
    
    # 保持正方形
    cx = (x_min + x_max) / 2
    cz = (z_min + z_max) / 2
    x_min, x_max = cx - max_range/2, cx + max_range/2
    z_min, z_max = cz - max_range/2, cz + max_range/2
    
    # 离散化为像素坐标
    xi = ((ground_pts[:, 0] - x_min) / (x_max - x_min) * (resolution - 1)).astype(int)
    zi = ((ground_pts[:, 2] - z_min) / (z_max - z_min) * (resolution - 1)).astype(int)
    
    # 边界检查
    xi = np.clip(xi, 0, resolution - 1)
    zi = np.clip(zi, 0, resolution - 1)
    
    # 创建BEV图像（统计每个像素的点密度）
    bev = np.zeros((resolution, resolution), dtype=np.float32)
    np.add.at(bev, (zi, xi), 1)
    
    return bev, (x_min, x_max, z_min, z_max)

def extract_road_direction(bev):
    """从BEV图像提取道路方向"""
    # 1. 二值化（基于点密度）
    threshold = np.percentile(bev, 70)
    binary = (bev > threshold).astype(np.uint8) * 255
    
    # 2. 形态学操作去除噪声
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # 3. 边缘检测
    edges = cv2.Canny(binary, 50, 150)
    
    # 4. 霍夫直线检测
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=50)
    
    if lines is None:
        # 如果霍夫变换失败，使用PCA分析点云方向
        return estimate_direction_by_pca(bev)
    
    # 5. 统计最主要的方向
    angles = []
    for line in lines:
        rho, theta = line[0]
        angles.append(theta)
    
    # 计算平均角度（考虑周期性）
    angles = np.array(angles)
    x_mean = np.cos(angles).mean()
    y_mean = np.sin(angles).mean()
    main_angle = np.arctan2(y_mean, x_mean)
    
    # 转换为角度（0-360度）
    main_angle_deg = np.rad2deg(main_angle) % 360
    
    # 可视化结果
    visualize_result(binary, edges, lines, main_angle_deg)
    
    return main_angle_deg

def estimate_direction_by_pca(bev):
    """使用PCA估计主方向"""
    # 获取非零点的坐标
    y_coords, x_coords = np.where(bev > 0)
    points = np.column_stack((x_coords, y_coords)).astype(np.float32)
    
    if len(points) < 2:
        return 0.0  # 默认方向
    
    # 中心化
    points -= points.mean(axis=0)
    
    # 计算协方差矩阵
    cov_matrix = np.cov(points.T)
    
    # 特征值分解
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # 主方向对应最大特征值的特征向量
    main_dir = eigenvectors[:, np.argmax(eigenvalues)]
    
    # 计算角度
    angle = np.arctan2(main_dir[1], main_dir[0])
    angle_deg = np.rad2deg(angle) % 360
    
    return angle_deg

def visualize_result(binary, edges, lines, direction):
    """可视化处理结果"""
    # 创建彩色图像
    result = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    
    # 绘制检测到的直线
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 1)
    
    # 保存可视化结果
    output_dir = "/home/suanzigao/PycharmProjects/street_view_global_localization/batch_streetview_results/2026-06-01_14-36-04_batch_run/2026-06-01_14-36-04_Qc7f8UCkWQSdu_0Wc7BDtQ_lat55_7132_lng12_5929/streetview_images/vggt_remote_output"
    cv2.imwrite(os.path.join(output_dir, "road_direction_bev.png"), result)
    cv2.imwrite(os.path.join(output_dir, "road_direction_edges.png"), edges)
    
    print(f"  可视化结果已保存")

def main():
    ply_path = "/home/suanzigao/PycharmProjects/street_view_global_localization/batch_streetview_results/2026-06-01_14-36-04_batch_run/2026-06-01_14-36-04_Qc7f8UCkWQSdu_0Wc7BDtQ_lat55_7132_lng12_5929/streetview_images/vggt_remote_output/merged_pointcloud_clean.ply"
    
    # 1. 加载点云
    pts = load_pointcloud(ply_path)
    
    # 2. 生成BEV
    print("\nGenerating BEV...")
    bev, bounds = generate_bev_points(pts)
    print(f"  BEV范围: X[{bounds[0]:.2f}, {bounds[1]:.2f}], Z[{bounds[2]:.2f}, {bounds[3]:.2f}]")
    
    # 3. 提取道路方向
    print("\nExtracting road direction...")
    direction = extract_road_direction(bev)
    
    # 4. 输出结果
    print(f"\n✅ 道路方向估计完成")
    print(f"   主方向角度: {direction:.1f}°")
    print(f"   方向描述: {get_direction_description(direction)}")

def get_direction_description(angle):
    """将角度转换为方向描述"""
    angle = angle % 360
    if angle < 22.5 or angle >= 337.5:
        return "正北 (North)"
    elif angle < 67.5:
        return "东北 (Northeast)"
    elif angle < 112.5:
        return "正东 (East)"
    elif angle < 157.5:
        return "东南 (Southeast)"
    elif angle < 202.5:
        return "正南 (South)"
    elif angle < 247.5:
        return "西南 (Southwest)"
    elif angle < 292.5:
        return "正西 (West)"
    else:
        return "西北 (Northwest)"

if __name__ == "__main__":
    main()
