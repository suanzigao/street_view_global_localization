"""
批量投影脚本

功能：
1. 对街景图像进行SAM3分割
2. 使用形态学操作优化地面mask（开运算去噪 + 闭运算填洞）
3. 保存SAM3所有中间结果
4. 投影生成BEV图像
5. 对BEV图像进行后处理（去除空洞、保留边缘）
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


def read_metadata(metadata_path):
    with open(metadata_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_intrinsics(width, height, fov_deg):
    fov = math.radians(fov_deg)
    fx = width / (2 * math.tan(fov / 2))
    fy = height / (2 * math.tan(fov / 2))
    cx = width / 2
    cy = height / 2
    return fx, fy, cx, cy


def pixel_to_world(x, y, fx, fy, cx, cy, heading_deg, pitch_deg, cam_x, cam_y, cam_z):
    yaw = -math.radians(heading_deg)
    pitch = math.radians(pitch_deg)
    
    u = (x - cx) / fx
    v = (y - cy) / fy
    
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    cos_pitch = math.cos(pitch)
    sin_pitch = math.sin(pitch)
    
    right_x = cos_yaw
    right_y = sin_yaw
    
    forward_x = -sin_yaw * cos_pitch
    forward_y = cos_yaw * cos_pitch
    forward_z = sin_pitch
    
    down_x = -sin_yaw * sin_pitch
    down_y = cos_yaw * sin_pitch
    down_z = -cos_pitch
    
    ray_world_x = u * right_x + v * down_x + forward_x
    ray_world_y = u * right_y + v * down_y + forward_y
    ray_world_z = u * 0 + v * down_z + forward_z
    
    if ray_world_z >= -1e-6:
        return None
    
    t = -cam_z / ray_world_z
    
    if t < 0:
        return None
    
    world_x = cam_x + t * ray_world_x
    world_y = cam_y + t * ray_world_y
    
    return (world_x, world_y, 0.0)


def normalize_mask(mask):
    mask_np = np.array(mask)
    if mask_np.max() > 1:
        mask_np = (mask_np > 127).astype(np.uint8) * 255
    else:
        mask_np = (mask_np * 255).astype(np.uint8)
    return mask_np

def save_sam3_intermediate_results(base_name, output_dir, image, result, ground_mask):
    os.makedirs(output_dir, exist_ok=True)
    
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_original.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    prompt_idx = 0
    for seg_result in result.get('results', []):
        if seg_result.get('success') and seg_result.get('masks'):
            for mask in seg_result.get('masks'):
                mask_normalized = normalize_mask(mask)
                cv2.imwrite(os.path.join(output_dir, f"{base_name}_prompt{prompt_idx}_mask.png"), mask_normalized)
                prompt_idx += 1
    
    ground_mask_normalized = normalize_mask(ground_mask)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_merged_mask.png"), ground_mask_normalized)
    
    masked_image = image.copy()
    masked_image[ground_mask_normalized == 0] = 0
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_masked_image.png"), cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR))


STEP_COLORS = [
    (255, 0, 0),      # step_0: 红色
    (0, 255, 0),      # step_1: 绿色
    (0, 0, 255),      # step_2: 蓝色
    (255, 255, 0),    # step_3: 黄色
    (0, 255, 255),    # step_4: 青色
    (255, 0, 255),    # step_5: 品红
    (255, 128, 0),    # step_6: 橙色
    (128, 0, 255),    # step_7: 紫色
    (255, 192, 203),  # step_8: 粉色
    (128, 128, 128),  # step_9: 灰色
    (0, 128, 255),    # step_10: 天蓝色（替代棕色，更明显）
]


def get_step_color(step_num):
    return STEP_COLORS[step_num % len(STEP_COLORS)]


def project_pointcloud(image_files, images_dir, output_dir, sam3_tool):
    all_points = []
    all_colors = []
    all_step_colors = []
    camera_positions = []
    
    for img_file in image_files:
        base_name = img_file.replace('_streetview.jpg', '')
        metadata_file = f"{base_name}_metadata.json"
        
        step_num = int(base_name.split('_')[1]) if '_' in base_name else 0
        
        image_path = os.path.join(images_dir, img_file)
        metadata_path = os.path.join(images_dir, metadata_file)
        
        if not os.path.exists(metadata_path):
            continue
        
        image_pil = Image.open(image_path).convert('RGB')
        image = np.array(image_pil)
        metadata = read_metadata(metadata_path)
        
        h, w = image.shape[:2]
        
        heading = metadata.get('heading', 0.0)
        pitch = metadata.get('pitch', 0.0)
        fov = metadata.get('fov', 60)
        relative_x = metadata.get('relative_x', 0.0)
        relative_y = metadata.get('relative_y', 0.0)
        cam_z = 2.5
        
        fx, fy, cx, cy = compute_intrinsics(w, h, fov)
        
        prompts = ["road", "street"]
        result = sam3_tool.get_segmented_images(image_pil, prompts)
        
        ground_mask = None
        for seg_result in result.get('results', []):
            if seg_result.get('success') and seg_result.get('masks'):
                for mask in seg_result.get('masks'):
                    if ground_mask is None:
                        ground_mask = mask.copy()
                    else:
                        ground_mask = np.logical_or(ground_mask, mask)
        
        if ground_mask is None:
            print(f"    ⚠️ {base_name}: SAM3未分割到道路，跳过投影")
            continue
        
        if ground_mask.max() <= 1:
            mask_255 = (ground_mask * 255).astype(np.uint8)
        else:
            mask_255 = ground_mask.astype(np.uint8)
        
        road_pixels = np.sum(mask_255 > 0)
        total_pixels = h * w
        road_ratio = road_pixels / total_pixels
        
        if road_ratio < 0.01:
            print(f"    ⚠️ {base_name}: 道路区域过小({road_ratio:.2%})，跳过投影")
            continue
        
        ground_mask_normalized = mask_255
        
        sam3_output_dir = os.path.join(output_dir, 'sam3_intermediate')
        save_sam3_intermediate_results(base_name, sam3_output_dir, image, result, ground_mask)
        
        points_3d = []
        colors = []
        
        max_distance = 200
        
        for y in range(h):
            for x in range(w):
                if ground_mask_normalized[y, x] == 0:
                    continue
                
                world_pt = pixel_to_world(x, y, fx, fy, cx, cy, heading, pitch, relative_x, relative_y, cam_z)
                
                if world_pt is not None:
                    dist = math.sqrt((world_pt[0] - relative_x)**2 + (world_pt[1] - relative_y)**2)
                    if dist < max_distance:
                        points_3d.append(world_pt)
                        colors.append(image[y, x])
        
        if points_3d:
            points_3d = np.array(points_3d)
            colors = np.array(colors)
            step_color = get_step_color(step_num)
            step_colors = np.full((len(points_3d), 3), step_color, dtype=np.uint8)
            all_points.append(points_3d)
            all_colors.append(colors)
            all_step_colors.append(step_colors)
            camera_positions.append({
                'step': step_num,
                'relative_x': relative_x,
                'relative_y': relative_y,
                'heading': heading,
                'pitch': pitch
            })
    
    return all_points, all_colors, all_step_colors, camera_positions


def optimize_bev(bev_image):
    gray = cv2.cvtColor(bev_image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    
    eroded = cv2.erode(binary, kernel_small, iterations=2)
    
    dilated = cv2.dilate(eroded, kernel_large, iterations=3)
    
    result = cv2.bitwise_and(bev_image, bev_image, mask=dilated)
    
    return result


def draw_camera_marker(bev_image, cam, x_min, y_min, padding, resolution):
    cam_x = cam['relative_x']
    cam_y = cam['relative_y']
    heading_deg = cam['heading']
    
    bev_height, bev_width = bev_image.shape[:2]
    
    px = int((cam_x - x_min + padding) * resolution)
    py = bev_height - 1 - int((cam_y - y_min + padding) * resolution)
    
    if 0 <= px < bev_width and 0 <= py < bev_height:
        cv2.circle(bev_image, (px, py), 8, (0, 255, 255), -1)
        cv2.circle(bev_image, (px, py), 12, (0, 255, 255), 2)
        
        arrow_len = 20
        end_x = int(px + arrow_len * math.sin(math.radians(heading_deg)))
        end_y = int(py - arrow_len * math.cos(math.radians(heading_deg)))
        cv2.arrowedLine(bev_image, (px, py), (end_x, end_y), (0, 255, 255), 2)
    
    return bev_image


def generate_bev(all_points, all_colors, output_path, camera_positions=None, all_step_colors=None):
    if not all_points:
        return None
    
    merged_points = np.vstack(all_points)
    merged_colors = np.vstack(all_colors)
    
    if all_step_colors:
        merged_step_colors = np.vstack(all_step_colors)
    else:
        merged_step_colors = None
    
    x_min = merged_points[:, 0].min()
    x_max = merged_points[:, 0].max()
    y_min = merged_points[:, 1].min()
    y_max = merged_points[:, 1].max()
    
    resolution = 10
    padding = 5
    
    bev_width = int((x_max - x_min + padding * 2) * resolution)
    bev_height = int((y_max - y_min + padding * 2) * resolution)
    
    bev_base = np.zeros((bev_height, bev_width, 3), dtype=np.uint8)
    
    for i in range(len(merged_points)):
        px = int((merged_points[i, 0] - x_min + padding) * resolution)
        py = bev_height - 1 - int((merged_points[i, 1] - y_min + padding) * resolution)
        
        if 0 <= px < bev_width and 0 <= py < bev_height:
            bev_base[py, px] = merged_colors[i]
    
    cv2.imwrite(output_path, cv2.cvtColor(bev_base, cv2.COLOR_RGB2BGR))
    
    if merged_step_colors is not None:
        bev_colorcoded = np.zeros((bev_height, bev_width, 3), dtype=np.uint8)
        for i in range(len(merged_points)):
            px = int((merged_points[i, 0] - x_min + padding) * resolution)
            py = bev_height - 1 - int((merged_points[i, 1] - y_min + padding) * resolution)
            
            if 0 <= px < bev_width and 0 <= py < bev_height:
                bev_colorcoded[py, px] = merged_step_colors[i]
        
        colorcoded_path = output_path.replace('.png', '_colorcoded.png')
        cv2.imwrite(colorcoded_path, cv2.cvtColor(bev_colorcoded, cv2.COLOR_RGB2BGR))
        
        alpha = 0.3
        bev_mixed = cv2.addWeighted(bev_base, 1 - alpha, bev_colorcoded, alpha, 0)
        mixed_path = output_path.replace('.png', '_colorcoded_mixed.png')
        cv2.imwrite(mixed_path, cv2.cvtColor(bev_mixed, cv2.COLOR_RGB2BGR))
    else:
        colorcoded_path = None
        mixed_path = None
    
    bev_optimized = optimize_bev(cv2.cvtColor(bev_base, cv2.COLOR_RGB2BGR))
    optimized_path = output_path.replace('.png', '_optimized.png')
    cv2.imwrite(optimized_path, bev_optimized)
    
    initial_px = int((0 - x_min + padding) * resolution)
    initial_py = bev_height - 1 - int((0 - y_min + padding) * resolution)
    
    bev_metadata = {
        'total_points': len(merged_points),
        'x_range': (x_min, x_max),
        'y_range': (y_min, y_max),
        'bev_size': (bev_width, bev_height),
        'padding': padding,
        'resolution': resolution,
        'initial_pixel_x': initial_px,
        'initial_pixel_y': initial_py,
        'optimized_path': optimized_path,
        'colorcoded_path': colorcoded_path,
        'mixed_path': mixed_path
    }
    
    output_dir = os.path.dirname(output_path)
    metadata_path = os.path.join(output_dir, 'bev_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(bev_metadata, f, indent=2)
    
    bev_metadata['bev_base'] = bev_base
    
    return bev_metadata


def main():
    import argparse
    parser = argparse.ArgumentParser(description='批量投影生成BEV')
    parser.add_argument('--point', type=str, default=None, help='指定单个点位目录名（默认处理所有点位）')
    args = parser.parse_args()

    import json
    batch_dir = 'manual_streetview_results'
    
    if args.point:
        point_dirs = [args.point] if os.path.isdir(os.path.join(batch_dir, args.point)) else []
    else:
        point_dirs = sorted([d for d in os.listdir(batch_dir) if os.path.isdir(os.path.join(batch_dir, d))])
    
    if not point_dirs:
        print("  ❌ 未找到任何点位")
        return
    
    sam3_tool = SAM3Tool()
    sam3_tool.initialize()
    
    results = []
    
    for point_dir in point_dirs:
        point_path = os.path.join(batch_dir, point_dir)
        images_dir = os.path.join(point_path, 'streetview_images')
        
        if not os.path.exists(images_dir):
            continue
        
        image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
        
        if not image_files:
            continue
        
        print(f"\n{'='*60}")
        print(f"=== 处理点位: {point_dir} ===")
        print(f"  图像数量: {len(image_files)}")
        
        output_dir = os.path.join(point_path, 'projection_output')
        os.makedirs(output_dir, exist_ok=True)
        
        all_points, all_colors, all_step_colors, camera_positions = project_pointcloud(image_files, images_dir, output_dir, sam3_tool)
        
        if not all_points:
            print("  ❌ 没有成功投影的图像")
            continue
        
        bev_path = os.path.join(output_dir, 'fused_bev.png')
        result = generate_bev(all_points, all_colors, bev_path, camera_positions, all_step_colors)
        
        if result:
            print(f"  ✅ BEV图像已保存")
            print(f"    总点数: {result['total_points']}")
            print(f"    X范围: {result['x_range'][0]:.2f} ~ {result['x_range'][1]:.2f}")
            print(f"    Y范围: {result['y_range'][0]:.2f} ~ {result['y_range'][1]:.2f}")
            print(f"    BEV尺寸: {result['bev_size'][0]}x{result['bev_size'][1]}")
            
            def save_three_versions(base_path, image, x_min, y_min, padding, resolution):
                cv2.imwrite(base_path, image)
                
                img_with_all = image.copy()
                if camera_positions:
                    for cam in camera_positions:
                        img_with_all = draw_camera_marker(img_with_all, cam, x_min, y_min, padding, resolution)
                path_all = base_path.replace('.png', '_with_cameras.png')
                cv2.imwrite(path_all, img_with_all)
                
                img_with_initial = image.copy()
                if camera_positions:
                    initial_cam = camera_positions[0]
                    img_with_initial = draw_camera_marker(img_with_initial, initial_cam, x_min, y_min, padding, resolution)
                path_initial = base_path.replace('.png', '_with_initial_camera.png')
                cv2.imwrite(path_initial, img_with_initial)
                
                return base_path, path_all, path_initial
            
            bev_base = result.get('bev_base')
            padding = result.get('padding', 5)
            resolution = result.get('resolution', 10)
            
            base_path = os.path.join(output_dir, 'fused_bev_base.png')
            cv2.imwrite(base_path, cv2.cvtColor(bev_base, cv2.COLOR_RGB2BGR))
            
            save_three_versions(
                os.path.join(output_dir, 'fused_bev.png'),
                cv2.cvtColor(bev_base, cv2.COLOR_RGB2BGR),
                result['x_range'][0], result['y_range'][0],
                padding, resolution
            )
            
            mixed_path = result.get('mixed_path')
            if mixed_path and os.path.exists(mixed_path):
                mixed_bev = cv2.imread(mixed_path)
                save_three_versions(
                    os.path.join(output_dir, 'fused_bev_colorcoded_mixed.png'),
                    mixed_bev,
                    result['x_range'][0], result['y_range'][0],
                    padding, resolution
                )
            
            optimized_path = result.get('optimized_path')
            if optimized_path and os.path.exists(optimized_path):
                opt_bev = cv2.imread(optimized_path)
                
                save_three_versions(
                    os.path.join(output_dir, 'fused_bev_optimized.png'),
                    opt_bev,
                    result['x_range'][0], result['y_range'][0],
                    padding, resolution
                )
                
                lowres_scale = 2 / 10
                lowres_width = int(opt_bev.shape[1] * lowres_scale)
                lowres_height = int(opt_bev.shape[0] * lowres_scale)
                lowres_bev = cv2.resize(opt_bev, (lowres_width, lowres_height), interpolation=cv2.INTER_AREA)
                
                lowres_resolution = resolution * lowres_scale
                
                initial_px_lowres = int(result['initial_pixel_x'] * lowres_scale)
                initial_py_lowres = int(result['initial_pixel_y'] * lowres_scale)
                
                lowres_metadata = {
                    'x_range': result['x_range'],
                    'y_range': result['y_range'],
                    'padding': padding,
                    'resolution': lowres_resolution,
                    'initial_pixel_x': initial_px_lowres,
                    'initial_pixel_y': initial_py_lowres
                }
                lowres_metadata_path = os.path.join(output_dir, 'bev_lowres_metadata.json')
                with open(lowres_metadata_path, 'w') as f:
                    json.dump(lowres_metadata, f, indent=2)
                
                save_three_versions(
                    os.path.join(output_dir, 'fused_bev_lowres.png'),
                    lowres_bev,
                    result['x_range'][0], result['y_range'][0],
                    padding, lowres_resolution
                )
                
                gray = cv2.cvtColor(lowres_bev, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
                
                coords = cv2.findNonZero(binary)
                if coords is not None:
                    x, y, w, h = cv2.boundingRect(coords)
                    
                    crop_padding = 20
                    
                    x_start = max(0, min(x - crop_padding, initial_px_lowres - crop_padding))
                    y_start = max(0, min(y - crop_padding, initial_py_lowres - crop_padding))
                    x_end = min(lowres_width, max(x + w + crop_padding, initial_px_lowres + crop_padding))
                    y_end = min(lowres_height, max(y + h + crop_padding, initial_py_lowres + crop_padding))
                    
                    cropped_bev = lowres_bev[y_start:y_end, x_start:x_end]
                    
                    cropped_x_min = result['x_range'][0] - padding + x_start / lowres_resolution
                    cropped_y_min = result['y_range'][0] - padding + y_start / lowres_resolution
                    
                    cropped_bev_height = cropped_bev.shape[0]
                    initial_px_cropped = initial_px_lowres - x_start
                    initial_py_cropped = initial_py_lowres - y_start
                    
                    cropped_metadata = {
                        'x_range': result['x_range'],
                        'y_range': result['y_range'],
                        'cropped_x_min': cropped_x_min,
                        'cropped_y_min': cropped_y_min,
                        'padding': padding,
                        'resolution': lowres_resolution,
                        'initial_pixel_x': initial_px_cropped,
                        'initial_pixel_y': initial_py_cropped
                    }
                    cropped_metadata_path = os.path.join(output_dir, 'bev_lowres_cropped_metadata.json')
                    with open(cropped_metadata_path, 'w') as f:
                        json.dump(cropped_metadata, f, indent=2)
                    
                    save_three_versions(
                        os.path.join(output_dir, 'fused_bev_lowres_cropped.png'),
                        cropped_bev,
                        cropped_x_min, cropped_y_min,
                        padding, lowres_resolution
                    )
                    print(f"  ✅ 裁剪后的低分辨率BEV图像已保存: {cropped_bev.shape[1]}x{cropped_bev.shape[0]}")
                else:
                    print(f"  ⚠️ BEV图像中没有道路区域")
            else:
                print(f"  ⚠️ 未找到优化后的BEV图像")
            
            optimized_bev_stats = None
            if optimized_path and os.path.exists(optimized_path):
                opt_bev = cv2.imread(optimized_path)
                opt_gray = cv2.cvtColor(opt_bev, cv2.COLOR_BGR2GRAY)
                _, opt_binary = cv2.threshold(opt_gray, 10, 255, cv2.THRESH_BINARY)
                
                opt_coords = cv2.findNonZero(opt_binary)
                if opt_coords is not None:
                    opt_x, opt_y, opt_w, opt_h = cv2.boundingRect(opt_coords)
                    
                    orig_resolution = 10
                    bev_x_min = result['x_range'][0] + opt_x / orig_resolution
                    bev_x_max = result['x_range'][0] + (opt_x + opt_w) / orig_resolution
                    bev_y_min = result['y_range'][0] + opt_y / orig_resolution
                    bev_y_max = result['y_range'][0] + (opt_y + opt_h) / orig_resolution
                    
                    bev_width_meters = bev_x_max - bev_x_min
                    bev_height_meters = bev_y_max - bev_y_min
                    
                    optimized_bev_stats = {
                        'bounding_box_pixel': (opt_x, opt_y, opt_w, opt_h),
                        'x_range_meters': (bev_x_min, bev_x_max),
                        'y_range_meters': (bev_y_min, bev_y_max),
                        'width_meters': bev_width_meters,
                        'height_meters': bev_height_meters,
                        'area_meters': bev_width_meters * bev_height_meters
                    }
    
            results.append({
                'point_dir': point_dir,
                'total_points': result['total_points'],
                'x_range': result['x_range'],
                'y_range': result['y_range'],
                'output_path': bev_path,
                'optimized_path': result['optimized_path'],
                'optimized_bev_stats': optimized_bev_stats
            })
    
    sam3_tool.close()
    
    print(f"\n{'='*60}")
    print(f"\n📊 批量投影完成")
    print(f"  成功处理点位: {len(results)}")
    
    for r in results:
        print(f"  - {r['point_dir']}: {r['total_points']}点, X={r['x_range'][0]:.1f}~{r['x_range'][1]:.1f}, Y={r['y_range'][0]:.1f}~{r['y_range'][1]:.1f}")
        if r['optimized_bev_stats']:
            stats = r['optimized_bev_stats']
            print(f"    优化后BEV覆盖: {stats['width_meters']:.1f}m × {stats['height_meters']:.1f}m = {stats['area_meters']:.1f}m²")
    
    import json
    stats_path = os.path.join(batch_dir, 'bev_coverage_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ BEV覆盖统计已保存: {stats_path}")


if __name__ == "__main__":
    main()
