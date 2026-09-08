"""
VGGT深度图 → 路面校准 → 招牌中心点BEV定位

流程:
1. 调用远程VGGT服务获取每张街景图像的相对深度图
2. SAM3分割路面(road)和招牌(shop sign, restaurant sign)
3. 路面像素 → 射线与Z=0平面求交 → 理论绝对深度
4. 路面像素的VGGT深度值 → 尺度因子k = median(d_abs / d_rel)
5. 招牌中心点 → VGGT深度 × k → 真实深度 → 反投影到3D世界坐标
6. 跨图像聚类 → BEV上绘制点标记
"""

import os
import sys
import json
import math
import base64
import io
import numpy as np
import cv2
import requests
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tools.sam3_tool import SAM3Tool

VGGT_SERVER = "http://127.0.0.1:18888"
TEXT_PROMPTS = ["shop sign", "restaurant sign", "storefront sign"]
PROMPT_COLORS = [(0, 255, 0), (255, 165, 0), (0, 0, 255)]
ROAD_PROMPTS = ["road", "street"]
MAX_DEPTH = 200.0
MIN_DEPTH = 0.5


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


def get_rotation_matrix(heading_deg, pitch_deg):
    yaw = -math.radians(heading_deg)
    pitch = math.radians(pitch_deg)

    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    cos_p, sin_p = math.cos(pitch), math.sin(pitch)

    right = np.array([cos_y, sin_y, 0.0])
    forward = np.array([-sin_y * cos_p, cos_y * cos_p, sin_p])
    down = np.array([-sin_y * sin_p, cos_y * sin_p, -cos_p])

    R = np.column_stack([right, down, forward])
    return R, right, forward, down


def compute_ray_direction(x, y, fx, fy, cx, cy, R):
    u = (x - cx) / fx
    v = (y - cy) / fy
    ray_cam = np.array([u, v, 1.0])
    ray_world = R @ ray_cam
    return ray_world / np.linalg.norm(ray_world)


def theoretical_ground_depth(x, y, fx, fy, cx, cy, R, cam_z):
    ray = compute_ray_direction(x, y, fx, fy, cx, cy, R)
    if ray[2] >= -1e-6:
        return None
    t = -cam_z / ray[2]
    if t < 0:
        return None
    return t


def pixel_to_world_with_depth(x, y, depth, fx, fy, cx, cy, R, cam_pos):
    u = (x - cx) / fx
    v = (y - cy) / fy
    ray_cam = np.array([u * depth, v * depth, depth])
    world_pt = R @ ray_cam + np.array(cam_pos)
    return world_pt


def call_vggt_infer(image_paths, server_url=VGGT_SERVER):
    imgs_b64 = []
    for p in image_paths:
        with open(p, 'rb') as f:
            imgs_b64.append(base64.b64encode(f.read()).decode())

    req = {"images": imgs_b64, "output_types": ["depth"]}
    r = requests.post(f"{server_url}/vggt/infer", json=req, timeout=300)
    result = r.json()

    if not result.get("success"):
        print(f"  ❌ VGGT infer failed: {result}")
        return None

    depth_raw = result.get("depth")
    if depth_raw is None:
        print(f"  ❌ VGGT returned no depth")
        return None

    depth_arr = np.array(depth_raw, dtype=np.float32)
    return depth_arr


def calibrate_depth_scale(road_mask, depth_rel, fx, fy, cx, cy, R, cam_z, target_h, target_w):
    """
    用路面像素校准VGGT相对深度图的尺度因子k
    
    k = median(理论绝对深度 / VGGT相对深度)
    
    返回: (k, calibration_stats_dict)
    """
    h, w = road_mask.shape[:2]
    rh, rw = depth_rel.shape[:2]
    
    if (rh, rw) != (h, w):
        depth_rel_full = cv2.resize(depth_rel, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        depth_rel_full = depth_rel.copy()

    valid_mask = road_mask > 0
    valid_pixels = np.where(valid_mask)
    
    if len(valid_pixels[0]) < 50:
        return None, {"reason": f"路面像素不足: {len(valid_pixels[0])}"}

    sample_size = min(3000, len(valid_pixels[0]))
    indices = np.random.choice(len(valid_pixels[0]), sample_size, replace=False)
    sample_ys = valid_pixels[0][indices]
    sample_xs = valid_pixels[1][indices]

    d_abs_list = []
    d_rel_list = []

    for i in range(len(sample_ys)):
        px = sample_xs[i]
        py = sample_ys[i]

        d_abs = theoretical_ground_depth(px, py, fx, fy, cx, cy, R, cam_z)
        if d_abs is None or d_abs < MIN_DEPTH or d_abs > MAX_DEPTH:
            continue

        d_rel = depth_rel_full[py, px]
        if d_rel < 1e-8 or not np.isfinite(d_rel):
            continue

        d_abs_list.append(d_abs)
        d_rel_list.append(d_rel)

    if len(d_abs_list) < 20:
        return None, {"reason": f"有效校准像素不足: {len(d_abs_list)}"}

    d_abs_arr = np.array(d_abs_list)
    d_rel_arr = np.array(d_rel_list)
    ratios = d_abs_arr / d_rel_arr
    k = float(np.median(ratios))

    d_abs_pred = k * d_rel_arr
    errors = np.abs(d_abs_pred - d_abs_arr)
    
    good_mask = errors < 10.0
    if np.sum(good_mask) > 50:
        k = float(np.median(d_abs_arr[good_mask] / d_rel_arr[good_mask]))

    final_errors = np.abs(k * d_rel_arr - d_abs_arr)
    
    stats = {
        "calibration_pixels": len(d_abs_list),
        "scale_factor_k": k,
        "error_median": float(np.median(final_errors)),
        "error_mean": float(np.mean(final_errors)),
        "error_max": float(np.max(final_errors)),
    }

    return k, stats


def detect_road_mask(image_pil, sam3_tool, target_size):
    result = sam3_tool.get_segmented_images(image_pil, ROAD_PROMPTS)
    
    h, w = target_size
    combined = np.zeros((h, w), dtype=np.uint8)
    
    for seg_result in result.get('results', []):
        if not seg_result.get('success') or not seg_result.get('masks'):
            continue
        for mask in seg_result['masks']:
            mask_np = np.array(mask)
            if mask_np.ndim == 3:
                mask_np = mask_np[0]
            if mask_np.max() <= 1:
                mask_np = (mask_np * 255).astype(np.uint8)
            else:
                mask_np = mask_np.astype(np.uint8)
            
            if mask_np.shape[:2] != (h, w):
                mask_img = Image.fromarray(mask_np, mode='L')
                mask_img = mask_img.resize((w, h), Image.NEAREST)
                mask_np = np.array(mask_img)
            
            combined = np.maximum(combined, mask_np)
    
    return combined


def detect_sign_objects(image_pil, sam3_tool, target_size):
    h, w = target_size
    results = []
    
    prompt_results = sam3_tool.get_segmented_images(image_pil, TEXT_PROMPTS)
    
    for idx, seg_result in enumerate(prompt_results.get('results', [])):
        prompt = seg_result.get('prompt', TEXT_PROMPTS[idx] if idx < len(TEXT_PROMPTS) else f"unknown_{idx}")
        if not seg_result.get('success') or not seg_result.get('masks'):
            continue
        
        masks = seg_result.get('masks', [])
        scores = seg_result.get('scores', [])
        
        for mask_idx, mask in enumerate(masks):
            mask_np = np.array(mask)
            if mask_np.ndim == 3:
                mask_np = mask_np[0]
            if mask_np.max() <= 1:
                mask_np = (mask_np * 255).astype(np.uint8)
            else:
                mask_np = mask_np.astype(np.uint8)
            
            if mask_np.shape[:2] != (h, w):
                mask_img = Image.fromarray(mask_np, mode='L')
                mask_img = mask_img.resize((w, h), Image.NEAREST)
                mask_np = np.array(mask_img)
            
            pixel_count = np.count_nonzero(mask_np)
            if pixel_count < 10:
                continue
            
            ys, xs = np.where(mask_np > 0)
            centroid_x = float(np.mean(xs))
            centroid_y = float(np.mean(ys))
            
            score = scores[mask_idx] if mask_idx < len(scores) else 0.5
            
            results.append({
                'prompt': prompt,
                'score': float(score),
                'centroid': {'x': centroid_x, 'y': centroid_y},
                'mask': mask_np,
            })
    
    return results


def deduplicate_masks(text_objects, overlap_threshold=0.5):
    if not text_objects:
        return []
    
    unique = []
    for obj in text_objects:
        mask = obj['mask']
        is_dup = False
        for existing in unique:
            existing_mask = existing['mask']
            intersection = np.logical_and(mask > 0, existing_mask > 0).sum()
            union = np.logical_or(mask > 0, existing_mask > 0).sum()
            if union > 0 and intersection / union > overlap_threshold:
                if obj['score'] > existing['score']:
                    unique.remove(existing)
                    unique.append(obj)
                is_dup = True
                break
        if not is_dup:
            unique.append(obj)
    return unique


def project_sign_with_depth(sign_obj, depth_rel, k, fx, fy, cx, cy, R, cam_pos, target_h, target_w):
    centroid_x = sign_obj['centroid']['x']
    centroid_y = sign_obj['centroid']['y']
    
    rh, rw = depth_rel.shape[:2]
    
    if (rh, rw) != (target_h, target_w):
        depth_scaled = cv2.resize(depth_rel, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    else:
        depth_scaled = depth_rel
    
    px = int(centroid_x)
    py = int(centroid_y)
    
    if not (0 <= px < depth_scaled.shape[1] and 0 <= py < depth_scaled.shape[0]):
        return None
    
    d_rel = float(depth_scaled[py, px])
    if d_rel < 1e-8 or not np.isfinite(d_rel):
        return None
    
    d_abs = k * d_rel
    if d_abs < MIN_DEPTH or d_abs > MAX_DEPTH:
        return None
    
    world_pt = pixel_to_world_with_depth(centroid_x, centroid_y, d_abs, fx, fy, cx, cy, R, cam_pos)
    
    return {
        'prompt': sign_obj['prompt'],
        'score': sign_obj['score'],
        'pixel_position': {'x': centroid_x, 'y': centroid_y},
        'relative_depth': d_rel,
        'absolute_depth': d_abs,
        'world_position': {'x': float(world_pt[0]), 'y': float(world_pt[1]), 'z': float(world_pt[2])},
        'camera_position': {'x': float(cam_pos[0]), 'y': float(cam_pos[1]), 'z': float(cam_pos[2])},
    }


def cluster_markers(markers, distance_threshold=5.0):
    if not markers:
        return []
    
    clusters = []
    for marker in markers:
        wp = marker['world_position']
        best_cluster = None
        best_dist = float('inf')
        
        for cluster in clusters:
            cx = np.mean([c['world_position']['x'] for c in cluster])
            cy = np.mean([c['world_position']['y'] for c in cluster])
            dist = math.sqrt((wp['x'] - cx)**2 + (wp['y'] - cy)**2)
            if dist < best_dist:
                best_dist = dist
                best_cluster = cluster
        
        if best_cluster is not None and best_dist < distance_threshold:
            best_cluster.append(marker)
        else:
            clusters.append([marker])
    
    result = []
    for cluster in clusters:
        if len(cluster) == 1:
            m = cluster[0]
            m['num_observations'] = 1
            result.append(m)
        else:
            xs = [m['world_position']['x'] for m in cluster]
            ys = [m['world_position']['y'] for m in cluster]
            zs = [m['world_position']['z'] for m in cluster]
            scores = [m['score'] for m in cluster]
            total_score = sum(scores)
            if total_score > 0:
                wx = sum(x * s for x, s in zip(xs, scores)) / total_score
                wy = sum(y * s for y, s in zip(ys, scores)) / total_score
                wz = sum(z * s for z, s in zip(zs, scores)) / total_score
            else:
                wx = np.mean(xs)
                wy = np.mean(ys)
                wz = np.mean(zs)
            
            best_marker = max(cluster, key=lambda m: m['score'])
            result.append({
                'prompt': best_marker['prompt'],
                'score': best_marker['score'],
                'world_position': {'x': float(wx), 'y': float(wy), 'z': float(wz)},
                'num_observations': len(cluster),
                'source_image': best_marker.get('source_image', 'multi_avg'),
                'absolute_depth': best_marker.get('absolute_depth'),
            })
    
    return result


def world_to_bev(world_x, world_y, x_min, y_min, padding, resolution, bev_height):
    px = int((world_x - x_min + padding) * resolution)
    py = bev_height - 1 - int((world_y - y_min + padding) * resolution)
    return px, py


def draw_markers_on_bev(bev_image, markers, x_min, y_min, padding, resolution):
    bev_height, bev_width = bev_image.shape[:2]
    marked = bev_image.copy()
    
    drawn = 0
    for marker in markers:
        wp = marker['world_position']
        px, py = world_to_bev(wp['x'], wp['y'], x_min, y_min, padding, resolution, bev_height)
        
        if not (0 <= px < bev_width and 0 <= py < bev_height):
            continue
        
        prompt = marker.get('prompt', 'unknown')
        color_idx = TEXT_PROMPTS.index(prompt) if prompt in TEXT_PROMPTS else 0
        color = PROMPT_COLORS[color_idx % len(PROMPT_COLORS)]
        
        n_obs = marker.get('num_observations', 1)
        radius = 6 + n_obs * 2
        
        cv2.circle(marked, (px, py), radius, color, -1)
        cv2.circle(marked, (px, py), radius + 3, color, 2)
        
        label = f"{prompt.split()[0]}({n_obs})"
        cv2.putText(marked, label, (px + radius + 5, py + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        drawn += 1
    
    return marked, drawn


def process_point(point_dir, sam3_tool, bev_metadata):
    images_dir = os.path.join(point_dir, 'streetview_images')
    output_dir = os.path.join(point_dir, 'text_output_vggt_depth')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'depth_vis'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'calibration'), exist_ok=True)
    
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('_streetview.jpg')])
    if not image_files:
        print("  ⚠️ 没有找到街景图像")
        return
    
    x_min = bev_metadata['x_range'][0]
    y_min = bev_metadata['y_range'][0]
    padding = bev_metadata['padding']
    resolution = bev_metadata['resolution']
    
    bev_path = bev_metadata.get('optimized_path', '')
    bev_image = None
    if bev_path and os.path.exists(bev_path):
        bev_image = cv2.imread(bev_path)
    if bev_image is None:
        for cand in ['fused_bev_optimized.png', 'fused_bev.png', 'fused_bev_base.png']:
            p = os.path.join(point_dir, 'projection_output', cand)
            if os.path.exists(p):
                bev_image = cv2.imread(p)
                if bev_image is not None:
                    break
    
    if bev_image is None:
        print("  ⚠️ 没有找到BEV图像")
        return
    
    bev_height = bev_image.shape[0]
    
    print(f"\n  调用VGGT获取 {len(image_files)} 张图像的深度图...")
    image_paths = [os.path.join(images_dir, f) for f in image_files]
    depth_result = call_vggt_infer(image_paths)
    if depth_result is None:
        print("  ❌ VGGT深度获取失败")
        return
    
    all_markers = []
    calibration_records = []
    
    for img_idx, img_file in enumerate(image_files):
        base_name = img_file.replace('_streetview.jpg', '')
        metadata_file = f"{base_name}_metadata.json"
        metadata_path = os.path.join(images_dir, metadata_file)
        
        if not os.path.exists(metadata_path):
            continue
        
        print(f"  [{img_idx+1}/{len(image_files)}] 处理 {img_file}...")
        
        image_pil = Image.open(os.path.join(images_dir, img_file)).convert('RGB')
        image_np = np.array(image_pil)
        h, w = image_np.shape[:2]
        
        metadata = read_metadata(metadata_path)
        
        heading = metadata.get('heading', 0.0)
        pitch = metadata.get('pitch', 0.0)
        fov = metadata.get('fov', 60)
        relative_x = metadata.get('relative_x', 0.0)
        relative_y = metadata.get('relative_y', 0.0)
        cam_z = 2.5
        
        fx, fy, cx, cy = compute_intrinsics(w, h, fov)
        R, right, forward, down = get_rotation_matrix(heading, pitch)
        cam_pos = (relative_x, relative_y, cam_z)
        
        depth_rel = depth_result[img_idx]
        drh, drw = depth_rel.shape[:2]
        if (drh, drw) != (h, w):
            depth_vis = cv2.resize(depth_rel, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            depth_vis = depth_rel.copy()
        
        depth_vis_save = (depth_vis - depth_vis.min()) / (depth_vis.max() - depth_vis.min() + 1e-8)
        depth_vis_save = (depth_vis_save * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, 'depth_vis', f"{base_name}_depth_rel.png"), depth_vis_save)
        
        print(f"    SAM3分割路面...")
        road_mask = detect_road_mask(image_pil, sam3_tool, (h, w))
        cv2.imwrite(os.path.join(output_dir, 'calibration', f"{base_name}_road_mask.png"), road_mask)
        
        print(f"    路面校准尺度因子...")
        k, cal_stats = calibrate_depth_scale(road_mask, depth_rel, fx, fy, cx, cy, R, cam_z, h, w)
        
        if k is None:
            print(f"    ⚠️ 校准失败: {cal_stats.get('reason', 'unknown')}")
            continue
        
        print(f"    校准结果: k={k:.4f}, 误差中位数={cal_stats['error_median']:.2f}m")
        calibration_records.append({"image": img_file, "k": k, **cal_stats})
        
        depth_calibrated = depth_vis * k
        depth_cal_save = np.clip(depth_calibrated, 0, 100)
        depth_cal_save = (depth_cal_save / depth_cal_save.max() * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, 'calibration', f"{base_name}_depth_calibrated.png"), depth_cal_save)
        
        print(f"    SAM3分割招牌...")
        sign_objects = detect_sign_objects(image_pil, sam3_tool, (h, w))
        sign_objects = deduplicate_masks(sign_objects)
        
        if not sign_objects:
            print(f"    未检测到招牌")
            continue
        
        print(f"    检测到 {len(sign_objects)} 个招牌")
        
        for sign in sign_objects:
            marker = project_sign_with_depth(
                sign, depth_rel, k, fx, fy, cx, cy, R, cam_pos, h, w
            )
            if marker is not None:
                marker['source_image'] = img_file
                all_markers.append(marker)
        
        img_markers_dir = os.path.join(output_dir, "sign_overlays")
        os.makedirs(img_markers_dir, exist_ok=True)
        overlay = image_np.copy()
        for sign in sign_objects:
            mask = sign['mask']
            color_idx = TEXT_PROMPTS.index(sign['prompt']) if sign['prompt'] in TEXT_PROMPTS else 0
            color = PROMPT_COLORS[color_idx % len(PROMPT_COLORS)]
            for c in range(3):
                overlay[:, :, c] = np.where(mask > 0, color[c], overlay[:, :, c])
            cx_s, cy_s = int(sign['centroid']['x']), int(sign['centroid']['y'])
            cv2.drawMarker(overlay, (cx_s, cy_s), color, cv2.MARKER_CROSS, 20, 2)
        overlay = cv2.addWeighted(overlay, 0.6, image_np, 0.4, 0)
        cv2.imwrite(os.path.join(img_markers_dir, f"{base_name}_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    
    if not all_markers:
        print("  ⚠️ 没有成功检测到任何招牌标记")
        return
    
    print(f"\n  跨图像聚类 ({len(all_markers)} → ...)...")
    final_markers = cluster_markers(all_markers, distance_threshold=5.0)
    
    multi = sum(1 for m in final_markers if m.get('num_observations', 1) > 1)
    print(f"  聚类后: {len(final_markers)} 个 (多观测={multi}, 单观测={len(final_markers)-multi})")
    
    with open(os.path.join(output_dir, 'calibration_records.json'), 'w') as f:
        json.dump(calibration_records, f, indent=2)
    
    markers_json_path = os.path.join(output_dir, 'sign_markers.json')
    with open(markers_json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'method': 'vggt_depth_road_calibration',
            'total_markers': len(final_markers),
            'markers': final_markers,
        }, f, indent=2, ensure_ascii=False)
    
    bev_marked, drawn = draw_markers_on_bev(bev_image, final_markers, x_min, y_min, padding, resolution)
    bev_markers_path = os.path.join(output_dir, 'bev_with_sign_markers.png')
    cv2.imwrite(bev_markers_path, bev_marked)
    
    print(f"\n  ✅ BEV标记图已保存: {bev_markers_path}")
    print(f"     共 {len(final_markers)} 个标记, {drawn} 个在BEV范围内")
    
    for m in final_markers:
        wp = m['world_position']
        n_obs = m.get('num_observations', 1)
        print(f"    - {m['prompt']}: ({wp['x']:.2f}, {wp['y']:.2f})m, z={wp['z']:.2f}m, n_obs={n_obs}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='VGGT深度+路面校准 → 招牌BEV定位')
    parser.add_argument('--input', type=str, required=True, help='街景点位目录')
    parser.add_argument('--vggt-server', type=str, default=VGGT_SERVER, help='VGGT服务地址')
    args = parser.parse_args()
    
    point_dir = args.input
    
    bev_metadata_path = os.path.join(point_dir, 'projection_output', 'bev_metadata.json')
    if not os.path.exists(bev_metadata_path):
        print(f"❌ 未找到BEV元数据: {bev_metadata_path}")
        return
    
    with open(bev_metadata_path, 'r') as f:
        bev_metadata = json.load(f)
    
    print(f"BEV元数据: X={bev_metadata['x_range']}, Y={bev_metadata['y_range']}, res={bev_metadata['resolution']}px/m")
    
    try:
        r = requests.get(f"{args.vggt_server}/health", timeout=5)
        health = r.json()
        print(f"VGGT服务: {health}")
    except Exception as e:
        print(f"❌ VGGT服务不可用: {e}")
        print(f"   请确认已启动服务并建立SSH隧道:")
        print(f"   ssh -CNg -L 18888:127.0.0.1:18888 -p 37859 root@connect.westb.seetacloud.com")
        return
    
    sam3_tool = SAM3Tool()
    sam3_tool.initialize()
    
    try:
        process_point(point_dir, sam3_tool, bev_metadata)
    finally:
        sam3_tool.close()


if __name__ == '__main__':
    main()
