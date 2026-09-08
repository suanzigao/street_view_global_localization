"""
Text-to-BEV 脚本

功能：
1. 使用SAM3的VLM能力判断街景图像中是否存在文字物体
2. 使用SAM3分割文字承载物体（路牌、招牌、广告牌、门牌等）
3. 将文字物体的像素坐标反投影到3D世界坐标
   - 优先使用mask底部像素投影到地面（Z=0）
   - 若无地面交点，则使用物体估算高度投影
4. 映射到BEV（鸟瞰图）坐标并标记
5. 跨prompt去重，优化结果
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


TEXT_PROMPTS = [
    "shop sign",
    "restaurant sign",
]

PROMPT_COLORS = [
    (0, 255, 0),
    (255, 165, 0),
]

OBJECT_HEIGHTS = {
    "shop sign": 4.0,
    "restaurant sign": 4.0,
}

MAX_PROJECT_DISTANCE = 150


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


def compute_ray_direction(x, y, fx, fy, cx, cy, heading_deg, pitch_deg):
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

    ray_x = u * right_x + v * down_x + forward_x
    ray_y = u * right_y + v * down_y + forward_y
    ray_z = v * 0 + v * down_z + forward_z

    return (ray_x, ray_y, ray_z)


def pixel_to_world(x, y, fx, fy, cx, cy, heading_deg, pitch_deg, cam_x, cam_y, cam_z):
    ray_x, ray_y, ray_z = compute_ray_direction(x, y, fx, fy, cx, cy, heading_deg, pitch_deg)

    if ray_z >= -1e-6:
        return None

    t = -cam_z / ray_z

    if t < 0:
        return None

    world_x = cam_x + t * ray_x
    world_y = cam_y + t * ray_y

    return (world_x, world_y, 0.0)


def pixel_to_world_at_height(x, y, fx, fy, cx, cy, heading_deg, pitch_deg, cam_x, cam_y, cam_z, target_z):
    ray_x, ray_y, ray_z = compute_ray_direction(x, y, fx, fy, cx, cy, heading_deg, pitch_deg)

    if abs(ray_z) < 1e-6:
        return None

    t = (target_z - cam_z) / ray_z

    if t < 0:
        return None

    world_x = cam_x + t * ray_x
    world_y = cam_y + t * ray_y

    return (world_x, world_y, target_z)


def detect_text_objects(image_pil, sam3_tool):
    results = []
    image_np = np.array(image_pil)
    h, w = image_np.shape[:2]

    prompt_results = sam3_tool.get_segmented_images(image_pil, TEXT_PROMPTS)

    for idx, seg_result in enumerate(prompt_results.get('results', [])):
        prompt = seg_result.get('prompt', TEXT_PROMPTS[idx] if idx < len(TEXT_PROMPTS) else f"unknown_{idx}")
        if not seg_result.get('success') or not seg_result.get('masks'):
            continue

        masks = seg_result.get('masks', [])
        bboxes = seg_result.get('bboxes', [])
        scores = seg_result.get('scores', [])

        for mask_idx, mask in enumerate(masks):
            mask_np = np.array(mask)
            if mask_np.ndim == 3 and mask_np.shape[0] == 1:
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

            y_max = float(np.max(ys))
            y_min = float(np.min(ys))

            bbox = bboxes[mask_idx] if mask_idx < len(bboxes) else [0, 0, w, h]
            score = scores[mask_idx] if mask_idx < len(scores) else 0.5

            results.append({
                'prompt': prompt,
                'score': float(score),
                'centroid': {'x': centroid_x, 'y': centroid_y},
                'bottom_y': y_max,
                'top_y': y_min,
                'bbox': bbox,
                'pixel_count': pixel_count,
                'mask': mask_np,
            })

    return results


def deduplicate_text_objects(text_objects, overlap_threshold=0.5):
    if not text_objects:
        return []

    unique = []

    for obj in text_objects:
        mask = obj['mask']
        is_duplicate = False

        for existing in unique:
            existing_mask = existing['mask']
            intersection = np.logical_and(mask > 0, existing_mask > 0).sum()
            union = np.logical_or(mask > 0, existing_mask > 0).sum()

            if union > 0 and intersection / union > overlap_threshold:
                if obj['score'] > existing['score']:
                    unique.remove(existing)
                    unique.append(obj)
                is_duplicate = True
                break

        if not is_duplicate:
            unique.append(obj)

    return unique


def deduplicate_markers_by_world_pos(markers, distance_threshold=5.0):
    if not markers:
        return []

    unique = []

    for marker in markers:
        world_pt = marker.get('elevated_world')
        if world_pt is None:
            world_pt = marker.get('ground_world')
        if world_pt is None:
            continue

        is_duplicate = False
        for existing in unique:
            ex_pt = existing.get('elevated_world')
            if ex_pt is None:
                ex_pt = existing.get('ground_world')
            if ex_pt is None:
                continue

            dist = math.sqrt((world_pt[0] - ex_pt[0]) ** 2 + (world_pt[1] - ex_pt[1]) ** 2)
            if dist < distance_threshold:
                if marker.get('score', 0) > existing.get('score', 0):
                    unique.remove(existing)
                    unique.append(marker)
                is_duplicate = True
                break

        if not is_duplicate:
            unique.append(marker)

    return unique


def cluster_and_average_markers(markers, distance_threshold=5.0):
    """
    按世界坐标聚类, 同一簇使用加权平均位置
    
    返回: 最终 markers 列表
    """
    if not markers:
        return []

    clusters = []

    for marker in markers:
        world_pt = marker.get('elevated_world')
        if world_pt is None:
            world_pt = marker.get('ground_world')
        if world_pt is None:
            continue

        best_cluster = None
        best_dist = float('inf')

        for cluster in clusters:
            cluster_pts = []
            for m in cluster:
                pt = m.get('elevated_world')
                if pt is None:
                    pt = m.get('ground_world')
                if pt is not None:
                    cluster_pts.append(pt)

            if not cluster_pts:
                continue

            cx = np.mean([p[0] for p in cluster_pts])
            cy = np.mean([p[1] for p in cluster_pts])

            dist = math.sqrt((world_pt[0] - cx) ** 2 + (world_pt[1] - cy) ** 2)
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
            pts = []
            scores = []
            for m in cluster:
                pt = m.get('elevated_world')
                if pt is None:
                    pt = m.get('ground_world')
                if pt is not None:
                    pts.append(pt)
                    scores.append(m.get('score', 0.5))

            best_marker = max(cluster, key=lambda m: m.get('score', 0))

            if pts:
                total_score = sum(scores)
                if total_score > 0:
                    wx = sum(pt[0] * s for pt, s in zip(pts, scores)) / total_score
                    wy = sum(pt[1] * s for pt, s in zip(pts, scores)) / total_score
                    wz = sum(pt[2] * s for pt, s in zip(pts, scores)) / total_score
                else:
                    wx = np.mean([pt[0] for pt in pts])
                    wy = np.mean([pt[1] for pt in pts])
                    wz = np.mean([pt[2] for pt in pts])

                result.append({
                    'prompt': best_marker.get('prompt', 'sign'),
                    'pixel_position': best_marker.get('pixel_position', {}),
                    'ground_world': (float(wx), float(wy), 0.0),
                    'elevated_world': (float(wx), float(wy), float(wz)),
                    'estimated_height': best_marker.get('estimated_height', 3.0),
                    'source_image': best_marker.get('source_image', 'multi_avg'),
                    'score': best_marker.get('score', 0),
                    'num_observations': len(cluster),
                })
            else:
                best_marker['num_observations'] = len(cluster)
                result.append(best_marker)

    return result


def project_text_object(text_obj, fx, fy, cx, cy, heading, pitch, cam_x, cam_y, cam_z):
    prompt = text_obj['prompt']
    mask = text_obj['mask']

    height_est = OBJECT_HEIGHTS.get(prompt, 3.0)

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None

    y_max = np.max(ys)
    y_threshold = y_max - max(1, int((y_max - np.min(ys)) * 0.1))
    bottom_indices = np.where(ys >= y_threshold)[0]

    ground_projections = []
    for idx in bottom_indices:
        px = int(xs[idx])
        py = int(ys[idx])
        pt = pixel_to_world(px, py, fx, fy, cx, cy, heading, pitch, cam_x, cam_y, cam_z)
        if pt is not None:
            dist = math.sqrt((pt[0] - cam_x) ** 2 + (pt[1] - cam_y) ** 2)
            if dist < MAX_PROJECT_DISTANCE:
                ground_projections.append(pt)

    if ground_projections:
        gx = float(np.mean([p[0] for p in ground_projections]))
        gy = float(np.mean([p[1] for p in ground_projections]))
        ground_pt = (gx, gy, 0.0)
    else:
        ground_pt = None

    centroid_x = text_obj['centroid']['x']
    centroid_y = text_obj['centroid']['y']

    ray_x, ray_y, ray_z = compute_ray_direction(centroid_x, centroid_y, fx, fy, cx, cy, heading, pitch)

    elevated_pt = pixel_to_world_at_height(
        centroid_x, centroid_y, fx, fy, cx, cy, heading, pitch, cam_x, cam_y, cam_z, height_est
    )

    if elevated_pt is not None:
        dist = math.sqrt((elevated_pt[0] - cam_x) ** 2 + (elevated_pt[1] - cam_y) ** 2)
        if dist >= MAX_PROJECT_DISTANCE:
            elevated_pt = None

    return {
        'prompt': prompt,
        'pixel_position': {'x': centroid_x, 'y': centroid_y},
        'ground_world': ground_pt,
        'elevated_world': elevated_pt,
        'estimated_height': height_est,
        'camera_position': (cam_x, cam_y, cam_z),
        'ray_direction': (ray_x, ray_y, ray_z),
    }


def world_to_bev(world_x, world_y, x_min, y_min, padding, resolution, bev_height):
    px = int((world_x - x_min + padding) * resolution)
    py = bev_height - 1 - int((world_y - y_min + padding) * resolution)
    return px, py


def draw_text_markers(bev_image, text_markers, x_min, y_min, padding, resolution):
    bev_height, bev_width = bev_image.shape[:2]
    marked = bev_image.copy()

    drawn_count = 0
    for marker in text_markers:
        world_pt = marker.get('ground_world')
        if world_pt is None:
            world_pt = marker.get('elevated_world')
        if world_pt is None:
            continue

        px, py = world_to_bev(world_pt[0], world_pt[1], x_min, y_min, padding, resolution, bev_height)

        if not (0 <= px < bev_width and 0 <= py < bev_height):
            continue

        prompt = marker.get('prompt', 'unknown')
        color_idx = TEXT_PROMPTS.index(prompt) if prompt in TEXT_PROMPTS else 0
        color = PROMPT_COLORS[color_idx % len(PROMPT_COLORS)]

        cv2.circle(marked, (px, py), 6, color, -1)
        cv2.circle(marked, (px, py), 10, color, 2)

        label = prompt
        text_x = min(px + 12, bev_width - len(label) * 7 - 5)
        text_y = max(py - 8, 12)
        cv2.putText(marked, label, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        drawn_count += 1

    return marked, drawn_count


def save_text_detection_visualizations(image_np, text_objects, save_dir, base_name):
    os.makedirs(save_dir, exist_ok=True)
    h, w = image_np.shape[:2]

    for obj_idx, obj in enumerate(text_objects):
        prompt = obj['prompt']
        mask = obj['mask']

        overlay = image_np.copy()
        color_idx = TEXT_PROMPTS.index(prompt) if prompt in TEXT_PROMPTS else 0
        color = PROMPT_COLORS[color_idx % len(PROMPT_COLORS)]

        for c in range(3):
            channel = overlay[:, :, c].copy()
            channel[mask > 0] = color[c]
            overlay[:, :, c] = channel

        alpha = 0.4
        overlay = cv2.addWeighted(overlay, 1 - alpha, image_np, alpha, 0)

        bbox = obj.get('bbox', [0, 0, w, h])
        if len(bbox) == 4:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        label = f"{prompt} ({obj['score']:.2f})"
        cv2.putText(overlay, label, (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        save_path = os.path.join(save_dir, f"{base_name}_text_{obj_idx}_{prompt.replace(' ', '_')}.png")
        cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    if text_objects:
        combined = image_np.copy()
        for obj in text_objects:
            mask = obj['mask']
            prompt = obj['prompt']
            color_idx = TEXT_PROMPTS.index(prompt) if prompt in TEXT_PROMPTS else 0
            color = PROMPT_COLORS[color_idx % len(PROMPT_COLORS)]

            for c in range(3):
                channel = combined[:, :, c].copy()
                channel[mask > 0] = color[c]
                combined[:, :, c] = channel

        save_path = os.path.join(save_dir, f"{base_name}_text_combined.png")
        cv2.imwrite(save_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))


def process_point(point_dir, sam3_tool, bev_metadata):
    images_dir = os.path.join(point_dir, 'streetview_images')
    output_dir = os.path.join(point_dir, 'text_output')
    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
    if not image_files:
        print("  ⚠️ 没有找到街景图像")
        return

    x_min = bev_metadata['x_range'][0]
    y_min = bev_metadata['y_range'][0]
    padding = bev_metadata['padding']
    resolution = bev_metadata['resolution']

    bev_path = bev_metadata.get('optimized_path', '')
    bev_image = None
    if bev_path:
        if os.path.exists(bev_path):
            bev_image = cv2.imread(bev_path)

    if bev_image is None:
        bev_candidates = [
            os.path.join(point_dir, 'projection_output', 'fused_bev_optimized.png'),
            os.path.join(point_dir, 'projection_output', 'fused_bev.png'),
            os.path.join(point_dir, 'projection_output', 'fused_bev_base.png'),
        ]
        for cand in bev_candidates:
            if os.path.exists(cand):
                bev_image = cv2.imread(cand)
                if bev_image is not None:
                    break

    if bev_image is None:
        print("  ⚠️ 没有找到BEV图像")
        return

    bev_height, bev_width = bev_image.shape[:2]

    all_text_markers = []
    per_image_results = []

    for img_file in image_files:
        base_name = img_file.replace('_streetview.jpg', '')
        metadata_file = f"{base_name}_metadata.json"

        image_path = os.path.join(images_dir, img_file)
        metadata_path = os.path.join(images_dir, metadata_file)

        if not os.path.exists(metadata_path):
            continue

        print(f"  处理 {img_file} ...")

        image_pil = Image.open(image_path).convert('RGB')
        image_np = np.array(image_pil)
        metadata = read_metadata(metadata_path)

        h, w = image_np.shape[:2]

        heading = metadata.get('heading', 0.0)
        pitch = metadata.get('pitch', 0.0)
        fov = metadata.get('fov', 60)
        relative_x = metadata.get('relative_x', 0.0)
        relative_y = metadata.get('relative_y', 0.0)
        cam_z = 2.5

        fx, fy, cx, cy = compute_intrinsics(w, h, fov)

        text_objects = detect_text_objects(image_pil, sam3_tool)

        if not text_objects:
            print(f"    未检测到文字物体")
            continue

        text_objects = deduplicate_text_objects(text_objects)
        print(f"    检测到 {len(text_objects)} 个文字物体（去重后）: {[o['prompt'] for o in text_objects]}")

        text_viz_dir = os.path.join(output_dir, 'text_detections')
        save_text_detection_visualizations(image_np, text_objects, text_viz_dir, base_name)

        for obj in text_objects:
            world_pt = project_text_object(
                obj, fx, fy, cx, cy, heading, pitch, relative_x, relative_y, cam_z
            )
            world_pt['source_image'] = img_file
            world_pt['score'] = obj['score']
            all_text_markers.append(world_pt)

        per_image_results.append({
            'image': img_file,
            'text_objects': [
                {
                    'prompt': o['prompt'],
                    'score': o['score'],
                    'centroid': o['centroid'],
                    'bbox': o['bbox'],
                    'pixel_count': o['pixel_count'],
                }
                for o in text_objects
            ],
        })

    if not all_text_markers:
        print("  ⚠️ 没有检测到任何文字标记")
        return

    before = len(all_text_markers)
    print(f"  开始多视角聚类与平均...")

    all_text_markers = cluster_and_average_markers(all_text_markers, distance_threshold=5.0)

    multi_count = sum(1 for m in all_text_markers if m.get('num_observations', 1) > 1)
    single_count = sum(1 for m in all_text_markers if m.get('num_observations', 1) == 1)
    print(f"  聚类: {before} 个标记 → {len(all_text_markers)} 个聚类")
    print(f"    - 多观测聚类 (≥2): {multi_count}")
    print(f"    - 单观测聚类 (1): {single_count}")

    text_markers_json_path = os.path.join(output_dir, 'text_markers.json')
    with open(text_markers_json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'total_markers': len(all_text_markers),
            'markers': all_text_markers,
        }, f, indent=2, ensure_ascii=False)

    bev_with_markers, drawn = draw_text_markers(
        bev_image, all_text_markers, x_min, y_min, padding, resolution
    )

    bev_markers_path = os.path.join(output_dir, 'bev_with_text_markers.png')
    cv2.imwrite(bev_markers_path, bev_with_markers)
    print(f"  ✅ BEV文字标记图已保存: {bev_markers_path}")
    print(f"     BEV尺寸: {bev_width}x{bev_height}, 实际绘制标记: {drawn}")

    print(f"  📊 共检测到 {len(all_text_markers)} 个文字标记（{drawn}个在BEV范围内）")
    for m in all_text_markers:
        g = m.get('ground_world')
        e = m.get('elevated_world')
        world = g if g else e
        if world:
            n_obs = m.get('num_observations', 1)
            avg_tag = " [加权平均]" if n_obs > 1 else ""
            print(f"    - {m['prompt']}: world=({world[0]:.2f}, {world[1]:.2f})m, n_obs={n_obs}{avg_tag}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='在BEV中标记文字物体位置')
    parser.add_argument('--input', type=str, required=True, help='街景点位目录')
    parser.add_argument('--sam3-weights', type=str, default=None, help='SAM3模型权重路径')
    args = parser.parse_args()

    point_dir = args.input

    bev_metadata_path = os.path.join(point_dir, 'projection_output', 'bev_metadata.json')
    if not os.path.exists(bev_metadata_path):
        print(f"❌ 未找到BEV元数据: {bev_metadata_path}")
        return

    with open(bev_metadata_path, 'r') as f:
        bev_metadata = json.load(f)

    print(f"加载BEV元数据: X范围={bev_metadata['x_range']}, Y范围={bev_metadata['y_range']}")
    print(f"BEV尺寸: {bev_metadata['bev_size']}, 分辨率: {bev_metadata['resolution']}px/m")

    sam3_tool = SAM3Tool()
    sam3_tool.initialize()

    try:
        process_point(point_dir, sam3_tool, bev_metadata)
    finally:
        sam3_tool.close()


if __name__ == '__main__':
    main()