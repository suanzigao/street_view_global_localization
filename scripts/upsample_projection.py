"""
上采样街景图像后投影生成 BEV

流程与原版 batch_projection 一致，但在投影前对街景图像做上采样：
1. 读入街景图像
2. 上采样 factor 倍（默认2x）
3. 上采样后的图像喂给 SAM3 分割地面（mask 也是上采样尺寸）
4. 用上采样后的尺寸计算内参（自动适配，几何不变）
5. 投影（地面像素更多，BEV 填充更密集）

输出到 projection_output_upsampled/，便于与原版对比
"""

import os
import sys
import json
import math
import argparse
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tools.sam3_tool import SAM3Tool
from scripts.batch_projection import (
    compute_intrinsics,
    pixel_to_world,
    read_metadata,
    generate_bev,
    draw_camera_marker,
    get_step_color,
)


def upsample_image(image_pil, factor):
    w, h = image_pil.size
    new_w, new_h = int(w * factor), int(h * factor)
    return image_pil.resize((new_w, new_h), Image.BICUBIC)


def project_pointcloud_upsampled(image_files, images_dir, output_dir, sam3_tool, factor):
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

        # 上采样
        image_pil = upsample_image(image_pil, factor)

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
                    mask_np = np.array(mask)
                    if mask_np.ndim == 3:
                        mask_np = mask_np[0]
                    if ground_mask is None:
                        ground_mask = mask_np.astype(bool)
                    else:
                        ground_mask = np.logical_or(ground_mask, mask_np.astype(bool))

        if ground_mask is None:
            print(f"    ⚠️ {base_name}: SAM3未分割到道路，跳过投影")
            continue

        mask_255 = (ground_mask * 255).astype(np.uint8) if ground_mask.max() <= 1 else ground_mask.astype(np.uint8)

        road_pixels = np.sum(mask_255 > 0)
        total_pixels = h * w
        road_ratio = road_pixels / total_pixels

        if road_ratio < 0.01:
            print(f"    ⚠️ {base_name}: 道路区域过小({road_ratio:.2%})，跳过投影")
            continue

        points_3d = []
        colors = []

        max_distance = 200

        ys, xs = np.where(mask_255 > 0)

        for i in range(len(ys)):
            x = xs[i]
            y = ys[i]

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

        print(f"    {base_name}: 上采样至 {w}x{h}, 投影 {len(points_3d)} 点")

    return all_points, all_colors, all_step_colors, camera_positions


def main():
    parser = argparse.ArgumentParser(description='上采样街景图像后投影生成BEV')
    parser.add_argument('--input', type=str, required=True, help='街景点位目录')
    parser.add_argument('--factor', type=float, default=2.0, help='上采样倍数，默认2x')
    parser.add_argument('--output-dir', type=str, default=None, help='输出目录名（相对点位目录），默认 projection_output_upsampled')
    args = parser.parse_args()

    point_dir = args.input
    factor = args.factor

    images_dir = os.path.join(point_dir, 'streetview_images')
    if args.output_dir:
        output_dir = os.path.join(point_dir, args.output_dir)
    else:
        output_dir = os.path.join(point_dir, 'projection_output_upsampled')
    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
    if not image_files:
        print("❌ 没有找到街景图像")
        return

    print(f"上采样倍数: {factor}x")
    print(f"图像数量: {len(image_files)}")

    sam3_tool = SAM3Tool()
    sam3_tool.initialize()

    try:
        all_points, all_colors, all_step_colors, camera_positions = project_pointcloud_upsampled(
            image_files, images_dir, output_dir, sam3_tool, factor
        )
    finally:
        sam3_tool.close()

    if not all_points:
        print("❌ 没有成功投影的图像")
        return

    bev_path = os.path.join(output_dir, 'fused_bev.png')
    result = generate_bev(all_points, all_colors, bev_path, camera_positions, all_step_colors)

    if result:
        print(f"\n✅ 上采样BEV已生成:")
        print(f"  总点数: {result['total_points']}")
        print(f"  X范围: {result['x_range'][0]:.2f} ~ {result['x_range'][1]:.2f}")
        print(f"  Y范围: {result['y_range'][0]:.2f} ~ {result['y_range'][1]:.2f}")
        print(f"  BEV尺寸: {result['bev_size'][0]}x{result['bev_size'][1]}")
        print(f"  输出目录: {output_dir}")


if __name__ == '__main__':
    main()
