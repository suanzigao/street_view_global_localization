"""
BEV道路图像与卫星地图叠加验证脚本
使用Google Maps Satellite Tile API获取卫星图像
修正：考虑经度/纬度方向分辨率差异
重要：使用Mercator投影精确计算真实经纬度在卫星图像中的像素位置
"""

import os
import json
import math
import cv2
import numpy as np
import requests
from PIL import Image
from io import BytesIO


def latlng_to_tile(lat, lng, zoom):
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    x = (lng + 180.0) / 360.0 * n
    y = (1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n
    return x, y


def get_session_token(api_key):
    url = f"https://tile.googleapis.com/v1/createSession?key={api_key}"
    payload = {
        "mapType": "satellite",
        "language": "zh-CN",
        "region": "CN"
    }
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()["session"]


def download_satellite_tile(api_key, session, zoom, x, y):
    url = f"https://tile.googleapis.com/v1/2dtiles/{zoom}/{x}/{y}?session={session}&key={api_key}"
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    return Image.open(BytesIO(response.content))


def get_resolutions(lat, zoom):
    cos_lat = math.cos(math.radians(lat))
    equator_res = 156543.03 / (2.0 ** zoom)
    meters_per_pixel_y = equator_res
    meters_per_pixel_x = equator_res * cos_lat
    return meters_per_pixel_x, meters_per_pixel_y, cos_lat


def create_overlay_image(sat_cropped, bev_img, bev_w, bev_h, initial_px, initial_py,
                         exact_px_cropped, exact_py_cropped, scale_x, scale_y,
                         mode_name, output_dir, zoom, match_point=None):
    bev_scaled_w = max(1, int(bev_w * scale_x))
    bev_scaled_h = max(1, int(bev_h * scale_y))

    print(f"  [{mode_name}] 缩放: X={scale_x:.4f}, Y={scale_y:.4f}")
    print(f"  [{mode_name}] BEV缩放后: {bev_scaled_w}x{bev_scaled_h}像素")

    if scale_x == scale_y:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_AREA

    bev_scaled = cv2.resize(bev_img, (bev_scaled_w, bev_scaled_h), interpolation=interpolation)

    scaled_initial_px = int(initial_px * bev_scaled_w / bev_w)
    scaled_initial_py = int(initial_py * bev_scaled_h / bev_h)

    overlay_x_start = int(exact_px_cropped - scaled_initial_px)
    overlay_y_start = int(exact_py_cropped - scaled_initial_py)

    print(f"  [{mode_name}] BEV叠加位置: ({overlay_x_start}, {overlay_y_start})")

    sat_h, sat_w = sat_cropped.shape[:2]
    overlay_img = sat_cropped.copy()

    src_x_start = max(0, -overlay_x_start)
    src_y_start = max(0, -overlay_y_start)
    src_x_end = min(bev_scaled_w, sat_w - overlay_x_start)
    src_y_end = min(bev_scaled_h, sat_h - overlay_y_start)

    dst_x_start = max(0, overlay_x_start)
    dst_y_start = max(0, overlay_y_start)
    dst_x_end = dst_x_start + (src_x_end - src_x_start)
    dst_y_end = dst_y_start + (src_y_end - src_y_start)

    if src_x_end > src_x_start and src_y_end > src_y_start:
        bev_patch = bev_scaled[src_y_start:src_y_end, src_x_start:src_x_end]
        overlay_patch = overlay_img[dst_y_start:dst_y_end, dst_x_start:dst_x_end]

        if len(bev_patch.shape) == 2:
            bev_colored = cv2.cvtColor(bev_patch, cv2.COLOR_GRAY2BGR)
        else:
            bev_colored = bev_patch

        mask = cv2.cvtColor(bev_colored, cv2.COLOR_BGR2GRAY)
        _, mask_binary = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)

        overlay_alpha = 0.5
        mask_normalized = cv2.cvtColor(mask_binary, cv2.COLOR_GRAY2BGR) / 255.0

        for c in range(3):
            overlay_img[dst_y_start:dst_y_end, dst_x_start:dst_x_end, c] = (
                overlay_patch[:, :, c] * (1 - mask_normalized[:, :, c]) * (1 - overlay_alpha) +
                bev_colored[:, :, c] * overlay_alpha * mask_normalized[:, :, c] +
                overlay_patch[:, :, c] * (1 - mask_normalized[:, :, c])
            )

    cv2.circle(overlay_img, (int(exact_px_cropped), int(exact_py_cropped)), 8, (0, 0, 255), -1)
    cv2.circle(overlay_img, (int(exact_px_cropped), int(exact_py_cropped)), 12, (0, 0, 255), 2)
    cv2.putText(overlay_img, 'ACTUAL POSITION', (int(exact_px_cropped) + 15, int(exact_py_cropped)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    if match_point is not None:
        match_px, match_py = match_point
        cv2.circle(overlay_img, (int(match_px), int(match_py)), 8, (255, 0, 0), -1)
        cv2.circle(overlay_img, (int(match_px), int(match_py)), 12, (255, 0, 0), 2)
        cv2.putText(overlay_img, 'OSM MATCH', (int(match_px) + 15, int(match_py)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        dx = match_px - exact_px_cropped
        dy = match_py - exact_py_cropped
        cv2.line(overlay_img, (int(exact_px_cropped), int(exact_py_cropped)),
                 (int(match_px), int(match_py)), (255, 255, 0), 2)
        print(f"  [{mode_name}] OSM匹配点偏移: ({dx:.1f}, {dy:.1f}) 像素")

    cv2.rectangle(overlay_img,
                  (overlay_x_start, overlay_y_start),
                  (overlay_x_start + bev_scaled_w, overlay_y_start + bev_scaled_h),
                  (0, 255, 0), 2)

    overlay_path = os.path.join(output_dir, f'bev_on_satellite_{mode_name}.png')
    cv2.imwrite(overlay_path, overlay_img)
    print(f"  ✅ [{mode_name}] 叠加图像已保存: {overlay_path}")

    overlay_only_bev = np.zeros((sat_h, sat_w, 3), dtype=np.uint8)
    if src_x_end > src_x_start and src_y_end > src_y_start:
        bev_patch = bev_scaled[src_y_start:src_y_end, src_x_start:src_x_end]
        if len(bev_patch.shape) == 2:
            bev_colored = cv2.cvtColor(bev_patch, cv2.COLOR_GRAY2BGR)
        else:
            bev_colored = bev_patch
        overlay_only_bev[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = bev_colored
    cv2.imwrite(os.path.join(output_dir, f'bev_only_on_black_{mode_name}_zoom{zoom}.png'), overlay_only_bev)

    max_display_dim = 1500
    if sat_w > max_display_dim or sat_h > max_display_dim:
        display_scale = min(max_display_dim / sat_w, max_display_dim / sat_h)
        display_w = int(sat_w * display_scale)
        display_h = int(sat_h * display_scale)
    else:
        display_scale = 1.0
        display_w, display_h = sat_w, sat_h

    overlay_display = cv2.resize(overlay_img, (display_w, display_h), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(output_dir, f'bev_on_satellite_{mode_name}_zoom{zoom}_display.png'), overlay_display)

    sat_display = cv2.resize(sat_cropped, (display_w, display_h), interpolation=cv2.INTER_AREA)
    bev_display = cv2.resize(bev_scaled, (int(bev_scaled_w * display_scale), int(bev_scaled_h * display_scale)),
                              interpolation=cv2.INTER_AREA)

    comp_h = max(sat_display.shape[0], bev_display.shape[0])
    comp_w = sat_display.shape[1] + bev_display.shape[1]
    comparison_img = np.zeros((comp_h, comp_w, 3), dtype=np.uint8)
    comparison_img[:sat_display.shape[0], :sat_display.shape[1]] = sat_display
    comparison_img[:bev_display.shape[0], sat_display.shape[1]:sat_display.shape[1]+bev_display.shape[1]] = bev_display

    cv2.putText(comparison_img, 'Satellite', (sat_display.shape[1]//2 - 50, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(comparison_img, 'BEV Road', (sat_display.shape[1] + bev_display.shape[1]//2 - 40, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imwrite(os.path.join(output_dir, f'satellite_and_bev_side_by_side_{mode_name}_zoom{zoom}.png'), comparison_img)

    return {
        'mode': mode_name,
        'scale_x': scale_x,
        'scale_y': scale_y,
        'bev_scaled_size': f"{bev_scaled_w}x{bev_scaled_h}",
        'overlay_position': [overlay_x_start, overlay_y_start]
    }


def download_and_overlay(api_key, center_lat, center_lng, bev_path, bev_meta_path, point_meta_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with open(bev_meta_path, 'r') as f:
        bev_meta = json.load(f)
    with open(point_meta_path, 'r') as f:
        point_meta = json.load(f)

    bev_resolution = bev_meta['resolution']
    initial_px = bev_meta['initial_pixel_x']
    initial_py = bev_meta['initial_pixel_y']
    actual_lat = point_meta['actual_lat']
    actual_lng = point_meta['actual_lng']

    x_range = bev_meta.get('x_range', [0, 100])
    y_range = bev_meta.get('y_range', [-100, 100])
    bev_width_m = x_range[1] - x_range[0]
    bev_height_m = y_range[1] - y_range[0]

    bev_h, bev_w = cv2.imread(bev_path).shape[:2]
    bev_meters_w = bev_w / bev_resolution
    bev_meters_h = bev_h / bev_resolution

    print(f"\n📍 坐标: ({actual_lat:.6f}, {actual_lng:.6f})")
    print(f"📏 BEV: {bev_w}x{bev_h}像素 = {bev_meters_w:.1f}m x {bev_meters_h:.1f}m")

    zoom = 18
    meters_px_x, meters_px_y, cos_lat = get_resolutions(actual_lat, zoom)

    print(f"🔍 Zoom={zoom}, cos(lat)={cos_lat:.4f}")
    print(f"📐 卫星分辨率: X(经度)={meters_px_x:.4f}m/px, Y(纬度)={meters_px_y:.4f}m/px")

    center_tile_x, center_tile_y = latlng_to_tile(center_lat, center_lng, zoom)
    tile_meters_x = meters_px_x * 256
    tile_meters_y = meters_px_y * 256

    range_meters = 500
    tiles_needed_x = math.ceil(range_meters / tile_meters_x) + 1
    tiles_needed_y = math.ceil(range_meters / tile_meters_y) + 1

    print(f"📏 每图块: {tile_meters_x:.1f}m x {tile_meters_y:.1f}m")
    print(f"📐 目标范围: 中心点周围 {range_meters}m")
    print(f"📥 下载卫星图块...")
    print(f"   X方向半径: {tiles_needed_x} 张瓦片")
    print(f"   Y方向半径: {tiles_needed_y} 张瓦片")

    session = get_session_token(api_key)

    min_tx = int(math.floor(center_tile_x)) - tiles_needed_x
    max_tx = int(math.floor(center_tile_x)) + tiles_needed_x
    min_ty = int(math.floor(center_tile_y)) - tiles_needed_y
    max_ty = int(math.floor(center_tile_y)) + tiles_needed_y

    tiles = {}
    total_tiles = (max_tx - min_tx + 1) * (max_ty - min_ty + 1)
    downloaded = 0

    for tx in range(min_tx, max_tx + 1):
        for ty in range(min_ty, max_ty + 1):
            if (tx, ty) not in tiles:
                try:
                    tile_img = download_satellite_tile(api_key, session, zoom, tx, ty)
                    tiles[(tx, ty)] = tile_img
                except Exception as e:
                    pass
                downloaded += 1
                if downloaded % 10 == 0 or downloaded == total_tiles:
                    print(f"  进度: {downloaded}/{total_tiles}")

    if not tiles:
        print("❌ 没有下载到任何图块")
        return None

    min_x = min(t[0] for t in tiles.keys())
    max_x = max(t[0] for t in tiles.keys())
    min_y = min(t[1] for t in tiles.keys())
    max_y = max(t[1] for t in tiles.keys())

    width = (max_x - min_x + 1) * 256
    height = (max_y - min_y + 1) * 256

    print(f"🖼️ 拼接卫星图像: {width}x{height}像素")

    combined = Image.new('RGB', (width, height))
    for (tx, ty), tile_img in tiles.items():
        x_offset = (tx - min_x) * 256
        y_offset = (ty - min_y) * 256
        combined.paste(tile_img, (x_offset, y_offset))

    sat_cv = cv2.cvtColor(np.array(combined), cv2.COLOR_RGB2BGR)
    sat_h, sat_w = sat_cv.shape[:2]

    exact_px = (center_tile_x - min_x) * 256
    exact_py = (center_tile_y - min_y) * 256

    print(f"📍 真实经纬度在卫星图像中的精确像素位置: ({exact_px:.1f}, {exact_py:.1f})")

    crop_half_x = int(range_meters / meters_px_x)
    crop_half_y = int(range_meters / meters_px_y)
    crop_x1 = max(0, int(exact_px - crop_half_x))
    crop_y1 = max(0, int(exact_py - crop_half_y))
    crop_x2 = min(sat_w, int(exact_px + crop_half_x))
    crop_y2 = min(sat_h, int(exact_py + crop_half_y))

    print(f"📐 裁剪范围: ({crop_x1}, {crop_y1}) - ({crop_x2}, {crop_y2})")
    print(f"📐 裁剪后尺寸: {crop_x2-crop_x1}x{crop_y2-crop_y1}像素")
    print(f"📐 实际覆盖: {(crop_x2-crop_x1)*meters_px_x:.1f}m x {(crop_y2-crop_y1)*meters_px_y:.1f}m")

    sat_cropped = sat_cv[crop_y1:crop_y2, crop_x1:crop_x2]
    sat_h, sat_w = sat_cropped.shape[:2]
    exact_px_cropped = exact_px - crop_x1
    exact_py_cropped = exact_py - crop_y1

    print(f"📍 裁剪后中心点位置: ({exact_px_cropped:.1f}, {exact_py_cropped:.1f})")

    sat_path = os.path.join(output_dir, f'satellite_zoom{zoom}_crop{range_meters}m.png')
    cv2.imwrite(sat_path, sat_cropped)
    print(f"✅ 裁剪后卫星图像已保存")

    bev_img = cv2.imread(bev_path)
    bev_meters_per_pixel = 1.0 / bev_resolution

    satellite_avg_meters_per_pixel = (meters_px_x + meters_px_y) / 2.0
    uniform_scale = bev_meters_per_pixel / satellite_avg_meters_per_pixel

    print(f"\n📐 BEV: {bev_meters_per_pixel:.3f}m/px")
    print(f"📐 卫星平均: {satellite_avg_meters_per_pixel:.3f}m/px")

    match_result_path = os.path.join(os.path.dirname(output_dir), 'matching_results', 'match_result.json')
    match_point = None
    match_info = None
    if os.path.exists(match_result_path):
        with open(match_result_path, 'r') as f:
            match_info = json.load(f)
        match_lat = match_info['match_lat']
        match_lng = match_info['match_lng']
        match_tile_x, match_tile_y = latlng_to_tile(match_lat, match_lng, zoom)
        match_px = (match_tile_x - min_x) * 256 - crop_x1
        match_py = (match_tile_y - min_y) * 256 - crop_y1
        match_point = (match_px, match_py)
        print(f"\n📍 OSM匹配点经纬度: ({match_lat:.6f}, {match_lng:.6f})")
        print(f"📍 OSM匹配点在卫星图中位置: ({match_px:.1f}, {match_py:.1f})")
        print(f"📏 OSM匹配误差: {match_info.get('distance_error', 'N/A')}m")
    else:
        print(f"\n⚠️ 未找到匹配结果文件: {match_result_path}")

    results = []

    results.append(create_overlay_image(
        sat_cropped, bev_img, bev_w, bev_h, initial_px, initial_py,
        exact_px_cropped, exact_py_cropped, uniform_scale, uniform_scale,
        "uniform", output_dir, zoom, match_point
    ))

    scale_x_stretched = bev_meters_per_pixel / meters_px_x
    scale_y_stretched = bev_meters_per_pixel / meters_px_y
    results.append(create_overlay_image(
        sat_cropped, bev_img, bev_w, bev_h, initial_px, initial_py,
        exact_px_cropped, exact_py_cropped, scale_x_stretched, scale_y_stretched,
        "stretched", output_dir, zoom, match_point
    ))

    sat_metadata = {
        'zoom': zoom,
        'center_lat': center_lat,
        'center_lng': center_lng,
        'meters_per_pixel_x': meters_px_x,
        'meters_per_pixel_y': meters_px_y,
        'cos_lat': cos_lat,
        'width': width,
        'height': height,
        'exact_pixel': {'x': exact_px, 'y': exact_py},
        'center_tile': {'x': math.floor(center_tile_x), 'y': math.floor(center_tile_y)},
        'min_tile': {'x': min_x, 'y': min_y}
    }
    meta_path = os.path.join(output_dir, f'satellite_zoom{zoom}_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(sat_metadata, f, indent=2)

    result = {
        'point_name': os.path.basename(os.path.dirname(output_dir)),
        'lat': actual_lat,
        'lng': actual_lng,
        'zoom': zoom,
        'meters_per_pixel_x': meters_px_x,
        'meters_per_pixel_y': meters_px_y,
        'exact_pixel': {'x': exact_px, 'y': exact_py},
        'bev_meters_per_pixel': bev_meters_per_pixel,
        'satellite_avg_meters_per_pixel': satellite_avg_meters_per_pixel,
        'overlay_modes': results,
        'match_info': {
            'match_lat': match_info['match_lat'] if match_info else None,
            'match_lng': match_info['match_lng'] if match_info else None,
            'distance_error': match_info.get('distance_error') if match_info else None,
            'match_pixel_in_satellite': [match_point[0], match_point[1]] if match_point else None
        } if match_info else None,
        'bev_orig_size': f"{bev_w}x{bev_h}",
        'satellite_size': f"{width}x{height}",
        'output_dir': output_dir
    }

    result_path = os.path.join(output_dir, 'verification_result.json')
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)

    return result


def main():
    env_path = "/home/suanzigao/PycharmProjects/street_view_global_localization/.env"
    with open(env_path, 'r') as f:
        for line in f:
            if line.startswith("GOOGLE_API_KEY="):
                api_key = line.strip().split("=")[1]
                break

    batch_dir = '/home/suanzigao/PycharmProjects/street_view_global_localization/manual_streetview_results'

    point_dirs = sorted([
        d for d in os.listdir(batch_dir)
        if os.path.isdir(os.path.join(batch_dir, d)) and not d.startswith('.')
    ])

    print(f"📊 共发现 {len(point_dirs)} 个点位目录")

    all_results = []

    for point_dir_name in point_dirs:
        point_dir = os.path.join(batch_dir, point_dir_name)

        bev_path = os.path.join(point_dir, 'projection_output', 'fused_bev_lowres_cropped.png')
        bev_meta_path = os.path.join(point_dir, 'projection_output', 'bev_lowres_cropped_metadata.json')
        point_meta_path = os.path.join(point_dir, 'streetview_images', 'step_0_metadata.json')

        if not os.path.exists(bev_path):
            bev_path = os.path.join(point_dir, 'projection_output', 'fused_bev_lowres.png')
            bev_meta_path = os.path.join(point_dir, 'projection_output', 'bev_lowres_metadata.json')

        if not os.path.exists(bev_path) or not os.path.exists(bev_meta_path) or not os.path.exists(point_meta_path):
            print(f"⚠️ 跳过 {point_dir_name}：缺少文件")
            continue

        output_dir = os.path.join(point_dir, 'satellite_verification')

        with open(point_meta_path, 'r') as f:
            point_meta = json.load(f)

        result = download_and_overlay(
            api_key,
            point_meta['actual_lat'],
            point_meta['actual_lng'],
            bev_path, bev_meta_path, point_meta_path, output_dir
        )

        if result:
            all_results.append(result)

    print(f"\n{'='*70}")
    print(f"卫星验证完成，成功 {len(all_results)} 个点位")

    summary = {'total_points': len(all_results), 'results': all_results}
    summary_path = os.path.join(batch_dir, 'satellite_verification_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✅ 汇总已保存: {summary_path}")


if __name__ == "__main__":
    main()
