"""
下载单个位置不同zoom级别的卫星图像
用于对比不同zoom级别的清晰度和范围
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
    x = int((lng + 180.0) / 360.0 * n)
    y = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
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
    meters_per_deg_lat = 111000.0
    meters_per_deg_lng = 111000.0 * cos_lat
    n = 2.0 ** zoom
    meters_per_pixel_y = meters_per_deg_lat / (256 * n)
    meters_per_pixel_x = meters_per_deg_lng / (256 * n)
    return meters_per_pixel_x, meters_per_pixel_y, cos_lat


def download_zoom_levels(api_key, lat, lng, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    session = get_session_token(api_key)
    print(f"📋 会话令牌获取成功\n")

    results = []

    for zoom in range(10, 19):
        print(f"{'='*60}")
        print(f"Zoom={zoom}")
        print(f"{'='*60}")

        meters_px_x, meters_px_y, cos_lat = get_resolutions(lat, zoom)
        tile_x_m = meters_px_x * 256
        tile_y_m = meters_px_y * 256

        center_x, center_y = latlng_to_tile(lat, lng, zoom)

        print(f"📐 分辨率: X(经度)={meters_px_x:.4f}m/px, Y(纬度)={meters_px_y:.4f}m/px")
        print(f"📏 每图块: {tile_x_m:.1f}m x {tile_y_m:.1f}m")
        print(f"📍 中心图块: ({center_x}, {center_y})")

        tile_img = download_satellite_tile(api_key, session, zoom, center_x, center_y)

        tile_path = os.path.join(output_dir, f'zoom{zoom}_tile.png')
        tile_img.save(tile_path)
        print(f"✅ 单图块已保存")

        tile_cv = cv2.cvtColor(np.array(tile_img), cv2.COLOR_RGB2BGR)

        overlay_text = f"Zoom={zoom} | X={meters_px_x:.3f}m/px Y={meters_px_y:.3f}m/px"
        overlay_img = tile_cv.copy()
        cv2.putText(overlay_img, overlay_text, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.circle(overlay_img, (128, 128), 5, (0, 0, 255), -1)
        cv2.circle(overlay_img, (128, 128), 10, (0, 0, 255), 2)
        annotated_path = os.path.join(output_dir, f'zoom{zoom}_annotated.png')
        cv2.imwrite(annotated_path, overlay_img)
        print(f"✅ 标注图已保存")

        results.append({
            'zoom': zoom,
            'meters_per_pixel_x': meters_px_x,
            'meters_per_pixel_y': meters_px_y,
            'tile_size_m': f"{tile_x_m:.1f}x{tile_y_m:.1f}",
            'center_tile': {'x': center_x, 'y': center_y}
        })

    summary_path = os.path.join(output_dir, 'zoom_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'lat': lat,
            'lng': lng,
            'results': results
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"所有zoom级别下载完成")
    print(f"{'='*60}")
    print(f"\n📊 各zoom级别对比:")
    print(f"{'Zoom':<6} {'X分辨率(m/px)':<18} {'Y分辨率(m/px)':<18} {'图块大小(m)':<15}")
    print(f"{'-'*57}")
    for r in results:
        print(f"{r['zoom']:<6} {r['meters_per_pixel_x']:<18.4f} {r['meters_per_pixel_y']:<18.4f} {r['tile_size_m']:<15}")

    print(f"\n📁 所有文件保存在: {output_dir}")


def main():
    env_path = "/home/suanzigao/PycharmProjects/street_view_global_localization/.env"
    with open(env_path, 'r') as f:
        for line in f:
            if line.startswith("GOOGLE_API_KEY="):
                api_key = line.strip().split("=")[1]
                break

    locations = [
        (47.573102, 19.120968, "Helsinki"),
        (60.174854, 24.776313, "Helsinki2"),
    ]

    for lat, lng, name in locations:
        print(f"\n{'#'*60}")
        print(f"📍 位置: {name} ({lat}, {lng})")
        print(f"{'#'*60}")

        output_dir = f'/home/suanzigao/PycharmProjects/street_view_global_localization/satellite_zoom_comparison/{name}_lat{lat:.4f}_lng{lng:.4f}'
        download_zoom_levels(api_key, lat, lng, output_dir)


if __name__ == "__main__":
    main()