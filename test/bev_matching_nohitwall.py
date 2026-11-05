"""无撞墙点的 BEV 路网匹配测试。

流程：
1. 从 geochain_mini_100.csv + stage1 batch 的 visualization.html 检测出"无撞墙"的点位。
2. 对每个点位：SAM3 分割道路 → 投影生成 BEV（含 lowres/cropped）。
3. 以真实坐标为搜索中心，用 BEV 模板在 OSM 路网（1km 范围）上做卷积模板匹配。
4. 汇总每个点的匹配误差（匹配坐标 vs 真实坐标）。

用法：
  .venv/bin/python test/bev_matching_nohitwall.py [--limit N] [--osm-radius 1000]
"""
import os
import sys
import re
import ast
import csv
import json
import math
import time
import argparse

import numpy as np
import cv2

os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"
os.environ["NO_PROXY"] = "localhost,127.0.0.1,localdomain,.localdomain"

BASE = "/home/suanzigao/PycharmProjects/street_view_global_localization"
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "localization"))

from dotenv import load_dotenv
load_dotenv()

from tools.sam3_tool import SAM3Tool
from batch_projection import project_pointcloud, generate_bev
from bev_osm_convolution_matcher import BEVOSMConvolutionMatcher

BATCH = os.path.join(BASE, "batch_streetview_results", "2026-10-22_01-41-41_stage1_batch")
CSV_PATH = os.path.join(BASE, "geochain_mini_100.csv")


def parse_trajectory(html_path):
    with open(html_path, encoding="utf-8") as f:
        content = f.read()
    m = re.search(r"const trajectory = (\[.*?\]);", content, re.DOTALL)
    if not m:
        return None
    try:
        return ast.literal_eval(m.group(1))
    except Exception:
        return None


def detect_hit_walls(traj):
    """move 动作但坐标没变 -> 撞墙。"""
    failures = []
    last_lat = last_lng = None
    for entry in traj:
        action = entry.get("action", "")
        lat = entry.get("lat")
        lng = entry.get("lng")
        if action == "Initial position":
            last_lat, last_lng = lat, lng
            continue
        if isinstance(action, str) and action.startswith("move"):
            if last_lat is not None and abs(lat - last_lat) < 1e-9 and abs(lng - last_lng) < 1e-9:
                failures.append(entry.get("step"))
            last_lat, last_lng = lat, lng
    return failures


def get_no_hit_wall_points():
    points = []
    with open(CSV_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            points.append((row["name"], float(row["lat"]), float(row["lng"])))

    no_fail = []
    for name, lat, lng in points:
        html = os.path.join(BATCH, name, "visualization.html")
        if not os.path.exists(html):
            continue
        traj = parse_trajectory(html)
        if traj is None:
            continue
        if not detect_hit_walls(traj):
            no_fail.append((name, lat, lng))
    return no_fail


def generate_lowres_bev(out_dir, result):
    """生成低分辨率 BEV + 裁剪版（与 three_stage_localization._generate_lowres_bev 一致）。"""
    optimized_path = os.path.join(out_dir, "fused_bev_optimized.png")
    if not os.path.exists(optimized_path):
        print("  ⚠️ 无优化 BEV，跳过 lowres 生成")
        return False

    opt_bev = cv2.imread(optimized_path)
    if opt_bev is None:
        print("  ⚠️ 无法读取优化 BEV")
        return False

    resolution = result.get("resolution", 10)
    padding = result.get("padding", 5)
    initial_px = result.get("initial_pixel_x", 0)
    initial_py = result.get("initial_pixel_y", 0)

    lowres_scale = 2.0 / resolution
    lowres_width = int(opt_bev.shape[1] * lowres_scale)
    lowres_height = int(opt_bev.shape[0] * lowres_scale)
    lowres_bev = cv2.resize(opt_bev, (lowres_width, lowres_height), interpolation=cv2.INTER_AREA)

    lowres_resolution = resolution * lowres_scale
    initial_px_lowres = int(initial_px * lowres_scale)
    initial_py_lowres = int(initial_py * lowres_scale)

    cv2.imwrite(os.path.join(out_dir, "fused_bev_lowres.png"), lowres_bev)
    lowres_metadata = {
        "x_range": result.get("x_range"),
        "y_range": result.get("y_range"),
        "padding": padding,
        "resolution": lowres_resolution,
        "initial_pixel_x": initial_px_lowres,
        "initial_pixel_y": initial_py_lowres,
    }
    with open(os.path.join(out_dir, "bev_lowres_metadata.json"), "w") as f:
        json.dump(lowres_metadata, f, indent=2)

    gray = cv2.cvtColor(lowres_bev, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(binary)
    if coords is None:
        print("  ⚠️ lowres BEV 无道路区域，跳过裁剪")
        return True

    x, y, w, h = cv2.boundingRect(coords)
    crop_padding = 20
    x_start = max(0, min(x - crop_padding, initial_px_lowres - crop_padding))
    y_start = max(0, min(y - crop_padding, initial_py_lowres - crop_padding))
    x_end = min(lowres_width, max(x + w + crop_padding, initial_px_lowres + crop_padding))
    y_end = min(lowres_height, max(y + h + crop_padding, initial_py_lowres + crop_padding))

    cropped_bev = lowres_bev[y_start:y_end, x_start:x_end]
    cv2.imwrite(os.path.join(out_dir, "fused_bev_lowres_cropped.png"), cropped_bev)

    cropped_x_min = result["x_range"][0] - padding + x_start / lowres_resolution
    cropped_y_min = result["y_range"][0] - padding + y_start / lowres_resolution
    cropped_metadata = {
        "x_range": result.get("x_range"),
        "y_range": result.get("y_range"),
        "cropped_x_min": cropped_x_min,
        "cropped_y_min": cropped_y_min,
        "padding": padding,
        "resolution": lowres_resolution,
        "initial_pixel_x": initial_px_lowres - x_start,
        "initial_pixel_y": initial_py_lowres - y_start,
    }
    with open(os.path.join(out_dir, "bev_lowres_cropped_metadata.json"), "w") as f:
        json.dump(cropped_metadata, f, indent=2)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 个无撞墙点")
    parser.add_argument("--osm-radius", type=int, default=1000, help="OSM 匹配搜索半径(米)")
    parser.add_argument("--interval", type=int, default=30, help="每个点位之间的间隔秒数（慢跑，避开 Overpass 过载）")
    parser.add_argument("--resume", action="store_true", help="断点续跑：跳过已有 match_result.json 的点")
    args = parser.parse_args()

    no_fail = get_no_hit_wall_points()
    print(f"无撞墙点位总数: {len(no_fail)}")

    if args.limit:
        no_fail = no_fail[: args.limit]
        print(f"仅处理前 {len(no_fail)} 个点")

    sam3_tool = SAM3Tool()
    sam3_tool.initialize()
    matcher = BEVOSMConvolutionMatcher()

    results = []
    try:
        for idx, (name, lat, lng) in enumerate(no_fail, 1):
            print(f"\n{'='*70}")
            print(f"[{idx}/{len(no_fail)}] {name} ({lat}, {lng})")
            print(f"{'='*70}")

            point_path = os.path.join(BATCH, name)
            images_dir = os.path.join(point_path, "streetview_images")
            out_dir = os.path.join(point_path, "projection_output")

            # 断点续跑：已有匹配结果则跳过
            match_result_path = os.path.join(point_path, "matching_results", "match_result.json")
            if args.resume and os.path.exists(match_result_path):
                try:
                    with open(match_result_path, encoding="utf-8") as f:
                        cached = json.load(f)
                    results.append(cached)
                    err = cached.get("distance_error")
                    err_str = f"{err:.2f} 米" if err is not None else "无误差"
                    print(f"  ♻️ 跳过（已有匹配结果，误差 {err_str}）")
                    continue
                except Exception as e:
                    print(f"  ⚠️ 读取已有结果失败，重新匹配: {e}")

            bev_cropped = os.path.join(out_dir, "fused_bev_lowres_cropped.png")
            if not os.path.exists(bev_cropped):
                os.makedirs(out_dir, exist_ok=True)
                image_files = sorted(
                    [f for f in os.listdir(images_dir) if f.endswith("_streetview.jpg")]
                )
                if not image_files:
                    print("  ❌ 无街景图，跳过")
                    continue

                print(f"  🔄 投影 {len(image_files)} 张街景图...")
                all_points, all_colors, all_step_colors, camera_positions = project_pointcloud(
                    image_files, images_dir, out_dir, sam3_tool
                )
                if not all_points:
                    print("  ❌ 投影失败，跳过")
                    continue

                bev_path = os.path.join(out_dir, "fused_bev.png")
                bev_result = generate_bev(
                    all_points, all_colors, bev_path, camera_positions, all_step_colors
                )
                if not bev_result:
                    print("  ❌ BEV 生成失败，跳过")
                    continue
                generate_lowres_bev(out_dir, bev_result)
            else:
                print("  ♻️ 复用已有 BEV")

            print(f"  🎯 1km 路网匹配...")
            point = {"name": name, "lat": lat, "lng": lng}
            match = matcher.match_point(point, BATCH, osm_radius=args.osm_radius)
            if match:
                results.append(match)

            # 慢跑：点位之间间隔，避开 Overpass 过载
            if idx < len(no_fail) and args.interval > 0:
                print(f"  ⏳ 慢跑间隔 {args.interval} 秒，避开 Overpass 过载...")
                time.sleep(args.interval)

    finally:
        sam3_tool.close()

    # 汇总
    print("\n" + "=" * 70)
    print("匹配结果汇总")
    print("=" * 70)

    valid = [r for r in results if r.get("distance_error") is not None]
    if valid:
        errors = [r["distance_error"] for r in valid]
        within_1km = [r for r in valid if r["distance_error"] <= 1000]
        within_100m = [r for r in valid if r["distance_error"] <= 100]
        print(f"成功匹配: {len(results)} 个")
        print(f"有效误差: {len(valid)} 个")
        print(f"误差 ≤ 1km: {len(within_1km)} 个")
        print(f"误差 ≤ 100m: {len(within_100m)} 个")
        print(f"平均误差: {sum(errors)/len(errors):.2f} 米")
        print(f"中位误差: {sorted(errors)[len(errors)//2]:.2f} 米")
        print(f"最小误差: {min(errors):.2f} 米")
        print(f"最大误差: {max(errors):.2f} 米")

        print("\n各点误差（按误差升序）:")
        for r in sorted(valid, key=lambda x: x["distance_error"]):
            print(f"  {r['name']}: {r['distance_error']:.2f} 米 (置信度 {r['match_confidence']:.3f})")

    summary = {
        "total": len(results),
        "valid": len(valid),
        "within_1km": len(within_1km) if valid else 0,
        "within_100m": len(within_100m) if valid else 0,
        "avg_error": sum(errors) / len(errors) if valid else None,
        "median_error": sorted(errors)[len(errors)//2] if valid else None,
        "min_error": min(errors) if valid else None,
        "max_error": max(errors) if valid else None,
        "results": results,
    }
    summary_path = os.path.join(BATCH, "bev_matching_nohitwall_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 汇总已保存: {summary_path}")


if __name__ == "__main__":
    main()
