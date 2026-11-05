import os
import re
import csv
import ast
import sys

BASE = "/home/suanzigao/PycharmProjects/street_view_global_localization"
BATCH = os.path.join(BASE, "batch_streetview_results", "2026-10-22_01-41-41_stage1_batch")
CSV_PATH = os.path.join(BASE, "geochain_mini_100.csv")


def parse_trajectory(html_path):
    with open(html_path, encoding="utf-8") as f:
        content = f.read()
    m = re.search(r"const trajectory = (\[.*?\]);", content, re.DOTALL)
    if not m:
        return None
    raw = m.group(1)
    raw = raw.replace("True", "True").replace("False", "False")
    try:
        traj = ast.literal_eval(raw)
    except Exception:
        return None
    return traj


def detect_hit_walls(traj):
    """检测撞墙（移动失败）：move 动作但坐标没变。返回 [(step, action, lat, lng), ...]"""
    failures = []
    last_lat = None
    last_lng = None
    for entry in traj:
        action = entry.get("action", "")
        lat = entry.get("lat")
        lng = entry.get("lng")
        if action == "Initial position":
            last_lat, last_lng = lat, lng
            continue
        if isinstance(action, str) and action.startswith("move"):
            if last_lat is not None and abs(lat - last_lat) < 1e-9 and abs(lng - last_lng) < 1e-9:
                failures.append((entry.get("step"), action, lat, lng))
            last_lat, last_lng = lat, lng
        # turn / fov / pitch / segment / search 不改变坐标
    return failures


def main():
    # 读取 100 个点
    points = []
    with open(CSV_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            points.append(row["name"])

    hit_wall_points = []
    no_fail_points = []
    missing = []

    for name in points:
        point_dir = os.path.join(BATCH, name)
        html = os.path.join(point_dir, "visualization.html")
        if not os.path.exists(html):
            missing.append(name)
            continue
        traj = parse_trajectory(html)
        if traj is None:
            missing.append(name)
            continue
        failures = detect_hit_walls(traj)
        if failures:
            hit_wall_points.append((name, failures))
        else:
            no_fail_points.append(name)

    print("=" * 70)
    print(f"检测结果：共 {len(points)} 个点")
    print(f"  有撞墙(移动失败)的点: {len(hit_wall_points)} 个")
    print(f"  无撞墙的点: {len(no_fail_points)} 个")
    if missing:
        print(f"  缺少可视化文件: {len(missing)} 个")
    print("=" * 70)

    if hit_wall_points:
        print("\n【有撞墙的点】")
        for name, failures in hit_wall_points:
            fail_steps = [f"step {s} ({a})" for s, a, _, _ in failures]
            print(f"  {name}: {len(failures)} 次撞墙 -> {', '.join(fail_steps)}")
    else:
        print("\n【无任何撞墙的点】")

    if no_fail_points:
        print("\n【无撞墙(全部移动成功)的点】")
        for name in no_fail_points:
            print(f"  {name}")

    if missing:
        print("\n【缺少 visualization.html 的点】")
        for name in missing:
            print(f"  {name}")


if __name__ == "__main__":
    main()
