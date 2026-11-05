"""针对历史撞墙点的 Stage 1 复测：验证撞墙反馈修复是否生效。

选取历史记录中撞墙最严重的 3 个点（各 20 次撞墙）：
  - dS9EPpbrXLeyyu1vVUvT5g   (45.3449811, -75.7235353) 渥太华
  - yJ39mtnttfZO_uSQ2Hjh6A   (60.1908444, 24.9142307)  赫尔辛基
  - 0RWA4OaNg-EVqWqcvt0j2w   (45.2666417, -75.94045)   渥太华

用法：
  .venv/bin/python test/hitwall_retest.py [max_steps]
"""
import os
import sys
import re
import ast
import datetime

os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"
os.environ["NO_PROXY"] = "localhost,127.0.0.1,localdomain,.localdomain"

BASE = "/home/suanzigao/PycharmProjects/street_view_global_localization"
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "localization"))

from dotenv import load_dotenv
load_dotenv()

from three_stage_localization import stage1_coarse_localize

TEST_POINTS = [
    {"name": "dS9EPpbrXLeyyu1vVUvT5g", "lat": 45.3449811, "lng": -75.7235353},
    {"name": "yJ39mtnttfZO_uSQ2Hjh6A", "lat": 60.1908444, "lng": 24.9142307},
    {"name": "0RWA4OaNg-EVqWqcvt0j2w", "lat": 45.2666417, "lng": -75.94045},
]


def detect_hit_walls(html_path):
    """检测撞墙：move 动作但坐标没变。返回 [(step, action), ...]"""
    with open(html_path, encoding="utf-8") as f:
        content = f.read()
    m = re.search(r"const trajectory = (\[.*?\]);", content, re.DOTALL)
    if not m:
        return None
    try:
        traj = ast.literal_eval(m.group(1))
    except Exception:
        return None

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
                failures.append((entry.get("step"), action))
            last_lat, last_lng = lat, lng
    return failures


def main():
    max_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    batch_dir = os.path.join(
        BASE, "batch_streetview_results",
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_hitwall_retest",
    )
    os.makedirs(batch_dir, exist_ok=True)

    print(f"🚀 撞墙点复测：共 {len(TEST_POINTS)} 个点，max_steps={max_steps}")
    print(f"   结果目录: {batch_dir}\n")

    for idx, p in enumerate(TEST_POINTS, 1):
        point_dir = os.path.join(batch_dir, p["name"])
        print(f"\n{'='*60}")
        print(f"[{idx}/{len(TEST_POINTS)}] {p['name']} ({p['lat']}, {p['lng']})")
        print(f"{'='*60}")
        try:
            stage1 = stage1_coarse_localize(
                point_dir, p["lat"], p["lng"], max_steps=max_steps, reuse=False
            )
        except Exception as e:
            print(f"❌ 该点运行失败: {e}")
            continue

        html_path = os.path.join(point_dir, "visualization.html")
        if not os.path.exists(html_path):
            print(f"⚠️ 未找到 visualization.html，跳过撞墙检测")
            continue

        failures = detect_hit_walls(html_path)
        if failures is None:
            print(f"⚠️ 无法解析 trajectory")
            continue

        if failures:
            fail_steps = [f"step {s}" for s, _ in failures]
            print(f"\n❌ 仍存在撞墙 {len(failures)} 次: {', '.join(fail_steps)}")
        else:
            print(f"\n✅ 无撞墙（移动全部成功）")

    print(f"\n✅ 复测完成，结果目录: {batch_dir}")


if __name__ == "__main__":
    main()
