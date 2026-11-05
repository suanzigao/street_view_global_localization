"""Stage 1 批量测试：读取 geochain_mini_100.csv 的所有点位，逐个跑 stage1_coarse_localize。

结果保存位置：
    batch_streetview_results/{时间戳}_stage1_batch/
        {name}/stage1_coarse.json      每个点位的 Stage 1 结果
        {name}/streetview_images/       街景图 + bev/*.png
        {name}/visualization.html       导航可视化
        stage1_summary.csv              全部点位的汇总（城市/粗定位/误差）

用法：
    .venv/bin/python test/batch_stage1.py [max_steps] [limit]
"""
import os
import sys
import csv
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
from utils.file_utils import read_gps_targets_from_csv
from utils.geometry import calculate_gps_distance

CSV_PATH = os.path.join(BASE, "geochain_mini_100.csv")
BATCH_ROOT = os.path.join(BASE, "batch_streetview_results")


def main():
    max_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    targets = read_gps_targets_from_csv(CSV_PATH)
    if limit:
        targets = targets[:limit]

    batch_dir = os.path.join(
        BATCH_ROOT,
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_stage1_batch",
    )
    os.makedirs(batch_dir, exist_ok=True)

    print(f"🚀 Stage 1 批量测试：共 {len(targets)} 个点位，max_steps={max_steps}")
    print(f"   结果目录: {batch_dir}\n")

    summary_rows = []
    for idx, p in enumerate(targets, 1):
        name = p["name"]
        point_dir = os.path.join(batch_dir, name)
        print(f"\n====== [{idx}/{len(targets)}] {name} ({p['lat']}, {p['lng']}) ======")
        try:
            stage1 = stage1_coarse_localize(
                point_dir, p["lat"], p["lng"], max_steps=max_steps, reuse=False
            )
            cg = stage1.get("coarse_gps") or {}
            error = None
            if cg.get("lat") is not None:
                error = calculate_gps_distance(
                    cg["lat"], cg["lng"], p["lat"], p["lng"]
                )
            summary_rows.append({
                "name": name,
                "city": stage1.get("city", ""),
                "city_gt": stage1.get("city_gt", ""),
                "coarse_lat": cg.get("lat"),
                "coarse_lng": cg.get("lng"),
                "error_m": round(error, 1) if error is not None else "",
            })
        except Exception as e:
            print(f"❌ 该点失败: {e}")
            summary_rows.append({
                "name": name, "city": "", "city_gt": "",
                "coarse_lat": "", "coarse_lng": "", "error_m": "FAILED",
            })

    summary_path = os.path.join(batch_dir, "stage1_summary.csv")
    fieldnames = ["name", "city", "city_gt", "coarse_lat", "coarse_lng", "error_m"]
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\n✅ 批量测试完成，汇总: {summary_path}")
    print(f"   各点位结果: {batch_dir}/<name>/stage1_coarse.json")


if __name__ == "__main__":
    main()
