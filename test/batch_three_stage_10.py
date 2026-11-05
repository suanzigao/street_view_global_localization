"""前 10 个点位完整三段式定位流程批量测试。

对 geochain_mini_10.csv 的每个点依次运行：
  Stage 1   Agent 自主导航 + 多模态推理（粗粒度城市/坐标）
  Stage 1.5 BEV 投影 + 招牌文字识别
  Stage 2   招牌空间投票 + 高斯核配准（街区级）
  Stage 3   BEV 路网 ↔ OSM 路网卷积匹配（亚十米级）

用法：
  .venv/bin/python test/batch_three_stage_10.py [max_steps]
"""
import os
import sys
import csv
import json
import datetime

os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"
os.environ["NO_PROXY"] = "localhost,127.0.0.1,localdomain,.localdomain"

BASE = "/home/suanzigao/PycharmProjects/street_view_global_localization"
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "localization"))

from dotenv import load_dotenv
load_dotenv()

from three_stage_localization import run_three_stage
from utils.geometry import calculate_gps_distance

CSV_PATH = os.path.join(BASE, "geochain_mini_10.csv")
BATCH_ROOT = os.path.join(BASE, "batch_streetview_results")


def read_points(csv_path):
    points = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row["name"].strip()
            lat = float(row["lat"])
            lng = float(row["lng"])
            if not name:
                continue
            points.append({"name": name, "lat": lat, "lng": lng})
    return points


def main():
    max_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    points = read_points(CSV_PATH)

    batch_dir = os.path.join(
        BATCH_ROOT,
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_three_stage_10",
    )
    os.makedirs(batch_dir, exist_ok=True)

    print(f"🚀 三段式完整流程批量测试：共 {len(points)} 个点，max_steps={max_steps}")
    print(f"   结果目录: {batch_dir}\n")

    summary_rows = []
    for idx, p in enumerate(points, 1):
        name = p["name"]
        point_dir = os.path.join(batch_dir, name)
        print(f"\n{'='*70}")
        print(f"[{idx}/{len(points)}] {name} ({p['lat']}, {p['lng']})")
        print(f"{'='*70}")

        row = {
            "name": name,
            "true_lat": p["lat"],
            "true_lng": p["lng"],
            "adopted_stage": "",
            "final_lat": None,
            "final_lng": None,
            "error_m": None,
        }
        try:
            summary = run_three_stage(
                point_dir, p["lat"], p["lng"], max_steps=max_steps, move_step=15.0
            )
            if summary:
                row["adopted_stage"] = summary.get("adopted_stage", "")
                fp = summary.get("final_position")
                if fp and fp.get("lat") is not None:
                    row["final_lat"] = fp["lat"]
                    row["final_lng"] = fp["lng"]
                    row["error_m"] = round(
                        calculate_gps_distance(fp["lat"], fp["lng"], p["lat"], p["lng"]), 1
                    )
        except Exception as e:
            print(f"❌ 该点失败: {e}")
            import traceback
            traceback.print_exc()
            row["adopted_stage"] = "FAILED"

        summary_rows.append(row)

    # 汇总
    print("\n" + "=" * 70)
    print("三段式定位结果汇总")
    print("=" * 70)
    valid = [r for r in summary_rows if r["error_m"] is not None]
    if valid:
        errors = [r["error_m"] for r in valid]
        print(f"成功定位: {len(valid)}/{len(summary_rows)} 个")
        print(f"平均误差: {sum(errors)/len(errors):.2f} 米")
        print(f"中位误差: {sorted(errors)[len(errors)//2]:.2f} 米")
        print(f"最小误差: {min(errors):.2f} 米")
        print(f"最大误差: {max(errors):.2f} 米")
        print("\n各点结果:")
        for r in sorted(valid, key=lambda x: x["error_m"]):
            print(f"  {r['name']}: {r['error_m']:.1f} 米 (采信 {r['adopted_stage']})")
    else:
        print("无有效定位结果")

    summary_path = os.path.join(batch_dir, "three_stage_summary.csv")
    fieldnames = ["name", "true_lat", "true_lng", "adopted_stage",
                  "final_lat", "final_lng", "error_m"]
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\n✅ 汇总已保存: {summary_path}")


if __name__ == "__main__":
    main()
