"""单点 Stage 1 测试（可指定坐标）。

用法：
  .venv/bin/python test/single_stage1_point.py <lat> <lng> <name> [max_steps]
"""
import os
import sys
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


def main():
    lat = float(sys.argv[1])
    lng = float(sys.argv[2])
    name = sys.argv[3]
    max_steps = int(sys.argv[4]) if len(sys.argv) > 4 else 20

    batch_dir = os.path.join(
        BASE, "batch_streetview_results",
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_stage1_single",
    )
    point_dir = os.path.join(batch_dir, name)

    print(f"🚀 单点 Stage 1 测试: {name} ({lat}, {lng}) max_steps={max_steps}")
    stage1 = stage1_coarse_localize(
        point_dir, lat, lng, max_steps=max_steps, move_step=15.0, reuse=False
    )
    print(f"\n✅ 完成: city={stage1.get('city')} coarse_gps={stage1.get('coarse_gps')}")
    print(f"结果目录: {point_dir}")


if __name__ == "__main__":
    main()
