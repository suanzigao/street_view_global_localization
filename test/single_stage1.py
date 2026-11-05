"""单个点位仅 Stage 1 快速测试。

用于验证导航提示词改动的行为效果（前方方位显式、路口环顾四周）。

用法：
  .venv/bin/python test/single_stage1.py [max_steps]
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

# 默认测第一个点（莫斯科，之前 Step 11 出现过带红绿灯的大路口）
LAT = 55.6463583
LNG = 37.7249981
NAME = "E-fIB3K6E-Z-GSW_xI-rSw"


def main():
    max_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 20

    batch_dir = os.path.join(
        BASE, "batch_streetview_results",
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_stage1_single",
    )
    point_dir = os.path.join(batch_dir, NAME)

    print(f"🚀 单点 Stage 1 测试: {NAME} ({LAT}, {LNG}) max_steps={max_steps}")
    stage1 = stage1_coarse_localize(
        point_dir, LAT, LNG, max_steps=max_steps, move_step=15.0, reuse=False
    )
    print(f"\n✅ 完成: city={stage1.get('city')} coarse_gps={stage1.get('coarse_gps')}")
    print(f"结果目录: {point_dir}")


if __name__ == "__main__":
    main()
