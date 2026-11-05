"""Stage 1 单点测试（方案 B：让模型真正思考 + 维护位置假设）。

跑一个点的 agent 导航 + 最终预测，验证：
1. 每步决策输出 hypothesis 并逐步收敛
2. 最终 prediction 利用收敛后的 location_hypothesis
"""
import os
import sys

# 代理配置（与 main.py 一致）
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"
os.environ["NO_PROXY"] = "localhost,127.0.0.1,localdomain,.localdomain"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "localization"))

from dotenv import load_dotenv
load_dotenv()

from three_stage_localization import stage1_coarse_localize

BASE = "/home/suanzigao/PycharmProjects/street_view_global_localization"

# geochain_mini_10.csv 点位（赫尔辛基）
NAME = "aDapL3J-jB-pQ3oVCzzAsw"
LAT = 60.1719367
LNG = 24.7787122
TRUE = (60.1719367, 24.7787122)

# 用全新目录，避免复用旧数据
point_dir = os.path.join(BASE, "manual_streetview_results",
                         "2026-10-20_promptv10_helsinki")

max_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 10

print(f"🚀 测试点位: {NAME} ({LAT}, {LNG})")
print(f"   探索步数: {max_steps}")
print(f"   输出目录: {point_dir}\n")

stage1 = stage1_coarse_localize(point_dir, LAT, LNG, max_steps=max_steps, reuse=False)

print("\n" + "=" * 60)
print("Stage 1 测试结果")
print("=" * 60)
print(f"城市: {stage1.get('city')}")
print(f"粗定位 coarse_gps: {stage1.get('coarse_gps')}")
print(f"\n📍 最终位置假设 location_hypothesis:\n  {stage1.get('location_hypothesis') or '(空)'}")
print(f"\n📝 最终预测 final_prediction:\n  {stage1.get('final_prediction')}")

# 计算误差
cg = stage1.get("coarse_gps")
if cg and cg.get("lat") is not None:
    import math
    R = 6371000.0
    dlat = math.radians(cg["lat"] - TRUE[0])
    dlng = math.radians(cg["lng"] - TRUE[1])
    a = math.sin(dlat/2)**2 + math.cos(math.radians(TRUE[0])) * math.cos(math.radians(cg["lat"])) * math.sin(dlng/2)**2
    err = R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    print(f"\n📏 Stage 1 定位误差: {err:.1f} 米")
