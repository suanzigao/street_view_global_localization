"""真实端到端验证：确认 GPT-4o 能稳定输出新的 hypothesis 数组格式。

跑 3 步（足以观察格式稳定性），输出到独立目录，不覆盖已有数据。
"""
import os
import sys

os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"
os.environ["NO_PROXY"] = "localhost,127.0.0.1,localdomain,.localdomain"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "localization"))

from dotenv import load_dotenv
load_dotenv()

from three_stage_localization import stage1_coarse_localize

BASE = "/home/suanzigao/PycharmProjects/street_view_global_localization"
LAT = 55.6463583
LNG = 37.7249981

point_dir = os.path.join(BASE, "manual_streetview_results",
                         "2026-10-20_hypothesis_array_verify")

print("🚀 真实端到端验证：hypothesis 数组格式（3 步）\n")
stage1 = stage1_coarse_localize(point_dir, LAT, LNG, max_steps=3, reuse=False)

print("\n" + "=" * 60)
print("验证结果")
print("=" * 60)
print(f"predicted_city: {stage1.get('city')}")
print(f"\nhypothesis_evidence_map (结构化累积):")
import json
print(json.dumps(stage1.get("hypothesis_evidence_map", []), ensure_ascii=False, indent=2))
print(f"\nlocation_hypothesis (文本摘要):\n{stage1.get('location_hypothesis')}")
print(f"\nevidence_bank 条数: {len(stage1.get('evidence_bank', []))}")
