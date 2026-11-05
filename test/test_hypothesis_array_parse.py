"""快速逻辑验证：hypothesis 数组格式解析 + 累积 + evidence_bank 结构化字段。

不联网、不实例化 env.__init__（绕过街景/SAM3 网络副作用），
直接构造最小实例验证 parse_gpt_decision 与证据库的核心逻辑。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.env import StreetViewAutoLocEnv


def _make_env():
    env = object.__new__(StreetViewAutoLocEnv)
    env.current_step = 0
    env.hypothesis_evidence_map = []
    env.location_hypothesis = ""
    env.evidence_bank = []
    env.predicted_city = ""
    env.segmented_image_history = {}
    env.ACTION_MAP = StreetViewAutoLocEnv.ACTION_MAP
    return env


def test_array_parse_and_accumulate():
    env = _make_env()

    r1 = '''```json
    {"reason": "look around", "hypothesis": [
        {"evidence": "German road signs", "region_city": "Zürich, Switzerland"},
        {"evidence": "FCZ graffiti", "region_city": "Zürich, Switzerland"}
    ], "city": "Zürich", "action": "turn right"}
    ```'''
    action, reason, seg, search = env.parse_gpt_decision(r1)
    assert action == "turn right", action
    assert len(env.hypothesis_evidence_map) == 2, env.hypothesis_evidence_map
    assert env.predicted_city == "Zürich", env.predicted_city

    # 第二步：累积不覆盖
    env.current_step = 1
    r2 = '''```json
    {"reason": "search sign", "hypothesis": [
        {"evidence": "Nykredit @ (55.682, 12.570)", "region_city": "Copenhagen, Denmark"}
    ], "city": "Copenhagen", "action": "search: Nykredit, Copenhagen"}
    ```'''
    env.parse_gpt_decision(r2)
    assert len(env.hypothesis_evidence_map) == 3, env.hypothesis_evidence_map
    assert env.predicted_city == "Copenhagen", env.predicted_city

    summary = env._build_hypothesis_summary()
    print("=== hypothesis summary ===")
    print(summary)
    assert "German road signs" in summary
    assert "Nykredit" in summary
    assert "Zürich" in summary and "Copenhagen" in summary

    print("\n[OK] 数组解析 + 跨步累积 + 城市覆盖 全部通过")
    return env


def test_string_compat():
    env = _make_env()
    r = '''```json
    {"reason": "x", "hypothesis": "Switzerland, Zürich (~47.37, 8.54), confidence: medium", "city": "Zürich", "action": "turn left"}
    ```'''
    env.parse_gpt_decision(r)
    assert len(env.hypothesis_evidence_map) == 1, env.hypothesis_evidence_map
    assert env.hypothesis_evidence_map[0]["region_city"] == ""
    print("[OK] 旧版字符串 hypothesis 兼容分支通过")


def test_evidence_bank_structured():
    env = _make_env()
    env.current_step = 2
    env.add_evidence("search", '搜索 "Nykredit, Copenhagen" → Nykredit Bank @ (55.6820, 12.5700)',
                     name="Nykredit Bank", lat=55.6820, lng=12.5700)
    env.add_evidence("segment", "分割 [building, road sign] → 检测到 3 个对象",
                     image_b64="FAKE_B64")
    e = env.evidence_bank
    assert e[0]["kind"] == "search" and e[0]["lat"] == 55.6820, e
    assert e[1]["kind"] == "segment" and e[1]["image_b64"] == "FAKE_B64", e
    print("[OK] evidence_bank 结构化字段（search 坐标 / segment 图像）通过")
    print("    evidence_bank =")
    import json
    print(json.dumps(e, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    test_array_parse_and_accumulate()
    test_string_compat()
    test_evidence_bank_structured()
    print("\n✅ 全部验证通过")
