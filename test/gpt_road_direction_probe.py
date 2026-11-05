"""测试 GPT 本身对街景图道路方向的视觉判断能力。

直接调用 GPT-4o，给它撞墙点的街景图，问它：这张图正对道路延伸方向吗？
对比 GPT 判断 vs Google 街景元数据的真实道路方向，评估 GPT 视觉判断的可靠性。

用法：
  .venv/bin/python test/gpt_road_direction_probe.py
"""
import os
import sys
import base64
import math
import requests
from PIL import Image
from io import BytesIO

os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"
os.environ["NO_PROXY"] = "localhost,127.0.0.1,localdomain,.localdomain"

BASE = "/home/suanzigao/PycharmProjects/street_view_global_localization"
sys.path.insert(0, BASE)

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

# 三个历史撞墙点（各 20 次撞墙）
TEST_POINTS = [
    {"name": "dS9EPpbrXLeyyu1vVUvT5g", "lat": 45.3449811, "lng": -75.7235353},
    {"name": "yJ39mtnttfZO_uSQ2Hjh6A", "lat": 60.1908444, "lng": 24.9142307},
    {"name": "0RWA4OaNg-EVqWqcvt0j2w", "lat": 45.2666417, "lng": -75.94045},
]

RETEST_DIR = os.path.join(BASE, "batch_streetview_results", "2026-10-27_00-34-39_hitwall_retest")

PROMPT = """Look at this street view image carefully and answer ONE question:

Is the road in this image running ALONG your viewing direction (extending away from you into the distance), or does it CROSS your view (running across the image from left to right)?

Judge by the road's two SIDE EDGES (curbs / lane lines / pavement borders):
- If the two side edges run AWAY from you and converge toward the image center (like train tracks going into the distance), the road is ALONG your heading.
- If the two side edges run ACROSS the image from left to right (a wide horizontal band), the road CROSSES your heading.

Answer ONLY in this JSON format:
{"road_orientation": "ahead" or "crossing", "confidence": "high/medium/low", "reason": "one short sentence"}
"""


def query_gpt(image_path):
    image = Image.open(image_path).convert("RGB")
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=95)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://api.laozhang.ai/v1",
    )

    content = [
        {"type": "text", "text": PROMPT},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}},
    ]
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": content}],
        temperature=0.1,
        max_tokens=300,
        timeout=60,
    )
    return resp.choices[0].message.content.strip()


def probe_road_truth(lat, lng, heading_deg=0.0):
    """用 Google 街景元数据探测：heading 方向 15m 处是否有街景。"""
    KEY = os.getenv("GOOGLE_API_KEY")
    R = 6378137.0
    ang = math.radians(heading_deg)
    d = 15.0
    lat2 = math.asin(
        math.sin(math.radians(lat)) * math.cos(d / R)
        + math.cos(math.radians(lat)) * math.sin(d / R) * math.cos(ang)
    )
    lng2 = math.radians(lng) + math.atan2(
        math.sin(ang) * math.sin(d / R) * math.cos(math.radians(lat)),
        math.cos(d / R) - math.sin(math.radians(lat)) * math.sin(lat2),
    )
    lat2 = math.degrees(lat2)
    lng2 = math.degrees(lng2)

    r = requests.get(
        "https://maps.googleapis.com/maps/api/streetview/metadata",
        params={"location": f"{lat2},{lng2}", "radius": 10, "key": KEY},
        timeout=15,
    )
    return r.json().get("status") == "OK"


def main():
    print("=" * 70)
    print("GPT 道路方向视觉判断测试")
    print("=" * 70)

    for p in TEST_POINTS:
        name = p["name"]
        img_path = os.path.join(RETEST_DIR, name, "streetview_images", "step_0_streetview.jpg")
        if not os.path.exists(img_path):
            print(f"\n⚠️ 找不到图像: {img_path}")
            continue

        print(f"\n{'='*70}")
        print(f"点位: {name} ({p['lat']}, {p['lng']})")
        print(f"图像: step_0 (初始 heading=0° 正北)")

        # GPT 判断
        try:
            gpt_ans = query_gpt(img_path)
            print(f"\n[GPT 判断]")
            print(f"  {gpt_ans}")
        except Exception as e:
            print(f"\n[GPT 判断] 失败: {e}")
            continue

        # 真实数据：朝北15m是否有街景
        north_ok = probe_road_truth(p["lat"], p["lng"], heading_deg=0.0)
        print(f"\n[真实数据] 初始 heading=0°(北) 15m 处街景: {'✅ 有' if north_ok else '❌ 无 (move forward 会撞墙)'}")

        print("-" * 70)

    print(f"\n✅ 测试完成")


if __name__ == "__main__":
    main()
