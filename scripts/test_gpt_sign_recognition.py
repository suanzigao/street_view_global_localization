"""用 GPT-4o 识别苏黎世点位的招牌裁剪图，对比 EasyOCR 效果。"""
import os
import sys
import glob
import json
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from PIL import Image
import numpy as np
import cv2
from openai import OpenAI
from dotenv import load_dotenv
from utils.file_utils import image_to_base64

load_dotenv()

LLM_MODEL = "gpt-4o"
LLM_BASE_URL = "https://api.laozhang.ai/v1"

CROP_DIR = "/home/suanzigao/PycharmProjects/street_view_global_localization/manual_streetview_results/2026-09-02_20-37-32_manual_lat47_3928_lng8_5308/text_output_vggt_depth/sign_crops"

PROMPT = """你是一个招牌文字识别助手。请识别图片中招牌、路牌、店铺标志、广告牌上最显眼、最有辨识度的文字（通常是店名、品牌名、地名）。

要求：
1. 只识别"能用于地点搜索"的文字（店名/品牌/地名），忽略装饰性文字、广告语、车牌等。
2. 如果图片中有可读的店名/品牌文字，text 填该文字。
3. 如果图片中没有可用于地点搜索的文字，text 填空字符串 ""。
4. confidence 填 0 到 1 之间的数字，表示你对识别结果的确信程度。

只输出 JSON，不要输出任何其他内容，格式如下：
{{
  "text": "识别到的招牌文字",
  "confidence": 0.9
}}
"""


def recognize(crop_np, client):
    if crop_np is None or crop_np.size == 0:
        return "", 0.0
    crop_pil = Image.fromarray(crop_np.astype(np.uint8))
    b64 = image_to_base64(crop_pil)
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }],
            temperature=0.1,
            max_tokens=100,
            timeout=60,
        )
        content = resp.choices[0].message.content.strip()
        m = re.search(r'\{[\s\S]*\}', content)
        if m:
            data = json.loads(m.group(0))
            text = data.get("text", "").strip()
            conf = data.get("confidence", 0.0)
            try:
                conf = float(conf)
            except (TypeError, ValueError):
                conf = 0.0
            conf = max(0.0, min(1.0, conf))
            return text, conf
    except Exception as e:
        print(f"    识别失败: {e}")
    return "", 0.0


def main():
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=LLM_BASE_URL,
    )

    files = sorted(glob.glob(os.path.join(CROP_DIR, '*.png')))
    print(f"共 {len(files)} 张裁剪图\n")

    results = []
    for f in files:
        name = os.path.basename(f)
        im = Image.open(f).convert('RGB')
        arr = np.array(im)
        h, w = arr.shape[:2]

        # 与 EasyOCR 一致的放大预处理
        scale = max(1.0, 320.0 / max(w, h))
        if scale > 1.0:
            arr = cv2.resize(arr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)

        text, conf = recognize(arr, client)
        results.append({"file": name, "size": [w, h], "text": text, "confidence": conf})
        if text:
            print(f"{name} ({w}x{h}) -> '{text}' conf={conf:.2f}")
        else:
            print(f"{name} ({w}x{h}) -> (无)")

    out_path = os.path.join(CROP_DIR, 'gpt_recognition_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    n_text = sum(1 for r in results if r['text'])
    print(f"\n共识别出 {n_text}/{len(results)} 张有文字，结果已保存到 {out_path}")


if __name__ == '__main__':
    main()
