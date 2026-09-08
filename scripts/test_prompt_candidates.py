"""测试不同提示词对step_2街景图像中文字招牌的分割效果"""
import os
import sys
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tools.sam3_tool import SAM3Tool

IMAGE_PATH = "/home/suanzigao/PycharmProjects/street_view_global_localization/manual_streetview_results/2026-09-02_20-37-32_manual_lat47_3928_lng8_5308/streetview_images/step_2_streetview.jpg"
OUTPUT_DIR = "/home/suanzigao/PycharmProjects/street_view_global_localization/prompt_test_results"

CANDIDATE_PROMPTS = [
    "shop sign",
    "restaurant sign",
    "storefront sign",
    "signboard",
    "text sign",
    "advertising sign",
    "business sign",
    "street sign",
    "sign",
    "facade sign",
    "lettering",
    "store name",
]

COLORS = [
    (0, 255, 0), (255, 165, 0), (0, 0, 255), (255, 0, 255),
    (255, 255, 0), (0, 255, 255), (128, 0, 128), (128, 128, 0),
    (0, 128, 128), (128, 128, 128), (255, 0, 0), (0, 128, 0),
]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    image_pil = Image.open(IMAGE_PATH).convert('RGB')
    image_np = np.array(image_pil)
    h, w = image_np.shape[:2]

    sam3_tool = SAM3Tool()
    sam3_tool.initialize()

    summary = []

    for idx, prompt in enumerate(CANDIDATE_PROMPTS):
        print(f"\n[{idx+1}/{len(CANDIDATE_PROMPTS)}] 测试提示词: '{prompt}'")
        try:
            result = sam3_tool.get_segmented_images(image_pil, [prompt])
        except Exception as e:
            print(f"  ❌ 调用失败: {e}")
            continue

        masks_found = 0
        overlay = image_np.copy()

        for seg_result in result.get('results', []):
            if not seg_result.get('success') or not seg_result.get('masks'):
                continue

            for mask_idx, mask in enumerate(seg_result.get('masks', [])):
                mask_np = np.array(mask)
                if mask_np.ndim == 3:
                    mask_np = mask_np[0]
                if mask_np.max() <= 1:
                    mask_np = (mask_np * 255).astype(np.uint8)
                else:
                    mask_np = mask_np.astype(np.uint8)

                if mask_np.shape[:2] != (h, w):
                    mask_img = Image.fromarray(mask_np, mode='L')
                    mask_img = mask_img.resize((w, h), Image.NEAREST)
                    mask_np = np.array(mask_img)

                pixel_count = np.count_nonzero(mask_np)
                if pixel_count < 20:
                    continue

                masks_found += 1
                color = COLORS[mask_idx % len(COLORS)]

                ys, xs = np.where(mask_np > 0)
                x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

                for c in range(3):
                    overlay[:, :, c] = np.where(mask_np > 0, color[c], overlay[:, :, c])

                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3)
                cv2.putText(overlay, f"{prompt}#{mask_idx} {pixel_count}px",
                            (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        alpha = 0.45
        blended = cv2.addWeighted(overlay, 1 - alpha, image_np, alpha, 0)

        safe_name = prompt.replace(' ', '_')
        save_path = os.path.join(OUTPUT_DIR, f"step_2_{safe_name}.png")
        cv2.imwrite(save_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))

        print(f"  检测到 {masks_found} 个区域, 已保存: {save_path}")
        summary.append({'prompt': prompt, 'detections': masks_found})

    sam3_tool.close()

    print("\n" + "=" * 50)
    print("测试结果汇总:")
    print("=" * 50)
    for s in sorted(summary, key=lambda x: -x['detections']):
        print(f"  '{s['prompt']}': {s['detections']} 个检测")


if __name__ == '__main__':
    main()
