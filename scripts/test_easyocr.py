"""用 EasyOCR 测试招牌裁剪图，支持放大预处理。"""
import os
import sys
import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from PIL import Image
import numpy as np

CROP_DIR = "/home/suanzigao/PycharmProjects/street_view_global_localization/manual_streetview_results/2026-07-22_19-25-23_manual_lat47_5731_lng19_1210/text_output_vggt_depth/sign_crops"


def main():
    import easyocr
    reader = easyocr.Reader(['en'], gpu=True)

    files = sorted(glob.glob(os.path.join(CROP_DIR, '*.png')))
    print(f"共 {len(files)} 张裁剪图\n")

    for f in files:
        name = os.path.basename(f)
        im = Image.open(f).convert('RGB')
        w, h = im.size

        # 放大到至少 160 像素宽
        scale = max(1.0, 160.0 / max(w, h))
        if scale > 1.0:
            im_big = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        else:
            im_big = im

        results = reader.readtext(f)
        texts_small = [r[1] for r in results]
        joined_small = ' | '.join(texts_small) if texts_small else '(无)'

        results_big = reader.readtext(np.array(im_big))
        texts_big = [r[1] for r in results_big]
        joined_big = ' | '.join(texts_big) if texts_big else '(无)'

        print(f"{name} ({w}x{h}, 放大{scale:.1f}x)")
        print(f"    原尺寸: {joined_small}")
        print(f"    放大后: {joined_big}")


if __name__ == '__main__':
    main()
