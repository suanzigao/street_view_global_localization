"""将两张 BEV 图按世界坐标对齐后叠加，体现上采样前后覆盖范围与细节的变化。"""
import os
import sys
import json
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def load_meta(meta_path):
    with open(meta_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def affine_from_meta(meta, res, ux_min, uy_max):
    """由 metadata 计算从该图像素坐标到统一画布像素坐标的平移 (tx, ty)。"""
    x_min, x_max = meta['x_range']
    y_min, y_max = meta['y_range']
    padding = meta.get('padding', 5)
    h = meta['bev_size'][1]
    tx = (x_min - padding - ux_min) * res
    ty = (uy_max - y_min + padding) * res - (h - 1)
    return tx, ty


def warp_to_canvas(img, meta, res, canvas_w, canvas_h, ux_min, uy_max):
    tx, ty = affine_from_meta(meta, res, ux_min, uy_max)
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    warped = cv2.warpAffine(img, M, (canvas_w, canvas_h), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return warped


def main():
    if len(sys.argv) > 1:
        base = sys.argv[1].rstrip('/')
    else:
        base = '/home/suanzigao/PycharmProjects/street_view_global_localization/manual_streetview_results/2026-07-22_19-25-23_manual_lat47_5731_lng19_1210'
    img1_path = os.path.join(base, 'projection_output', 'fused_bev_optimized.png')
    img2_path = os.path.join(base, 'projection_output_upsampled_2x', 'fused_bev_optimized.png')
    meta1_path = os.path.join(base, 'projection_output', 'bev_metadata.json')
    meta2_path = os.path.join(base, 'projection_output_upsampled_2x', 'bev_metadata.json')

    img1 = cv2.imread(img1_path)  # BGR
    img2 = cv2.imread(img2_path)
    meta1 = load_meta(meta1_path)
    meta2 = load_meta(meta2_path)

    res = 10
    # 统一世界范围 = 两张图有数据范围的并集（留 2m 边距）
    ux_min = min(meta1['x_range'][0], meta2['x_range'][0]) - 2
    ux_max = max(meta1['x_range'][1], meta2['x_range'][1]) + 2
    uy_min = min(meta1['y_range'][0], meta2['y_range'][0]) - 2
    uy_max = max(meta1['y_range'][1], meta2['y_range'][1]) + 2
    canvas_w = int((ux_max - ux_min) * res)
    canvas_h = int((uy_max - uy_min) * res)

    w1 = warp_to_canvas(img1, meta1, res, canvas_w, canvas_h, ux_min, uy_max)
    w2 = warp_to_canvas(img2, meta2, res, canvas_w, canvas_h, ux_min, uy_max)

    # 亮度
    L1 = cv2.cvtColor(w1, cv2.COLOR_BGR2GRAY)
    L2 = cv2.cvtColor(w2, cv2.COLOR_BGR2GRAY)

    # 覆盖 mask（非黑色像素）
    m1 = (L1 > 10).astype(np.uint8)
    m2 = (L2 > 10).astype(np.uint8)
    both = (m1 & m2).astype(np.uint8)
    only1 = (m1 & (1 - m2)).astype(np.uint8)
    only2 = (m2 & (1 - m1)).astype(np.uint8)

    # 1) 双色叠加：青 = 只有1x，红 = 只有2x，白 = 重叠
    overlay = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    overlay[:, :, 0] = L2  # B/R 通道放 2x 亮度（cv2 是 BGR，这里用 BGR 顺序）
    overlay[:, :, 1] = L1
    overlay[:, :, 2] = L1
    # 纠正：用 BGR 顺序，青色 = (255,255,0) 在 BGR 里是 (255,255,0)，红色 = (0,0,255)
    # 重叠区域增强为亮色
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)  # 先不用

    # 重新构造：BGR 顺序
    # 青 = 只有1x -> B=高,G=高,R=低
    # 红 = 只有2x -> B=低,G=低,R=高
    # 白 = 重叠 -> B,G,R 都高
    ov = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    ov[:, :, 0] = L1  # B
    ov[:, :, 1] = L1  # G
    ov[:, :, 2] = L2  # R
    # 这样：只有1x -> B=L1,G=L1,R=0 青色；只有2x -> B=0,G=0,R=L2 红色；重叠 -> 白(若L1~L2)

    out_overlay = os.path.join(base, 'projection_output', 'bev_compare_overlay.png')
    cv2.imwrite(out_overlay, ov)

    # 2) 差异标注图：1x 灰底 + 2x 新增覆盖区域标红
    diff = cv2.cvtColor(w1, cv2.COLOR_BGR2GRAY)
    diff_bgr = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
    diff_bgr[only2 > 0] = (0, 0, 255)  # 红色标注 2x 新增区域
    out_diff = os.path.join(base, 'projection_output', 'bev_compare_diff.png')
    cv2.imwrite(out_diff, diff_bgr)

    # 3) 并排对比图
    h = max(img1.shape[0], img2.shape[0])
    gap = 20
    canvas = np.full((h, img1.shape[1] + gap + img2.shape[1], 3), 255, dtype=np.uint8)
    canvas[:img1.shape[0], :img1.shape[1]] = img1
    canvas[:img2.shape[0], img1.shape[1] + gap:] = img2
    cv2.putText(canvas, '1x (original)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    cv2.putText(canvas, '2x (upsampled)', (img1.shape[1] + gap + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    out_side = os.path.join(base, 'projection_output', 'bev_compare_side_by_side.png')
    cv2.imwrite(out_side, canvas)

    # 统计
    total1 = int(m1.sum())
    total2 = int(m2.sum())
    both_n = int(both.sum())
    only1_n = int(only1.sum())
    only2_n = int(only2.sum())

    print(f'统一画布: {canvas_w}x{canvas_h}px, 世界范围 x[{ux_min:.1f},{ux_max:.1f}] y[{uy_min:.1f},{uy_max:.1f}]')
    print(f'1x 覆盖像素: {total1}  ({meta1["x_range"][1]-meta1["x_range"][0]:.1f} x {meta1["y_range"][1]-meta1["y_range"][0]:.1f} m)')
    print(f'2x 覆盖像素: {total2}  ({meta2["x_range"][1]-meta2["x_range"][0]:.1f} x {meta2["y_range"][1]-meta2["y_range"][0]:.1f} m)')
    print(f'重叠像素: {both_n} ({both_n/max(total1,1)*100:.1f}% of 1x)')
    print(f'只有1x: {only1_n} ({only1_n/max(total1,1)*100:.1f}% of 1x)')
    print(f'只有2x(新增覆盖): {only2_n} ({only2_n/max(total2,1)*100:.1f}% of 2x)')
    print(f'\n输出:')
    print(f'  叠加图: {out_overlay}')
    print(f'  差异图: {out_diff}')
    print(f'  并排图: {out_side}')


if __name__ == '__main__':
    main()
