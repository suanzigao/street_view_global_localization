"""
手动调整BEV在卫星图上的位置、大小和旋转角度
交互操作：
  鼠标左键拖动：移动BEV
  鼠标滚轮：缩放BEV
  鼠标右键拖动：旋转BEV
  按键 [s]：保存当前结果
  按键 [r]：重置
  按键 [q] 或 ESC：退出
  按键 +/-：微调缩放
  按键 ↑↓←→：微调位置
  按键 [/]：逆时针旋转1度
  按键 [']：顺时针旋转1度
"""

import os
import json
import math
import cv2
import numpy as np


class BEValigner:
    def __init__(self, satellite_path, bev_path, output_dir, point_dir):
        self.sat_img = cv2.imread(satellite_path)
        self.bev_orig = cv2.imread(bev_path)
        self.output_dir = output_dir
        self.point_dir = point_dir

        if self.sat_img is None or self.bev_orig is None:
            raise ValueError("无法读取图像")

        self.sat_h, self.sat_w = self.sat_img.shape[:2]
        self.bev_h, self.bev_w = self.bev_orig.shape[:2]

        # 加载元数据
        bev_meta_path = os.path.join(point_dir, 'projection_output', 'bev_lowres_cropped_metadata.json')
        if not os.path.exists(bev_meta_path):
            bev_meta_path = os.path.join(point_dir, 'projection_output', 'bev_lowres_metadata.json')

        with open(bev_meta_path, 'r') as f:
            bev_meta = json.load(f)

        point_meta_path = os.path.join(point_dir, 'streetview_images', 'step_0_metadata.json')
        with open(point_meta_path, 'r') as f:
            point_meta = json.load(f)

        self.initial_px = bev_meta['initial_pixel_x']
        self.initial_py = bev_meta['initial_pixel_y']
        self.actual_lat = point_meta['actual_lat']
        self.actual_lng = point_meta['actual_lng']

        # 读取匹配结果
        match_result_path = os.path.join(point_dir, 'matching_results', 'match_result.json')
        self.match_point = None
        if os.path.exists(match_result_path):
            with open(match_result_path, 'r') as f:
                match_info = json.load(f)
            self.match_lat = match_info['match_lat']
            self.match_lng = match_info['match_lng']
            self.match_error = match_info.get('distance_error', None)

            # 从verification_result读取卫星图分辨率
            veri_path = os.path.join(output_dir, 'verification_result.json')
            if os.path.exists(veri_path):
                with open(veri_path, 'r') as f:
                    veri = json.load(f)
                self.meters_px_x = veri['meters_per_pixel_x']
                self.meters_px_y = veri['meters_per_pixel_y']
                self.zoom = veri['zoom']

                sat_meta_path = os.path.join(output_dir, f'satellite_zoom{self.zoom}_metadata.json')
                with open(sat_meta_path, 'r') as f:
                    sat_meta = json.load(f)

                exact_px = sat_meta['exact_pixel']['x']
                exact_py = sat_meta['exact_pixel']['y']
                min_x = sat_meta['min_tile']['x']
                min_y = sat_meta['min_tile']['y']
                crop_x1 = int(exact_px - (self.sat_w / 2))
                crop_y1 = int(exact_py - (self.sat_h / 2))

                # 重新计算匹配点在裁剪后卫星图中的位置
                from bev_satellite_overlay import latlng_to_tile
                match_tile_x, match_tile_y = latlng_to_tile(self.match_lat, self.match_lng, self.zoom)
                self.match_px = (match_tile_x - min_x) * 256 - crop_x1
                self.match_py = (match_tile_y - min_y) * 256 - crop_y1
                self.match_point = (self.match_px, self.match_py)
        else:
            self.match_point = None

        # 初始参数
        self.scale = 1.0
        self.angle = 0.0
        self.offset_x = self.sat_w / 2 - self.initial_px
        self.offset_y = self.sat_h / 2 - self.initial_py

        # 鼠标状态
        self.dragging = False
        self.rotating = False
        self.last_mouse = (0, 0)

        # 显示窗口
        self.window_name = 'Manual BEV Align (drag=move, wheel=scale, right-drag=rotate)'
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1400, 1000)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        self.update_display()

    def get_transformed_bev(self):
        center_x = self.bev_w / 2
        center_y = self.bev_h / 2

        M = cv2.getRotationMatrix2D((center_x, center_y), self.angle, self.scale)

        # 计算旋转后图像大小
        cos_a = abs(math.cos(math.radians(self.angle)))
        sin_a = abs(math.sin(math.radians(self.angle)))
        new_w = int(self.bev_w * self.scale * cos_a + self.bev_h * self.scale * sin_a)
        new_h = int(self.bev_w * self.scale * sin_a + self.bev_h * self.scale * cos_a)

        M[0, 2] += (new_w - self.bev_w) / 2
        M[1, 2] += (new_h - self.bev_h) / 2

        rotated = cv2.warpAffine(self.bev_orig, M, (new_w, new_h), borderValue=(0, 0, 0))
        return rotated, M

    def overlay(self):
        result = self.sat_img.copy()
        bev_transformed, M = self.get_transformed_bev()
        bev_h, bev_w = bev_transformed.shape[:2]

        # 计算BEV左上角在卫星图中的位置
        # 初始点对应卫星图中心
        top_left_x = int(self.sat_w / 2 - self.initial_px * self.scale)
        top_left_y = int(self.sat_h / 2 - self.initial_py * self.scale)

        # 考虑旋转后的中心偏移
        cx = self.bev_w / 2
        cy = self.bev_h / 2
        new_cx = M[0, 0] * cx + M[0, 1] * cy + M[0, 2]
        new_cy = M[1, 0] * cx + M[1, 1] * cy + M[1, 2]
        offset_cx = new_cx - bev_w / 2
        offset_cy = new_cy - bev_h / 2

        # 修正：我们希望初始点在self.offset_x, self.offset_y处
        # 初始点在BEV中位置为 (self.initial_px, self.initial_py)
        # 旋转后初始点位置：
        init_new_x = M[0, 0] * self.initial_px + M[0, 1] * self.initial_py + M[0, 2]
        init_new_y = M[1, 0] * self.initial_px + M[1, 1] * self.initial_py + M[1, 2]

        overlay_x = int(self.offset_x - init_new_x)
        overlay_y = int(self.offset_y - init_new_y)

        # 裁剪范围
        src_x_start = max(0, -overlay_x)
        src_y_start = max(0, -overlay_y)
        src_x_end = min(bev_w, self.sat_w - overlay_x)
        src_y_end = min(bev_h, self.sat_h - overlay_y)

        dst_x_start = max(0, overlay_x)
        dst_y_start = max(0, overlay_y)

        if src_x_end > src_x_start and src_y_end > src_y_start:
            bev_patch = bev_transformed[src_y_start:src_y_end, src_x_start:src_x_end]
            overlay_patch = result[dst_y_start:dst_y_start + (src_y_end - src_y_start),
                                   dst_x_start:dst_x_start + (src_x_end - src_x_start)]

            mask = cv2.cvtColor(bev_patch, cv2.COLOR_BGR2GRAY)
            _, mask_binary = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
            mask_normalized = cv2.cvtColor(mask_binary, cv2.COLOR_GRAY2BGR) / 255.0
            alpha = 0.5

            for c in range(3):
                overlay_patch[:, :, c] = (
                    overlay_patch[:, :, c] * (1 - mask_normalized[:, :, c]) * (1 - alpha) +
                    bev_patch[:, :, c] * alpha * mask_normalized[:, :, c] +
                    overlay_patch[:, :, c] * (1 - mask_normalized[:, :, c])
                )

        # 标注初始点
        cv2.circle(result, (int(self.offset_x), int(self.offset_y)), 8, (0, 0, 255), -1)
        cv2.circle(result, (int(self.offset_x), int(self.offset_y)), 12, (0, 0, 255), 2)

        # 标注OSM匹配点
        if self.match_point is not None:
            mp_x, mp_y = self.match_point
            cv2.circle(result, (int(mp_x), int(mp_y)), 8, (255, 0, 0), -1)
            cv2.circle(result, (int(mp_x), int(mp_y)), 12, (255, 0, 0), 2)
            cv2.line(result, (int(self.offset_x), int(self.offset_y)), (int(mp_x), int(mp_y)), (255, 255, 0), 2)

        # 绘制BEV边界框
        pts = np.array([
            [overlay_x, overlay_y],
            [overlay_x + bev_w, overlay_y],
            [overlay_x + bev_w, overlay_y + bev_h],
            [overlay_x, overlay_y + bev_h]
        ], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(result, [pts], True, (0, 255, 0), 2)

        # 添加文字信息
        info_text = f"scale={self.scale:.4f} angle={self.angle:.1f} offset=({self.offset_x:.1f}, {self.offset_y:.1f})"
        cv2.putText(result, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return result

    def update_display(self):
        result = self.overlay()
        cv2.imshow(self.window_name, result)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.last_mouse = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.rotating = True
            self.last_mouse = (x, y)
        elif event == cv2.EVENT_RBUTTONUP:
            self.rotating = False
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging:
                dx = x - self.last_mouse[0]
                dy = y - self.last_mouse[1]
                self.offset_x += dx
                self.offset_y += dy
                self.last_mouse = (x, y)
                self.update_display()
            elif self.rotating:
                dx = x - self.last_mouse[0]
                self.angle += dx * 0.1
                self.last_mouse = (x, y)
                self.update_display()
        elif event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                self.scale *= 1.1
            else:
                self.scale *= 0.9
            self.update_display()

    def save_result(self):
        result = self.overlay()
        save_path = os.path.join(self.output_dir, 'bev_on_satellite_manual_aligned.png')
        cv2.imwrite(save_path, result)

        params = {
            'scale': self.scale,
            'angle': self.angle,
            'offset_x': self.offset_x,
            'offset_y': self.offset_y,
            'initial_px_in_bev': self.initial_px,
            'initial_py_in_bev': self.initial_py,
            'satellite_size': [self.sat_w, self.sat_h]
        }
        params_path = os.path.join(self.output_dir, 'manual_alignment_params.json')
        with open(params_path, 'w') as f:
            json.dump(params, f, indent=2)

        print(f"✅ 已保存: {save_path}")
        print(f"✅ 参数已保存: {params_path}")

    def run(self):
        print("\n操作说明：")
        print("  鼠标左键拖动：移动BEV")
        print("  鼠标滚轮：缩放BEV")
        print("  鼠标右键拖动：旋转BEV")
        print("  [s] 保存当前结果")
        print("  [r] 重置")
        print("  [+/-] 微调缩放")
        print("  方向键 微调位置")
        print("  [/] 逆时针旋转1度, ['] 顺时针旋转1度")
        print("  [q] 或 ESC 退出")

        while True:
            key = cv2.waitKey(50) & 0xFF

            if key == ord('q') or key == 27:
                break
            elif key == ord('s'):
                self.save_result()
            elif key == ord('r'):
                self.scale = 1.0
                self.angle = 0.0
                self.offset_x = self.sat_w / 2 - self.initial_px
                self.offset_y = self.sat_h / 2 - self.initial_py
                self.update_display()
            elif key == ord('+') or key == ord('='):
                self.scale *= 1.05
                self.update_display()
            elif key == ord('-') or key == ord('_'):
                self.scale *= 0.95
                self.update_display()
            elif key == 81:  # 左
                self.offset_x -= 2
                self.update_display()
            elif key == 82:  # 上
                self.offset_y -= 2
                self.update_display()
            elif key == 83:  # 右
                self.offset_x += 2
                self.update_display()
            elif key == 84:  # 下
                self.offset_y += 2
                self.update_display()
            elif key == ord('/'):
                self.angle -= 1.0
                self.update_display()
            elif key == ord("'"):
                self.angle += 1.0
                self.update_display()

        cv2.destroyAllWindows()


def main():
    import sys

    if len(sys.argv) > 1:
        point_dir = sys.argv[1]
    else:
        point_dir = '/home/suanzigao/PycharmProjects/street_view_global_localization/manual_streetview_results/2026-07-22_19-25-23_manual_lat47_5731_lng19_1210'

    output_dir = os.path.join(point_dir, 'satellite_verification')
    satellite_path = os.path.join(output_dir, 'satellite_zoom18_crop500m.png')
    bev_path = os.path.join(point_dir, 'projection_output', 'fused_bev_lowres_cropped.png')

    if not os.path.exists(bev_path):
        bev_path = os.path.join(point_dir, 'projection_output', 'fused_bev_lowres.png')

    print(f"点位: {point_dir}")
    print(f"卫星图: {satellite_path}")
    print(f"BEV图: {bev_path}")

    aligner = BEValigner(satellite_path, bev_path, output_dir, point_dir)
    aligner.run()


if __name__ == "__main__":
    main()
