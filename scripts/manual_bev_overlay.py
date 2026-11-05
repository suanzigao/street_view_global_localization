"""用 matplotlib 交互式地将 BEV 标注图叠加到底图（如卫星截图）上。

交互操作：
  鼠标左键拖动：移动 BEV
  鼠标滚轮：缩放 BEV
  鼠标右键拖动：旋转 BEV
  键盘 [a]/[z]：增大/减小透明度
  键盘 [s]：保存当前结果
  键盘 [r]：重置
  键盘 [q] 或 [Esc]：退出
  键盘 [←→↑↓]：微调位置（1 像素）
  键盘 [,] [.]：微调旋转（1 度）
"""

import os
import math
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton


class BEVOverlayGUI:
    def __init__(self, background_path, overlay_path, output_path):
        self.bg = cv2.imread(background_path)
        self.bg = cv2.cvtColor(self.bg, cv2.COLOR_BGR2RGB)
        self.overlay_orig = cv2.imread(overlay_path)
        self.overlay_orig = cv2.cvtColor(self.overlay_orig, cv2.COLOR_BGR2RGB)

        if self.bg is None or self.overlay_orig is None:
            raise ValueError("无法读取图像")

        self.output_path = output_path
        self.bg_h, self.bg_w = self.bg.shape[:2]
        self.ov_h, self.ov_w = self.overlay_orig.shape[:2]

        # 初始参数
        self.scale = 0.35 * self.bg_w / self.ov_w
        self.angle = 0.0
        self.alpha = 0.5
        # overlay 中心放底图中心
        self.cx = self.bg_w / 2.0
        self.cy = self.bg_h / 2.0

        self.dragging = False
        self.rotating = False
        self.last_xy = None

        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.ax.set_title('BEV 叠加对齐器：左键拖动移动 / 滚轮缩放 / 右键拖动旋转 / a,z透明度 / s保存 / q退出')
        self.ax.imshow(self.bg)

        self.overlay_img = None
        self.redraw()

        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        print("\n操作说明：")
        print("  左键拖动：移动 BEV")
        print("  滚轮：缩放 BEV")
        print("  右键拖动：旋转 BEV")
        print("  [a] 增大透明度  [z] 减小透明度")
        print("  [s] 保存  [r] 重置  [q]/Esc 退出")
        print("  [←→↑↓] 微调位置  [,] [.] 微调旋转")
        plt.show()

    def get_offsets(self):
        # overlay 中心 (cx, cy)，缩放+旋转后画布尺寸
        cos_a = abs(math.cos(math.radians(self.angle)))
        sin_a = abs(math.sin(math.radians(self.angle)))
        new_w = self.ov_w * self.scale * cos_a + self.ov_h * self.scale * sin_a
        new_h = self.ov_w * self.scale * sin_a + self.ov_h * self.scale * cos_a
        extent_w = new_w
        extent_h = new_h
        left = self.cx - extent_w / 2
        bottom = self.cy - extent_h / 2
        return left, bottom, extent_w, extent_h

    def get_transformed(self):
        center_x = self.ov_w / 2
        center_y = self.ov_h / 2
        M = cv2.getRotationMatrix2D((center_x, center_y), self.angle, self.scale)
        cos_a = abs(math.cos(math.radians(self.angle)))
        sin_a = abs(math.sin(math.radians(self.angle)))
        new_w = int(self.ov_w * self.scale * cos_a + self.ov_h * self.scale * sin_a)
        new_h = int(self.ov_w * self.scale * sin_a + self.ov_h * self.scale * cos_a)
        new_w = max(new_w, 1)
        new_h = max(new_h, 1)
        M[0, 2] += (new_w - self.ov_w) / 2
        M[1, 2] += (new_h - self.ov_h) / 2
        return cv2.warpAffine(self.overlay_orig, M, (new_w, new_h), borderValue=(0, 0, 0))

    def redraw(self):
        if self.overlay_img is not None:
            self.overlay_img.remove()
            self.overlay_img = None

        transformed = self.get_transformed()
        # 非黑色像素作为 alpha mask
        gray = cv2.cvtColor(transformed, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        rgba = np.dstack([transformed, mask.astype(np.float32) / 255.0 * self.alpha])

        left, bottom, extent_w, extent_h = self.get_offsets()
        self.overlay_img = self.ax.imshow(
            rgba,
            extent=[left, left + extent_w, bottom + extent_h, bottom],
            interpolation='nearest', zorder=2,
        )
        self.ax.set_xlim(0, self.bg_w)
        self.ax.set_ylim(self.bg_h, 0)
        self.ax.set_title(
            f'scale={self.scale:.4f} angle={self.angle:.1f} alpha={self.alpha:.2f} center=({self.cx:.0f},{self.cy:.0f})'
        )
        self.fig.canvas.draw_idle()

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        if event.button == MouseButton.LEFT:
            self.dragging = True
            self.last_xy = (event.xdata, event.ydata)
        elif event.button == MouseButton.RIGHT:
            self.rotating = True
            self.last_xy = (event.xdata, event.ydata)

    def on_release(self, event):
        if event.button == MouseButton.LEFT:
            self.dragging = False
        elif event.button == MouseButton.RIGHT:
            self.rotating = False

    def on_motion(self, event):
        if not self.last_xy or event.xdata is None:
            return
        dx = event.xdata - self.last_xy[0]
        dy = event.ydata - self.last_xy[1]
        if self.dragging:
            self.cx += dx
            self.cy += dy
            self.last_xy = (event.xdata, event.ydata)
            self.redraw()
        elif self.rotating:
            self.angle += dx * 0.2
            self.last_xy = (event.xdata, event.ydata)
            self.redraw()

    def on_scroll(self, event):
        factor = 1.1 if event.button == 'up' else 0.9
        self.scale *= factor
        self.redraw()

    def on_key(self, event):
        k = event.key
        if k == 's':
            self.save()
        elif k == 'r':
            self.scale = 0.35 * self.bg_w / self.ov_w
            self.angle = 0.0
            self.alpha = 0.5
            self.cx = self.bg_w / 2.0
            self.cy = self.bg_h / 2.0
            self.redraw()
        elif k == 'a':
            self.alpha = min(1.0, self.alpha + 0.05)
            self.redraw()
        elif k == 'z':
            self.alpha = max(0.0, self.alpha - 0.05)
            self.redraw()
        elif k == 'left':
            self.cx -= 1
            self.redraw()
        elif k == 'right':
            self.cx += 1
            self.redraw()
        elif k == 'up':
            self.cy -= 1
            self.redraw()
        elif k == 'down':
            self.cy += 1
            self.redraw()
        elif k == ',':
            self.angle -= 1.0
            self.redraw()
        elif k == '.':
            self.angle += 1.0
            self.redraw()
        elif k == 'q' or k == 'escape':
            plt.close(self.fig)

    def save(self):
        transformed = self.get_transformed()
        gray = cv2.cvtColor(transformed, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) / 255.0

        result = self.bg.copy()
        t_h, t_w = transformed.shape[:2]
        left, bottom, extent_w, extent_h = self.get_offsets()
        ox = int(round(self.cx - extent_w / 2))
        oy = int(round(self.cy - extent_h / 2))

        src_x_start = max(0, -ox)
        src_y_start = max(0, -oy)
        src_x_end = min(t_w, self.bg_w - ox)
        src_y_end = min(t_h, self.bg_h - oy)
        dst_x_start = max(0, ox)
        dst_y_start = max(0, oy)

        if src_x_end > src_x_start and src_y_end > src_y_start:
            patch = transformed[src_y_start:src_y_end, src_x_start:src_x_end]
            patch_mask = mask_3ch[src_y_start:src_y_end, src_x_start:src_x_end]
            bg_patch = result[dst_y_start:dst_y_start + (src_y_end - src_y_start),
                              dst_x_start:dst_x_start + (src_x_end - src_x_start)]
            blended = bg_patch * (1 - self.alpha * patch_mask) + patch * (self.alpha * patch_mask)
            result[dst_y_start:dst_y_start + (src_y_end - src_y_start),
                   dst_x_start:dst_x_start + (src_x_end - src_x_start)] = blended.astype(np.uint8)

        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        cv2.imwrite(self.output_path, result_bgr)
        print(f"已保存: {self.output_path}")


def main():
    import sys
    base = '/home/suanzigao/PycharmProjects/street_view_global_localization/manual_streetview_results/2026-09-09_17-39-52_manual_lat47_3928_lng8_5308/text_output_vggt_depth'
    background = os.path.join(base, '截图 2026-09-09 20-29-49.png')
    overlay = os.path.join(base, 'bev_with_sign_markers.png')
    output = os.path.join(base, 'bev_on_satellite_overlay.png')

    if len(sys.argv) >= 2:
        background = sys.argv[1]
    if len(sys.argv) >= 3:
        overlay = sys.argv[2]
    if len(sys.argv) >= 4:
        output = sys.argv[3]

    print(f"底图: {background}")
    print(f"叠加图: {overlay}")
    print(f"输出: {output}")

    gui = BEVOverlayGUI(background, overlay, output)


if __name__ == '__main__':
    main()
