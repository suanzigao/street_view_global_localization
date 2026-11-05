"""BEV 网格功能验证：初始点居中 + 网格线 + 固定分辨率。

不联网、不加载 SAM3：用最小 env 实例 + 预填道路 mask 缓存，
直接验证 _generate_bev_image 的栅格化逻辑。
"""
import os
import sys
import tempfile
import base64
from io import BytesIO

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.env import StreetViewAutoLocEnv


def _make_env():
    env = object.__new__(StreetViewAutoLocEnv)
    env.use_bev = True
    env.bev_pixels_per_meter = 1.0
    env.move_step = 15.0
    env.use_sam3 = False
    env.sam3_tool = object()  # 非 None，配合预填 _road_mask_cache 跳过真实 SAM3
    env.fov_default = 60
    env.image_b64_history = {}
    env.image_path_history = {}
    env.image_meta_history = {}
    env._road_mask_cache = {}
    env._bev_cache_b64 = None
    env._bev_cache_key = None
    return env


def test_bev_grid():
    env = _make_env()

    # 构造道路 mask：640x640 图像下方中央一小块，代表正前方路面（heading=0, pitch=0）
    mask = np.zeros((640, 640), dtype=bool)
    mask[400:520, 300:340] = True

    tmpdir = tempfile.mkdtemp()
    for s, (rx, ry) in [(0, (0.0, 0.0)), (1, (10.0, 0.0))]:
        p = os.path.join(tmpdir, f"step{s}.jpg")
        Image.new("RGB", (640, 640), (0, 0, 0)).save(p)
        env.image_b64_history[s] = "x"
        env.image_path_history[s] = p
        env.image_meta_history[s] = {
            "fov": 60, "heading": 0.0, "pitch": 0.0,
            "relative_x": rx, "relative_y": ry,
        }
        env._road_mask_cache[s] = mask.copy()

    b64 = env._generate_bev_image()
    assert b64, "BEV 生成返回空"

    img = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
    arr = np.array(img)
    h, w = arr.shape[:2]
    print(f"BEV 尺寸: {w} x {h}")

    # 1) 正方形
    assert h == w, f"应为正方形，实际 {w}x{h}"

    # 2) 中心像素是初始点（红色）
    center = arr[h // 2, w // 2]
    print(f"中心像素 (初始点): {tuple(center)}")
    assert (center == np.array([255, 0, 0])).all(), f"中心应为红色初始点，实际 {tuple(center)}"

    # 3) 网格线浅灰存在
    gray_mask = (arr == np.array([232, 232, 232])).all(axis=-1)
    assert gray_mask.any(), "应存在浅灰网格线"
    print(f"浅灰网格像素数: {int(gray_mask.sum())}")

    # 4) 网格间距应等于 move_step(15) * 有效尺度
    #    取顶行（无道路点/相机点），竖网格线应出现在间距≈15px 的列上
    top_row = arr[0]
    gray_cols = np.where((top_row == np.array([232, 232, 232])).all(axis=-1))[0]
    print(f"顶行竖网格线列索引: {gray_cols.tolist()}")
    assert len(gray_cols) >= 2, "至少应有两条竖网格线"
    gaps = np.diff(gray_cols)
    print(f"竖线间距: {gaps.tolist()}px (期望≈15px)")
    assert all(abs(int(g) - 15) <= 1 for g in gaps), f"网格间距异常: {gaps.tolist()}"

    print("\n✅ BEV 网格功能验证通过")


if __name__ == "__main__":
    test_bev_grid()
