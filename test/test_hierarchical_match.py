"""分层匹配（Coarse-to-Fine）单元测试。

覆盖纯函数：_downsample_bev / _extract_top_peaks / pixel_to_gps 往返。
可选集成测试：--integration 对单个点位跑完整分层匹配（需联网下载 OSM）。
"""
import os
import sys
import math

# 代理配置（与 main.py 一致，osmnx 下载 OSM 需要走代理）
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"
os.environ["NO_PROXY"] = "localhost,127.0.0.1,localdomain,.localdomain"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'localization'))

import numpy as np
from bev_osm_convolution_matcher import BEVOSMConvolutionMatcher


def test_downsample_bev():
    m = BEVOSMConvolutionMatcher()
    bev = np.zeros((100, 200), dtype=np.uint8)
    bev[40:60, 80:120] = 255

    out = m._downsample_bev(bev, 0.5)
    assert out.shape == (50, 100), f"期望 (50,100)，实际 {out.shape}"

    out2 = m._downsample_bev(bev, 1.0)
    assert out2.shape == (100, 200), f"scale=1 应原样返回，实际 {out2.shape}"
    print("✅ _downsample_bev 通过")


def test_extract_top_peaks():
    m = BEVOSMConvolutionMatcher()
    # 构造 100x100 相关图，设 3 个峰：值 0.9(0,0)、0.8(50,50)、0.7(90,90)
    result = np.zeros((100, 100), dtype=np.float32)
    result[0, 0] = 0.9
    result[50, 50] = 0.8
    result[90, 90] = 0.7

    peaks = m._extract_top_peaks(result, top_k=3, min_dist_px=10.0)
    assert len(peaks) == 3
    assert abs(peaks[0]['confidence'] - 0.9) < 1e-5
    assert (peaks[0]['pixel_x'], peaks[0]['pixel_y']) == (0, 0)
    assert (peaks[1]['pixel_x'], peaks[1]['pixel_y']) == (50, 50)
    print("✅ _extract_top_peaks 通过")

    # min_dist 抑制：两峰距离 < min_dist_px 时，次高峰应被抑制
    result2 = np.zeros((50, 50), dtype=np.float32)
    result2[10, 10] = 0.9
    result2[12, 12] = 0.85
    peaks2 = m._extract_top_peaks(result2, top_k=5, min_dist_px=20.0)
    coords = [(p['pixel_x'], p['pixel_y']) for p in peaks2]
    assert (10, 10) in coords
    assert (12, 12) not in coords, f"距离<20px 的次高峰应被抑制，实际选中 {coords}"
    print("✅ _extract_top_peaks 抑制逻辑通过")


def test_pixel_gps_roundtrip():
    m = BEVOSMConvolutionMatcher()
    center_lat, center_lng = 47.3928, 8.5308
    radius, resolution = 5000, 0.5

    for (px, py) in [(1234, 5678), (0, 0), (5000, 5000)]:
        lat, lng = m.pixel_to_gps(px, py, center_lat, center_lng, radius=radius, resolution=resolution)
        x2, y2 = m._gps_to_pixel(lat, lng, center_lat, center_lng, radius=radius, resolution=resolution)
        assert abs(x2 - px) <= 1 and abs(y2 - py) <= 1, \
            f"往返误差过大: ({px},{py}) -> ({x2},{y2})"
    print("✅ pixel_to_gps ↔ _gps_to_pixel 往返通过")


def test_integration():
    """对单个点位跑完整分层匹配（需联网 + 已有 BEV 投影数据）。"""
    batch_dir = '/home/suanzigao/PycharmProjects/street_view_global_localization/manual_streetview_results'
    point = {'name': 'E-fIB3K6E-Z-GSW_xI-rSw', 'lat': 55.6463583, 'lng': 37.7249981}

    m = BEVOSMConvolutionMatcher()
    result = m.match_point_hierarchical(
        point, batch_dir, coarse_radius=5000, coarse_resolution=0.5,
        fine_window=500, top_k=5)
    if result:
        print(f"\n最终匹配: ({result['match_lat']:.6f}, {result['match_lng']:.6f}) "
              f"置信度={result['match_confidence']:.4f} 误差={result.get('distance_error')}")
        assert result['match_lat'] is not None and result['match_lng'] is not None
    else:
        print("⚠️ 分层匹配返回 None（可能缺 BEV 数据或网络失败）")
    print("✅ 集成测试完成")


def test_integration_15km():
    """对单个点位跑 15km 三层分层匹配（需联网 + 已有 BEV 投影数据）。"""
    batch_dir = '/home/suanzigao/PycharmProjects/street_view_global_localization/manual_streetview_results'
    point = {'name': 'E-fIB3K6E-Z-GSW_xI-rSw', 'lat': 55.6463583, 'lng': 37.7249981}

    m = BEVOSMConvolutionMatcher()
    result = m.match_point_15km(point, batch_dir, top_k=5)
    if result:
        print(f"\n15km最终匹配: ({result['match_lat']:.6f}, {result['match_lng']:.6f}) "
              f"置信度={result['match_confidence']:.4f} 误差={result.get('distance_error')}")
        assert result['match_lat'] is not None and result['match_lng'] is not None
    else:
        print("⚠️ 15km 分层匹配返回 None（可能缺 BEV 数据或网络失败）")
    print("✅ 15km 集成测试完成")


if __name__ == "__main__":
    test_downsample_bev()
    test_extract_top_peaks()
    test_pixel_gps_roundtrip()

    if "--integration" in sys.argv:
        test_integration()
    elif "--integration-15km" in sys.argv:
        test_integration_15km()
    else:
        print("\n(跳过集成测试；加 --integration 或 --integration-15km 运行完整匹配)")
