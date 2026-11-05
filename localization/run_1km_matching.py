"""
只运行1km范围的BEV与OSM路网匹配
"""

import os
import json
from bev_osm_convolution_matcher import BEVOSMConvolutionMatcher


def main():
    matcher = BEVOSMConvolutionMatcher()

    batch_dir = '/home/suanzigao/PycharmProjects/street_view_global_localization/manual_streetview_results'

    point_dirs = sorted([d for d in os.listdir(batch_dir) if os.path.isdir(os.path.join(batch_dir, d)) and not d.startswith('.')])

    points = []
    for point_dir in point_dirs:
        images_dir = os.path.join(batch_dir, point_dir, 'streetview_images')
        metadata_path = os.path.join(images_dir, 'step_0_metadata.json')

        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                meta = json.load(f)
                actual_lat = meta.get('actual_lat')
                actual_lng = meta.get('actual_lng')
                if actual_lat and actual_lng:
                    points.append({
                        'name': point_dir,
                        'lat': actual_lat,
                        'lng': actual_lng
                    })
                    print(f"📍 找到点位: {point_dir}, 坐标: ({actual_lat:.6f}, {actual_lng:.6f})")

    print(f"\n📊 共读取 {len(points)} 个点位")

    print(f"\n{'='*70}")
    print(f"1km范围匹配")
    print(f"{'='*70}")

    all_results = []
    for point in points:
        result = matcher.match_point(point, batch_dir, osm_radius=1000)
        if result:
            all_results.append(result)

    print(f"\n{'='*70}")
    print(f"1km批量匹配完成")
    print(f"{'='*70}")

    print(f"\n📊 1km范围匹配结果:")
    for result in all_results:
        print(f"  {result['name']}: 误差 {result['distance_error']:.2f}米" if result['distance_error'] else f"  {result['name']}: 无误差")

    if all_results:
        errors = [r['distance_error'] for r in all_results if r['distance_error']]
        if errors:
            print(f"\n📊 1km统计: 平均{sum(errors)/len(errors):.2f}米, 最小{min(errors):.2f}米, 最大{max(errors):.2f}米")

    summary_path = os.path.join(batch_dir, 'matching_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ 1km匹配汇总已保存: {summary_path}")


if __name__ == "__main__":
    main()
