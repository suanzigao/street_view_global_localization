"""
下载OSM路网并渲染为图像，保留道路粗细和细节
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


class OSMRoadRenderer:
    def __init__(self):
        self.ox = None
        self._try_import_osmnx()

    def _try_import_osmnx(self):
        try:
            import osmnx as ox
            self.ox = ox
            print("✅ osmnx 已安装")
        except ImportError:
            print("⚠️ osmnx 未安装，请运行: pip install osmnx")

    def download_and_render(
        self,
        lat: float,
        lng: float,
        radius: float = 500,
        output_dir: str = "osm_output",
        resolution: float = 10,
        show_plot: bool = False
    ) -> dict:
        """
        下载OSM路网并渲染为图像

        Args:
            lat: 中心纬度
            lng: 中心经度
            radius: 搜索半径（米）
            output_dir: 输出目录
            resolution: 像素/米
            show_plot: 是否显示绘图

        Returns:
            包含图像路径和元数据的字典
        """
        if self.ox is None:
            print("❌ osmnx 未安装，无法下载OSM路网")
            return {}

        os.makedirs(output_dir, exist_ok=True)

        print(f"\n📡 正在下载OSM路网...")
        print(f"   中心坐标: ({lat:.6f}, {lng:.6f})")
        print(f"   搜索半径: {radius}米")

        try:
            G = self.ox.graph_from_point((lat, lng), dist=radius, network_type='drive')
        except Exception as e:
            print(f"❌ OSM下载失败: {e}")
            return {}

        print(f"✅ OSM路网下载完成")
        print(f"   节点数: {len(G.nodes)}")
        print(f"   边数: {len(G.edges)}")

        fig, ax = self.ox.plot_graph(
            G,
            show=False,
            save=False,
            close=False,
            edge_color='black',
            edge_linewidth=2,
            node_size=0,
            figsize=(15, 15),
            dpi=100
        )

        plt.tight_layout()

        plot_path = os.path.join(output_dir, "osm_network_plot.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✅ 路网图像已保存: {plot_path}")

        if not show_plot:
            plt.close()

        image_path = os.path.join(output_dir, "osm_road_image.png")
        self._render_road_image(G, lat, lng, radius, resolution, image_path)

        return {
            'plot_path': plot_path,
            'image_path': image_path,
            'node_count': len(G.nodes),
            'edge_count': len(G.edges),
            'lat': lat,
            'lng': lng,
            'radius': radius,
            'resolution': resolution
        }

    def _get_edge_linewidth(self, data):
        """根据道路类型返回线宽"""
        highway_type = data.get('highway', 'residential')

        width_map = {
            'motorway': 6,
            'motorway_link': 4,
            'trunk': 5,
            'trunk_link': 3,
            'primary': 4,
            'primary_link': 2.5,
            'secondary': 3.5,
            'secondary_link': 2,
            'tertiary': 3,
            'tertiary_link': 1.5,
            'residential': 2.5,
            'service': 1.5,
            'unclassified': 2,
        }

        if isinstance(highway_type, list):
            highway_type = highway_type[0]

        return width_map.get(highway_type, 2)

    def _render_road_image(
        self,
        G,
        center_lat: float,
        center_lng: float,
        radius: float,
        resolution: float,
        output_path: str
    ):
        """渲染为二值道路图像"""
        meters_per_deg_lat = 111000.0
        meters_per_deg_lng = 111000.0 * np.cos(np.radians(center_lat))

        image_width = int(2 * radius * resolution)
        image_height = int(2 * radius * resolution)

        road_image = np.zeros((image_height, image_width), dtype=np.uint8)

        def lat_lng_to_pixel(lat, lng):
            dy_meters = (lat - center_lat) * meters_per_deg_lat
            dx_meters = (lng - center_lng) * meters_per_deg_lng

            x = int((dx_meters + radius) * resolution)
            y = int((radius - dy_meters) * resolution)

            return x, y

        for u, v, data in G.edges(data=True):
            highway_type = data.get('highway', 'residential')
            if isinstance(highway_type, list):
                highway_type = highway_type[0]

            width_map = {
                'motorway': 8,
                'motorway_link': 5,
                'trunk': 7,
                'trunk_link': 4,
                'primary': 6,
                'primary_link': 3,
                'secondary': 5,
                'secondary_link': 2.5,
                'tertiary': 4,
                'tertiary_link': 2,
                'residential': 3,
                'service': 2,
                'unclassified': 2.5,
            }

            linewidth = width_map.get(highway_type, 3)
            linewidth = int(linewidth)

            if 'geometry' in data:
                coords = list(data['geometry'].coords)
                points = [lat_lng_to_pixel(c[1], c[0]) for c in coords]
            else:
                u_node = G.nodes[u]
                v_node = G.nodes[v]
                points = [
                    lat_lng_to_pixel(u_node['y'], u_node['x']),
                    lat_lng_to_pixel(v_node['y'], v_node['x'])
                ]

            for i in range(len(points) - 1):
                x1, y1 = points[i]
                x2, y2 = points[i + 1]

                if 0 <= x1 < image_width and 0 <= y1 < image_height and \
                   0 <= x2 < image_width and 0 <= y2 < image_height:
                    cv2.line(road_image, (x1, y1), (x2, y2), 255, linewidth)

        cv2.imwrite(output_path, road_image)
        print(f"✅ 二值道路图像已保存: {output_path}")
        print(f"   图像尺寸: {image_width}×{image_height}")
        print(f"   分辨率: {resolution}像素/米")


def main():
    test_points = [
        (55.713202, 12.592934),
    ]

    renderer = OSMRoadRenderer()

    for i, (lat, lng) in enumerate(test_points, 1):
        print(f"\n{'='*60}")
        print(f"处理点位 {i}/{len(test_points)}")
        print(f"坐标: ({lat:.6f}, {lng:.6f})")
        print(f"{'='*60}")

        output_dir = f"osm_output/point_{i}"
        result = renderer.download_and_render(
            lat=lat,
            lng=lng,
            radius=5000,
            output_dir=output_dir,
            resolution=2
        )

        if result:
            print(f"\n📊 渲染结果:")
            print(f"   路网图像: {result['plot_path']}")
            print(f"   二值图像: {result['image_path']}")
            print(f"   节点数: {result['node_count']}")
            print(f"   边数: {result['edge_count']}")


if __name__ == "__main__":
    main()