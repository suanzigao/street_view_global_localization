"""绘制误差较小点位的对比图。

背景为 OSM 路网，标记三个点：
  绿色 = 真实位置 (GT)
  蓝色 = 三段式级联定位预测 (新框架)
  红色 = 纯大模型预测 (2026-06-30 batch_run 原始数据)

纯大模型坐标来自 batch_streetview_results/2026-06-30_19-39-05_batch_run 各点
final_prediction.txt 中的 predicted_initial_coords。
"""
import os
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D

import osmnx as ox

for fp in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"]:
    try:
        fm.fontManager.addfont(fp)
    except Exception:
        pass

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Droid Sans Fallback"]
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = "/home/suanzigao/PycharmProjects/street_view_global_localization/batch_streetview_results/point_comparison_maps"

# 误差较小的点位（新框架误差 < 1.5km）：城市名, GT, 新框架预测, 纯大模型预测, OSM 半径(米)
POINTS = [
    {
        "name": "8GEv-Se8N55o4t4ee88gJg",
        "city": "苏黎世 (Zürich)",
        "gt": (47.3928056, 8.5308389),
        "new": (47.3927400764791, 8.530828178213337),
        "baseline": (47.3769, 8.5417),
        "radius": 2500,
    },
    {
        "name": "tDT-oOhnKoJSL3qzi2f17A",
        "city": "布达佩斯 (Budapest)",
        "gt": (47.5309819, 19.0696085),
        "new": (47.53088325601592, 19.069630563583154),
        "baseline": (47.4979, 19.0402),
        "radius": 5000,
    },
    {
        "name": "juYqx8PDME4OG5u_2iJFOQ",
        "city": "圣保罗 (São Paulo)",
        "gt": (-23.5548322, -46.6248183),
        "new": (-23.547123603603602, -46.62648275586014),
        "baseline": (-23.5505, -46.6333),
        "radius": 1500,
    },
    {
        "name": "aDapL3J-jB-pQ3oVCzzAsw",
        "city": "赫尔辛基 (Helsinki)",
        "gt": (60.1719367, 24.7787122),
        "new": (60.176, 24.804),
        "baseline": (60.184, 24.832),
        "radius": 4000,
    },
]


def haversine(a, b):
    R = 6371000.0
    p1, l1 = math.radians(a[0]), math.radians(a[1])
    p2, l2 = math.radians(b[0]), math.radians(b[1])
    dp, dl = p2 - p1, l2 - l1
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


def draw_point(p):
    gt = p["gt"]
    center_lat, center_lng = gt
    m_per_deg_lat = 111000.0
    m_per_deg_lng = 111000.0 * math.cos(math.radians(center_lat))

    def to_xy(lat, lng):
        x = (lng - center_lng) * m_per_deg_lng
        y = (lat - center_lat) * m_per_deg_lat
        return x, y

    print(f"\n[{p['city']}] 下载 OSM 路网 (radius={p['radius']}m)...")
    try:
        G = ox.graph_from_point(gt, dist=p["radius"], network_type="drive")
    except Exception as e:
        print(f"  ❌ OSM 下载失败: {e}")
        return None

    fig, ax = plt.subplots(figsize=(9, 9))

    # 画路网
    for u, v, data in G.edges(data=True):
        if "geometry" in data:
            xs, ys = zip(*[(c[0], c[1]) for c in data["geometry"].coords])
        else:
            xs = [G.nodes[u]["x"], G.nodes[v]["x"]]
            ys = [G.nodes[u]["y"], G.nodes[v]["y"]]
        px = [(lng - center_lng) * m_per_deg_lng for lng in xs]
        py = [(lat - center_lat) * m_per_deg_lat for lat in ys]
        hw = data.get("highway", "")
        if isinstance(hw, list):
            hw = hw[0] if hw else ""
        lw = 1.6 if hw in ("primary", "secondary", "trunk", "motorway") else 0.8
        color = "#b0b0b0" if hw in ("primary", "secondary", "trunk", "motorway") else "#d8d8d8"
        ax.plot(px, py, color=color, linewidth=lw, solid_capstyle="round", zorder=1)

    # 三个点
    gx, gy = to_xy(*gt)
    nx, ny = to_xy(*p["new"])
    bx, by = to_xy(*p["baseline"])

    # 三种点大小不同，便于重叠时区分（大→小：真实 > 新框架 > 纯大模型）
    # 较小的点 zorder 更高，重叠时小点浮在大点之上，都能看清
    ax.scatter([gx], [gy], s=260, c="#2ecc71", marker="o", edgecolors="white", linewidths=1.6, zorder=5, label="真实位置 (GT)")
    ax.scatter([nx], [ny], s=150, c="#2f80ed", marker="o", edgecolors="white", linewidths=1.6, zorder=6, label="三段式级联定位 (新框架)")
    ax.scatter([bx], [by], s=70, c="#e74c3c", marker="X", edgecolors="white", linewidths=1.2, zorder=7, label="纯大模型 (之前结果)")

    # 连线：真实 -> 新框架、真实 -> 纯大模型
    ax.plot([gx, nx], [gy, ny], color="#2f80ed", linewidth=1.2, linestyle="--", zorder=4)
    ax.plot([gx, bx], [gy, by], color="#e74c3c", linewidth=1.2, linestyle="--", zorder=4)

    err_new = haversine(gt, p["new"])
    err_base = haversine(gt, p["baseline"])

    # 比例尺
    half = p["radius"]
    scale_len = max(100, round(p["radius"] / 5 / 100) * 100)
    sx = half - scale_len - 100
    sy = -half + 150
    ax.plot([sx, sx + scale_len], [sy, sy], color="black", linewidth=2)
    ax.text(sx + scale_len / 2, sy + 120, f"{scale_len} m", ha="center", fontsize=9, color="black")

    ax.set_aspect("equal")
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_xlabel("东 ↔ 西 (m)", fontsize=9)
    ax.set_ylabel("北 ↔ 南 (m)", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_title(f"{p['city']}\n真实 vs 新框架 vs 纯大模型", fontsize=13, weight="bold", pad=10)
    ax.text(0.5, -0.06,
            f"新框架误差 {err_new:.1f} m  |  纯大模型误差 {err_base:.0f} m",
            transform=ax.transAxes, ha="center", fontsize=10, color="#555555")

    ax.legend(loc="upper left", fontsize=9, framealpha=0.9, markerscale=0.8)

    out_path = os.path.join(OUT_DIR, f"{p['name']}_comparison.png")
    os.makedirs(OUT_DIR, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✅ 已保存: {out_path}  (新框架误差 {err_new:.1f}m, 纯大模型误差 {err_base:.0f}m)")
    return out_path


def main():
    for p in POINTS:
        draw_point(p)


if __name__ == "__main__":
    main()
