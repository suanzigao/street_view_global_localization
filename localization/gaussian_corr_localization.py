"""
高斯核卷积精定位：用招牌 BEV 相对位置 + 招牌 GPS，通过 20-channel 高斯核图互相关，
精修粗定位 C0，得到更准的采集点坐标。

流程：
1. 读 sign_markers.json 提取招牌 BEV 位置 (x,y) 与 GPS (lat,lng)
2. 读空间投票 top-3 作为 C0
3. 对每个 C0：
   - B 图：招牌 GPS 投影到以 C0 为原点的局部平面 (east,north)
   - A 图：招牌 BEV 位置 (x,y)（相对采集点）
   - 互相关找峰值偏移 (dx*, dy*) = 采集点相对 C0 的偏移
   - 精定位 = C0 + (dx*, dy*)

互相关说明：等价于 20 张高斯核图 A_j 与 B_j 做卷积后求和，
    C(dx,dy) = Σ_j Σ_{x,y} A_j(x,y) · B_j(x+dx, y+dy)
峰值位置即两张图对齐所需的平移量。
"""
import os
import json
import math
import numpy as np

SIGMA_M = 30.0
SEARCH_RANGE_M = 300.0
STEP_M = 1.0


def latlng_to_local(lat, lng, lat0, lng0):
    """把 (lat,lng) 投影到以 (lat0,lng0) 为原点的局部米制平面，x=东, y=北。"""
    x = (lng - lng0) * 111320.0 * math.cos(math.radians(lat0))
    y = (lat - lat0) * 110540.0
    return x, y


def local_to_latlng(x, y, lat0, lng0):
    """把局部米制平面坐标反投影回 (lat,lng)。"""
    lat = lat0 + y / 110540.0
    lng = lng0 + x / (111320.0 * math.cos(math.radians(lat0)))
    return lat, lng


def gaussian_corr_refine(signs, C0, sigma=SIGMA_M, search_range=SEARCH_RANGE_M, step=STEP_M):
    """对粗定位 C0 做高斯核互相关精定位。

    signs: list of {'x': float, 'y': float, 'gps_lat': float, 'gps_lng': float, 'weight': float}
    C0: (lat, lng) 粗定位坐标（空间投票 top-k）

    返回 (score, dx_star, dy_star)，其中精定位 = C0 + (dx_star, dy_star) 米。
    """
    lat0, lng0 = C0
    xs, ys, easts, norths, weights = [], [], [], [], []
    for s in signs:
        east, north = latlng_to_local(s['gps_lat'], s['gps_lng'], lat0, lng0)
        xs.append(s['x'])
        ys.append(s['y'])
        easts.append(east)
        norths.append(north)
        weights.append(s.get('weight', 1.0))

    n = len(xs)
    best_score = -1.0
    best_dx = 0.0
    best_dy = 0.0
    offsets = np.arange(-search_range, search_range + step, step)

    corr_map = np.zeros((len(offsets), len(offsets)))
    for i, dx in enumerate(offsets):
        for jj, dy in enumerate(offsets):
            score = 0.0
            for j in range(n):
                ex = easts[j] - dx
                ny = norths[j] - dy
                d2 = (xs[j] - ex) ** 2 + (ys[j] - ny) ** 2
                score += weights[j] * math.exp(-d2 / (2.0 * sigma * sigma))
            corr_map[jj, i] = score
            if score > best_score:
                best_score = score
                best_dx = float(dx)
                best_dy = float(dy)

    return best_score, best_dx, best_dy, corr_map, offsets


def find_top_peaks(corr_map, offsets, top_n=10, min_dist=20.0):
    """在相关图上找 top-n 个高点（按值降序，带最小间隔抑制）。

    返回 list of {'dx': float, 'dy': float, 'score': float}，按 score 降序。
    """
    step = float(offsets[1] - offsets[0]) if len(offsets) > 1 else 1.0
    min_dist_px = max(1, int(round(min_dist / step)))

    # 所有网格点按相关值降序排列
    flat_idx = np.argsort(corr_map.ravel())[::-1]
    ys_idx, xs_idx = np.unravel_index(flat_idx, corr_map.shape)

    selected = []
    for yi, xi in zip(ys_idx, xs_idx):
        dx = float(offsets[xi])
        dy = float(offsets[yi])
        score = float(corr_map[yi, xi])
        ok = True
        for q in selected:
            if math.hypot(dx - q["dx"], dy - q["dy"]) < min_dist:
                ok = False
                break
        if ok:
            selected.append({"dx": dx, "dy": dy, "score": score})
        if len(selected) >= top_n:
            break

    return selected


def build_density_map(points_x, points_y, weights, sigma, res, margin):
    """在米制平面上，把一组点用高斯核叠加成一张密度图。

    points_x/y: 各点的局部米制坐标
    weights: 各点权重
    sigma: 高斯核带宽（米）
    res: 网格分辨率（米/格）
    margin: 网格四周留白（米）

    返回 (density, xs_grid, ys_grid)，其中 xs_grid/ys_grid 是一维坐标轴。
    """
    xmin = min(points_x) - margin
    xmax = max(points_x) + margin
    ymin = min(points_y) - margin
    ymax = max(points_y) + margin
    xs_grid = np.arange(xmin, xmax, res)
    ys_grid = np.arange(ymin, ymax, res)
    X, Y = np.meshgrid(xs_grid, ys_grid)
    density = np.zeros_like(X)
    for px, py, w in zip(points_x, points_y, weights):
        density += w * np.exp(-((X - px) ** 2 + (Y - py) ** 2) / (2.0 * sigma * sigma))
    return density, xs_grid, ys_grid


def main():
    base = "/home/suanzigao/PycharmProjects/street_view_global_localization"
    point_dir = os.path.join(base, "manual_streetview_results/2026-09-09_17-39-52_manual_lat47_3928_lng8_5308")
    out_dir = os.path.join(point_dir, "text_output_vggt_depth")

    markers_path = os.path.join(out_dir, "sign_markers.json")
    compare_path = os.path.join(out_dir, "spatial_vote_compare.json")
    summary_path = os.path.join(point_dir, "collection_summary.json")

    with open(markers_path) as f:
        markers_data = json.load(f)
    with open(compare_path) as f:
        compare_data = json.load(f)
    with open(summary_path) as f:
        summary = json.load(f)

    true_lat = summary.get("actual_initial_lat") or summary.get("initial_lat")
    true_lng = summary.get("actual_initial_lng") or summary.get("initial_lng")

    # 提取有 GPS 的招牌：BEV 相对位置 (x,y) + GPS
    signs = []
    for m in markers_data["markers"]:
        gps = m.get("gps")
        if not gps or gps.get("lat") is None or gps.get("lng") is None:
            continue
        wp = m.get("world_position")
        if not wp:
            continue
        signs.append({
            "text": m.get("text", ""),
            "x": wp["x"],
            "y": wp["y"],
            "gps_lat": gps["lat"],
            "gps_lng": gps["lng"],
            "weight": 1.0,
        })

    print(f"参与精定位的招牌数: {len(signs)}")
    for s in signs:
        print(f"  - {s['text']:20s} BEV=({s['x']:7.2f}, {s['y']:7.2f}) GPS=({s['gps_lat']:.5f}, {s['gps_lng']:.5f})")

    # 取空间投票 top-3 作为 C0
    top3 = compare_data["sort_weighted_support"][:3]

    results = []
    corr_maps = []
    density_infos = []
    print("\n=== 高斯核卷积精定位结果 ===")
    for i, t in enumerate(top3):
        C0 = (t["lat"], t["lng"])
        # 只使用支持该 C0 的招牌点位（来自空间投票的 supporting_signs）
        supporting = set(t.get("supporting_signs", []))
        use_signs = [s for s in signs if s["text"] in supporting]
        if not use_signs:
            use_signs = signs

        score, dx, dy, corr_map, offsets = gaussian_corr_refine(use_signs, C0)
        ref_lat, ref_lng = local_to_latlng(dx, dy, C0[0], C0[1])

        # 提取 top-10 峰值（每个峰值对应一个候选精定位坐标）
        peaks = find_top_peaks(corr_map, offsets, top_n=10, min_dist=20.0)
        for p in peaks:
            plat, plng = local_to_latlng(p["dx"], p["dy"], C0[0], C0[1])
            p["lat"] = plat
            p["lng"] = plng
            if true_lat is not None and true_lng is not None:
                p["err_m"] = round(math.sqrt(
                    ((plat - true_lat) * 110540.0) ** 2
                    + ((plng - true_lng) * 111320.0 * math.cos(math.radians(true_lat))) ** 2
                ), 1)

        err = None
        if true_lat is not None and true_lng is not None:
            err = math.sqrt(
                ((ref_lat - true_lat) * 110540.0) ** 2
                + ((ref_lng - true_lng) * 111320.0 * math.cos(math.radians(true_lat))) ** 2
            )

        results.append({
            "coarse_C0": {"lat": C0[0], "lng": C0[1]},
            "coarse_err_m": t.get("distance_to_true_m"),
            "used_signs": [s["text"] for s in use_signs],
            "offset_m": {"dx": dx, "dy": dy},
            "corr_peak_score": score,
            "refined": {"lat": ref_lat, "lng": ref_lng},
            "refined_err_m": round(err, 1) if err is not None else None,
            "top10_peaks": peaks,
        })
        corr_maps.append((i + 1, C0, dx, dy, corr_map, offsets, peaks))
        density_infos.append((i + 1, C0, use_signs))

        print(f"  C0=top-{i+1} ({C0[0]:.6f}, {C0[1]:.6f})")
        print(f"      使用支持招牌({len(use_signs)}): {', '.join(s['text'] for s in use_signs)}")
        print(f"      偏移=({dx:+.1f}, {dy:+.1f})m  峰值相关={score:.4f}")
        print(f"      精定位=({ref_lat:.6f}, {ref_lng:.6f})  误差={results[-1]['refined_err_m']}m")
        print(f"      top-10 候选:")
        for k, p in enumerate(peaks):
            print(f"        #{k+1} ({p['lat']:.6f}, {p['lng']:.6f}) 偏移=({p['dx']:+.0f},{p['dy']:+.0f})m 相关={p['score']:.4f} 误差={p.get('err_m')}m")

    out_path = os.path.join(out_dir, "gaussian_corr_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "method": "gaussian_kernel_correlation_refinement",
            "true_position": {"lat": true_lat, "lng": true_lng},
            "params": {"sigma_m": SIGMA_M, "search_range_m": SEARCH_RANGE_M, "step_m": STEP_M},
            "num_signs": len(signs),
            "results": results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n已保存: {out_path}")

    # 可视化三张相关热力图，标出峰值
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (idx, C0, dx, dy, corr_map, offsets, peaks) in zip(axes, corr_maps):
        extent = [offsets[0], offsets[-1], offsets[0], offsets[-1]]
        im = ax.imshow(corr_map, origin="lower", extent=extent, cmap="hot", aspect="auto")
        # top-10 峰值散点
        pk_dx = [p["dx"] for p in peaks]
        pk_dy = [p["dy"] for p in peaks]
        ax.scatter(pk_dx, pk_dy, s=40, edgecolors="white", facecolors="none", linewidths=1.2, zorder=3)
        for k, p in enumerate(peaks):
            ax.annotate(str(k + 1), (p["dx"], p["dy"]), textcoords="offset points",
                        xytext=(4, 4), fontsize=8, color="white", zorder=4)
        # 全局峰值（top-1）
        ax.axvline(dx, color="cyan", linestyle="--", linewidth=1)
        ax.axhline(dy, color="cyan", linestyle="--", linewidth=1)
        ax.plot(dx, dy, "x", color="cyan", markersize=12, markeredgewidth=2)
        ax.set_title(f"top-{idx}  C0=({C0[0]:.5f},{C0[1]:.5f})\npeak=({dx:+.0f},{dy:+.0f})m (白圈=top10峰值)")
        ax.set_xlabel("dx (east, m)")
        ax.set_ylabel("dy (north, m)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Gaussian Kernel Cross-Correlation (BEV vs GPS)", fontsize=14)
    fig.tight_layout()
    corr_img_path = os.path.join(out_dir, "gaussian_corr_heatmaps.png")
    fig.savefig(corr_img_path, dpi=120)
    plt.close(fig)
    print(f"已保存热力图: {corr_img_path}")

    # 可视化 A（BEV 高斯核密度图）与 B（GPS 高斯核密度图），每列用各自 C0 的支持招牌
    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
    for col, (idx, C0, use_signs) in enumerate(density_infos):
        axs = [s['x'] for s in use_signs]
        ays = [s['y'] for s in use_signs]
        aws = [s['weight'] for s in use_signs]
        texts = [s['text'] for s in use_signs]
        dens_A, axs_grid, ays_grid = build_density_map(axs, ays, aws, SIGMA_M, res=3.0, margin=80.0)

        bxs = []
        bys = []
        for s in use_signs:
            east, north = latlng_to_local(s['gps_lat'], s['gps_lng'], C0[0], C0[1])
            bxs.append(east)
            bys.append(north)
        dens_B, bxs_grid, bys_grid = build_density_map(bxs, bys, aws, SIGMA_M, res=3.0, margin=80.0)

        axA = axes2[0, col]
        imA = axA.imshow(dens_A, origin="lower", extent=[axs_grid[0], axs_grid[-1], ays_grid[0], ays_grid[-1]],
                         cmap="hot", aspect="auto")
        axA.scatter(axs, ays, s=60, edgecolors="cyan", facecolors="none", linewidths=1.5, zorder=3)
        for x, y, txt in zip(axs, ays, texts):
            axA.annotate(txt, (x, y), textcoords="offset points", xytext=(5, 5),
                         fontsize=8, color="cyan", zorder=4)
        axA.set_title(f"A: BEV 密度图 (top-{idx}, {len(use_signs)}招牌, 原点=采集点)")
        axA.set_xlabel("x (东, m)")
        axA.set_ylabel("y (北, m)")
        fig2.colorbar(imA, ax=axA, fraction=0.046, pad=0.04)

        axB = axes2[1, col]
        imB = axB.imshow(dens_B, origin="lower", extent=[bxs_grid[0], bxs_grid[-1], bys_grid[0], bys_grid[-1]],
                         cmap="hot", aspect="auto")
        axB.scatter(bxs, bys, s=60, edgecolors="cyan", facecolors="none", linewidths=1.5, zorder=3)
        for x, y, txt in zip(bxs, bys, texts):
            axB.annotate(txt, (x, y), textcoords="offset points", xytext=(5, 5),
                         fontsize=8, color="cyan", zorder=4)
        axB.set_title(f"B: GPS 密度图 (top-{idx}, 原点=C0=({C0[0]:.5f},{C0[1]:.5f}))")
        axB.set_xlabel("east (m)")
        axB.set_ylabel("north (m)")
        fig2.colorbar(imB, ax=axB, fraction=0.046, pad=0.04)

    fig2.suptitle("A/B 高斯核密度图（仅使用各 C0 的支持招牌；青色空心点=招牌位置）", fontsize=14)
    fig2.tight_layout()
    density_img_path = os.path.join(out_dir, "gaussian_density_AB.png")
    fig2.savefig(density_img_path, dpi=120)
    plt.close(fig2)
    print(f"已保存密度图: {density_img_path}")


if __name__ == "__main__":
    main()
