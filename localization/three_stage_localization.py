"""
三段式级联定位编排脚本 (Three-Stage Localization Pipeline)

将定位过程拆分为三个递进的粒度，每一段缩小一个数量级的搜索空间：

  Stage 1 (粗粒度 / 城市级):  Agent 自主导航 + 多模态推理
       输入: 初始 GPS (lat, lng)
       输出: 城市名 + 粗 GPS + 探索到的地点 (found_places)
       → 写入 stage1_coarse.json

  Stage 2 (中粒度 / 街区级):  招牌语义匹配 + 空间投票 + 高斯核招牌配准
       输入: 街景图 + 城市名 (来自 Stage 1)
       输出: top-10 候选位置 + 每个位置的支持招牌集合 + 高斯核精化坐标
       → 写入 stage2_spatial_vote.json

  Stage 3 (细粒度 / 亚十米级): BEV 路网 ↔ OSM 路网卷积匹配
       输入: BEV 道路模板 + Stage 2 的 final_estimate 作为搜索中心
       输出: 卷积匹配坐标 + top-5 候选
       → 写入 stage3_osm_match.json

最终汇总写入 three_stage_result.json。

段间数据契约:
  stage1_coarse.json   = { "stage":1, "city":str, "coarse_gps":{lat,lng}, "found_places":[...] }
  stage2_spatial_vote.json = { "stage":2, "top10":[ {rank,lat,lng,support_count,weighted_support,supporting_signs,...} ],
                                "gaussian_refined":[...], "final_estimate":{lat,lng} }
  stage3_osm_match.json = { "stage":3, "center":{lat,lng}, "match_lat", "match_lng", "match_confidence",
                            "top5_matches":[...], "final_position":{lat,lng} }
"""
import os
import sys
import json
import math
import re
import argparse

LOCALIZATION_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(LOCALIZATION_DIR)
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, LOCALIZATION_DIR)

# ---- Stage 1 依赖 ----
from core.env import StreetViewAutoLocEnv

# ---- Stage 2 依赖 ----
from vggt_depth_to_bev import (
    spatial_vote_localize,
    reverse_geocode_city,
    _init_places_tool,
    process_point,
)

# ---- BEV 投影 (Stage 1.5) 依赖 ----
from batch_projection import project_pointcloud, generate_bev
import cv2
import numpy as np
from PIL import Image as PILImage

# ---- Stage 2 高斯核招牌配准依赖 ----
from gaussian_corr_localization import (
    gaussian_corr_refine,
    find_top_peaks,
    latlng_to_local,
    local_to_latlng,
    SIGMA_M,
    SEARCH_RANGE_M,
    STEP_M,
)

# ---- Stage 3 依赖 (BEV 路网 ↔ OSM 路网卷积匹配) ----
from bev_osm_convolution_matcher import BEVOSMConvolutionMatcher

# 段间数据契约文件名
STAGE1_FILENAME = "stage1_coarse.json"
STAGE2_FILENAME = "stage2_spatial_vote.json"
STAGE3_FILENAME = "stage3_osm_match.json"
FINAL_FILENAME = "three_stage_result.json"

# Stage 1 参数（与 main.py 保持一致）
TURN_ANGLE = 45
FOV_STEP = 20
PITCH_STEP = 10
FOV_DEFAULT = 60
USE_BEV = True

# ---- 置信度门控阈值 ----
# 各阶段输出一个置信度指标，低于阈值则该阶段结果不采信，回退到上一阶段。
# Stage 2 高斯核招牌配准：confidence = corr_peak_score / 参与配准的招牌数。
#   归一化后取值 [0, 1]，1.0 表示所有招牌完全对齐。低于 0.5 说明平均每个招牌
#   有效对齐不足一半，配准不可信。
STAGE2_MIN_PEAK_SCORE = 0.5
# Stage 3 OSM 卷积匹配：match_confidence 是 TM_CCOEFF_NORMED 归一化互相关系数，
#   取值 [-1, 1]，0.6 以下一般认为模板匹配不可信（苏黎世样例仅 0.495）。
STAGE3_MIN_CONFIDENCE = 0.6


def _parse_coarse_gps(text):
    """从 agent 的最终预测文本中解析出粗定位 GPS。"""
    if not text:
        return None
    patterns = [
        r'predicted_initial_coords:\s*([\d.-]+),([\d.-]+)',
        r'([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)',
        r'lat[\s:]*([-+]?\d*\.?\d+)[^,]*lng[\s:]*([-+]?\d*\.?\d+)',
        r'latitude[\s:]*([-+]?\d*\.?\d+)[^,]*longitude[\s:]*([-+]?\d*\.?\d+)',
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            try:
                return {"lat": float(m.group(1)), "lng": float(m.group(2))}
            except ValueError:
                continue
    return None


def load_signs_from_markers(point_dir):
    """从 sign_markers.json 读取招牌，返回 (stage2_signs, stage3_signs)。

    stage2_signs: [{text, search_query, text_confidence}]  用于空间投票
    stage3_signs: [{text, x, y, weight}]                   用于 BEV 互相关
       （只带 BEV 相对坐标 x/y；GPS 由投票结果 supporting_signs 的坐标提供，
        见 _gaussian_refine_top3，保证招牌 GPS 与投票互证一致）
    """
    markers_path = os.path.join(point_dir, "text_output_vggt_depth", "sign_markers.json")
    if not os.path.exists(markers_path):
        return [], []

    with open(markers_path, encoding="utf-8") as f:
        data = json.load(f)

    stage2_signs = []
    stage3_signs = []
    for m in data.get("markers", []):
        text = (m.get("text") or "").strip()
        if not text:
            continue
        stage2_signs.append({
            "text": text,
            "search_query": m.get("search_query") or "",
            "text_confidence": m.get("text_confidence", 0.0),
        })

        wp = m.get("world_position")
        if wp:
            stage3_signs.append({
                "text": text,
                "x": wp["x"],
                "y": wp["y"],
                "weight": 1.0,
            })
    return stage2_signs, stage3_signs


# =====================================================================
# Stage 1: Agent 自主导航 + 多模态推理（粗粒度）
# =====================================================================
def stage1_coarse_localize(point_dir, lat, lng, max_steps=10, move_step=15.0, reuse=True):
    """运行 agent 导航，得到城市名 + 粗 GPS。返回 stage1 数据字典。"""
    out_path = os.path.join(point_dir, STAGE1_FILENAME)
    if reuse and os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            print(f"[Stage 1] 复用已有结果: {out_path}")
            return json.load(f)

    img_dir = os.path.join(point_dir, "streetview_images")
    os.makedirs(img_dir, exist_ok=True)

    env = StreetViewAutoLocEnv(
        initial_lat=lat,
        initial_lng=lng,
        move_step=move_step,
        max_steps=max_steps,
        point_dir=point_dir,
        img_save_dir=img_dir,
        turn_angle=TURN_ANGLE,
        fov_step=FOV_STEP,
        pitch_step=PITCH_STEP,
        fov_default=FOV_DEFAULT,
        use_bev=USE_BEV,
    )
    print("[Stage 1] 预加载 SAM3 模型...")
    env.preload_sam3()
    env.reset()

    try:
        while env.current_step < env.max_steps:
            if env.current_image is None:
                print("[Stage 1] 无图像，跳过该点")
                break
            action_str = env.gpt4o_decision(env.current_image, env.current_step + 1)
            action, reason, seg_prompts, search_query = env.parse_gpt_decision(action_str)
            env.reason_history.append(reason)
            if env.current_step in env.image_meta_history:
                env.image_meta_history[env.current_step]["reason"] = reason

            # 在 step() 递增步数之前读取本步决策帧的 observation（决策时 current_step 即对应帧）
            obs_text = (env.image_meta_history.get(env.current_step, {}).get("observation") or "").strip()

            obs, reward, terminated, truncated, info = env.step(action)
            info["reason"] = reason
            action_type = info.get("action_type")

            if action_type == "invalid":
                print(f"[Stage 1] 检测到死循环，强制右转...")
                env.step("turn right")
                continue

            print(f"[Stage 1] Step {env.current_step} | {action} | {reason}")
            if obs_text:
                print(f"[Stage 1]   👁 {obs_text}")
            if env.plan:
                print(f"[Stage 1]   🧭 PLAN: {env.plan}")

            if action_type == "search":
                print(f"[Stage 1]   {info.get('msg', '')}")

        if not env.final_prediction:
            env._final_prediction()

        city = env.predicted_city  # 用模型预测的城市（跨步收敛），而非真值逆地理编码
        city_gt = reverse_geocode_city(lat, lng)  # 仅作审计/评估对比用，不传给下游
        coarse_gps = _parse_coarse_gps(env.final_prediction)

        stage1 = {
            "stage": 1,
            "method": "agent_navigation_multimodal_reasoning",
            "city": city or "",
            "city_gt": city_gt or "",
            "coarse_gps": coarse_gps,
            "final_prediction": env.final_prediction,
            "location_hypothesis": env.location_hypothesis,
            "hypothesis_evidence_map": env.hypothesis_evidence_map,
            "evidence_bank": env.evidence_bank,
            "found_places": env.found_places,
            "max_steps": max_steps,
        }

        # 生成导航可视化 HTML（复用 main.py 的 generate_visualization）
        try:
            from main import generate_visualization
            generate_visualization(env, point_dir)
            print(f"[Stage 1] 可视化已生成: {os.path.join(point_dir, 'visualization.html')}")
        except Exception as e:
            print(f"[Stage 1] ⚠️ 生成可视化失败: {e}")
    finally:
        env.close(release_sam3=False)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stage1, f, indent=2, ensure_ascii=False)
    print(f"[Stage 1] 结果已保存: {out_path}")
    return stage1


# =====================================================================
# Stage 1.5: BEV 投影（Stage 1 → Stage 2 中间必需步骤）
# =====================================================================
def _generate_lowres_bev(out_dir, result):
    """在 projection_output 目录里生成低分辨率 BEV 图（供 Stage 3 卷积匹配使用）。

    与 batch_projection.py 的 lowres 生成逻辑一致：
    resolution 10 -> 2（下采样 5 倍），并额外生成裁剪版。
    产出 fused_bev_lowres.png / fused_bev_lowres_cropped.png 及对应 metadata。
    """
    optimized_path = os.path.join(out_dir, "fused_bev_optimized.png")
    if not os.path.exists(optimized_path):
        print("[BEV] ⚠️ 无优化 BEV 图像，跳过 lowres 生成")
        return

    opt_bev = cv2.imread(optimized_path)
    if opt_bev is None:
        print("[BEV] ⚠️ 无法读取优化 BEV，跳过 lowres 生成")
        return

    resolution = result.get("resolution", 10)
    padding = result.get("padding", 5)
    initial_px = result.get("initial_pixel_x", 0)
    initial_py = result.get("initial_pixel_y", 0)

    lowres_scale = 2.0 / resolution
    lowres_width = int(opt_bev.shape[1] * lowres_scale)
    lowres_height = int(opt_bev.shape[0] * lowres_scale)
    lowres_bev = cv2.resize(opt_bev, (lowres_width, lowres_height), interpolation=cv2.INTER_AREA)

    lowres_resolution = resolution * lowres_scale
    initial_px_lowres = int(initial_px * lowres_scale)
    initial_py_lowres = int(initial_py * lowres_scale)

    cv2.imwrite(os.path.join(out_dir, "fused_bev_lowres.png"), lowres_bev)
    lowres_metadata = {
        "x_range": result.get("x_range"),
        "y_range": result.get("y_range"),
        "padding": padding,
        "resolution": lowres_resolution,
        "initial_pixel_x": initial_px_lowres,
        "initial_pixel_y": initial_py_lowres,
    }
    with open(os.path.join(out_dir, "bev_lowres_metadata.json"), "w") as f:
        json.dump(lowres_metadata, f, indent=2)

    # 裁剪版：仅保留道路覆盖区域 + 初始点，缩小模板尺寸
    gray = cv2.cvtColor(lowres_bev, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(binary)
    if coords is None:
        print("[BEV] ⚠️ lowres BEV 无道路区域，跳过裁剪版生成")
        return

    x, y, w, h = cv2.boundingRect(coords)
    crop_padding = 20
    x_start = max(0, min(x - crop_padding, initial_px_lowres - crop_padding))
    y_start = max(0, min(y - crop_padding, initial_py_lowres - crop_padding))
    x_end = min(lowres_width, max(x + w + crop_padding, initial_px_lowres + crop_padding))
    y_end = min(lowres_height, max(y + h + crop_padding, initial_py_lowres + crop_padding))

    cropped_bev = lowres_bev[y_start:y_end, x_start:x_end]
    cv2.imwrite(os.path.join(out_dir, "fused_bev_lowres_cropped.png"), cropped_bev)

    cropped_x_min = result["x_range"][0] - padding + x_start / lowres_resolution
    cropped_y_min = result["y_range"][0] - padding + y_start / lowres_resolution
    cropped_metadata = {
        "x_range": result.get("x_range"),
        "y_range": result.get("y_range"),
        "cropped_x_min": cropped_x_min,
        "cropped_y_min": cropped_y_min,
        "padding": padding,
        "resolution": lowres_resolution,
        "initial_pixel_x": initial_px_lowres - x_start,
        "initial_pixel_y": initial_py_lowres - y_start,
    }
    with open(os.path.join(out_dir, "bev_lowres_cropped_metadata.json"), "w") as f:
        json.dump(cropped_metadata, f, indent=2)

    print(f"[BEV] ✅ lowres BEV 已生成: {lowres_width}x{lowres_height}, 裁剪={cropped_bev.shape[1]}x{cropped_bev.shape[0]}")


def bev_projection(point_dir, reuse=True):
    """对 stage1 采集到的街景图做 SAM3 分割 → 投影 → 生成 BEV。
    返回 projection_output 目录路径。"""
    out_dir = os.path.join(point_dir, "projection_output")
    bev_meta_path = os.path.join(out_dir, "bev_metadata.json")
    lowres_meta_path = os.path.join(out_dir, "bev_lowres_metadata.json")
    cropped_meta_path = os.path.join(out_dir, "bev_lowres_cropped_metadata.json")
    if reuse and os.path.exists(bev_meta_path):
        # 已有投影但缺 lowres 版本（旧数据）→ 从 bev_metadata 补生成，避免重跑 SAM3
        if not (os.path.exists(lowres_meta_path) and os.path.exists(cropped_meta_path)):
            print("[BEV] 复用已有投影，但缺 lowres 版本，补生成...")
            with open(bev_meta_path) as f:
                result = json.load(f)
            _generate_lowres_bev(out_dir, result)
        else:
            print(f"[BEV] 复用已有投影: {bev_meta_path}")
        return out_dir

    img_dir = os.path.join(point_dir, "streetview_images")
    if not os.path.isdir(img_dir):
        print("[BEV] ❌ 无 streetview_images 目录")
        return None

    image_files = sorted([f for f in os.listdir(img_dir) if f.endswith("_streetview.jpg")])
    if not image_files:
        print("[BEV] ❌ 无街景图像")
        return None

    os.makedirs(out_dir, exist_ok=True)

    from tools.sam3_tool import SAM3Tool
    sam3_tool = SAM3Tool()
    sam3_tool.initialize()
    try:
        print(f"[BEV] 开始投影 {len(image_files)} 张街景图像...")
        all_points, all_colors, all_step_colors, camera_positions = \
            project_pointcloud(image_files, img_dir, out_dir, sam3_tool)

        if not all_points:
            print("[BEV] ❌ 投影失败，无有效点云")
            return None

        bev_path = os.path.join(out_dir, "fused_bev.png")
        result = generate_bev(all_points, all_colors, bev_path, camera_positions, all_step_colors)
        if result:
            print(f"[BEV] ✅ 投影完成: {result['total_points']} 点")
            print(f"       X={result['x_range'][0]:.1f}~{result['x_range'][1]:.1f}  "
                  f"Y={result['y_range'][0]:.1f}~{result['y_range'][1]:.1f}  "
                  f"尺寸={result['bev_size'][0]}x{result['bev_size'][1]}")
            _generate_lowres_bev(out_dir, result)
    finally:
        sam3_tool.close()

    return out_dir


# =====================================================================
# Stage 2: 招牌文字空间投票（中粒度）
# =====================================================================
def stage2_sign_vote(point_dir, city="", reuse=True, top_n=10):
    """运行招牌空间投票，得到 top-10 候选位置。返回 stage2 数据字典。"""
    out_path = os.path.join(point_dir, STAGE2_FILENAME)
    if reuse and os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            print(f"[Stage 2] 复用已有结果: {out_path}")
            return json.load(f)

    # 若 sign_markers.json 不存在，需要先跑 process_point（VGGT + SAM3 + GPT 识别）
    markers_path = os.path.join(point_dir, "text_output_vggt_depth", "sign_markers.json")
    if not os.path.exists(markers_path):
        bev_metadata_path = os.path.join(point_dir, "projection_output", "bev_metadata.json")
        if not os.path.exists(bev_metadata_path):
            print("[Stage 2] ❌ 缺少 BEV 元数据，请先运行 batch_projection.py 生成 BEV")
            return {"stage": 2, "error": "missing bev_metadata"}
        from tools.sam3_tool import SAM3Tool
        with open(bev_metadata_path) as f:
            bev_metadata = json.load(f)
        sam3_tool = SAM3Tool()
        sam3_tool.initialize()
        try:
            process_point(point_dir, sam3_tool, bev_metadata, city=city)
        finally:
            sam3_tool.close()

    stage2_signs, stage3_signs = load_signs_from_markers(point_dir)
    if not stage2_signs:
        print("[Stage 2] ❌ 无可用招牌")
        return {"stage": 2, "error": "no signs", "top10": []}

    places_tool = _init_places_tool()
    top = spatial_vote_localize(stage2_signs, places_tool, city=city, top_n=top_n)

    # ---- Stage 2 收尾: 高斯核招牌配准 (BEV 相对坐标 ↔ 招牌 GPS) ----
    # 对 top-3 候选分别做高斯核互相关，把带 GPS 的招牌在 BEV 与 GPS 两套
    # 坐标系间做刚体对齐，输出更精细的街区级锚点。
    gaussian_refined, refined_estimate, best_peak_score = _gaussian_refine_top3(point_dir, top, stage3_signs)

    stage2 = {
        "stage": 2,
        "method": "sign_spatial_voting + gaussian_kernel_sign_alignment",
        "city": city,
        "num_signs": len(stage2_signs),
        "top10": top,
        "gaussian_refined": gaussian_refined,
        "confidence": best_peak_score,
        "final_estimate": refined_estimate
                          or ({"lat": top[0]["lat"], "lng": top[0]["lng"]} if top else None),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stage2, f, indent=2, ensure_ascii=False)
    print(f"[Stage 2] 结果已保存: {out_path}")
    return stage2


def _gaussian_refine_top3(point_dir, top10, stage3_signs):
    """对 Stage 2 空间投票的 top-3 分别做 BEV 高斯核互相关精化。

    返回 (refined_list, best_estimate, best_peak_score)。这是 Stage 2 的
    内部收尾步骤，本质仍是招牌语义锚点的对齐，不引入路网等新几何信息。
    """
    if not stage3_signs or not top10:
        return [], None, 0.0

    refined_list = []
    for t in top10[:3]:
        C0 = (t["lat"], t["lng"])
        # 投票结果 supporting_signs: [{text, name, lat, lng, place_id}]
        # 招牌 GPS 取"投票获胜点附近该招牌投的候选点坐标"，与投票互证一致
        vote_gps = {}
        for s in t.get("supporting_signs", []):
            if isinstance(s, dict):
                txt = (s.get("text") or "").strip()
                if txt and s.get("lat") is not None and s.get("lng") is not None:
                    vote_gps[txt] = (s["lat"], s["lng"])
            else:
                vote_gps[str(s)] = None

        use_signs = []
        for s in stage3_signs:
            if s["text"] in vote_gps and vote_gps[s["text"]] is not None:
                lat, lng = vote_gps[s["text"]]
                use_signs.append({
                    "text": s["text"],
                    "x": s["x"],
                    "y": s["y"],
                    "gps_lat": lat,
                    "gps_lng": lng,
                    "weight": s.get("weight", 1.0),
                })
        # 无投票坐标可用的招牌 → 该 C0 无法配准，跳过
        if not use_signs:
            print(f"[Stage 2] C0=top-{t.get('rank')} 无投票坐标招牌，跳过配准")
            continue

        score, dx, dy, corr_map, offsets = gaussian_corr_refine(use_signs, C0)
        peaks = find_top_peaks(corr_map, offsets, top_n=10, min_dist=20.0)
        for p in peaks:
            plat, plng = local_to_latlng(p["dx"], p["dy"], C0[0], C0[1])
            p["lat"] = plat
            p["lng"] = plng

        # 归一化峰值：除以参与配准的招牌数，得到 [0,1] 的平均对齐程度
        n_used = len(use_signs)
        norm_score = (score / n_used) if n_used else 0.0

        ref_lat, ref_lng = local_to_latlng(dx, dy, C0[0], C0[1])
        refined_list.append({
            "C0_rank": t.get("rank"),
            "coarse_C0": {"lat": C0[0], "lng": C0[1]},
            "used_signs": [s["text"] for s in use_signs],
            "offset_m": {"dx": dx, "dy": dy},
            "corr_peak_score": score,
            "normalized_score": norm_score,
            "refined": {"lat": ref_lat, "lng": ref_lng},
            "top10_peaks": peaks,
        })
        print(f"[Stage 2] C0=top-{t.get('rank')} 招牌配准=({ref_lat:.6f}, {ref_lng:.6f}) "
              f"峰值相关={score:.4f} 归一化={norm_score:.4f} (n={n_used})")

    best = max(refined_list, key=lambda r: r["normalized_score"]) if refined_list else None
    best_estimate = {"lat": best["refined"]["lat"], "lng": best["refined"]["lng"]} if best else None
    best_peak_score = best["normalized_score"] if best else 0.0
    return refined_list, best_estimate, best_peak_score


# =====================================================================
# Stage 3: BEV 路网 ↔ OSM 路网卷积匹配（细粒度）
# =====================================================================
def stage3_osm_road_match(point_dir, reuse=True, center=None, fallback_center=None):
    """以 Stage 2 的 final_estimate 为搜索中心，将 BEV 道路模板与 OSM
    路网做卷积模板匹配，输出亚十米级精定位。返回 stage3 数据字典。

    若 Stage 2 无有效结果（无招牌/无候选），则回退用 fallback_center
    （Stage 1 的 coarse_gps）作为搜索中心。
    """
    out_path = os.path.join(point_dir, STAGE3_FILENAME)
    if reuse and os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            print(f"[Stage 3] 复用已有结果: {out_path}")
            return json.load(f)

    center_source = "stage2"
    # 读取 Stage 2 的 final_estimate 作为 OSM 下载/匹配中心
    if center is None:
        stage2_path = os.path.join(point_dir, STAGE2_FILENAME)
        if os.path.exists(stage2_path):
            with open(stage2_path, encoding="utf-8") as f:
                stage2 = json.load(f)
            center = stage2.get("final_estimate")

    # Stage 2 无有效中心 → 回退到 Stage 1 的粗定位
    if not center:
        center = fallback_center
        center_source = "stage1_fallback"

    if not center:
        print("[Stage 3] ❌ Stage 2 与 Stage 1 均无有效中心坐标")
        return {"stage": 3, "error": "no center"}

    print(f"[Stage 3] 搜索中心来自 {center_source}: ({center['lat']:.6f}, {center['lng']:.6f})")

    # BEV 道路模板：优先低分辨率裁剪图，其次低分辨率图
    proj_dir = os.path.join(point_dir, "projection_output")
    bev_candidates = [
        os.path.join(proj_dir, "fused_bev_lowres_cropped.png"),
        os.path.join(proj_dir, "fused_bev_lowres.png"),
        os.path.join(proj_dir, "fused_bev.png"),
    ]
    bev_path = next((p for p in bev_candidates if os.path.exists(p)), None)
    if bev_path is None:
        print("[Stage 3] ❌ 未找到 BEV 图像")
        return {"stage": 3, "error": "no bev image"}

    # 以 Stage 2 中心构造 point，调用卷积匹配器（center 不再用真值）
    name = os.path.basename(point_dir.rstrip(os.sep))
    point = {"name": name, "lat": center["lat"], "lng": center["lng"]}
    parent_dir = os.path.dirname(point_dir.rstrip(os.sep))

    matcher = BEVOSMConvolutionMatcher()
    # 分层匹配：粗层低分辨率覆盖 5km 全局搜索 + 细层高分辨率局部精修。
    # 解决旧单层方案在 osm_radius 增大到 5000m 时 matchTemplate 结果矩阵
    # 内存爆炸（约 1.6GB）的问题，每一层内存都保持在 ~100MB 量级。
    result = matcher.match_point_hierarchical(
        point, parent_dir, coarse_radius=5000, coarse_resolution=0.5,
        fine_window=500, top_k=5)

    if not result:
        print("[Stage 3] ❌ 卷积匹配失败")
        return {"stage": 3, "error": "matching failed"}

    stage3 = {
        "stage": 3,
        "method": "bev_osm_road_network_convolution",
        "center": {"lat": center["lat"], "lng": center["lng"]},
        "center_source": center_source,
        "match_lat": result.get("match_lat"),
        "match_lng": result.get("match_lng"),
        "match_confidence": result.get("match_confidence"),
        "top5_matches": result.get("top5_matches", []),
        "final_position": {
            "lat": result.get("match_lat"),
            "lng": result.get("match_lng"),
        } if result.get("match_lat") is not None else None,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stage3, f, indent=2, ensure_ascii=False)
    print(f"[Stage 3] 结果已保存: {out_path}")
    return stage3


# =====================================================================
# 总编排
# =====================================================================
def run_three_stage(point_dir, lat, lng, max_steps=10, move_step=15.0):
    """依次运行三段定位，返回最终汇总。"""
    os.makedirs(point_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("Stage 1: Agent 自主导航（粗粒度）")
    print("=" * 60)
    stage1 = stage1_coarse_localize(point_dir, lat, lng, max_steps=max_steps, move_step=move_step)

    print("\n" + "=" * 60)
    print("Stage 1.5: BEV 投影 + 文字识别")
    print("=" * 60)
    bev_dir = bev_projection(point_dir)
    if bev_dir is None:
        print("❌ BEV 投影失败，终止后续阶段")
        return None

    print("\n" + "=" * 60)
    print("Stage 2: 招牌语义匹配 + 空间投票 + 高斯核招牌配准（中粒度）")
    print("=" * 60)
    stage2 = stage2_sign_vote(point_dir, city=stage1.get("city", ""))

    print("\n" + "=" * 60)
    print("Stage 3: BEV 路网 ↔ OSM 路网卷积匹配（细粒度）")
    print("=" * 60)
    # 若 Stage 2 无结果，Stage 3 回退用 Stage 1 粗定位（或原始输入坐标）作为搜索中心
    fallback = stage1.get("coarse_gps") or {"lat": lat, "lng": lng}
    stage3 = stage3_osm_road_match(point_dir, fallback_center=fallback)

    # ---- 置信度门控：低置信度阶段不采信，回退到上一阶段 ----
    final_position, adopted_stage = _select_final_position(stage1, stage2, stage3)

    summary = {
        "method": "three_stage_cascade_localization",
        "point_dir": point_dir,
        "stages": {
            "stage1_coarse": stage1,
            "stage2_medium": stage2,
            "stage3_fine": stage3,
        },
        "adopted_stage": adopted_stage,
        "final_position": final_position,
    }
    with open(os.path.join(point_dir, FINAL_FILENAME), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("三段式定位完成:")
    print(f"  采信阶段: {adopted_stage}")
    print(f"  最终坐标: ({final_position['lat']:.6f}, {final_position['lng']:.6f})"
          if final_position else "  最终坐标: (无结果)")
    print(f"汇总已保存: {os.path.join(point_dir, FINAL_FILENAME)}")
    return summary


def _select_final_position(stage1, stage2, stage3):
    """按置信度门控从三段结果中选出最终位置。

    优先采信 Stage 3（最细），若其置信度低于阈值则不采信并回退 Stage 2，
    Stage 2 置信度不足则继续回退 Stage 1 粗定位。

    返回 (final_position, adopted_stage)。
    """
    # Stage 3: 卷积匹配置信度够高才采信
    s3_conf = stage3.get("match_confidence")
    s3_pos = stage3.get("final_position")
    if s3_pos and s3_conf is not None and s3_conf >= STAGE3_MIN_CONFIDENCE:
        return s3_pos, "stage3_osm_match"

    # Stage 2: 高斯核招牌配准峰值够高才采信
    s2_conf = stage2.get("confidence")
    s2_pos = stage2.get("final_estimate")
    if s2_pos and s2_conf is not None and s2_conf >= STAGE2_MIN_PEAK_SCORE:
        return s2_pos, "stage2_sign_vote"

    # Stage 1: 粗定位兜底
    s1_pos = stage1.get("coarse_gps")
    if s1_pos:
        return s1_pos, "stage1_coarse"

    return None, "none"


def main():
    parser = argparse.ArgumentParser(description="三段式级联定位")
    parser.add_argument("--lat", type=float, required=True, help="初始纬度")
    parser.add_argument("--lng", type=float, required=True, help="初始经度")
    parser.add_argument("--name", type=str, default="point", help="点位名称")
    parser.add_argument("--point-dir", type=str, default=None, help="点位目录（默认自动生成）")
    parser.add_argument("--max-steps", type=int, default=10, help="agent 导航步数")
    parser.add_argument("--move-step", type=float, default=15.0, help="单步移动距离(米)")
    args = parser.parse_args()

    if args.point_dir:
        point_dir = args.point_dir
    else:
        import datetime
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        lat_s = f"{args.lat:.4f}".replace(".", "_")
        lng_s = f"{args.lng:.4f}".replace(".", "_")
        point_dir = os.path.join(PROJECT_DIR, "manual_streetview_results",
                                 f"{ts}_{args.name}_lat{lat_s}_lng{lng_s}")

    run_three_stage(point_dir, args.lat, args.lng, max_steps=args.max_steps, move_step=args.move_step)


if __name__ == "__main__":
    main()
