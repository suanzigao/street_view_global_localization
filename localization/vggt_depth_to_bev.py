"""
VGGT深度图 → 路面校准 → 招牌中心点BEV定位

流程:
1. 调用远程VGGT服务获取每张街景图像的相对深度图
2. SAM3分割路面(road)和招牌(shop sign, restaurant sign)
3. 路面像素 → 射线与Z=0平面求交 → 理论绝对深度
4. 路面像素的VGGT深度值 → 尺度因子k = median(d_abs / d_rel)
5. 招牌中心点 → VGGT深度 × k → 真实深度 → 反投影到3D世界坐标
6. 跨图像聚类 → BEV上绘制点标记
"""

import os
import sys
import json
import math
import base64
import io
import re
import difflib
import numpy as np
import cv2
import requests
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tools.sam3_tool import SAM3Tool
from tools.places_tool import PlacesAPITool
from utils.file_utils import image_to_base64

load_dotenv()

VGGT_SERVER = "http://127.0.0.1:18888"
TEXT_PROMPTS = ["shop sign", "restaurant sign", "storefront sign"]
PROMPT_COLORS = [(0, 255, 0), (255, 165, 0), (0, 0, 255)]
ROAD_PROMPTS = ["road", "street"]
MAX_DEPTH = 200.0
MIN_DEPTH = 0.5

LLM_MODEL = "gpt-4o"
LLM_BASE_URL = "https://api.laozhang.ai/v1"

TEXT_CONFIDENCE_THRESHOLD = 0.9
TEXT_SIMILARITY_THRESHOLD = 0.6

SPATIAL_VOTE_K = 20
SPATIAL_VOTE_R_UNIQUE = 200.0
SPATIAL_VOTE_R_VOTE = 300.0
SPATIAL_VOTE_TOP_N = 10

TEXT_RECOGNITION_PROMPT = """你是一个招牌文字识别助手。请识别图片中招牌、路牌、店铺标志、广告牌上最显眼、最有辨识度的文字（通常是店名、品牌名、地名）。

要求：
1. 只识别"能用于地点搜索"的文字（店名/品牌/地名），忽略装饰性文字、广告语、车牌等。
2. 如果图片中有可读的店名/品牌文字，text 填该文字；search_query 填"文字, 城市或国家"（如果你能根据图片判断所在城市/国家就加上，否则只写文字本身）。
3. 如果图片中没有可用于地点搜索的文字，text 填空字符串 ""。
4. confidence 填 0 到 1 之间的数字，表示你对识别结果的确信程度。

只输出 JSON，不要输出任何其他内容，格式如下：
{{
  "text": "识别到的招牌文字",
  "search_query": "用于地点搜索的query",
  "confidence": 0.9
}}
"""


def _init_llm_client():
    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=LLM_BASE_URL,
    )


def _init_places_tool():
    return PlacesAPITool()


def read_metadata(metadata_path):
    with open(metadata_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_intrinsics(width, height, fov_deg):
    fov = math.radians(fov_deg)
    fx = width / (2 * math.tan(fov / 2))
    fy = height / (2 * math.tan(fov / 2))
    cx = width / 2
    cy = height / 2
    return fx, fy, cx, cy


def get_rotation_matrix(heading_deg, pitch_deg):
    yaw = -math.radians(heading_deg)
    pitch = math.radians(pitch_deg)

    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    cos_p, sin_p = math.cos(pitch), math.sin(pitch)

    right = np.array([cos_y, sin_y, 0.0])
    forward = np.array([-sin_y * cos_p, cos_y * cos_p, sin_p])
    down = np.array([-sin_y * sin_p, cos_y * sin_p, -cos_p])

    R = np.column_stack([right, down, forward])
    return R, right, forward, down


def compute_ray_direction(x, y, fx, fy, cx, cy, R):
    u = (x - cx) / fx
    v = (y - cy) / fy
    ray_cam = np.array([u, v, 1.0])
    ray_world = R @ ray_cam
    return ray_world / np.linalg.norm(ray_world)


def theoretical_ground_depth(x, y, fx, fy, cx, cy, R, cam_z):
    ray = compute_ray_direction(x, y, fx, fy, cx, cy, R)
    if ray[2] >= -1e-6:
        return None
    t = -cam_z / ray[2]
    if t < 0:
        return None
    return t


def pixel_to_world_with_depth(x, y, depth, fx, fy, cx, cy, R, cam_pos):
    u = (x - cx) / fx
    v = (y - cy) / fy
    ray_cam = np.array([u * depth, v * depth, depth])
    world_pt = R @ ray_cam + np.array(cam_pos)
    return world_pt


def call_vggt_infer(image_paths, server_url=VGGT_SERVER):
    imgs_b64 = []
    for p in image_paths:
        with open(p, 'rb') as f:
            imgs_b64.append(base64.b64encode(f.read()).decode())

    req = {"images": imgs_b64, "output_types": ["depth"]}
    r = requests.post(f"{server_url}/vggt/infer", json=req, timeout=300)
    result = r.json()

    if not result.get("success"):
        print(f"  ❌ VGGT infer failed: {result}")
        return None

    depth_raw = result.get("depth")
    if depth_raw is None:
        print(f"  ❌ VGGT returned no depth")
        return None

    depth_arr = np.array(depth_raw, dtype=np.float32)
    return depth_arr


def calibrate_depth_scale(road_mask, depth_rel, fx, fy, cx, cy, R, cam_z, target_h, target_w):
    """
    用路面像素校准VGGT相对深度图的尺度因子k
    
    k = median(理论绝对深度 / VGGT相对深度)
    
    返回: (k, calibration_stats_dict)
    """
    h, w = road_mask.shape[:2]
    rh, rw = depth_rel.shape[:2]
    
    if (rh, rw) != (h, w):
        depth_rel_full = cv2.resize(depth_rel, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        depth_rel_full = depth_rel.copy()

    valid_mask = road_mask > 0
    valid_pixels = np.where(valid_mask)
    
    if len(valid_pixels[0]) < 50:
        return None, {"reason": f"路面像素不足: {len(valid_pixels[0])}"}

    sample_size = min(3000, len(valid_pixels[0]))
    indices = np.random.choice(len(valid_pixels[0]), sample_size, replace=False)
    sample_ys = valid_pixels[0][indices]
    sample_xs = valid_pixels[1][indices]

    d_abs_list = []
    d_rel_list = []

    for i in range(len(sample_ys)):
        px = sample_xs[i]
        py = sample_ys[i]

        d_abs = theoretical_ground_depth(px, py, fx, fy, cx, cy, R, cam_z)
        if d_abs is None or d_abs < MIN_DEPTH or d_abs > MAX_DEPTH:
            continue

        d_rel = depth_rel_full[py, px]
        if d_rel < 1e-8 or not np.isfinite(d_rel):
            continue

        d_abs_list.append(d_abs)
        d_rel_list.append(d_rel)

    if len(d_abs_list) < 20:
        return None, {"reason": f"有效校准像素不足: {len(d_abs_list)}"}

    d_abs_arr = np.array(d_abs_list)
    d_rel_arr = np.array(d_rel_list)
    ratios = d_abs_arr / d_rel_arr
    k = float(np.median(ratios))

    d_abs_pred = k * d_rel_arr
    errors = np.abs(d_abs_pred - d_abs_arr)
    
    good_mask = errors < 10.0
    if np.sum(good_mask) > 50:
        k = float(np.median(d_abs_arr[good_mask] / d_rel_arr[good_mask]))

    final_errors = np.abs(k * d_rel_arr - d_abs_arr)
    
    stats = {
        "calibration_pixels": len(d_abs_list),
        "scale_factor_k": k,
        "error_median": float(np.median(final_errors)),
        "error_mean": float(np.mean(final_errors)),
        "error_max": float(np.max(final_errors)),
    }

    return k, stats


def detect_road_mask(image_pil, sam3_tool, target_size):
    result = sam3_tool.get_segmented_images(image_pil, ROAD_PROMPTS)
    
    h, w = target_size
    combined = np.zeros((h, w), dtype=np.uint8)
    
    for seg_result in result.get('results', []):
        if not seg_result.get('success') or not seg_result.get('masks'):
            continue
        for mask in seg_result['masks']:
            mask_np = np.array(mask)
            if mask_np.ndim == 3:
                mask_np = mask_np[0]
            if mask_np.max() <= 1:
                mask_np = (mask_np * 255).astype(np.uint8)
            else:
                mask_np = mask_np.astype(np.uint8)
            
            if mask_np.shape[:2] != (h, w):
                mask_img = Image.fromarray(mask_np, mode='L')
                mask_img = mask_img.resize((w, h), Image.NEAREST)
                mask_np = np.array(mask_img)
            
            combined = np.maximum(combined, mask_np)
    
    return combined


def detect_sign_objects(image_pil, sam3_tool, target_size):
    h, w = target_size
    results = []
    
    prompt_results = sam3_tool.get_segmented_images(image_pil, TEXT_PROMPTS)
    
    for idx, seg_result in enumerate(prompt_results.get('results', [])):
        prompt = seg_result.get('prompt', TEXT_PROMPTS[idx] if idx < len(TEXT_PROMPTS) else f"unknown_{idx}")
        if not seg_result.get('success') or not seg_result.get('masks'):
            continue
        
        masks = seg_result.get('masks', [])
        scores = seg_result.get('scores', [])
        
        for mask_idx, mask in enumerate(masks):
            mask_np = np.array(mask)
            if mask_np.ndim == 3:
                mask_np = mask_np[0]
            if mask_np.max() <= 1:
                mask_np = (mask_np * 255).astype(np.uint8)
            else:
                mask_np = mask_np.astype(np.uint8)
            
            if mask_np.shape[:2] != (h, w):
                mask_img = Image.fromarray(mask_np, mode='L')
                mask_img = mask_img.resize((w, h), Image.NEAREST)
                mask_np = np.array(mask_img)
            
            pixel_count = np.count_nonzero(mask_np)
            if pixel_count < 10:
                continue
            
            ys, xs = np.where(mask_np > 0)
            centroid_x = float(np.mean(xs))
            centroid_y = float(np.mean(ys))
            
            score = scores[mask_idx] if mask_idx < len(scores) else 0.5
            
            results.append({
                'prompt': prompt,
                'score': float(score),
                'centroid': {'x': centroid_x, 'y': centroid_y},
                'bbox': [int(np.min(xs)), int(np.min(ys)), int(np.max(xs)), int(np.max(ys))],
                'mask': mask_np,
            })
    
    return results


def deduplicate_masks(text_objects, overlap_threshold=0.5):
    if not text_objects:
        return []
    
    unique = []
    for obj in text_objects:
        mask = obj['mask']
        is_dup = False
        for existing in unique:
            existing_mask = existing['mask']
            intersection = np.logical_and(mask > 0, existing_mask > 0).sum()
            union = np.logical_or(mask > 0, existing_mask > 0).sum()
            if union > 0 and intersection / union > overlap_threshold:
                if obj['score'] > existing['score']:
                    unique.remove(existing)
                    unique.append(obj)
                is_dup = True
                break
        if not is_dup:
            unique.append(obj)
    return unique


def crop_sign_region(image_np, bbox, pad_ratio=0.5):
    """根据 bbox 裁剪招牌区域，四周按比例扩展留出上下文。"""
    h, w = image_np.shape[:2]
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    pad_x = int(bw * pad_ratio)
    pad_y = int(bh * pad_ratio)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)
    return image_np[y1:y2, x1:x2]


def recognize_sign_text(crop_image_np, llm_client):
    """调用 GPT 识别招牌裁剪图中的文字，返回 (text, search_query, confidence)。"""
    if crop_image_np is None or crop_image_np.size == 0:
        return "", "", 0.0
    crop_pil = Image.fromarray(crop_image_np.astype(np.uint8))
    b64 = image_to_base64(crop_pil)

    try:
        resp = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": TEXT_RECOGNITION_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }],
            temperature=0.1,
            max_tokens=100,
            timeout=60,
        )
        content = resp.choices[0].message.content.strip()
        m = re.search(r'\{[\s\S]*\}', content)
        if m:
            data = json.loads(m.group(0))
            text = data.get("text", "").strip()
            query = data.get("search_query", "").strip()
            conf = data.get("confidence", 0.0)
            try:
                conf = float(conf)
            except (TypeError, ValueError):
                conf = 0.0
            conf = max(0.0, min(1.0, conf))
            return text, query, conf
    except Exception as e:
        print(f"      ⚠️ 文字识别失败: {e}")
    return "", "", 0.0


def reverse_geocode_city(lat, lng):
    """逆地理编码得到城市名（英文），失败返回空字符串。"""
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        return ""
    try:
        url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {"latlng": f"{lat},{lng}", "key": api_key, "language": "en"}
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        if data.get("status") != "OK":
            return ""
        for res in data.get("results", []):
            for comp in res.get("address_components", []):
                if "locality" in comp.get("types", []):
                    return comp.get("long_name", "")
    except Exception as e:
        print(f"      ⚠️ 逆地理编码失败: {e}")
    return ""


def search_place_gps(query, places_tool, city="", ref_coord=None):
    """用识别文字搜索 Places API，返回招牌 GPS 坐标。city 非空时拼入搜索限定城市。

    与空间投票一致使用 top-20 候选（find_place_top_k）。若提供 ref_coord
    （粗定位参考坐标），从 top-20 里选距离参考坐标最近的分店，避免连锁店
    （如 GORILLA SPORTS / Starbucks）取到其它城市的错误分店；否则取第一个。
    """
    if not query:
        return None
    search_input = query
    if city and city.lower() not in query.lower():
        search_input = f"{query}, {city}"
    try:
        candidates = places_tool.find_place_top_k(search_input, k=20)
        if not candidates:
            return None
        if ref_coord is not None:
            best = min(
                candidates,
                key=lambda c: haversine_distance_m(
                    ref_coord[0], ref_coord[1],
                    c.get('lat') if c.get('lat') is not None else 0.0,
                    c.get('lng') if c.get('lng') is not None else 0.0,
                ),
            )
        else:
            best = candidates[0]
        if best.get('lat') is None or best.get('lng') is None:
            return None
        return {
            "name": best.get("name", query),
            "address": best.get("address", ""),
            "lat": best.get("lat"),
            "lng": best.get("lng"),
        }
    except Exception as e:
        print(f"      ⚠️ 地点搜索失败: {e}")
    return None


def haversine_distance_m(lat1, lng1, lat2, lng2):
    """计算两个 GPS 坐标间的球面距离（米）。"""
    R = 6371000.0
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlng / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _weighted_median(values, weights):
    """加权中位数：按权重累计到一半时返回对应值。"""
    pairs = sorted(zip(values, weights))
    total = sum(weights)
    if total <= 0:
        return sum(values) / len(values)
    cum = 0.0
    for v, w in pairs:
        cum += w
        if cum * 2 >= total:
            return v
    return pairs[-1][0]


def filter_outlier_gps(markers, cluster_radius_m=5000.0):
    """对标记的 GPS 坐标做离群滤除。

    思路：按地理距离对 GPS 做聚类，保留点数最多的簇（真实招牌都聚在采集点所在城市），
    其余簇视为幻觉/错误坐标，将其 gps 置空并标记 gps_filtered。
    返回 (filtered_markers, final_estimate)，final_estimate 为最大簇的加权中位数 (lat, lng) 或 None。
    """
    valid_idx = [i for i, m in enumerate(markers)
                 if m.get('gps') and m['gps'].get('lat') is not None and m['gps'].get('lng') is not None]
    if not valid_idx:
        return markers, None

    n = len(valid_idx)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    coords = [(markers[i]['gps']['lat'], markers[i]['gps']['lng']) for i in valid_idx]
    for a in range(n):
        for b in range(a + 1, n):
            if haversine_distance_m(coords[a][0], coords[a][1], coords[b][0], coords[b][1]) < cluster_radius_m:
                union(a, b)

    groups = {}
    for k in range(n):
        groups.setdefault(find(k), []).append(valid_idx[k])

    largest = max(groups.values(), key=len)
    largest_set = set(largest)

    for i in valid_idx:
        if i not in largest_set:
            markers[i]['gps'] = None
            markers[i]['gps_filtered'] = True

    lats, lngs, weights = [], [], []
    for i in largest:
        m = markers[i]
        lats.append(m['gps']['lat'])
        lngs.append(m['gps']['lng'])
        weights.append(m.get('num_observations', 1))

    final_estimate = (_weighted_median(lats, weights), _weighted_median(lngs, weights))
    return markers, final_estimate


def spatial_vote_localize(signs, places_tool, city="", k=SPATIAL_VOTE_K,
                          r_unique=SPATIAL_VOTE_R_UNIQUE, r_vote=SPATIAL_VOTE_R_VOTE,
                          top_n=SPATIAL_VOTE_TOP_N, sort_by="weighted_support",
                          return_both=False):
    """空间投票定位：每个招牌搜索 top-k 候选 → 去重 → 聚类 → 统计支持度 → 打分 → 输出 top-n 位置。

    sort_by: 排序主键，可选 "weighted_support"（特异性加权，默认）或 "support_count"（纯招牌数）。
    return_both: 为 True 时，基于同一批候选/聚类结果，同时返回两种排序的 top-n，
                 返回 (top_weighted_support, top_support_count)；否则按 sort_by 返回单个列表。

    返回 list of {
        'rank', 'lat', 'lng', 'support_count',
        'supporting_signs', 'avg_confidence', 'score'
    }
    """
    from sklearn.cluster import DBSCAN

    n_search = len(signs)
    if n_search == 0:
        return ([], []) if return_both else []

    # 1. 每个招牌搜索 top-k 候选（按招牌分组，便于计算特异性权重）
    sign_results = {}
    for sign_idx, s in enumerate(signs):
        text = (s.get('text') or '').strip()
        query = (s.get('search_query') or '').strip() or text
        if not query:
            continue
        search_input = query
        if city and city.lower() not in query.lower():
            search_input = f"{query}, {city}"
        results = places_tool.find_place_top_k(search_input, k=k)
        cands = []
        for rank, c in enumerate(results):
            if c.get('lat') is None or c.get('lng') is None:
                continue
            cands.append({
                'sign_id': sign_idx,
                'text': text,
                'text_confidence': s.get('text_confidence', 0.0),
                'rank': rank + 1,
                'place_id': c.get('place_id', ''),
                'name': c.get('name', ''),
                'lat': c['lat'],
                'lng': c['lng'],
            })
        if cands:
            sign_results[sign_idx] = cands

    if not sign_results:
        return ([], []) if return_both else []

    # 1.5 计算每个招牌的特异性权重：候选越集中(越像独有店)权重越高，
    #     越分散(越像连锁店)权重越低。用候选到质心的平均距离衡量。
    sign_weight = {}
    for sign_idx, cands in sign_results.items():
        if len(cands) <= 1:
            spread_km = 0.0
        else:
            lat0 = sum(c['lat'] for c in cands) / len(cands)
            lng0 = sum(c['lng'] for c in cands) / len(cands)
            dists = [haversine_distance_m(lat0, lng0, c['lat'], c['lng']) for c in cands]
            spread_km = (sum(dists) / len(dists)) / 1000.0
        sign_weight[sign_idx] = 1.0 / (1.0 + spread_km)

    candidates = [c for cands in sign_results.values() for c in cands]

    # 2. 按 place_id 去重（保留 rank 更靠前者）
    seen = {}
    for c in candidates:
        pid = c['place_id'] or f"{c['lat']:.6f},{c['lng']:.6f}"
        if pid not in seen or c['rank'] < seen[pid]['rank']:
            seen[pid] = c
    unique_candidates = list(seen.values())

    # 3. DBSCAN 空间聚类，把近邻候选合并成"唯一位置"
    coords_rad = np.radians([[c['lat'], c['lng']] for c in unique_candidates])
    eps_rad = r_unique / 6371000.0
    if len(unique_candidates) == 1:
        labels = [0]
    else:
        db = DBSCAN(eps=eps_rad, min_samples=1, metric='haversine').fit(coords_rad)
        labels = db.labels_.tolist()

    clusters = {}
    for c, lab in zip(unique_candidates, labels):
        clusters.setdefault(lab, []).append(c)

    # 聚类中心（用 text_confidence 加权）
    locations = []
    for members in clusters.values():
        lats = [m['lat'] for m in members]
        lngs = [m['lng'] for m in members]
        weights = [max(m['text_confidence'], 0.01) for m in members]
        total = sum(weights)
        clat = sum(la * w for la, w in zip(lats, weights)) / total
        clng = sum(ln * w for ln, w in zip(lngs, weights)) / total
        locations.append({'lat': clat, 'lng': clng})

    # 4. 统计每个位置的支持度（半径 r_vote 内不同招牌数）
    scored = []
    for L in locations:
        best_conf = {}
        for c in unique_candidates:
            d = haversine_distance_m(L['lat'], L['lng'], c['lat'], c['lng'])
            if d < r_vote:
                sid = c['sign_id']
                if sid not in best_conf or c['rank'] < best_conf[sid]['rank']:
                    best_conf[sid] = c

        support = len(best_conf)
        conf_sum = sum(c['text_confidence'] for c in best_conf.values())
        rank_sum = sum(1.0 / c['rank'] for c in best_conf.values())
        weighted_support = sum(sign_weight[sid] for sid in best_conf)

        # 4.5 支持数上限校验：超过实际搜索招牌数说明统计异常，放弃
        if support > n_search:
            continue

        # 支持招牌：文字 + 每个招牌投票到的候选点坐标（投票获胜点附近该招牌最近的 GPS）
        supporting_signs = []
        for sid in sorted(best_conf):
            c = best_conf[sid]
            supporting_signs.append({
                'text': c.get('text', ''),
                'name': c.get('name', ''),
                'lat': c.get('lat'),
                'lng': c.get('lng'),
                'place_id': c.get('place_id', ''),
            })
        supporting_texts = [s['text'] for s in supporting_signs if s.get('text')]

        # 主排序用加权支持度（特异性加权），置信度/排名仅作 tie-breaker
        score = weighted_support + 0.01 * conf_sum + 0.01 * rank_sum
        scored.append({
            'lat': L['lat'],
            'lng': L['lng'],
            'support_count': support,
            'weighted_support': weighted_support,
            'supporting_signs': supporting_signs,
            'supporting_texts': supporting_texts,
            'avg_confidence': (conf_sum / support) if support else 0.0,
            'score': score,
        })

    # 5. 排序取 top-n
    def _rank_top(items, sort_key):
        ranked = [dict(t) for t in sorted(items, key=sort_key)[:top_n]]
        for i, t in enumerate(ranked):
            t['rank'] = i + 1
        return ranked

    if return_both:
        top_weighted = _rank_top(scored, sort_key=lambda x: (-x['weighted_support'], -x['score']))
        top_count = _rank_top(scored, sort_key=lambda x: (-x['support_count'], -x['weighted_support'], -x['score']))
        return top_weighted, top_count

    if sort_by == "support_count":
        return _rank_top(scored, sort_key=lambda x: (-x['support_count'], -x['weighted_support'], -x['score']))
    return _rank_top(scored, sort_key=lambda x: (-x['weighted_support'], -x['score']))


def project_sign_with_depth(sign_obj, depth_rel, k, fx, fy, cx, cy, R, cam_pos, target_h, target_w):
    centroid_x = sign_obj['centroid']['x']
    centroid_y = sign_obj['centroid']['y']
    
    rh, rw = depth_rel.shape[:2]
    
    if (rh, rw) != (target_h, target_w):
        depth_scaled = cv2.resize(depth_rel, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    else:
        depth_scaled = depth_rel
    
    px = int(centroid_x)
    py = int(centroid_y)
    
    if not (0 <= px < depth_scaled.shape[1] and 0 <= py < depth_scaled.shape[0]):
        return None
    
    d_rel = float(depth_scaled[py, px])
    if d_rel < 1e-8 or not np.isfinite(d_rel):
        return None
    
    d_abs = k * d_rel
    if d_abs < MIN_DEPTH or d_abs > MAX_DEPTH:
        return None
    
    world_pt = pixel_to_world_with_depth(centroid_x, centroid_y, d_abs, fx, fy, cx, cy, R, cam_pos)
    
    return {
        'prompt': sign_obj['prompt'],
        'score': sign_obj['score'],
        'text': sign_obj.get('text', ''),
        'search_query': sign_obj.get('search_query', ''),
        'text_confidence': sign_obj.get('text_confidence', 0.0),
        'gps': sign_obj.get('gps'),
        'pixel_position': {'x': centroid_x, 'y': centroid_y},
        'relative_depth': d_rel,
        'absolute_depth': d_abs,
        'world_position': {'x': float(world_pt[0]), 'y': float(world_pt[1]), 'z': float(world_pt[2])},
        'camera_position': {'x': float(cam_pos[0]), 'y': float(cam_pos[1]), 'z': float(cam_pos[2])},
    }


def _normalize_text(t):
    """归一化招牌文字：小写、去空格、去非字母数字，用于一致性比较。"""
    return re.sub(r'[^a-z0-9]', '', t.lower())


def _texts_similar(a, b):
    """判断两个归一化文本是否相似（完全相同或高相似度）。"""
    if a == b:
        return True
    if not a or not b:
        return False
    return difflib.SequenceMatcher(None, a, b).ratio() >= TEXT_SIMILARITY_THRESHOLD


def _resolve_cluster_text(cluster):
    """多视角一致性校验：同一聚类的多个观测，识别文字必须一致才采纳。

    返回 (text, search_query, text_confidence, gps)：
    - 若聚类内所有带文字的观测归一化后相互一致，选置信度最高/文字最长的观测，
      其 gps 一并返回，保证文字与坐标来自同一观测；
    - 若存在明显冲突（不同文字），返回空，视为不可靠丢弃。
    """
    with_text = [m for m in cluster if m.get('text', '').strip()]
    if not with_text:
        return '', '', 0.0, None

    normalized = [_normalize_text(m['text']) for m in with_text]
    groups = []
    group_repr = []
    for i, nt in enumerate(normalized):
        placed = False
        for gi, rep in enumerate(group_repr):
            if _texts_similar(nt, rep):
                groups[gi].append(with_text[i])
                placed = True
                break
        if not placed:
            groups.append([with_text[i]])
            group_repr.append(nt)

    best_group = max(groups, key=len)

    # 多数派规则：最大一致组需覆盖至少 2/3 的带文字观测，否则视为冲突丢弃
    if len(best_group) * 3 < len(with_text) * 2:
        return '', '', 0.0, None

    has_conf = [m for m in best_group if m.get('text_confidence', 0.0) > 0]
    pool = has_conf if has_conf else best_group
    if has_conf:
        best = max(pool, key=lambda m: (m.get('text_confidence', 0.0), len(m.get('text', '').strip())))
    else:
        best = max(pool, key=lambda m: len(m.get('text', '').strip()))
    return best.get('text', '').strip(), best.get('search_query', ''), best.get('text_confidence', 0.0), best.get('gps')


def _select_best_marker_for_text(cluster):
    """从同一聚类的多个观测中选择文字命名来源。

    策略：
    1. 优先在识别出非空文字的观测中，选 text_confidence 最高的；
    2. 若都没有置信度信息，选文字最长的（正面视角通常最完整）；
    3. 若都未识别出文字，退回 SAM3 分割置信度最高的。
    """
    with_text = [m for m in cluster if m.get('text', '').strip()]
    if with_text:
        has_conf = [m for m in with_text if m.get('text_confidence', 0.0) > 0]
        pool = has_conf if has_conf else with_text
        if has_conf:
            return max(pool, key=lambda m: (m.get('text_confidence', 0.0), len(m.get('text', '').strip())))
        return max(pool, key=lambda m: len(m.get('text', '').strip()))
    return max(cluster, key=lambda m: m.get('score', 0))


def cluster_markers(markers, distance_threshold=5.0):
    if not markers:
        return []
    
    clusters = []
    for marker in markers:
        wp = marker['world_position']
        best_cluster = None
        best_dist = float('inf')
        
        for cluster in clusters:
            cx = np.mean([c['world_position']['x'] for c in cluster])
            cy = np.mean([c['world_position']['y'] for c in cluster])
            dist = math.sqrt((wp['x'] - cx)**2 + (wp['y'] - cy)**2)
            if dist < best_dist:
                best_dist = dist
                best_cluster = cluster
        
        if best_cluster is not None and best_dist < distance_threshold:
            best_cluster.append(marker)
        else:
            clusters.append([marker])
    
    result = []
    for cluster in clusters:
        source_images = sorted({m.get('source_image', '') for m in cluster if m.get('source_image')})
        if len(cluster) == 1:
            m = cluster[0]
            m['num_observations'] = 1
            m['source_images'] = source_images
            result.append(m)
        else:
            xs = [m['world_position']['x'] for m in cluster]
            ys = [m['world_position']['y'] for m in cluster]
            zs = [m['world_position']['z'] for m in cluster]
            scores = [m['score'] for m in cluster]
            total_score = sum(scores)
            if total_score > 0:
                wx = sum(x * s for x, s in zip(xs, scores)) / total_score
                wy = sum(y * s for y, s in zip(ys, scores)) / total_score
                wz = sum(z * s for z, s in zip(zs, scores)) / total_score
            else:
                wx = np.mean(xs)
                wy = np.mean(ys)
                wz = np.mean(zs)
            
            best_marker = _select_best_marker_for_text(cluster)
            resolved_text, resolved_query, resolved_conf, resolved_gps = _resolve_cluster_text(cluster)
            if not resolved_text:
                continue
            result.append({
                'prompt': best_marker['prompt'],
                'score': best_marker['score'],
                'text': resolved_text,
                'search_query': resolved_query,
                'text_confidence': resolved_conf,
                'gps': resolved_gps,
                'world_position': {'x': float(wx), 'y': float(wy), 'z': float(wz)},
                'num_observations': len(cluster),
                'source_image': best_marker.get('source_image', 'multi_avg'),
                'source_images': source_images,
                'absolute_depth': best_marker.get('absolute_depth'),
            })
    
    return result


def world_to_bev(world_x, world_y, x_min, y_min, padding, resolution, bev_height):
    px = int((world_x - x_min + padding) * resolution)
    py = bev_height - 1 - int((world_y - y_min + padding) * resolution)
    return px, py


def draw_markers_on_bev(bev_image, markers, x_min, y_min, padding, resolution):
    bev_height, bev_width = bev_image.shape[:2]
    marked = bev_image.copy()
    
    drawn = 0
    for marker in markers:
        wp = marker['world_position']
        px, py = world_to_bev(wp['x'], wp['y'], x_min, y_min, padding, resolution, bev_height)
        
        if not (0 <= px < bev_width and 0 <= py < bev_height):
            continue
        
        prompt = marker.get('prompt', 'unknown')
        color_idx = TEXT_PROMPTS.index(prompt) if prompt in TEXT_PROMPTS else 0
        color = PROMPT_COLORS[color_idx % len(PROMPT_COLORS)]
        
        n_obs = marker.get('num_observations', 1)
        radius = 6 + n_obs * 2
        
        cv2.circle(marked, (px, py), radius, color, -1)
        cv2.circle(marked, (px, py), radius + 3, color, 2)
        
        # 优先用识别到的文字命名；否则退回 prompt
        text = marker.get('text', '')
        label = text if text else prompt.split()[0]
        if n_obs > 1:
            label = f"{label}({n_obs})"
        cv2.putText(marked, label, (px + radius + 5, py + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 若已获得 GPS，标注坐标
        gps = marker.get('gps')
        if gps and gps.get('lat') is not None and gps.get('lng') is not None:
            gps_label = f"({gps['lat']:.4f}, {gps['lng']:.4f})"
            cv2.putText(marked, gps_label, (px + radius + 5, py + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
        
        drawn += 1
    
    return marked, drawn


def _read_city(point_dir, explicit_city=""):
    """解析文字搜索限定城市名。

    优先级：显式传入 city > stage1_coarse.json 的 city > collection_summary.json
    逆地理编码。
    """
    if explicit_city:
        print(f"  文字搜索限定城市: {explicit_city}")
        return explicit_city

    # Stage 1 结果（agent 导航流程产物，含 city 字段）
    stage1_path = os.path.join(point_dir, 'stage1_coarse.json')
    if os.path.exists(stage1_path):
        try:
            with open(stage1_path, 'r', encoding='utf-8') as f:
                stage1 = json.load(f)
            c = stage1.get('city')
            if c:
                print(f"  文字搜索限定城市(来自 stage1): {c}")
                return c
        except (json.JSONDecodeError, OSError):
            pass

    # collection_summary（手动采集流程产物，需逆地理编码）
    summary_path = os.path.join(point_dir, 'collection_summary.json')
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        lat = summary.get('initial_lat') or summary.get('actual_initial_lat')
        lng = summary.get('initial_lng') or summary.get('actual_initial_lng')
        if lat is not None and lng is not None:
            c = reverse_geocode_city(lat, lng)
            if c:
                print(f"  逆地理编码城市: {c}")
                return c
            print(f"  ⚠️ 逆地理编码城市失败，文字搜索不限定城市")
            return ""
    print(f"  ⚠️ 未找到 stage1_coarse.json / collection_summary.json，文字搜索不限定城市")
    return ""


def _read_coarse_gps(point_dir):
    """解析粗定位参考坐标 (lat, lng)，用于招牌 GPS 搜索时选最近分店。

    优先级：stage1_coarse.json 的 coarse_gps > collection_summary.json 的 initial 坐标。
    返回 (lat, lng) 或 None。
    """
    stage1_path = os.path.join(point_dir, 'stage1_coarse.json')
    if os.path.exists(stage1_path):
        try:
            with open(stage1_path, 'r', encoding='utf-8') as f:
                stage1 = json.load(f)
            cg = stage1.get('coarse_gps')
            if cg and cg.get('lat') is not None and cg.get('lng') is not None:
                return (cg['lat'], cg['lng'])
        except (json.JSONDecodeError, OSError):
            pass

    summary_path = os.path.join(point_dir, 'collection_summary.json')
    if os.path.exists(summary_path):
        try:
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            lat = summary.get('initial_lat') or summary.get('actual_initial_lat')
            lng = summary.get('initial_lng') or summary.get('actual_initial_lng')
            if lat is not None and lng is not None:
                return (lat, lng)
        except (json.JSONDecodeError, OSError):
            pass
    return None


def process_point(point_dir, sam3_tool, bev_metadata, city=""):
    images_dir = os.path.join(point_dir, 'streetview_images')
    output_dir = os.path.join(point_dir, 'text_output_vggt_depth')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'depth_vis'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'calibration'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'sign_crops'), exist_ok=True)
    
    # 初始化大模型与 Places API
    llm_client = _init_llm_client()
    places_tool = _init_places_tool()

    # 解析文字搜索限定城市名（供 Places API 搜索限定城市）
    city = _read_city(point_dir, city)
    # 解析粗定位参考坐标（供招牌 GPS 搜索时选最近分店）
    coarse_gps = _read_coarse_gps(point_dir)
    if coarse_gps is not None:
        print(f"  招牌 GPS 搜索参考坐标: ({coarse_gps[0]:.5f}, {coarse_gps[1]:.5f})")
    
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('_streetview.jpg')])
    if not image_files:
        print("  ⚠️ 没有找到街景图像")
        return
    
    x_min = bev_metadata['x_range'][0]
    y_min = bev_metadata['y_range'][0]
    padding = bev_metadata['padding']
    resolution = bev_metadata['resolution']
    
    bev_path = bev_metadata.get('optimized_path', '')
    bev_image = None
    if bev_path and os.path.exists(bev_path):
        bev_image = cv2.imread(bev_path)
    if bev_image is None:
        for cand in ['fused_bev_optimized.png', 'fused_bev.png', 'fused_bev_base.png']:
            p = os.path.join(point_dir, 'projection_output', cand)
            if os.path.exists(p):
                bev_image = cv2.imread(p)
                if bev_image is not None:
                    break
    
    if bev_image is None:
        print("  ⚠️ 没有找到BEV图像")
        return
    
    bev_height = bev_image.shape[0]
    
    print(f"\n  调用VGGT获取 {len(image_files)} 张图像的深度图...")
    image_paths = [os.path.join(images_dir, f) for f in image_files]
    depth_result = call_vggt_infer(image_paths)
    if depth_result is None:
        print("  ❌ VGGT深度获取失败")
        return
    
    all_markers = []
    calibration_records = []
    collected_ks = []  # 已成功校准的尺度因子，供校准失败的图借用
    
    for img_idx, img_file in enumerate(image_files):
        base_name = img_file.replace('_streetview.jpg', '')
        metadata_file = f"{base_name}_metadata.json"
        metadata_path = os.path.join(images_dir, metadata_file)
        
        if not os.path.exists(metadata_path):
            continue
        
        print(f"  [{img_idx+1}/{len(image_files)}] 处理 {img_file}...")
        
        image_pil = Image.open(os.path.join(images_dir, img_file)).convert('RGB')
        image_np = np.array(image_pil)
        h, w = image_np.shape[:2]
        
        metadata = read_metadata(metadata_path)
        
        heading = metadata.get('heading', 0.0)
        pitch = metadata.get('pitch', 0.0)
        fov = metadata.get('fov', 60)
        relative_x = metadata.get('relative_x', 0.0)
        relative_y = metadata.get('relative_y', 0.0)
        cam_z = 2.5
        
        fx, fy, cx, cy = compute_intrinsics(w, h, fov)
        R, right, forward, down = get_rotation_matrix(heading, pitch)
        cam_pos = (relative_x, relative_y, cam_z)
        
        depth_rel = depth_result[img_idx]
        drh, drw = depth_rel.shape[:2]
        if (drh, drw) != (h, w):
            depth_vis = cv2.resize(depth_rel, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            depth_vis = depth_rel.copy()
        
        depth_vis_save = (depth_vis - depth_vis.min()) / (depth_vis.max() - depth_vis.min() + 1e-8)
        depth_vis_save = (depth_vis_save * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, 'depth_vis', f"{base_name}_depth_rel.png"), depth_vis_save)
        
        print(f"    SAM3分割路面...")
        road_mask = detect_road_mask(image_pil, sam3_tool, (h, w))
        cv2.imwrite(os.path.join(output_dir, 'calibration', f"{base_name}_road_mask.png"), road_mask)
        
        print(f"    路面校准尺度因子...")
        k, cal_stats = calibrate_depth_scale(road_mask, depth_rel, fx, fy, cx, cy, R, cam_z, h, w)
        
        if k is None:
            # 方案A：借用同批次其它图已校准的尺度因子
            if collected_ks:
                k = float(np.median(collected_ks))
                print(f"    ⚠️ 校准失败: {cal_stats.get('reason', 'unknown')}，借用邻居k={k:.4f}")
                cal_stats = {"borrowed_k": True, **cal_stats}
            else:
                print(f"    ⚠️ 校准失败: {cal_stats.get('reason', 'unknown')}，且无可借用的k，跳过")
                continue
        
        print(f"    校准结果: k={k:.4f}" + (f", 误差中位数={cal_stats['error_median']:.2f}m" if 'error_median' in cal_stats else " (借用)"))
        calibration_records.append({"image": img_file, "k": k, **cal_stats})
        collected_ks.append(k)
        
        depth_calibrated = depth_vis * k
        depth_cal_save = np.clip(depth_calibrated, 0, 100)
        depth_cal_save = (depth_cal_save / depth_cal_save.max() * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, 'calibration', f"{base_name}_depth_calibrated.png"), depth_cal_save)
        
        print(f"    SAM3分割招牌...")
        sign_objects = detect_sign_objects(image_pil, sam3_tool, (h, w))
        sign_objects = deduplicate_masks(sign_objects)
        
        if not sign_objects:
            print(f"    未检测到招牌")
            continue
        
        print(f"    检测到 {len(sign_objects)} 个招牌")
        
        for sign in sign_objects:
            marker = project_sign_with_depth(
                sign, depth_rel, k, fx, fy, cx, cy, R, cam_pos, h, w
            )
            if marker is not None:
                marker['source_image'] = img_file
                
                # 裁剪招牌区域，用大模型识别文字，再搜索 GPS
                crop = crop_sign_region(image_np, sign['bbox'])
                crop_save_path = os.path.join(output_dir, 'sign_crops', f"{base_name}_crop_{len(all_markers)}.png")
                if crop.size > 0:
                    cv2.imwrite(crop_save_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                
                text, search_query, confidence = recognize_sign_text(crop, llm_client)
                if not text or confidence < TEXT_CONFIDENCE_THRESHOLD:
                    continue

                marker['text'] = text
                marker['search_query'] = search_query
                marker['text_confidence'] = confidence

                # 用识别文字搜索 Places API，得到招牌 GPS（供 Stage 2 高斯核配准使用）
                gps = search_place_gps(search_query or text, places_tool, city, ref_coord=coarse_gps)
                if gps and gps.get('lat') is not None and gps.get('lng') is not None:
                    marker['gps'] = gps
                    print(f"      招牌GPS: ({gps['lat']:.5f}, {gps['lng']:.5f})")
                else:
                    print(f"      ⚠️ 未找到招牌 GPS")

                print(f"      招牌文字: '{text}'")

                all_markers.append(marker)
        
        img_markers_dir = os.path.join(output_dir, "sign_overlays")
        os.makedirs(img_markers_dir, exist_ok=True)
        overlay = image_np.copy()
        for sign in sign_objects:
            mask = sign['mask']
            color_idx = TEXT_PROMPTS.index(sign['prompt']) if sign['prompt'] in TEXT_PROMPTS else 0
            color = PROMPT_COLORS[color_idx % len(PROMPT_COLORS)]
            for c in range(3):
                overlay[:, :, c] = np.where(mask > 0, color[c], overlay[:, :, c])
            cx_s, cy_s = int(sign['centroid']['x']), int(sign['centroid']['y'])
            cv2.drawMarker(overlay, (cx_s, cy_s), color, cv2.MARKER_CROSS, 20, 2)
        overlay = cv2.addWeighted(overlay, 0.6, image_np, 0.4, 0)
        cv2.imwrite(os.path.join(img_markers_dir, f"{base_name}_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    
    if not all_markers:
        print("  ⚠️ 没有成功检测到任何招牌标记")
        return
    
    print(f"\n  跨图像聚类 ({len(all_markers)} → ...)...")
    final_markers = cluster_markers(all_markers, distance_threshold=5.0)
    
    multi = sum(1 for m in final_markers if m.get('num_observations', 1) > 1)
    print(f"  聚类后: {len(final_markers)} 个 (多观测={multi}, 单观测={len(final_markers)-multi})")

    # 离群 GPS 过滤：把远离主簇（采集点所在城市）的错误搜索坐标置空
    final_markers, gps_estimate = filter_outlier_gps(final_markers)
    n_filtered = sum(1 for m in final_markers if m.get('gps_filtered'))
    if n_filtered:
        print(f"  离群 GPS 过滤: 置空 {n_filtered} 个")

    # 空间投票定位：得到 top-n 候选位置
    top_positions = spatial_vote_localize(final_markers, places_tool, city)
    final_estimate = None
    if top_positions:
        final_estimate = (top_positions[0]['lat'], top_positions[0]['lng'])
        print(f"  空间投票定位: 得到 {len(top_positions)} 个候选位置")
        for t in top_positions:
            texts = t.get('supporting_texts') or [s['text'] for s in t.get('supporting_signs', []) if s.get('text')]
            print(f"    #{t['rank']} ({t['lat']:.6f}, {t['lng']:.6f}) 支持招牌={t['support_count']} → {', '.join(texts)}")
    else:
        print("  ⚠️ 空间投票定位: 无候选位置")
    
    with open(os.path.join(output_dir, 'calibration_records.json'), 'w') as f:
        json.dump(calibration_records, f, indent=2)
    
    markers_json_path = os.path.join(output_dir, 'sign_markers.json')
    with open(markers_json_path, 'w', encoding='utf-8') as f:
        payload = {
            'method': 'vggt_depth_road_calibration',
            'total_markers': len(final_markers),
            'markers': final_markers,
        }
        if final_estimate is not None:
            payload['final_estimate'] = {
                'lat': final_estimate[0],
                'lng': final_estimate[1],
            }
        if top_positions:
            payload['top_positions'] = top_positions
        json.dump(payload, f, indent=2, ensure_ascii=False)
    
    bev_marked, drawn = draw_markers_on_bev(bev_image, final_markers, x_min, y_min, padding, resolution)
    bev_markers_path = os.path.join(output_dir, 'bev_with_sign_markers.png')
    cv2.imwrite(bev_markers_path, bev_marked)
    
    print(f"\n  ✅ BEV标记图已保存: {bev_markers_path}")
    print(f"     共 {len(final_markers)} 个标记, {drawn} 个在BEV范围内")
    
    for m in final_markers:
        wp = m['world_position']
        n_obs = m.get('num_observations', 1)
        text = m.get('text', '')
        gps = m.get('gps')
        name = text if text else m['prompt']
        gps_str = ""
        if gps and gps.get('lat') is not None and gps.get('lng') is not None:
            gps_str = f" → GPS({gps['lat']:.5f}, {gps['lng']:.5f})"
        src_images = m.get('source_images', [])
        src_str = f"  [出现在: {', '.join(src_images)}]" if src_images else ""
        print(f"    - {name}: ({wp['x']:.2f}, {wp['y']:.2f})m, z={wp['z']:.2f}m, n_obs={n_obs}{gps_str}{src_str}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='VGGT深度+路面校准 → 招牌BEV定位')
    parser.add_argument('--input', type=str, required=True, help='街景点位目录')
    parser.add_argument('--vggt-server', type=str, default=VGGT_SERVER, help='VGGT服务地址')
    args = parser.parse_args()
    
    point_dir = args.input
    
    bev_metadata_path = os.path.join(point_dir, 'projection_output', 'bev_metadata.json')
    if not os.path.exists(bev_metadata_path):
        print(f"❌ 未找到BEV元数据: {bev_metadata_path}")
        return
    
    with open(bev_metadata_path, 'r') as f:
        bev_metadata = json.load(f)
    
    print(f"BEV元数据: X={bev_metadata['x_range']}, Y={bev_metadata['y_range']}, res={bev_metadata['resolution']}px/m")
    
    try:
        r = requests.get(f"{args.vggt_server}/health", timeout=5)
        health = r.json()
        print(f"VGGT服务: {health}")
    except Exception as e:
        print(f"❌ VGGT服务不可用: {e}")
        print(f"   请确认已启动服务并建立SSH隧道:")
        print(f"   ssh -CNg -L 18888:127.0.0.1:18888 -p 37859 root@connect.westb.seetacloud.com")
        return
    
    sam3_tool = SAM3Tool()
    sam3_tool.initialize()
    
    try:
        process_point(point_dir, sam3_tool, bev_metadata)
    finally:
        sam3_tool.close()


if __name__ == '__main__':
    main()
