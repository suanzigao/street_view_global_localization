"""
Gym环境封装模块 - 增强版

集成SAM3图像分割工具，让大模型可以自主决定识别哪些视觉特征来辅助定位
"""

import os
import math
import traceback
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from PIL import Image
from typing import Tuple, Optional, Dict, Any, List
from openai import OpenAI
import json
import re
from io import BytesIO
import base64

from utils.file_utils import save_streetview_image, image_to_base64
from utils.decorators import retry_on_failure
from prompts import NAVIGATION_PROMPT_TEMPLATE, FINAL_PREDICTION_PROMPT

try:
    from tools.sam3_tool import SAM3Tool
    SAM3_AVAILABLE = True
except ImportError:
    SAM3_AVAILABLE = False

try:
    from tools.places_tool import PlacesAPITool
    PLACES_API_AVAILABLE = True
except ImportError:
    PLACES_API_AVAILABLE = False


def _compute_intrinsics(width, height, fov_deg):
    fov = math.radians(fov_deg)
    fx = width / (2 * math.tan(fov / 2))
    fy = height / (2 * math.tan(fov / 2))
    cx = width / 2
    cy = height / 2
    return fx, fy, cx, cy


def _pixel_to_world(x, y, fx, fy, cx, cy, heading_deg, pitch_deg, cam_x, cam_y, cam_z):
    """将路面像素反投影到世界坐标（Z=0 平面）。"""
    yaw = -math.radians(heading_deg)
    pitch = math.radians(pitch_deg)

    u = (x - cx) / fx
    v = (y - cy) / fy

    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    cos_pitch = math.cos(pitch)
    sin_pitch = math.sin(pitch)

    ray_world_x = u * cos_yaw + v * (-sin_yaw * sin_pitch) + (-sin_yaw * cos_pitch)
    ray_world_y = u * sin_yaw + v * (cos_yaw * sin_pitch) + (cos_yaw * cos_pitch)
    ray_world_z = v * (-cos_pitch) + sin_pitch

    if ray_world_z >= -1e-6:
        return None

    t = -cam_z / ray_world_z
    if t < 0:
        return None

    world_x = cam_x + t * ray_world_x
    world_y = cam_y + t * ray_world_y
    return (world_x, world_y, 0.0)


class StreetViewAutoLocEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    ACTION_MAP = {
        "move forward": ("move", "forward"),
        "move left": ("move", "left"),
        "move right": ("move", "right"),
        "move backward": ("move", "backward"),
        "turn left": ("turn", "left"),
        "turn right": ("turn", "right"),
        "zoom in": ("fov", "zoom_in"),
        "zoom out": ("fov", "zoom_out"),
        "look up": ("pitch", "up"),
        "look down": ("pitch", "down")
    }
    
    # 分割操作 - 独立的基础操作类型
    # 格式: "segment: object1, object2"
    SEGMENT_ACTION_PREFIX = "segment:"
    
    # 地点搜索操作 - 通过 Google Places API 查询看到的建筑/店铺名称
    # 格式: "search: query text"
    SEARCH_ACTION_PREFIX = "search:"
    
    def __init__(
            self, 
            initial_lat: float, 
            initial_lng: float,
            move_step: float = 15.0, 
            max_retries: int = 3, 
            max_steps: int = 10,
            point_dir: str = "./", 
            img_save_dir: str = "./images",
            turn_angle: int = 45,
            fov_step: int = 20,
            pitch_step: int = 10,
            fov_default: int = 60,
            use_sam3: bool = True,
            use_bev: bool = False
    ):
        super().__init__()
        
        from core.street_view import GoogleStreetViewData
        
        self.data_handler = GoogleStreetViewData(
            max_retries=max_retries,
            move_step=move_step,
            radius=10
        )
        self.data_handler.init_position(initial_lat, initial_lng, fov=fov_default)
        
        self.initial_lat = initial_lat
        self.initial_lng = initial_lng
        
        self.move_step = move_step
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.point_dir = point_dir
        self.img_save_dir = img_save_dir
        self.turn_angle = turn_angle
        self.fov_step = fov_step
        self.pitch_step = pitch_step
        self.fov_default = fov_default
        
        os.makedirs(img_save_dir, exist_ok=True)
        
        self.openai_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url="https://api.laozhang.ai/v1",
        )
        
        self.action_space = spaces.Discrete(len(self.ACTION_MAP))
        self.observation_space = spaces.Dict({
            "image": spaces.Box(0, 255, (640, 640, 3), int),
            "lat": spaces.Box(-90, 90, (1,), float),
            "lng": spaces.Box(-180, 180, (1,), float),
            "heading": spaces.Box(0, 360, (1,), float),
            "pitch": spaces.Box(-90, 90, (1,), float)
        })
        
        self.current_step = 0
        self.current_image = None
        self.final_prediction = ""
        self.action_history = []
        self.reason_history = []
        self.image_path_history = {}
        self.image_b64_history = {}
        self.image_meta_history = {}
        self.segmented_image_history = {}
        self.coord_correction_history = []
        self.trajectory_history = []
        
        self.use_sam3 = use_sam3 and SAM3_AVAILABLE
        self.sam3_tool = None
        self.segmentation_history = []
        
        self.use_bev = use_bev
        self.bev_pixels_per_meter = 3.0  # BEV 固定分辨率：3 像素 = 1 米（导航阶段道路结构理解用）
        self._road_mask_cache = {}
        self._bev_cache_b64 = None
        self._bev_cache_key = None
        
        self.use_places_api = PLACES_API_AVAILABLE
        self.places_tool = None
        self.found_places = []  # 探索过程中通过 search 工具找到的地点列表
        
        self.last_forced_action = None  # 记录上次是否执行过强制动作
        self.force_action_step = 0  # 记录强制动作发生的步骤，用于重置死循环检测窗口
        
        self.current_theoretical_lat = None
        self.current_theoretical_lng = None
        self.current_actual_lat = None
        self.current_actual_lng = None
        self.current_correction_diff = None
        
        self.relative_x = 0.0
        self.relative_y = 0.0
        self.relative_history = []
        self.last_move_feedback = ""  # 上一次 move 动作的结果反馈（撞墙时告知 GPT）
        
        self.actual_initial_lat = None
        self.actual_initial_lng = None
        self.location_hypothesis = ""  # 大模型持续维护的位置假设（"证据 → 地区/城市"的文本摘要）
        self.evidence_bank = []  # 跨步累积的定位证据（search 地点坐标 + segment 分割证据图像）
        self.hypothesis_evidence_map = []  # 结构化累积：每条 {"step", "evidence", "region_city"}
        self.predicted_city = ""  # 大模型跨步收敛的"预测城市名"（供下游 Stage 2 限定搜索）
        self.plan = ""  # 大模型当前的多步计划（跨步持久化，回传给下一轮决策，解决"意图遗忘"）

        # 离散探索图：节点=街景采样点（离散），边=从某节点朝某方向 move 的结果
        self._graph_nodes = {}      # node_key -> {"id", "lat", "lng", "step"}
        self._graph_node_counter = 0
        self._graph_edges = {}      # (from_node_key, quantized_dir) -> {"to": node_key|None, "step"}
        self._current_node_key = None
        self._prev_node_key = None  # 上一次 move 前的节点（用于标注"来路"）
        self._last_move_geo_ang = None

        try:
            _, actual_lat, actual_lng = self.data_handler.get_street_view_image()
            if actual_lat and actual_lng:
                self.actual_initial_lat = actual_lat
                self.actual_initial_lng = actual_lng
        except Exception as e:
            print(f"初始化时获取实际初始采样点失败: {e}")
    
    def _init_sam3(self):
        if self.use_sam3 and self.sam3_tool is None:
            self.sam3_tool = SAM3Tool()
            self.sam3_tool.initialize()
    
    def _call_sam3(self, image: Image.Image, prompts: List[str], save_path: str = None, output_format: str = 'PNG'):
        """
        调用SAM3工具进行图像分割

        Args:
            image: PIL Image对象
            prompts: 分割提示词列表
            save_path: 保存路径（可选）
            output_format: 输出格式，'PNG' 或 'JPG'

        Returns:
            dict: 包含分割结果的字典
                - 'success': bool
                - 'combined_image_b64': str (所有掩码叠加后的图像Base64)
                - 'combined_image_path': str (保存路径)
                - 'results': list (每个类别的分割结果)
                - 'prompt_count': int (成功分割的类别数)
                - 'combined_image': PIL.Image (叠加后的图像对象，用于传递给GPT)
        """
        if not self.use_sam3:
            return {'success': False, 'error': 'SAM3 not enabled', 'combined_image_b64': None}

        self._init_sam3()
        result = self.sam3_tool.get_segmented_images(
            image,
            prompts,
            save_path=save_path,
            output_format=output_format
        )

        if result.get('success') and result.get('combined_image_b64'):
            result['combined_image'] = Image.open(BytesIO(base64.b64decode(result['combined_image_b64'])))

        return result
    
    def _call_places_search(self, query: str) -> dict:
        """
        调用 Google Places API 搜索地点
        
        Args:
            query: 搜索文本
            
        Returns:
            dict: {'success': bool, 'candidates': [...], 'error': str}
        """
        if not self.use_places_api:
            return {"success": False, "error": "Places API not available", "candidates": []}
        
        if self.places_tool is None:
            self.places_tool = PlacesAPITool()
        
        return self.places_tool.find_place(query)
    
    def _build_history_summary(self) -> str:
        """生成每一步的文字摘要（不含图像），作为轻量级记忆，替代回传全部历史图。

        每步附带该步的视觉观察（observation），让模型记住"之前每一步看到了什么"。
        """
        lines = []
        for s in sorted(self.image_meta_history.keys()):
            meta = self.image_meta_history[s]
            line = (
                f"Step {meta['step']}: action={meta['action']}, "
                f"heading={meta['heading']:.0f}°, "
                f"rel=({meta['relative_x']:.0f}m, {meta['relative_y']:.0f}m)"
            )
            obs = (meta.get("observation") or "").strip()
            if obs:
                line += f" | saw: {obs}"
            lines.append(line)
        return "\n".join(lines)
    
    def add_evidence(self, kind: str, content: str, **extra):
        """往证据库追加一条定位证据（跨步累积，最终预测时用于验证假设）。

        kind: segment / search
        extra: 结构化附加字段，如 segment 的 image_b64；search 的 name/lat/lng。
        """
        entry = {
            "step": self.current_step,
            "kind": kind,
            "content": content,
        }
        entry.update(extra)
        self.evidence_bank.append(entry)

    def _build_evidence_summary(self) -> str:
        """把证据库整理成文字清单，供最终预测做验证。"""
        if not self.evidence_bank:
            return "(无累积证据)"
        lines = []
        for e in self.evidence_bank:
            lines.append(f"- [Step {e['step']}] {e['kind']}: {e['content']}")
        return "\n".join(lines)

    def _build_hypothesis_summary(self) -> str:
        """把 hypothesis_evidence_map 整理成"证据 → 地区/城市"清单，供回传/最终预测。"""
        if not self.hypothesis_evidence_map:
            return "None yet — form your first hypothesis now."
        lines = []
        for h in self.hypothesis_evidence_map:
            region = h.get("region_city") or "(unknown region)"
            lines.append(f"- [Step {h['step']}] {h['evidence']} → {region}")
        return "\n".join(lines)

    def _count_consecutive_failed_moves(self):
        """统计最近连续 move 失败的次数，以及最近一次失败时的朝向。

        只从 relative_history 末尾往前数连续的 moved=False 条目（turn 不会进入
        relative_history，因此这里的"连续"跨过了中间的 turn，含义是"连续多次尝试
        move 都没有移动位置"，即卡在死点）。返回 (连续失败次数, 最近失败朝向)。
        """
        fails = []
        for h in reversed(self.relative_history):
            if h.get("moved", True):
                break
            fails.append(h)
        last_heading = fails[0]["heading"] if fails else None
        return len(fails), last_heading

    def _quantize_heading(self, h: float) -> int:
        """把任意 heading 量化到最近的 8 个绝对方位之一（0/45/.../315）。"""
        return int((h % 360 + 22.5) // 45) % 8 * 45

    def _bearing_deg(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """从 (lat1,lng1) 指向 (lat2,lng2) 的初始方位角（度，0=北，顺时针）。"""
        import math
        lat1, lng1 = math.radians(lat1), math.radians(lng1)
        lat2, lng2 = math.radians(lat2), math.radians(lng2)
        dlng = lng2 - lng1
        y = math.sin(dlng) * math.cos(lat2)
        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlng)
        return (math.degrees(math.atan2(y, x)) + 360) % 360

    def _register_node(self, lat: float, lng: float, step: int):
        """登记（或复用）一个离散采样点节点，返回 node_key。

        复用已有节点时累加访问次数，用于全局去重（识别"绕圈回到已访问节点"）。
        """
        key = (round(lat, 6), round(lng, 6))
        if key not in self._graph_nodes:
            self._graph_node_counter += 1
            self._graph_nodes[key] = {
                "id": self._graph_node_counter,
                "lat": lat, "lng": lng, "step": step,
                "first_step": step,
                "visits": 1,
            }
        else:
            self._graph_nodes[key]["visits"] += 1
        return key

    def _record_edge(self, from_key, geo_ang: float, to_key):
        """记录"从 from_key 朝 geo_ang 方向 move 到 to_key"这条边（to_key 为 None 表示死路）。"""
        if from_key is None or geo_ang is None:
            return
        dir_q = self._quantize_heading(geo_ang)
        self._graph_edges[(from_key, dir_q)] = {"to": to_key, "step": self.current_step}

    def _build_exploration_summary(self) -> str:
        """把离散探索图转成模型可直接读懂的语义清单（egocentric：前/左/右/后）。

        对每个"未尝试"方向额外标注"需 turn 几次 left/right 才能面对它"，
        并在末尾给出一个推荐动作，避免模型自己数 turn 次数。
        """
        if self._current_node_key is None:
            return ""

        cur = self._current_node_key
        node_id = self._graph_nodes[cur]["id"]
        heading = self.data_handler.heading % 360
        cur_q = self._quantize_heading(heading)

        rel_names = {
            0: "ahead", 45: "ahead-right", 90: "right", 135: "behind-right",
            180: "behind", 225: "behind-left", 270: "left", 315: "ahead-left",
        }
        abs_names = {0: "N", 45: "NE", 90: "E", 135: "SE",
                     180: "S", 225: "SW", 270: "W", 315: "NW"}

        def _turn_hint(offset: int) -> str:
            """offset 是相对当前朝向的顺时针角度差，返回"turn left/right N times"提示。"""
            if offset == 0:
                return "no turn"
            if offset <= 180:
                return f"turn right {offset // 45} time(s)"
            return f"turn left {(360 - offset) // 45} time(s)"

        untried = []  # (offset, rel_name, abs_name)
        lines = []
        lines.append("EXPLORATION GRAPH (discrete nodes you have visited):")
        lines.append(f"You are at node #{node_id} (heading {abs_names[cur_q]}/{cur_q:.0f}°). "
                     f"{len(self._graph_nodes)} node(s) visited, {len(self._graph_edges)} direction(s) tried.")

        # 全局已访问节点集合（去重：让你知道哪些节点你已经到过，避免绕圈回去）
        visited_desc = []
        for key, node in sorted(self._graph_nodes.items(), key=lambda kv: kv[1]["id"]):
            tag = "START" if node["id"] == 1 else f"step {node['first_step']}"
            extra = f", visited {node['visits']}x" if node["visits"] > 1 else ""
            marker = " <-- YOU ARE HERE" if key == cur else ""
            visited_desc.append(f"#{node['id']}({tag}{extra}){marker}")
        lines.append("ALL VISITED NODES (do NOT go back to a node you already visited unless no other way): "
                     + ", ".join(visited_desc))

        # 若当前节点是重复访问，明确警告"绕圈"
        cur_node = self._graph_nodes[cur]
        if cur_node.get("visits", 1) > 1:
            lines.append(f"⚠️ You are at node #{node_id} for the {cur_node['visits']}th time "
                         f"(first visited at step {cur_node['first_step']}). You have LOOPED back to an explored node — "
                         f"next, prefer a direction that leads AWAY from nodes you have already visited.")

        # 来路方向 = 从当前节点指向上一节点的绝对方位（量化到 8 方向）。
        # 该方向必然通向已访问的上一节点，即使图里还没有这条边，也必须标记为"已探索"，避免走回头路。
        incoming_abs = None
        if self._prev_node_key is not None and self._prev_node_key in self._graph_nodes:
            p = self._graph_nodes[self._prev_node_key]
            incoming_abs = self._quantize_heading(
                self._bearing_deg(cur_node["lat"], cur_node["lng"], p["lat"], p["lng"])
            )

        lines.append("From your CURRENT node, by direction (relative to where you face):")

        for offset in range(0, 360, 45):
            rel_name = rel_names[offset]
            abs_dir = (cur_q + offset) % 360
            abs_name = abs_names[abs_dir]
            edge = self._graph_edges.get((cur, abs_dir))
            if edge is None:
                if incoming_abs is not None and abs_dir == incoming_abs:
                    status = "the road you CAME FROM (already explored — do NOT backtrack)"
                else:
                    status = f"NOT TRIED yet (to face it: {_turn_hint(offset)})"
                    untried.append((offset, rel_name, abs_name))
            elif edge["to"] is None:
                status = "BLOCKED (move failed, no street view)"
            else:
                to_id = self._graph_nodes[edge["to"]]["id"]
                if edge["to"] == self._prev_node_key:
                    status = f"node #{to_id} — the road you CAME FROM (already explored, do NOT backtrack)"
                else:
                    status = f"node #{to_id} (already visited)"
            lines.append(f"  {rel_name:12s} ({abs_name:>2s}) : {status}")

        # 推荐动作：优先选一个非 behind 的未探索方向
        if untried:
            prefer = [x for x in untried if x[1] not in ("behind", "behind-left", "behind-right")]
            pick = prefer[0] if prefer else untried[0]
            lines.append(f">> RECOMMENDED NEXT MOVE: face '{pick[1]}' ({pick[2]}), which is {_turn_hint(pick[0])}, "
                         f"then 'move forward'.")
        else:
            lines.append(">> No unexplored direction from here: turn around and move forward back the way you came.")

        return "\n".join(lines)

    def _generate_bev_image(self, max_size: int = 4096) -> Optional[str]:
        """
        基于已走过帧的道路分割结果，增量生成 BEV（鸟瞰图）并返回 base64。

        采用道路 mask 缓存 + 向量化投影，只在历史帧集合变化时重新生成，
        避免每步决策都全量重算。返回 None 表示当前尚不足以生成 BEV。
        """
        if not self.use_bev:
            return None
        
        steps = sorted(self.image_b64_history.keys())
        if len(steps) < 1:
            return None
        
        # 缓存 key 加入 current_step：转向(turn)会改变步数但可能不新增帧，
        # 加入 current_step 确保转向后也重新生成并落盘一张新 BEV。
        cache_key = (tuple(steps), self.current_step)
        if self._bev_cache_key == cache_key and self._bev_cache_b64:
            return self._bev_cache_b64
        
        self._init_sam3()
        if self.sam3_tool is None:
            return None
        
        road_color = (110, 110, 110)  # 道路统一颜色（不再按 step 染色）
        
        all_wx, all_wy = [], []
        camera_pts = []
        
        for step in steps:
            img_path = self.image_path_history.get(step)
            meta = self.image_meta_history.get(step)
            if not img_path or not meta or not os.path.exists(img_path):
                continue
            
            # 道路 mask（带缓存，避免对同一帧重复 SAM3 分割）
            if step not in self._road_mask_cache:
                image_pil = Image.open(img_path).convert('RGB')
                result = self.sam3_tool.get_segmented_images(image_pil, ["road", "street"])
                ground_mask = None
                for seg in result.get('results', []):
                    if seg.get('success') and seg.get('masks'):
                        for m in seg.get('masks'):
                            ground_mask = m if ground_mask is None else np.logical_or(ground_mask, m)
                self._road_mask_cache[step] = ground_mask
            
            ground_mask = self._road_mask_cache.get(step)
            if ground_mask is None:
                continue
            mask_bin = (ground_mask > 0) if ground_mask.max() > 1 else ground_mask.astype(bool)
            if not mask_bin.any():
                continue
            
            h, w = mask_bin.shape
            fx, fy, cx, cy = _compute_intrinsics(w, h, meta.get('fov', self.fov_default))
            
            heading = meta.get('heading', 0.0)
            pitch = meta.get('pitch', 0.0)
            rel_x = meta.get('relative_x', 0.0)
            rel_y = meta.get('relative_y', 0.0)
            cam_z = 2.5
            
            ys, xs = np.where(mask_bin)
            # 下采样控制点数
            stride = max(1, len(xs) // 4000)
            xs = xs[::stride]
            ys = ys[::stride]
            
            yaw = -math.radians(heading)
            pitch_r = math.radians(pitch)
            cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
            cos_pitch, sin_pitch = math.cos(pitch_r), math.sin(pitch_r)
            
            u = (xs - cx) / fx
            v = (ys - cy) / fy

            # 方案 B：收窄横向范围——只保留视野中心窄带，丢弃两侧人行道/绿化/建筑，
            # 避免把整块地面投影成巨大的扇形虚影。
            center_band = 0.4
            center_mask = np.abs(u) < center_band
            u = u[center_mask]
            v = v[center_mask]
            xs = xs[center_mask]
            ys = ys[center_mask]

            ray_x = u * cos_yaw + v * (-sin_yaw * sin_pitch) + (-sin_yaw * cos_pitch)
            ray_y = u * sin_yaw + v * (cos_yaw * sin_pitch) + (cos_yaw * cos_pitch)
            ray_z = v * (-cos_pitch) + sin_pitch
            
            valid = ray_z < -1e-6
            ray_x, ray_y, ray_z = ray_x[valid], ray_y[valid], ray_z[valid]
            xs, ys = xs[valid], ys[valid]
            
            t = -cam_z / ray_z
            valid = t > 0
            t = t[valid]
            ray_x, ray_y = ray_x[valid], ray_y[valid]
            xs, ys = xs[valid], ys[valid]
            
            world_x = rel_x + t * ray_x
            world_y = rel_y + t * ray_y
            
            dist = np.sqrt((world_x - rel_x) ** 2 + (world_y - rel_y) ** 2)
            valid = dist < 50.0
            world_x, world_y = world_x[valid], world_y[valid]
            
            if len(world_x) == 0:
                continue
            
            all_wx.append(world_x)
            all_wy.append(world_y)
            camera_pts.append((rel_x, rel_y, heading))
        
        if not all_wx:
            return None
        
        world_x = np.concatenate(all_wx)
        world_y = np.concatenate(all_wy)
        
        # 栅格化（以初始点为图像中心，固定分辨率，每 move_step 画网格）
        scale = self.bev_pixels_per_meter  # 像素/米
        grid_m = self.move_step            # 网格间距 = 单步移动距离
        padding = 5.0                      # 边缘留白（米）

        # 覆盖范围：所有道路点 + 所有相机位置（相对初始点的最大绝对值）
        max_extent = 0.0
        if len(world_x):
            max_extent = max(float(np.abs(world_x).max()), float(np.abs(world_y).max()))
        for rx, ry, _ in camera_pts:
            max_extent = max(max_extent, abs(rx), abs(ry))
        half_m = max(max_extent + padding, grid_m)  # 至少覆盖一个网格

        full_w = int(2 * half_m * scale)
        downscale = min(1.0, max_size / full_w) if full_w > 0 else 1.0
        bev_w = max(1, int(full_w * downscale))
        bev_h = bev_w
        eff_scale = scale * downscale  # 有效像素/米（下采样后）

        cx = bev_w // 2
        cy = bev_h // 2

        img = np.full((bev_h, bev_w, 3), 255, dtype=np.uint8)

        # 1) 网格（最底层，深灰）
        grid_px = grid_m * eff_scale
        if grid_px >= 2:
            n = int(bev_w / grid_px) + 2
            for k in range(-n, n + 1):
                x = int(round(cx + k * grid_px))
                if 0 <= x < bev_w:
                    img[:, x] = (80, 80, 80)
                y = int(round(cy + k * grid_px))
                if 0 <= y < bev_h:
                    img[y, :] = (80, 80, 80)

        # 2) 道路点：先低分辨率栅格化 → 形态学去噪/填洞 → 上采样（与三阶段 optimize_bev 类似）
        #    避免直接画散点导致的孤立噪点与道路断裂，让道路连续、干净。
        import cv2

        up_factor = 2  # 先以 1/up_factor 分辨率栅格化，再上采样，道路更连续
        low_w = max(1, bev_w // up_factor)
        low_h = max(1, bev_h // up_factor)
        low_scale = eff_scale / up_factor
        low_cx = cx / up_factor
        low_cy = cy / up_factor

        lpx = (low_cx + world_x * low_scale).astype(int)
        lpy = (low_cy - world_y * low_scale).astype(int)
        valid = (lpx >= 0) & (lpx < low_w) & (lpy >= 0) & (lpy < low_h)

        road_mask = np.zeros((low_h, low_w), dtype=np.uint8)
        road_mask[lpy[valid], lpx[valid]] = 255

        def _meter_kernel(radius_m: float):
            """按物理米数换算形态学核半径（像素），避免不同分辨率下核物理尺寸漂移。"""
            r = max(1, int(round(radius_m * low_scale)))
            return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))

        # 稀疏点云必须先"膨胀连接"再"腐蚀去噪"：若先腐蚀，1 像素宽的细线会被整条擦除。
        dilated = cv2.dilate(road_mask, _meter_kernel(1.0), iterations=2)
        eroded = cv2.erode(dilated, _meter_kernel(0.6), iterations=1)

        # 上采样回目标尺寸（线性插值，让道路平滑连续）
        road_clean = cv2.resize(eroded, (bev_w, bev_h), interpolation=cv2.INTER_LINEAR)
        road_clean = road_clean > 0

        img[road_clean] = road_color

        # 3) 相机位置（黑点），初始点用红色标记，并记录像素坐标用于画朝向箭头
        camera_px_pts = []
        for rel_x, rel_y, heading in camera_pts:
            cpx = int(round(cx + rel_x * eff_scale))
            cpy = int(round(cy - rel_y * eff_scale))
            if 0 <= cpx < bev_w and 0 <= cpy < bev_h:
                is_initial = (abs(rel_x) < 1e-6 and abs(rel_y) < 1e-6)
                color = (255, 0, 0) if is_initial else (0, 0, 0)
                img[max(0, cpy - 2):cpy + 3, max(0, cpx - 2):cpx + 3] = color
                camera_px_pts.append((cpx, cpy, heading, is_initial))

        # 4) 尺度标注（左下角）+ 每个相机位置的朝向箭头
        bev_pil = Image.fromarray(img)
        try:
            from PIL import ImageDraw
            draw = ImageDraw.Draw(bev_pil)
            draw.text((5, bev_h - 16), f"grid = {grid_m:.0f}m / cell", fill=(0, 0, 0))
            # 朝向箭头：箭头长 ≈ 0.8 个网格，便于估读方向
            arrow_len = max(6, int(grid_m * eff_scale * 0.8))
            n_pts = len(camera_px_pts)
            for i, (cpx, cpy, heading, is_initial) in enumerate(camera_px_pts):
                rad = math.radians(heading)
                dx = math.sin(rad)
                dy = -math.cos(rad)  # 图像 y 向下；heading 0°=北=图像向上
                ex = int(round(cpx + dx * arrow_len))
                ey = int(round(cpy + dy * arrow_len))
                is_current = (i == n_pts - 1)
                color = (255, 0, 0) if is_current else (0, 0, 0)
                width = 3 if is_current else 2
                draw.line([(cpx, cpy), (ex, ey)], fill=color, width=width)
                # 箭头头（V 形）
                head_len = max(3, int(arrow_len * 0.35))
                base_x = ex - dx * head_len
                base_y = ey - dy * head_len
                perp_x, perp_y = -dy, dx
                half_w = max(2, head_len // 2)
                draw.line([(ex, ey), (int(base_x + perp_x * half_w), int(base_y + perp_y * half_w))], fill=color, width=width)
                draw.line([(ex, ey), (int(base_x - perp_x * half_w), int(base_y - perp_y * half_w))], fill=color, width=width)
        except Exception:
            pass

        # 5) 每新增一帧就落盘一张 BEV（以当前步命名，回溯 BEV 生长过程）
        try:
            bev_dir = os.path.join(self.img_save_dir, "bev")
            os.makedirs(bev_dir, exist_ok=True)
            bev_pil.save(os.path.join(bev_dir, f"bev_step{self.current_step}.png"), format='PNG')
        except Exception:
            pass

        buf = BytesIO()
        bev_pil.save(buf, format='PNG')
        self._bev_cache_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        self._bev_cache_key = cache_key

        return self._bev_cache_b64
    
    @retry_on_failure(max_retries=3, delay=2.0)
    def gpt4o_decision(self, image: Image.Image, step_num: int) -> str:
        b64 = image_to_base64(image)
        action_history_str = "\n- ".join(self.action_history) if self.action_history else "No navigation actions yet"
        
        # 已找到的地点信息
        if self.found_places:
            places_str = "📍 Locations found: " + "; ".join(
                f"{p['name']} @ ({p['lat']:.4f}, {p['lng']:.4f})" for p in self.found_places
            )
        else:
            places_str = "No locations searched yet"
        
        # 先格式化导航提示词（处理 step_num、max_steps 和 move_step）
        nav_prompt = NAVIGATION_PROMPT_TEMPLATE.format(
            step_num=self.current_step, 
            max_steps=self.max_steps,
            move_step=self.move_step
        )
        
        relative_history_str = "\n- ".join(
            [f"Step {h['step']}: ({h['x']:.1f}m, {h['y']:.1f}m), heading={h['heading']:.1f}°, action={h['action']}"
             + ("" if h.get('moved', True) else " [FAILED - position unchanged]")
             for h in self.relative_history[-10:]]
        ) if self.relative_history else "No movement yet"
        
        # 当前已维护的位置假设（让模型在上一轮基础上持续修正）
        hypothesis_str = self._build_hypothesis_summary()

        # 连续 move 失败计数（代码替模型"数数"：它自己数不清连续撞墙几次）
        consecutive_fails, last_failed_heading = self._count_consecutive_failed_moves()
        stuck_warning = ""
        if consecutive_fails >= 2:
            stuck_warning = (
                f"- ⚠️ STUCK WARNING (computed by the system): your last {consecutive_fails} move attempts "
                f"ALL FAILED and your position did not change (most recently facing heading={last_failed_heading:.0f}°). "
                f"You are hitting a dead end. Do NOT retry that direction — TURN to a clearly DIFFERENT heading, then move forward."
            )

        # 离散探索图总结（代码替模型维护：哪些节点/方向已探索、哪些未尝试）
        exploration_summary = self._build_exploration_summary()

        prompt = f"""
        Your Role: Autonomous Google Street View Navigator & Visual Location Predictor
        CORE MISSION: Predict the GPS coordinates of your INITIAL starting point using ONLY VISUAL FEATURES.
        
        CONTEXT:
        - Action parameters: {self.move_step}m per move | {self.turn_angle}° per turn | FOV step: {self.fov_step}° | Pitch step: {self.pitch_step}°
        - Current heading: {self.data_handler.heading:.1f}°
        - Relative position from start: X={self.relative_x:.1f}m, Y={self.relative_y:.1f}m
        - LAST ACTION RESULT: {self.last_move_feedback or 'last action succeeded / no move yet'}
        {stuck_warning}
        {exploration_summary}
        - Position history (last 10 steps): {relative_history_str}
        - Taken actions: {action_history_str}
        - Found places: {places_str}
        - Progress: {self.current_step}/{self.max_steps}
        - Your previous location hypothesis: {hypothesis_str}
        - YOUR CURRENT PLAN (you set this earlier — keep following it unless it is finished; update/clear it as needed): {self.plan or 'no active plan yet — set one'}
        
        {nav_prompt}
        """
        
        content = [{"type": "text", "text": prompt}]
        
        # 1) 文字摘要：完整的轨迹记忆（每一步一行，不含图像，成本极低）
        history_summary = self._build_history_summary()
        if history_summary:
            content.append({
                "type": "text",
                "text": "Below is a text summary of every step so far (positions & actions).\n"
                        f"{history_summary}"
            })
        
        # 2) 证据库：跨步累积的定位证据（分割结论 + 搜索到的地点坐标），
        #    用文字回传替代历史关键帧图像，避免上下文爆炸，同时保留关键事实。
        evidence_summary = self._build_evidence_summary()
        content.append({
            "type": "text",
            "text": (
                "Below is your accumulated EVIDENCE BANK (facts you have already "
                "established during this exploration). Use it to update your location "
                "hypothesis. Do not repeat tools you have already used successfully — "
                "keep gathering NEW evidence.\n"
                f"{evidence_summary}"
            )
        })
        
        # 3) BEV 鸟瞰图：把走过的道路拓扑压成一张图，作为空间结构辅助信息
        if self.use_bev:
            bev_b64 = self._generate_bev_image()
            if bev_b64:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{bev_b64}", "detail": "high"}
                })
                content.append({
                    "type": "text",
                    "text": "This is YOUR EXPLORATION MEMORY MAP (bird's-eye view of the roads you have already walked). "
                            "Its ONLY purpose is to tell you WHERE YOU HAVE BEEN and WHERE YOU HAVE NOT, so you never walk the same road twice.\n"
                            "HOW TO READ IT:\n"
                            "  - Every BLACK DOT is one of YOUR OWN past camera positions (your footprints). "
                            "The RED dot is where you started. The RED arrow is where you are NOW and which way you face.\n"
                            "  - A road that has black dots along it = ALREADY EXPLORED by you. "
                            "A road/branch that continues but has NO black dots = NOT YET EXPLORED.\n"
                            "THE ONE RULE: at a junction, look at the map, pick a branch that has NO black dots, and walk into it. "
                            "NEVER walk back along a branch that already has black dots — that is backtracking and wastes steps.\n"
                            "(To judge which way to turn and where a road actually runs, use the current street view image, not this map — "
                            "this map may be rotated/skewed. But for remembering where you have been, this map is exactly what you need.)"
                })
        
        # 4) 当前图像单独回传并明确标注
        cur_meta = self.image_meta_history.get(self.current_step, {})
        content.append({"type": "image_url", "image_url": {
            "url": f"data:image/jpeg;base64,{b64}",
            "detail": "high"
        }})
        content.append({"type": "text", "text": (
            f"Current Image (Step {self.current_step}, "
            f"heading={cur_meta.get('heading', self.data_handler.heading):.0f}°)"
        )})
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": content}],
            temperature=0.1,
            max_tokens=1200,
            timeout=60
        )
        
        return response.choices[0].message.content.strip()
    
    def _is_segment_action(self, action: str) -> bool:
        """检查动作是否是分割操作"""
        return action.lower().startswith(self.SEGMENT_ACTION_PREFIX)
    
    def _is_search_action(self, action: str) -> bool:
        """检查动作是否是地点搜索操作"""
        return action.lower().startswith(self.SEARCH_ACTION_PREFIX)
    
    def _parse_segment_targets(self, action: str) -> List[str]:
        """从分割动作中提取目标对象"""
        if not self._is_segment_action(action):
            return []
        targets_str = action[len(self.SEGMENT_ACTION_PREFIX):].strip()
        targets = [t.strip() for t in targets_str.split(',')]
        return [t for t in targets if t]
    
    def parse_gpt_decision(self, gpt_response: str) -> Tuple[str, str, List[str], str]:
        import re
        
        try:
            # 查找JSON块（处理可能有多个块的情况）
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', gpt_response)
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                # 如果没有找到 ```json 格式，尝试找普通的 ```
                json_match = re.search(r'```\s*([\s\S]*?)\s*```', gpt_response)
                if json_match:
                    json_str = json_match.group(1).strip()
                else:
                    # 如果都没有，尝试直接解析整个内容
                    json_str = gpt_response
            
            response_data = json.loads(json_str)
            action = response_data.get("action", "turn right")
            reason = response_data.get("reason", "No reason provided")

            # 解析"本步观察到的画面描述"（observation），作为历史视觉记忆存入当前步 meta。
            # 决策时针对的是 self.current_step 这一帧，该帧的 image_meta_history 已由 reset/上一步 step 写入，
            # 因此这里可直接写回，后续 _build_history_summary 会把它拼进历史摘要回传给下一轮决策。
            observation = (response_data.get("observation") or "").strip()
            if observation:
                if self.current_step in self.image_meta_history:
                    self.image_meta_history[self.current_step]["observation"] = observation
                else:
                    self.image_meta_history[self.current_step] = {
                        "step": self.current_step,
                        "action": "observe",
                        "heading": self.data_handler.heading,
                        "relative_x": self.relative_x,
                        "relative_y": self.relative_y,
                        "observation": observation,
                    }

            # 维护大模型持续更新的位置假设（方案 B：让模型真正思考、逐步收敛）
            # hypothesis 现在是结构化 JSON 数组：每一条 {"evidence": ..., "region_city": ...}
            # 逐条累积到 hypothesis_evidence_map，避免覆盖丢失；location_hypothesis 存其文本摘要。
            hypothesis = response_data.get("hypothesis")
            if isinstance(hypothesis, list):
                for item in hypothesis:
                    if isinstance(item, dict):
                        evidence = (item.get("evidence") or "").strip()
                        region = (item.get("region_city") or item.get("region") or "").strip()
                        if evidence or region:
                            self.hypothesis_evidence_map.append({
                                "step": self.current_step,
                                "evidence": evidence,
                                "region_city": region,
                            })
            elif isinstance(hypothesis, str) and hypothesis.strip():
                # 兼容旧版字符串格式：整段视为一条证据
                self.hypothesis_evidence_map.append({
                    "step": self.current_step,
                    "evidence": hypothesis.strip(),
                    "region_city": "",
                })
            self.location_hypothesis = self._build_hypothesis_summary()

            # 维护大模型跨步持久化的当前计划（方案：解决"意图遗忘"，让路口探索等
            # 多步意图能在下一步继续执行，而不是每步重新采样后遗忘）。
            plan = (response_data.get("plan") or "").strip()
            if plan:
                self.plan = plan
                # 把本步的 plan 也写进 image_meta_history，供可视化按步回放意图演进
                if self.current_step in self.image_meta_history:
                    self.image_meta_history[self.current_step]["plan"] = plan
                else:
                    self.image_meta_history[self.current_step] = {
                        "step": self.current_step,
                        "action": "observe",
                        "heading": self.data_handler.heading,
                        "relative_x": self.relative_x,
                        "relative_y": self.relative_y,
                        "plan": plan,
                    }

            # 维护大模型跨步收敛的预测城市（结构化字段，供下游 Stage 2 限定搜索）
            city = response_data.get("city")
            if city and city.strip():
                self.predicted_city = city.strip()
            
            # 检查是否是分割动作
            if self._is_segment_action(action):
                seg_prompts = self._parse_segment_targets(action)
                search_query = ""
            elif self._is_search_action(action):
                seg_prompts = []
                search_query = action[len(self.SEARCH_ACTION_PREFIX):].strip()
            else:
                seg_prompts = []
                search_query = ""
                # 如果是普通动作，确保在ACTION_MAP中
                if action not in self.ACTION_MAP:
                    action = "turn right"
            
            return action, reason, seg_prompts, search_query
        
        except (json.JSONDecodeError, ValueError) as e:
            print(f"⚠️ JSON解析失败: {e}")
            print(f"   尝试解析的内容: {gpt_response[:200]}...")
            
            # 降级方案：从文本中提取动作
            # 首先检查是否有 search 操作
            search_match = re.search(r'search:\s*([^\n]+)', gpt_response, re.IGNORECASE)
            if search_match:
                action = f"search:{search_match.group(1).strip()}"
                return action, "No reason provided", [], search_match.group(1).strip()
            
            # 检查是否有分割操作
            seg_match = re.search(r'segment:\s*([^\n]+)', gpt_response, re.IGNORECASE)
            if seg_match:
                action = f"segment:{seg_match.group(1).strip()}"
                seg_prompts = self._parse_segment_targets(action)
                return action, "No reason provided", seg_prompts, ""
            
            # 检查普通动作
            action_names = list(self.ACTION_MAP.keys())
            for action_name in action_names:
                if action_name.lower() in gpt_response.lower():
                    return action_name, "No reason provided", [], ""
            
            return "turn right", "Failed to parse response", [], ""
    
    def get_observation(self) -> Dict[str, Any]:
        obs = {
            "lat": self.data_handler.current_lat,
            "lng": self.data_handler.current_lng,
            "heading": self.data_handler.heading,
            "pitch": self.data_handler.pitch,
            "fov": self.data_handler.fov,
            "step": self.current_step
        }
        
        if self.current_image is not None:
            obs["image"] = np.array(self.current_image)
        
        return obs
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        self.current_image = None
        self.final_prediction = ""
        self.action_history = []
        self.reason_history = []
        self.image_path_history = {}
        self.image_b64_history = {}
        self.image_meta_history = {}
        self.segmented_image_history = {}
        self.coord_correction_history = []
        self.trajectory_history = []
        self.segmentation_history = []
        self.found_places = []
        self._road_mask_cache = {}
        self._bev_cache_b64 = None
        self._bev_cache_key = None
        self.relative_x = 0.0
        self.relative_y = 0.0
        self.relative_history = []
        self.last_move_feedback = ""
        self.actual_initial_lat = None
        self.actual_initial_lng = None
        self.location_hypothesis = ""
        self.evidence_bank = []
        self.hypothesis_evidence_map = []
        self.predicted_city = ""
        self.plan = ""
        self._graph_nodes = {}
        self._graph_node_counter = 0
        self._graph_edges = {}
        self._current_node_key = None
        self._prev_node_key = None
        self._last_move_geo_ang = None
        
        self.data_handler.init_position(self.initial_lat, self.initial_lng, fov=self.fov_default)
        
        try:
            self.current_image, actual_lat, actual_lng = self.data_handler.get_street_view_image()
            if actual_lat and actual_lng:
                self.actual_initial_lat = actual_lat
                self.actual_initial_lng = actual_lng
                self._current_node_key = self._register_node(actual_lat, actual_lng, 0)
            if self.current_image:
                img_path, img_b64 = save_streetview_image(
                    self.current_image, 0, self.img_save_dir,
                    metadata={
                        'heading': self.data_handler.heading,
                        'pitch': self.data_handler.pitch,
                        'fov': self.data_handler.fov,
                        'relative_x': 0.0,
                        'relative_y': 0.0,
                        'actual_lat': self.data_handler.current_lat,
                        'actual_lng': self.data_handler.current_lng,
                        'action': 'reset'
                    }
                )
                self.image_path_history[0] = img_path
                self.image_b64_history[0] = img_b64
                self.image_meta_history[0] = {
                    'step': 0,
                    'heading': self.data_handler.heading,
                    'pitch': self.data_handler.pitch,
                    'fov': self.data_handler.fov,
                    'relative_x': 0.0,
                    'relative_y': 0.0,
                    'action': 'reset',
                    'action_type': 'reset'
                }
        except Exception as e:
            print(f"重置时获取图像失败: {e}")
        
        return self.get_observation(), {}
    
    def _detect_infinite_loop(self, action: str) -> bool:
        """检测是否陷入死循环模式"""
        # 只考虑强制动作发生之后的动作历史
        if self.force_action_step != 0:
            # 强制动作刚发生，下一次操作不检测死循环
            if len(self.action_history) == 0:
                return False
            
            # 检查最后一个动作是否是强制动作（以负数步骤号标识）
            last_action = self.action_history[-1]
            if "force_action" in last_action:
                # 强制动作之后的第一次操作不检测死循环
                return False
            
            # 强制动作发生后，只检查强制动作之后的历史
            # force_action_step 为负数时表示强制动作发生在该步骤
            target_step = abs(self.force_action_step)
            start_idx = -1
            for i, a in enumerate(self.action_history):
                if f"Step {target_step}:" in a:
                    start_idx = i + 1  # 从强制动作的下一个动作开始
                    break
            
            if start_idx > 0:
                relevant_history = self.action_history[start_idx:]
            else:
                relevant_history = self.action_history
        else:
            relevant_history = self.action_history
        
        if len(relevant_history) < 1:
            return False
        
        is_segment_action = action.lower().startswith(self.SEGMENT_ACTION_PREFIX)
        is_search_action = self._is_search_action(action)
        current_action_type = "segment" if is_segment_action else "search" if is_search_action else "move"
        
        # 获取最近几次操作的类型（只考虑强制动作之后的）
        recent_actions = []
        for a in relevant_history[-5:]:
            if "segment" in a:
                recent_actions.append("segment")
            elif "search" in a:
                recent_actions.append("search")
            else:
                recent_actions.append("move")
        
        # 模式1：连续多次相同工具调用（3次及以上）
        if current_action_type != "move":
            consecutive_count = 0
            for a in reversed(recent_actions):
                if a == current_action_type:
                    consecutive_count += 1
                else:
                    break
            if consecutive_count >= 2:
                return True
        
        # 模式2：连续两次相同参数的工具调用（只检查强制动作之后的）
        if len(relevant_history) >= 1 and current_action_type != "move":
            last_action = relevant_history[-1]
            if current_action_type == "segment":
                segment_targets = action[len(self.SEGMENT_ACTION_PREFIX):].strip().lower()
                if segment_targets in last_action.lower():
                    return True
            elif current_action_type == "search":
                search_query = action[len(self.SEARCH_ACTION_PREFIX):].strip().lower()
                if search_query in last_action.lower():
                    return True
        
        # 模式3：工具轮流调用（segment→search→segment→search 或 search→segment→search→segment）
        if len(recent_actions) >= 4:
            pattern1 = ["segment", "search", "segment", "search"]
            pattern2 = ["search", "segment", "search", "segment"]
            if recent_actions[-4:] == pattern1 or recent_actions[-4:] == pattern2:
                return True
        
        # 模式4：分割失败后继续分割
        if current_action_type == "segment" and len(self.segmentation_history) >= 1:
            last_segmentation = self.segmentation_history[-1]
            if not last_segmentation.get('success', False) or last_segmentation.get('prompt_count', 1) == 0:
                return True
        
        # 模式5：工具调用次数过多（连续5次工具调用）
        tool_call_count = sum(1 for a in recent_actions if a != "move")
        if tool_call_count >= 4:
            return True
        
        return False

    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        # 初始化返回值
        terminated = False
        truncated = False
        reward = 0.0
        ok = True
        
        # 死循环检测
        if self._detect_infinite_loop(action):
            info = {"action_type": "invalid", "msg": "ERROR: Detected infinite loop! Please explore the environment instead of repeatedly calling tools."}
            return self.get_observation(), reward, terminated, truncated, info
        
        # 检查是否是分割或搜索操作（这些不增加步数）
        is_segment_action = action.lower().startswith(self.SEGMENT_ACTION_PREFIX)
        is_search_action = self._is_search_action(action)
        
        if is_segment_action:
            # 分割操作：不增加步数，只进行分析
            a_type = "segment"
            # 提取分割目标
            segment_targets = action[len(self.SEGMENT_ACTION_PREFIX):].strip()
            a_dir = segment_targets
            
            info = {"action_type": a_type, "segment_targets": segment_targets, "current_step": self.current_step}
            
            # 实际调用SAM3进行分割
            if self.current_image and segment_targets:
                targets_list = [t.strip() for t in segment_targets.split(',')]
                save_path = os.path.join(self.img_save_dir, f"step{self.current_step}_segmented.png")
                sam3_result = self._call_sam3(self.current_image, targets_list, save_path=save_path)
                
                if sam3_result.get('success'):
                    prompt_count = sam3_result.get('prompt_count', 0)
                    info["msg"] = f"Segmented: {segment_targets} (found {prompt_count} objects)"
                    info["segmentation_result"] = sam3_result
                    self.segmentation_history.append({
                        "step": self.current_step,
                        "targets": segment_targets,
                        "success": True,
                        "prompt_count": prompt_count
                    })
                    # 保存分割图像证据（与文字证据同源，供最终预测交叉判断）
                    seg_image_b64 = sam3_result.get("combined_image_b64")
                    if seg_image_b64:
                        self.segmented_image_history[self.current_step] = [{
                            "prompt": ", ".join(targets_list),
                            "image_b64": seg_image_b64,
                            "saved_path": sam3_result.get("combined_image_path"),
                            "prompt_count": prompt_count,
                            "results": sam3_result.get("results", []),
                        }]
                    self.add_evidence("segment", f"分割 [{segment_targets}] → 检测到 {prompt_count} 个对象",
                                      image_b64=seg_image_b64)
                else:
                    info["msg"] = f"Segmentation failed: {sam3_result.get('error', 'Unknown error')}"
                    info["segmentation_result"] = sam3_result
                    self.segmentation_history.append({
                        "step": self.current_step,
                        "targets": segment_targets,
                        "success": False,
                        "error": sam3_result.get('error', 'Unknown error')
                    })
            else:
                info["msg"] = "Segmentation skipped: no image or empty targets"
                self.segmentation_history.append({
                    "step": self.current_step,
                    "targets": segment_targets,
                    "success": False,
                    "error": "No image or empty targets"
                })
            
            self.action_history.append(f"Step {self.current_step}: {a_type} {segment_targets}")
            reward = 0.0
        elif is_search_action:
            # 搜索操作：不增加步数，调用 Places API
            a_type = "search"
            search_query = action[len(self.SEARCH_ACTION_PREFIX):].strip()
            a_dir = search_query
            
            info = {"action_type": a_type, "search_query": search_query, "current_step": self.current_step}
            self.action_history.append(f"Step {self.current_step}: {a_type} \"{search_query}\"")
            
            # 调用 Places API
            search_result = self._call_places_search(search_query)
            info["search_result"] = search_result
            
            if search_result.get("success"):
                candidates = search_result.get("candidates", [])
                if candidates:
                    best = candidates[0]
                    place_info = {
                        "step": self.current_step,
                        "query": search_query,
                        "name": best.get("name"),
                        "address": best.get("address"),
                        "lat": best.get("lat"),
                        "lng": best.get("lng"),
                    }
                    self.found_places.append(place_info)
                    info["msg"] = f"Found: {best.get('name')} at {best.get('address')} ({best.get('lat')}, {best.get('lng')})"
                    info["found_place"] = place_info
                    self.add_evidence("search", f"搜索 \"{search_query}\" → {best.get('name')} @ ({best.get('lat')}, {best.get('lng')})",
                                      name=best.get("name"), lat=best.get("lat"), lng=best.get("lng"))
                else:
                    info["msg"] = f"No results for '{search_query}'"
            else:
                info["msg"] = f"Search failed: {search_result.get('error', 'Unknown')}"
            
            reward = 0.0
        else:
            # 普通操作：增加步数
            self.current_step += 1
            
            a_type, a_dir = self.ACTION_MAP.get(action, ("turn", "right"))
            
            info = {"action_type": a_type, "relative_dir": a_dir, "current_step": self.current_step}
            
            self.current_theoretical_lat = None
            self.current_theoretical_lng = None
            self.current_actual_lat = None
            self.current_actual_lng = None
            self.current_correction_diff = None
            
            if a_type != "move":
                self.action_history.append(f"Step {self.current_step}: {a_type} {a_dir}")
            
            if a_type == "move":
                import math
                from utils.geometry import get_geo_angle, calculate_relative_offset
                
                geo_ang = get_geo_angle(self.data_handler.heading, a_dir)
                self._last_move_geo_ang = geo_ang
                ok, msg, actual_distance = self.data_handler.move(geo_ang, self.move_step)
                info["msg"] = msg
                info["actual_distance"] = actual_distance
                
                if ok:
                    self.action_history.append(f"Step {self.current_step}: {a_type} {a_dir}")
                    self.last_move_feedback = ""
                    self.current_theoretical_lat = self.data_handler.current_lat
                    self.current_theoretical_lng = self.data_handler.current_lng
                    info["theoretical_lat"] = self.current_theoretical_lat
                    info["theoretical_lng"] = self.current_theoretical_lng
                else:
                    self.action_history.append(f"Step {self.current_step}: {a_type} {a_dir} [FAILED: {msg}]")
                    self.last_move_feedback = f"上一步 move {a_dir} 失败：{msg}（位置未变化，请转向或换个方向探索）"
                    self._record_edge(self._current_node_key, geo_ang, None)
                
                self.relative_history.append({
                    'step': self.current_step,
                    'x': self.relative_x,
                    'y': self.relative_y,
                    'heading': self.data_handler.heading,
                    'action': action,
                    'moved': ok
                })
                
                reward = 0.0 if ok else -1.0
            
            elif a_type == "turn":
                delta = -self.turn_angle if a_dir == "left" else self.turn_angle
                self.data_handler.turn(delta)
                info["msg"] = f"Turned {a_dir.upper()} {abs(delta)}°"
            
            elif a_type == "fov":
                current_fov = self.data_handler.fov
                new_fov = max(10, current_fov - self.fov_step) if a_dir == "zoom_in" else min(120, current_fov + self.fov_step)
                self.data_handler.fov = new_fov
                info["msg"] = f"FOV {a_dir.upper()} to {new_fov}°"
            
            elif a_type == "pitch":
                current_pitch = self.data_handler.pitch
                new_pitch = min(90, current_pitch + self.pitch_step) if a_dir == "up" else max(-90, current_pitch - self.pitch_step)
                self.data_handler.pitch = new_pitch
                info["msg"] = f"Pitch {a_dir.upper()} to {new_pitch}°"
        
        try:
            # 只有非分割操作才需要获取新图像
            if not is_segment_action:
                self.current_image, actual_lat, actual_lng = self.data_handler.get_street_view_image()
                
                if a_type == "move" and ok and actual_lat and actual_lng and self.current_theoretical_lat:
                    from utils.geometry import calculate_gps_distance, calculate_relative_offset
                    
                    self.current_actual_lat = actual_lat
                    self.current_actual_lng = actual_lng
                    self.current_correction_diff = calculate_gps_distance(
                        self.current_theoretical_lat, self.current_theoretical_lng,
                        actual_lat, actual_lng
                    )
                    
                    correction = {
                        "step": self.current_step,
                        "theoretical_lat": self.current_theoretical_lat,
                        "theoretical_lng": self.current_theoretical_lng,
                        "actual_lat": actual_lat,
                        "actual_lng": actual_lng,
                        "lat_diff": actual_lat - self.current_theoretical_lat,
                        "lng_diff": actual_lng - self.current_theoretical_lng,
                        "distance_diff": self.current_correction_diff
                    }
                    self.coord_correction_history.append(correction)
                    
                    self.data_handler.current_lat = actual_lat
                    self.data_handler.current_lng = actual_lng
                    
                    info["actual_lat"] = actual_lat
                    info["actual_lng"] = actual_lng
                    info["coord_correction_diff_m"] = self.current_correction_diff

                    # 离散探索图：move 成功 → 登记新采样点节点，记录当前节点→新节点的边
                    to_key = self._register_node(actual_lat, actual_lng, self.current_step)
                    self._record_edge(self._current_node_key, self._last_move_geo_ang, to_key)
                    self._prev_node_key = self._current_node_key
                    self._current_node_key = to_key
                    
                    ref_lat = self.actual_initial_lat if self.actual_initial_lat else self.initial_lat
                    ref_lng = self.actual_initial_lng if self.actual_initial_lng else self.initial_lng
                    self.relative_x, self.relative_y = calculate_relative_offset(
                        ref_lat, ref_lng,
                        actual_lat, actual_lng
                    )
                
                if self.current_image:
                    img_path, img_b64 = save_streetview_image(
                        self.current_image, self.current_step, self.img_save_dir,
                        metadata={
                            'heading': self.data_handler.heading,
                            'pitch': self.data_handler.pitch,
                            'fov': self.data_handler.fov,
                            'relative_x': self.relative_x,
                            'relative_y': self.relative_y,
                            'actual_lat': actual_lat if actual_lat else self.data_handler.current_lat,
                            'actual_lng': actual_lng if actual_lng else self.data_handler.current_lng,
                            'action': action
                        }
                    )
                    self.image_path_history[self.current_step] = img_path
                    self.image_b64_history[self.current_step] = img_b64
                    self.image_meta_history[self.current_step] = {
                        'step': self.current_step,
                        'heading': self.data_handler.heading,
                        'pitch': self.data_handler.pitch,
                        'fov': self.data_handler.fov,
                        'relative_x': self.relative_x,
                        'relative_y': self.relative_y,
                        'action': action,
                        'action_type': a_type
                    }
        
        except Exception as e:
            self.current_image = None
            reward = -1.0
            info["error"] = f"获取图像失败: {str(e)}"
        
        self.trajectory_history.append({
            "step": self.current_step,
            "lat": self.data_handler.current_lat,
            "lng": self.data_handler.current_lng,
            "heading": self.data_handler.heading,
            "fov": self.data_handler.fov,
            "pitch": self.data_handler.pitch,
            "action": action
        })
        
        if not is_segment_action and not is_search_action and self.current_step >= self.max_steps:
            terminated = True
            self._final_prediction()
        
        return self.get_observation(), reward, terminated, truncated, info
    
    def _final_prediction(self):
        if not self.image_b64_history:
            self.final_prediction = "No images available for prediction"
            return
        
        prompt = FINAL_PREDICTION_PROMPT.format(max_steps=self.max_steps)
        
        # 添加搜索找到的地点信息
        if self.found_places:
            places_text = "\n**📍 PLACES FOUND DURING EXPLORATION (via Google Places API):**\n"
            for place in self.found_places:
                places_text += f"- \"{place['query']}\" → {place['name']}: {place['address']} ({place['lat']}, {place['lng']})\n"
            places_text += "\nUse these location hints to refine your coordinate prediction.\n"
        else:
            places_text = ""
        
        prompt = prompt.replace("{{PLACES_HINTS}}", places_text)
        
        # 把探索过程中逐步收敛的位置假设（证据 → 地区/城市）作为强先验喂给最终预测
        if self.hypothesis_evidence_map:
            hypothesis_text = (
                "\n**🧭 YOUR LOCATION HYPOTHESIS (evidence → region/city, refined over the whole exploration):**\n"
                f"{self._build_hypothesis_summary()}\n"
                "Use this as your strongest prior — it was built up step by step from visual evidence. "
                "Refine it into a final coordinate prediction.\n"
            )
            prompt = prompt.replace("{{HYPOTHESIS}}", hypothesis_text)
        else:
            prompt = prompt.replace("{{HYPOTHESIS}}", "")
        
        # 把跨步累积的证据库喂给最终预测，作为"最后验证"的依据：
        # 搜索结果是真实世界坐标，分割结果是已确认存在的视觉对象，最终答案必须与它们自洽。
        if self.evidence_bank:
            evidence_text = (
                "\n**🔎 EVIDENCE BANK (facts verified during exploration):**\n"
                f"{self._build_evidence_summary()}\n"
                "Before you answer, VERIFY your prediction against these facts: "
                "every search result is a REAL-WORLD coordinate (your prediction must be "
                "within ~1km of it if any search succeeded); every segmentation result tells "
                "you which visual objects were confirmed present. Your final answer must be "
                "consistent with them.\n"
            )
            prompt = prompt.replace("{{EVIDENCE_BANK}}", evidence_text)
        else:
            prompt = prompt.replace("{{EVIDENCE_BANK}}", "")
        
        content = [{"type": "text", "text": prompt}]
        
        for step in sorted(self.image_b64_history):
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{self.image_b64_history[step]}", "detail": "high"}
            })
            content.append({"type": "text", "text": f"Step {step}"})
            
            if step in self.segmented_image_history:
                for seg in self.segmented_image_history[step]:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{seg['image_b64']}", "detail": "high"}
                    })
                    content.append({"type": "text", "text": f"Segmented: {seg['prompt']}"})
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": content}],
                temperature=0.1,
                max_tokens=1000,
                timeout=120
            )
            self.final_prediction = response.choices[0].message.content.strip()
        except Exception as e:
            self.final_prediction = f"Prediction failed: {str(e)}"
    
    def render(self, mode="human"):
        if mode == "human":
            print(f"Step {self.current_step}/{self.max_steps}")
            print(f"Position: ({self.data_handler.current_lat:.6f}, {self.data_handler.current_lng:.6f})")
            print(f"Heading: {self.data_handler.heading:.1f}°")
    
    def close(self, release_sam3=True):
        if release_sam3 and self.sam3_tool:
            self.sam3_tool.close()
            self.sam3_tool = None

    def preload_sam3(self):
        """预加载SAM3模型（批量处理时调用一次即可）"""
        if self.use_sam3 and self.sam3_tool is None:
            self.sam3_tool = SAM3Tool()
            self.sam3_tool.initialize()
            print("✅ SAM3模型已预加载")
