"""
Gym环境封装模块 - 增强版

集成SAM3图像分割工具，让大模型可以自主决定识别哪些视觉特征来辅助定位
"""

import os
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
from prompts import NAVIGATION_PROMPT_TEMPLATE, SEGMENTATION_EXAMPLES, FINAL_PREDICTION_PROMPT

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
            use_sam3: bool = True
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
        self.segmented_image_history = {}
        self.coord_correction_history = []
        self.trajectory_history = []
        
        self.use_sam3 = use_sam3 and SAM3_AVAILABLE
        self.sam3_tool = None
        self.segmentation_history = []
        
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
        
        self.actual_initial_lat = None
        self.actual_initial_lng = None
        
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
        
        # 先格式化导航提示词（处理 step_num 和 max_steps）
        nav_prompt = NAVIGATION_PROMPT_TEMPLATE.format(
            step_num=self.current_step + 1, 
            max_steps=self.max_steps
        )
        # 然后替换 SEGMENTATION_EXAMPLES（注意是双重花括号）
        nav_prompt = nav_prompt.replace('{{SEGMENTATION_EXAMPLES}}', SEGMENTATION_EXAMPLES)
        
        relative_history_str = "\n- ".join(
            [f"Step {h['step']}: ({h['x']:.1f}m, {h['y']:.1f}m), heading={h['heading']:.1f}°, action={h['action']}" 
             for h in self.relative_history[-10:]]
        ) if self.relative_history else "No movement yet"
        
        prompt = f"""
        Your Role: Autonomous Google Street View Navigator & Visual Location Predictor
        CORE MISSION: Predict the GPS coordinates of your INITIAL starting point using ONLY VISUAL FEATURES.
        
        CONTEXT:
        - Action parameters: {self.move_step}m per move | {self.turn_angle}° per turn | FOV step: {self.fov_step}° | Pitch step: {self.pitch_step}°
        - Current heading: {self.data_handler.heading:.1f}°
        - Relative position from start: X={self.relative_x:.1f}m, Y={self.relative_y:.1f}m
        - Position history (last 10 steps): {relative_history_str}
        - Taken actions: {action_history_str}
        - Found places: {places_str}
        - Progress: {self.current_step}/{self.max_steps}
        
        {nav_prompt}
        """
        
        content = [{"type": "text", "text": prompt}]
        
        for step in sorted(self.image_b64_history):
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{self.image_b64_history[step]}", "detail": "high"}
            })
            content.append({"type": "text", "text": f"Image from Step {step}"})
            
            if step in self.segmented_image_history:
                for seg in self.segmented_image_history[step]:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{seg['image_b64']}", "detail": "high"}
                    })
                    content.append({"type": "text", "text": f"Segmented: {seg['prompt']}"})
        
        content.append({"type": "image_url", "image_url": {
            "url": f"data:image/jpeg;base64,{b64}",
            "detail": "high"
        }})
        content.append({"type": "text", "text": f"Current Image (Step {self.current_step + 1})"})
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": content}],
            temperature=0.1,
            max_tokens=150,
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
        self.segmented_image_history = {}
        self.coord_correction_history = []
        self.trajectory_history = []
        self.segmentation_history = []
        self.found_places = []
        self.relative_x = 0.0
        self.relative_y = 0.0
        self.relative_history = []
        self.actual_initial_lat = None
        self.actual_initial_lng = None
        
        self.data_handler.init_position(self.initial_lat, self.initial_lng, fov=self.fov_default)
        
        try:
            self.current_image, actual_lat, actual_lng = self.data_handler.get_street_view_image()
            if actual_lat and actual_lng:
                self.actual_initial_lat = actual_lat
                self.actual_initial_lng = actual_lng
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
            
            self.action_history.append(f"Step {self.current_step}: {a_type} {a_dir}")
            
            if a_type == "move":
                import math
                from utils.geometry import get_geo_angle, calculate_relative_offset
                
                geo_ang = get_geo_angle(self.data_handler.heading, a_dir)
                ok, msg, actual_distance = self.data_handler.move(geo_ang, self.move_step)
                info["msg"] = msg
                info["actual_distance"] = actual_distance
                
                if ok:
                    self.current_theoretical_lat = self.data_handler.current_lat
                    self.current_theoretical_lng = self.data_handler.current_lng
                    info["theoretical_lat"] = self.current_theoretical_lat
                    info["theoretical_lng"] = self.current_theoretical_lng
                
                self.relative_history.append({
                    'step': self.current_step,
                    'x': self.relative_x,
                    'y': self.relative_y,
                    'heading': self.data_handler.heading,
                    'action': action
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
                max_tokens=200,
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
