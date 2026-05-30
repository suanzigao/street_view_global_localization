"""
Gym环境封装模块

将街景定位任务封装为Gym风格的强化学习环境，便于与决策模型集成
"""

import os
import traceback
import httpx
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from PIL import Image
from typing import Tuple, Optional, Dict, Any
from openai import OpenAI

from utils.file_utils import save_streetview_image, image_to_base64
from utils.decorators import retry_on_failure
from prompts import NAVIGATION_PROMPT, FINAL_PREDICTION_PROMPT


class StreetViewAutoLocEnv(gym.Env):
    """
    街景自主定位环境
    
    封装街景导航和定位任务为标准Gym环境接口
    """
    
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
            fov_default: int = 60
    ):
        """
        初始化街景定位环境
        
        Args:
            initial_lat: 初始纬度
            initial_lng: 初始经度
            move_step: 每次移动步长（米）
            max_retries: 最大重试次数
            max_steps: 最大导航步数
            point_dir: 点位数据保存目录
            img_save_dir: 图像保存目录
            turn_angle: 每次转向角度（度）
            fov_step: FOV调整步长（度）
            pitch_step: 俯仰调整步长（度）
            fov_default: 默认FOV值
        """
        super().__init__()
        
        # 导入放在这里避免循环依赖
        from core.street_view import GoogleStreetViewData
        
        self.data_handler = GoogleStreetViewData(
            max_retries=max_retries,
            move_step=move_step,
            radius=10
        )
        self.data_handler.init_position(initial_lat, initial_lng)
        
        # 配置参数
        self.max_steps = max_steps
        self.move_step = move_step
        self.turn_angle = turn_angle
        self.fov_step = fov_step
        self.pitch_step = pitch_step
        self.fov_default = fov_default
        
        # 目录配置
        self.point_dir = point_dir
        self.img_save_dir = img_save_dir
        
        # OpenAI客户端（用于决策）
        transport = httpx.HTTPTransport(proxy="http://127.0.0.1:7897")
        self.openai_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url="https://api.laozhang.ai/v1",
            http_client=httpx.Client(transport=transport)
        )
        
        # 动作空间和观察空间
        self.action_space = spaces.Discrete(len(self.ACTION_MAP))
        self.observation_space = spaces.Dict({
            "image": spaces.Box(0, 255, (640, 640, 3), int),
            "lat": spaces.Box(-90, 90, (1,), float),
            "lng": spaces.Box(-180, 180, (1,), float),
            "heading": spaces.Box(0, 360, (1,), float),
            "pitch": spaces.Box(-90, 90, (1,), float)
        })
        
        # 状态变量
        self.current_step = 0
        self.current_image = None
        self.final_prediction = ""
        self.action_history = []
        self.reason_history = []
        self.image_path_history = {}
        self.image_b64_history = {}
        self.coord_correction_history = []
        self.trajectory_history = []
        
        # 临时坐标记录
        self.current_theoretical_lat = None
        self.current_theoretical_lng = None
        self.current_actual_lat = None
        self.current_actual_lng = None
        self.current_correction_diff = None
    
    @retry_on_failure(max_retries=3, delay=2.0)
    def gpt4o_decision(self, image: Image.Image, step_num: int) -> str:
        """
        调用GPT-4o进行导航决策
        
        Args:
            image: 当前街景图像
            step_num: 当前步数
        
        Returns:
            GPT-4o返回的决策结果（JSON格式）
        """
        b64 = image_to_base64(image)
        action_history_str = "\n- ".join(self.action_history) if self.action_history else "No navigation actions yet"
        
        prompt = f"""
        Your Role: Autonomous Google Street View Navigator & Visual Location Predictor
        CORE MISSION: Predict the GPS coordinates of your INITIAL starting point using ONLY VISUAL FEATURES.
        
        CONTEXT:
        - Action parameters: {self.move_step}m per move | {self.turn_angle}° per turn | FOV step: {self.fov_step}° | Pitch step: {self.pitch_step}°
        - Current heading: {self.data_handler.heading:.1f}°
        - Taken actions: {action_history_str}
        - Progress: {self.current_step}/{self.max_steps}
        
        {NAVIGATION_PROMPT.format(step_num=self.current_step + 1, max_steps=self.max_steps)}
        """
        
        content = [{"type": "text", "text": prompt}]
        
        for step in sorted(self.image_b64_history):
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{self.image_b64_history[step]}", "detail": "high"}
            })
            content.append({"type": "text", "text": f"Image from Step {step}"})
        
        content.append({"type": "image_url", "image_url": {
            "url": f"data:image/jpeg;base64,{b64}",
            "detail": "high"
        }})
        content.append({"type": "text", "text": f"Current Image (Step {self.current_step + 1})"})
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": content}],
            temperature=0.1,
            max_tokens=100,
            timeout=60
        )
        
        return response.choices[0].message.content.strip()
    
    def parse_gpt_decision(self, gpt_response: str) -> Tuple[str, str]:
        """
        解析GPT-4o的决策响应
        
        Args:
            gpt_response: GPT-4o返回的响应
        
        Returns:
            (action: str, reason: str)
        """
        import re
        import json
        
        try:
            cleaned_response = gpt_response.strip()
            
            if cleaned_response.startswith('```json') and cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[7:-3].strip()
            elif cleaned_response.startswith('```') and cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[3:-3].strip()
            
            response_data = json.loads(cleaned_response)
            action = response_data.get("action", "turn right")
            reason = response_data.get("reason", "No reason provided")
            
            if action not in self.ACTION_MAP:
                action = "turn right"
            
            return action, reason
        
        except (json.JSONDecodeError, ValueError):
            action_names = list(self.ACTION_MAP.keys())
            for action_name in action_names:
                if action_name in gpt_response.lower():
                    return action_name, "No reason provided"
            
            return "turn right", "Failed to parse response"
    
    def get_observation(self) -> Dict[str, Any]:
        """
        获取当前观察
        
        Returns:
            观察字典，包含图像、GPS坐标、朝向等信息
        """
        if self.current_image:
            img_arr = np.array(self.current_image)
        else:
            img_arr = np.zeros((640, 640, 3), int)
        
        return {
            "image": img_arr,
            "lat": np.array([self.data_handler.current_lat]),
            "lng": np.array([self.data_handler.current_lng]),
            "heading": np.array([self.data_handler.heading]),
            "pitch": np.array([self.data_handler.pitch])
        }
    
    def step(self, action: str) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        执行一步动作
        
        Args:
            action: 动作名称
        
        Returns:
            (observation, reward, terminated, truncated, info)
        """
        self.current_step += 1
        a_type, a_dir = self.ACTION_MAP.get(action, ("turn", "right"))
        
        info = {"action_type": a_type, "relative_dir": a_dir, "current_step": self.current_step}
        reward = 0.0
        terminated = truncated = False
        
        # 重置坐标记录
        self.current_theoretical_lat = None
        self.current_theoretical_lng = None
        self.current_actual_lat = None
        self.current_actual_lng = None
        self.current_correction_diff = None
        
        # 记录动作历史
        self.action_history.append(f"Step {self.current_step}: {a_type} {a_dir}")
        
        # 执行动作
        if a_type == "move":
            from utils.geometry import get_geo_angle
            
            geo_ang = get_geo_angle(self.data_handler.heading, a_dir)
            ok, msg, actual_distance = self.data_handler.move(geo_ang, self.move_step)
            info["msg"] = msg
            info["actual_distance"] = actual_distance
            
            if ok:
                self.current_theoretical_lat = self.data_handler.current_lat
                self.current_theoretical_lng = self.data_handler.current_lng
                info["theoretical_lat"] = self.current_theoretical_lat
                info["theoretical_lng"] = self.current_theoretical_lng
            
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
        
        # 获取图像
        try:
            self.current_image, actual_lat, actual_lng = self.data_handler.get_street_view_image()
            
            if self.current_image:
                img_path, img_b64 = save_streetview_image(self.current_image, self.current_step, self.img_save_dir)
                self.image_path_history[self.current_step] = img_path
                self.image_b64_history[self.current_step] = img_b64
            
            # 坐标修正（仅移动动作）
            if a_type == "move" and ok and actual_lat and actual_lng and self.current_theoretical_lat:
                from utils.geometry import calculate_gps_distance
                
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
        
        except Exception as e:
            self.current_image = None
            reward = -1.0
            info["error"] = f"获取图像失败: {str(e)}"
        
        # 记录轨迹
        if not hasattr(self, 'trajectory_history'):
            self.trajectory_history = []
        
        action_desc = self._get_action_description(a_type, a_dir, info)
        self.trajectory_history.append({
            "step": self.current_step,
            "lat": self.data_handler.current_lat,
            "lng": self.data_handler.current_lng,
            "heading": self.data_handler.heading,
            "fov": self.data_handler.fov,
            "pitch": self.data_handler.pitch,
            "action": action_desc
        })
        
        # 检查是否达到最大步数
        if self.current_step >= self.max_steps:
            truncated = True
            self._force_final_prediction()
        
        return self.get_observation(), reward, terminated, truncated, info
    
    def _get_action_description(self, a_type: str, a_dir: str, info: Dict) -> str:
        """
        获取动作描述
        
        Args:
            a_type: 动作类型
            a_dir: 动作方向
            info: 动作信息
        
        Returns:
            动作描述字符串
        """
        if a_type == "move":
            return f"Move {a_dir} {info.get('actual_distance', 0):.2f}m"
        elif a_type == "turn":
            delta = -self.turn_angle if a_dir == "left" else self.turn_angle
            return f"Turn {a_dir} {abs(delta)}°"
        elif a_type == "fov":
            return f"FOV {a_dir} to {self.data_handler.fov}°"
        elif a_type == "pitch":
            return f"Pitch {a_dir} to {self.data_handler.pitch}°"
        else:
            return f"{a_type} {a_dir}"
    
    def _force_final_prediction(self):
        """
        强制进行最终GPS预测
        """
        if not self.current_image:
            self.final_prediction = "No image for final prediction"
            return
        
        try:
            full_hist = "\n".join(self.action_history) if self.action_history else "No actions"
            actual_max_step = max(self.image_b64_history.keys()) if self.image_b64_history else 0
            
            force_prompt = f"""
            MAX NAVIGATION STEPS ({self.max_steps}) REACHED! YOU MUST MAKE A FINAL PREDICTION.
            
            CONTEXT:
            1. Navigation History: {full_hist}
            2. Visual Data: All street view images from Step 0 to Step {actual_max_step}.
            3. Movement Rules: {self.move_step}m per move | {self.turn_angle}° per turn.
            
            {FINAL_PREDICTION_PROMPT.format(max_steps=actual_max_step)}
            """
            
            content = [{"type": "text", "text": force_prompt}]
            for step in sorted(self.image_b64_history):
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{self.image_b64_history[step]}", "detail": "high"}
                })
                content.append({"type": "text", "text": f"Image from Step {step}"})
            
            resp = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": content}],
                temperature=0.1,
                max_tokens=1000,
                timeout=120
            )
            
            self.final_prediction = resp.choices[0].message.content.strip()
        
        except Exception as e:
            self.final_prediction = f"Final prediction error: {e}"
    
    def reset(self, seed=None, options=None):
        """
        重置环境
        
        Args:
            seed: 随机种子
            options: 额外选项
        
        Returns:
            初始观察
        """
        super().reset(seed=seed)
        
        self.current_step = 0
        self.final_prediction = ""
        self.action_history = []
        self.image_path_history = {}
        self.image_b64_history = {}
        self.coord_correction_history = []
        self.trajectory_history = []
        
        self.current_theoretical_lat = None
        self.current_theoretical_lng = None
        self.current_actual_lat = None
        self.current_actual_lng = None
        self.current_correction_diff = None
        
        self.data_handler.init_position(
            self.data_handler.initial_lat, 
            self.data_handler.initial_lng
        )
        
        try:
            self.current_image, _, _ = self.data_handler.get_street_view_image()
            
            if self.current_image:
                img_path, img_b64 = save_streetview_image(self.current_image, 0, self.img_save_dir)
                self.image_path_history[0] = img_path
                self.image_b64_history[0] = img_b64
        
        except Exception:
            self.current_image = None
        
        return self.get_observation()
    
    def render(self, mode="human"):
        """
        渲染当前图像
        
        Args:
            mode: 渲染模式
        """
        if mode == "human" and self.current_image:
            self.current_image.show(title=f"Step {self.current_step}")