import os
import re
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv

# 加载.env文件中的环境变量
load_dotenv()

from google_street_view import GoogleStreetViewData
from config import GLOBAL_MOVE_STEP, TURN_ANGLE, FOV_STEP, PITCH_STEP
from utils import save_streetview_image, get_geo_angle, calculate_gps_distance, retry_on_failure, image_to_base64
from visualization import generate_visualization

# ====================== StreetViewAutoLocEnv ======================
class StreetViewAutoLocEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}
    ACTION_MAP = {
        0: ("move", "forward"),
        1: ("move", "left"),
        2: ("move", "right"),
        3: ("move", "backward"),
        4: ("turn", "left"),
        5: ("turn", "right"),
        6: ("fov", "zoom_in"),  # 新增：FOV缩小（放大视角）
        7: ("fov", "zoom_out"),  # 新增：FOV增大（缩小视角）
        8: ("pitch", "up"),  # 新增：抬头（pitch增加）
        9: ("pitch", "down")  # 新增：低头（pitch减少）
    }

    def __init__(
            self, initial_lat: float, initial_lng: float,
            move_step: float = GLOBAL_MOVE_STEP, max_retries: int = 3, max_steps: int = 10,
            point_dir: str = "./", img_save_dir: str = "./images"
    ):
        super().__init__()
        self.data_handler = GoogleStreetViewData(max_retries=max_retries)
        self.data_handler.move_step = move_step
        self.data_handler.init_position(initial_lat, initial_lng)
        self.max_steps = max_steps
        self.current_step = 0
        self.point_dir = point_dir
        self.img_save_dir = img_save_dir

        self.openai_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url="https://api.laozhang.ai/v1"
        )

        self.action_space = spaces.Discrete(6)
        # 使用640x640作为观察空间的尺寸，因为Google API可能返回这个尺寸
        self.observation_space = spaces.Dict({
            "image": spaces.Box(0, 255, (640, 640, 3), int),  # 匹配Google API返回的尺寸
            "lat": spaces.Box(-90, 90, (1,), float),
            "lng": spaces.Box(-180, 180, (1,), float),
            "heading": spaces.Box(0, 360, (1,), float),
            "pitch": spaces.Box(-90, 90, (1,), float)
        })

        self.current_image = None
        self.final_prediction = ""
        self.action_history = []
        self.image_path_history = {}
        self.image_b64_history = {}
        self.coord_correction_history = []  # 记录坐标修正历史
        self.trajectory_history = []  # 记录轨迹历史
        # 重置临时坐标记录
        self.current_theoretical_lat = None
        self.current_theoretical_lng = None
        self.current_actual_lat = None
        self.current_actual_lng = None
        self.current_correction_diff = None

    # ====================== 完全原版提示词，一字未改 ======================
    @retry_on_failure(3, 2.0)
    def gpt4o_decision(self, image: Image.Image, step_num: int) -> str:
        b64 = image_to_base64(image)
        action_history_str = "\n- ".join(self.action_history) if self.action_history else "No navigation actions yet"

        # ========== 你原版英文提示词，完全保留 ==========
        prompt = f"""
        Your Role: Autonomous Google Street View Navigator & Visual Location Predictor
        CORE MISSION: Predict the GPS coordinates of your INITIAL starting point using ONLY VISUAL FEATURES.
        ACTION VALUES CONTEXT (FOR NAVIGATION ONLY - NO CALCULATION):
        - Exact action params: {self.data_handler.move_step}m per move | {TURN_ANGLE}° per turn | Current heading: {self.data_handler.heading:.1f}°
        - Taken actions: {action_history_str}
        - Current steps / Max steps: {self.current_step}/{self.max_steps}

        ABSOLUTE RULES (VIOLATION = INVALID PREDICTION):
        1. NO GPS coordinates provided, NO geometric/spatial calculations allowed!
        2. All decisions/predictions MUST be based SOLELY on visual clues from the street view image.
        3. Navigate toward unique visual landmarks (street signs, buildings, landmarks) to find identifiable features.
        4. All MOVE actions are RELATIVE to your CURRENT VIEW DIRECTION (not fixed geographic north/south)!
        5. You MUST ONLY move ALONG THE ROAD. Do NOT walk into buildings, grass, walls, rivers, or off-road.

        # 仅有的6个Action Options（CHOOSE ONLY ONE NUMBER, NO EXTRA TEXT, NO OTHER WORDS）
        0 = Move FORWARD {self.data_handler.move_step}m (current view front)
        1 = Move LEFT {self.data_handler.move_step}m (current view left side)
        2 = Move RIGHT {self.data_handler.move_step}m (current view right side)
        3 = Move BACKWARD {self.data_handler.move_step}m (current view back)
        4 = Turn LEFT {TURN_ANGLE}° (view left, no movement)
        5 = Turn RIGHT {TURN_ANGLE}° (view right, no movement)

        # OUTPUT FORMAT (STRICTLY FOLLOW, NO DEVIATIONS, NO SYMBOLS, NO EXPLANATION)
        Output ONLY the number of your chosen action (0/1/2/3/4/5), e.g., 0, 4, 5 (nothing else)!
        You will be forced to make a final GPS prediction when max navigation steps are reached.
        """

        response = self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}",
                        "detail": "high"
                    }}
                ]
            }],
            temperature=0.1,
            max_tokens=100,
            timeout=60
        )
        return response.choices[0].message.content.strip()

    def parse_gpt_decision(self, gpt_response: str) -> int:
        import re
        match = re.search(r'^(\d+)', gpt_response.strip())
        if not match:
            return 5
        return max(0, min(5, int(match.group(1))))

    def get_observation(self) -> Dict[str, Any]:
        if self.current_image:
            img_arr = np.array(self.current_image)
        else:
            # 使用640x640作为默认尺寸，因为Google API可能返回这个尺寸
            img_arr = np.zeros((640, 640, 3), int)
        return {
            "image": img_arr,
            "lat": np.array([self.data_handler.current_lat]),
            "lng": np.array([self.data_handler.current_lng]),
            "heading": np.array([self.data_handler.heading]),
            "pitch": np.array([self.data_handler.pitch])
        }

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        self.current_step += 1
        a_type, a_dir = self.ACTION_MAP.get(action, ("turn", "right"))
        info = {"action_type": a_type, "relative_dir": a_dir, "current_step": self.current_step}
        reward = 0.0
        terminated = truncated = False

        # 重置当前步的坐标记录
        self.current_theoretical_lat = None
        self.current_theoretical_lng = None
        self.current_actual_lat = None
        self.current_actual_lng = None
        self.current_correction_diff = None

        # 记录动作历史
        self.action_history.append(f"Step {self.current_step}: {a_type} {a_dir}")

        # 保存当前状态用于轨迹记录
        current_lat = self.data_handler.current_lat
        current_lng = self.data_handler.current_lng
        current_heading = self.data_handler.heading
        current_fov = self.data_handler.fov if hasattr(self.data_handler, 'fov') else 30
        current_pitch = self.data_handler.pitch

        # ========== 1. 执行对应动作 ==========
        if a_type == "move":
            # 移动动作：计算理论GPS并更新
            geo_ang = get_geo_angle(self.data_handler.heading, a_dir)
            ok, msg, actual_distance = self.data_handler.move(geo_ang, self.data_handler.move_step)
            info["msg"] = msg
            info["actual_distance"] = actual_distance
            # 保存理论坐标
            self.current_theoretical_lat = self.data_handler.current_lat
            self.current_theoretical_lng = self.data_handler.current_lng
            info["theoretical_lat"] = self.current_theoretical_lat
            info["theoretical_lng"] = self.current_theoretical_lng
            reward = 0.0 if ok else -1.0
        elif a_type == "turn":
            # 转向动作：仅调整朝向，不修改坐标
            delta = -TURN_ANGLE if a_dir == "left" else TURN_ANGLE
            self.data_handler.turn(delta)
            info["msg"] = f"Turned {a_dir.upper()} {abs(delta)}°"
        elif a_type == "fov":
            # FOV缩放：仅调整视角，不修改坐标
            current_fov = self.data_handler.fov if hasattr(self.data_handler, 'fov') else 30
            if a_dir == "zoom_in":
                new_fov = max(10, current_fov - FOV_STEP)
            else:
                new_fov = min(120, current_fov + FOV_STEP)
            self.data_handler.fov = new_fov
            info["msg"] = f"FOV {a_dir.upper()} to {new_fov}° (视角{'放大' if a_dir == 'zoom_in' else '缩小'})"
        elif a_type == "pitch":
            # 俯仰调整：仅调整角度，不修改坐标
            current_pitch = self.data_handler.pitch
            if a_dir == "up":
                new_pitch = min(90, current_pitch + PITCH_STEP)
            else:
                new_pitch = max(-90, current_pitch - PITCH_STEP)
            self.data_handler.pitch = new_pitch
            info["msg"] = f"Pitch {a_dir.upper()} to {new_pitch}° ({'抬头' if a_dir == 'up' else '低头'})"

        # ========== 2. 获取图像（所有动作都需要更新图像） ==========
        try:
            self.current_image, actual_lat, actual_lng = self.data_handler.get_street_view_image()
            
            # 保存图像路径到历史记录
            if self.current_image:
                img_path, img_b64 = save_streetview_image(self.current_image, self.current_step, self.img_save_dir)
                self.image_path_history[self.current_step] = img_path
                self.image_b64_history[self.current_step] = img_b64

            # ========== 3. 仅在移动动作时执行坐标修正 ==========
            if a_type == "move" and actual_lat and actual_lng:
                # 保存实际坐标
                self.current_actual_lat = actual_lat
                self.current_actual_lng = actual_lng
                self.current_correction_diff = calculate_gps_distance(
                    self.current_theoretical_lat, self.current_theoretical_lng,
                    actual_lat, actual_lng
                )
                # 记录修正前后的差异
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

                # 更新为实际采样点坐标
                self.data_handler.current_lat = actual_lat
                self.data_handler.current_lng = actual_lng

                # 补充info信息
                info["actual_lat"] = actual_lat
                info["actual_lng"] = actual_lng
                info["coord_correction_diff_m"] = self.current_correction_diff
        except Exception as e:
            self.current_image = None
            reward = -1.0
            info["error"] = f"获取图像失败: {str(e)}"
        
        # 记录所有操作到轨迹历史
        if not hasattr(self, 'trajectory_history'):
            self.trajectory_history = []
        
        # 确定动作描述
        if a_type == "move":
            action_desc = f"Move {a_dir} {info.get('actual_distance', 0):.2f}m"
        elif a_type == "turn":
            delta = -TURN_ANGLE if a_dir == "left" else TURN_ANGLE
            action_desc = f"Turn {a_dir} {abs(delta)}°"
        elif a_type == "fov":
            new_fov = self.data_handler.fov if hasattr(self.data_handler, 'fov') else 30
            action_desc = f"FOV {a_dir} to {new_fov}°"
        elif a_type == "pitch":
            action_desc = f"Pitch {a_dir} to {self.data_handler.pitch}°"
        else:
            action_desc = f"{a_type} {a_dir}"
        
        # 记录轨迹点
        self.trajectory_history.append({
            "step": self.current_step,
            "lat": self.data_handler.current_lat,
            "lng": self.data_handler.current_lng,
            "heading": self.data_handler.heading,
            "fov": self.data_handler.fov if hasattr(self.data_handler, 'fov') else 30,
            "pitch": self.data_handler.pitch,
            "action": action_desc
        })

        # ========== 4. 检查是否达到最大步数 ==========
        if self.current_step >= self.max_steps:
            truncated = True
            self._force_final_prediction()

        return self.get_observation(), reward, terminated, truncated, info

    # ====================== 完全原版最终预测提示词，一字未改 ======================
    def _force_final_prediction(self):
        if not self.current_image:
            self.final_prediction = "No image for final prediction"
            return

        try:
            final_step = self.max_steps + 1
            img_path, img_b64 = save_streetview_image(self.current_image, final_step, self.img_save_dir)
            self.image_b64_history[final_step] = img_b64

            full_hist = "\n".join(self.action_history) if self.action_history else "No actions"

            # ========== 你原版最终预测提示词，完全保留 ==========
            force_prompt = f"""
            MAX NAVIGATION STEPS ({self.max_steps}) REACHED! YOU MUST MAKE A FINAL PREDICTION OF THE INITIAL GPS COORDINATES.

            # CONTEXT (USE ALL INFORMATION BELOW FOR PREDICTION):
            1. Navigation History (all actions taken):
            {full_hist}

            2. Visual Data: You will receive ALL street view images from the entire navigation process (Step 1 to Step {final_step}).
            3. Movement Rules: {self.data_handler.move_step}m per move | {TURN_ANGLE}° per turn | All moves are relative to current view direction.

            # STRICT RULES (VIOLATION = INVALID PREDICTION):
            1. NO geometric/spatial calculations - ONLY use visual clues from ALL street view images and navigation history.
            2. Synthesize information from ALL images (not just the last one) - look for consistent landmarks, street signs, architecture styles, road patterns.
            3. Base your prediction on identifiable visual features (landmarks, buildings, street signs, city architecture, road style, language on signs).
            4. Do NOT guess randomly - use the most obvious and consistent visual clues across all images.
            5. 🛑 NAVIGATION CONSTRAINT REVIEW: All movements were done **ALONG THE ROAD ONLY**, no off-road, no crossing buildings/walls.

            # OUTPUT FORMAT (EXACTLY THIS, NO OTHER TEXT, NO SYMBOLS, NO EXTRA LINES):
            "predicted_initial_coords: [LATITUDE],[LONGITUDE] | reasoning: [DETAILED VISUAL CLUES FROM ALL IMAGES]"
            - [LATITUDE/LONGITUDE]: Valid GPS float numbers (e.g., 40.7580,-73.9855)
            - [REASONING]: Specific visual features from MULTIPLE images that led to your prediction (2-4 concise sentences)
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

    def reset(self, seed=None, options=None, skip_image=False):
        super().reset(seed=seed)
        self.current_step = 0
        self.final_prediction = ""
        self.action_history = []
        self.image_path_history = {}
        self.image_b64_history = {}
        self.coord_correction_history = []  # 重置坐标修正历史
        self.trajectory_history = []  # 重置轨迹历史
        # 重置临时坐标记录
        self.current_theoretical_lat = None
        self.current_theoretical_lng = None
        self.current_actual_lat = None
        self.current_actual_lng = None
        self.current_correction_diff = None
        self.data_handler.init_position(self.data_handler.initial_lat, self.data_handler.initial_lng)
        if not skip_image:
            try:
                # 重置时获取初始图像（仅获取，不修正坐标，保持初始传入的GPS）
                self.current_image, _, _ = self.data_handler.get_street_view_image()
                # 保存初始图像到历史记录
                if self.current_image:
                    img_path, img_b64 = save_streetview_image(self.current_image, 0, self.img_save_dir)
                    self.image_path_history[0] = img_path
                    self.image_b64_history[0] = img_b64
            except:
                self.current_image = None
        else:
            self.current_image = None
        return self.get_observation()

    def render(self, mode="human"):
        if mode == "human" and self.current_image:
            self.current_image.show(title=f"Step {self.current_step}")
