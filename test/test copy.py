import os
import csv
import math
import time
import base64
import traceback
import warnings
import requests
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from PIL import Image
from io import BytesIO
from typing import Tuple, Optional, Dict, Any, List
from openai import OpenAI
from dotenv import load_dotenv

# ====================== 屏蔽警告 ==============================
warnings.filterwarnings("ignore", category=UserWarning, message="Gym has been unmaintained")
warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()

# ====================== 全局配置 ==============================
BATCH_ROOT_DIR = "batch_streetview_results"
GLOBAL_MAX_STEPS = 15
GLOBAL_MOVE_STEP = 15.0
GLOBAL_MAX_RETRIES = 3
# 新增：FOV/俯仰控制参数
FOV_STEP = 20
PITCH_STEP = 10
TURN_ANGLE = 45
FOV_DEFAULT = 60
# 新增：Google API参数
GOOGLE_API_RADIUS = 10
GPS_CSV_PATH = "1.csv"

# 当前批处理目录
CURRENT_BATCH_DIR = None


# ====================== 读取GPS CSV ======================
def read_gps_targets_from_csv(csv_path: str) -> List[Dict]:
    gps_targets = []
    if not os.path.exists(csv_path):
        print(f"❌ GPS CSV文件不存在: {os.path.abspath(csv_path)}")
        raise FileNotFoundError(f"GPS CSV file not found: {csv_path}")

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_headers = ["name", "lat", "lng"]
        if not all(h in reader.fieldnames for h in required_headers):
            raise ValueError(f"CSV必须包含表头: {required_headers}")

        for line_num, row in enumerate(reader, start=2):
            try:
                name = row["name"].strip()
                lat_str = row["lat"].strip()
                lng_str = row["lng"].strip()
                if not name or not lat_str or not lng_str:
                    continue

                lat = float(lat_str)
                lng = float(lng_str)
                if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
                    continue

                max_steps = int(row["max_steps"].strip()) if row["max_steps"].strip() else None
                move_step = float(row["move_step"].strip()) if row["move_step"].strip() else None

                point_info = {"name": name, "lat": lat, "lng": lng}
                if max_steps:
                    point_info["max_steps"] = max_steps
                if move_step:
                    point_info["move_step"] = move_step

                gps_targets.append(point_info)
            except:
                continue

    if not gps_targets:
        raise ValueError("CSV中无有效GPS点位")
    return gps_targets


BATCH_GPS_TARGETS = read_gps_targets_from_csv(GPS_CSV_PATH)


# ====================== 重试装饰器 ======================
def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries >= max_retries:
                        raise
                    time.sleep(delay)
            return None

        return wrapper

    return decorator


# ====================== 文件工具 ======================
def init_batch_dir():
    global CURRENT_BATCH_DIR
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    CURRENT_BATCH_DIR = os.path.join(BATCH_ROOT_DIR, f"{timestamp}_batch_run")
    os.makedirs(CURRENT_BATCH_DIR, exist_ok=True)
    return CURRENT_BATCH_DIR


def get_point_dir_name(point_info: Dict) -> str:
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    lat_str = f"{point_info['lat']:.4f}".replace(".", "_")
    lng_str = f"{point_info['lng']:.4f}".replace(".", "_")
    return f"{timestamp}_{point_info['name']}_lat{lat_str}_lng{lng_str}"


def init_point_dir(point_info: Dict) -> Tuple[str, str]:
    global CURRENT_BATCH_DIR
    if CURRENT_BATCH_DIR is None:
        # 如果CURRENT_BATCH_DIR未初始化，使用默认目录
        point_dir = os.path.join(BATCH_ROOT_DIR, get_point_dir_name(point_info))
    else:
        # 使用当前批处理目录
        point_dir = os.path.join(CURRENT_BATCH_DIR, get_point_dir_name(point_info))
    img_dir = os.path.join(point_dir, "streetview_images")
    os.makedirs(img_dir, exist_ok=True)
    return point_dir, img_dir


def image_to_base64(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format='JPEG', quality=100)  # 提升Base64编码的图片质量
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def save_streetview_image(image: Image.Image, step_num: int, img_save_dir: str) -> Tuple[str, str]:
    img_path = os.path.join(img_save_dir, f"step_{step_num}_streetview.jpg")
    # 提升保存质量：100%无压缩 + 关闭色度子采样
    image.save(
        img_path,
        format='JPEG',
        quality=100,
        subsampling=0,
        optimize=False
    )
    return img_path, image_to_base64(image)


# ====================== 地理角度工具 ======================
def get_geo_angle(current_heading: float, relative_dir: str) -> float:
    angle_map = {
        "forward": current_heading,
        "left": current_heading - 90,
        "right": current_heading + 90,
        "backward": current_heading + 180
    }
    return angle_map[relative_dir] % 360


# ====================== Google Street View ======================
class GoogleStreetViewData:
    def __init__(self, api_key: str = None, max_retries: int = 3, move_step: float = GLOBAL_MOVE_STEP):
        self.google_api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.static_url = "https://maps.googleapis.com/maps/api/streetview"
        self.metadata_url = "https://maps.googleapis.com/maps/api/streetview/metadata"
        self.max_retries = max_retries
        self.initial_lat = self.initial_lng = 0.0
        self.initial_heading = 0.0
        self.initial_pitch = 0.0
        self.initial_fov = FOV_DEFAULT
        self.current_lat = self.current_lng = 0.0
        self.heading = 0.0
        self.pitch = 0.0
        self.move_step = move_step
        self.fov = FOV_DEFAULT

    def init_position(self, lat: float, lng: float, heading: float = 0.0, fov: float = FOV_DEFAULT, pitch: float = 0.0):
        self.initial_lat = lat
        self.initial_lng = lng
        self.initial_heading = heading
        self.initial_pitch = pitch
        self.initial_fov = fov
        self.current_lat = lat
        self.current_lng = lng
        self.heading = heading
        self.pitch = pitch
        self.fov = fov

    @retry_on_failure(3, 1.0)
    def get_metadata(self, lat: float, lng: float) -> Optional[Dict]:
        params = {"location": f"{lat},{lng}", "radius": GOOGLE_API_RADIUS, "key": self.google_api_key}
        r = requests.get(self.metadata_url, params=params, timeout=10)
        data = r.json()
        if data.get("status") == "OK" and "location" in data:
            # 检查实际街景点距离理论坐标的距离
            actual_lat = data["location"]["lat"]
            actual_lng = data["location"]["lng"]
            distance = calculate_gps_distance(lat, lng, actual_lat, actual_lng)
            if distance <= GOOGLE_API_RADIUS:
                return data
        return None

    def calculate_new_coords(self, move_direction: float) -> Tuple[float, float]:
        R = 6378137.0
        lat_r = math.radians(self.current_lat)
        lng_r = math.radians(self.current_lng)
        ang_r = math.radians(move_direction)
        d = self.move_step

        new_lat_r = math.asin(
            math.sin(lat_r) * math.cos(d / R) +
            math.cos(lat_r) * math.sin(d / R) * math.cos(ang_r)
        )
        new_lng_r = lng_r + math.atan2(
            math.sin(ang_r) * math.sin(d / R) * math.cos(lat_r),
            math.cos(d / R) - math.sin(lat_r) * math.sin(new_lat_r)
        )
        return math.degrees(new_lat_r), math.degrees(new_lng_r)

    def turn(self, delta_angle: float):
        self.heading = (self.heading + delta_angle) % 360

    @retry_on_failure(3, 1.0)
    def get_street_view_image(self) -> Tuple[Optional[Image.Image], Optional[float], Optional[float]]:
        """
        修改：返回图像 + 实际采样点的GPS坐标
        return: (图像, 实际lat, 实际lng)
        """
        params = {
            "location": f"{self.current_lat},{self.current_lng}",
            "heading": self.heading,
            "pitch": self.pitch,
            "fov": self.fov,
            "radius": GOOGLE_API_RADIUS,
            "size": "640x640",
            "key": self.google_api_key
        }
        # 1. 先获取元数据（包含实际采样点坐标）
        metadata = self.get_metadata(self.current_lat, self.current_lng)
        actual_lat, actual_lng = None, None
        if metadata and "location" in metadata:
            actual_lat = metadata["location"]["lat"]
            actual_lng = metadata["location"]["lng"]

        # 2. 获取街景图像
        r = requests.get(self.static_url, params=params, stream=True, timeout=10)
        image = Image.open(BytesIO(r.content)).convert("RGB")

        return image, actual_lat, actual_lng

    def move(self, direction: float, target_distance: float = None) -> Tuple[bool, str, float]:
        """
        支持分段移动的move方法，兼容原有调用逻辑
        :param direction: 移动方向（地理角度）
        :param target_distance: 目标移动距离（可选，默认用self.move_step）
        :return: (是否成功, 提示信息, 实际移动距离)
        """
        # 用传入的目标距离，无则用self.move_step
        target_distance = target_distance or self.move_step
        # 计算新坐标（用目标距离）
        R = 6378137.0
        lat_r = math.radians(self.current_lat)
        lng_r = math.radians(self.current_lng)
        ang_r = math.radians(direction)
        d = target_distance

        new_lat_r = math.asin(
            math.sin(lat_r) * math.cos(d / R) +
            math.cos(lat_r) * math.sin(d / R) * math.cos(ang_r)
        )
        new_lng_r = lng_r + math.atan2(
            math.sin(ang_r) * math.sin(d / R) * math.cos(lat_r),
            math.cos(d / R) - math.sin(lat_r) * math.sin(new_lat_r)
        )
        nlat, nlng = math.degrees(new_lat_r), math.degrees(new_lng_r)

        # 检查是否有街景
        if not self.get_metadata(nlat, nlng):
            # 移动失败，保持当前坐标不变
            return False, "No street view available", 0.0

        # 计算实际移动距离
        actual_distance = calculate_gps_distance(
            self.current_lat, self.current_lng,
            nlat, nlng
        )
        # 更新坐标（先更新为理论计算值）
        self.current_lat, self.current_lng = nlat, nlng
        return True, "Move successful", actual_distance


# ====================== StreetViewAutoLocEnv ======================
class StreetViewAutoLocEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}
    ACTION_MAP = {
        "move forward": ("move", "forward"),
        "move left": ("move", "left"),
        "move right": ("move", "right"),
        "move backward": ("move", "backward"),
        "turn left": ("turn", "left"),
        "turn right": ("turn", "right"),
        "zoom in": ("fov", "zoom_in"),  # 新增：FOV缩小（放大视角）
        "zoom out": ("fov", "zoom_out"),  # 新增：FOV增大（缩小视角）
        "look up": ("pitch", "up"),  # 新增：抬头（pitch增加）
        "look down": ("pitch", "down")  # 新增：低头（pitch减少）
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

        self.action_space = spaces.Discrete(len(self.ACTION_MAP))
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
        self.reason_history = []  # 新增：存储动作原因历史
        self.image_path_history = {}
        self.image_b64_history = {}
        # 新增：记录理论坐标和实际坐标的差异
        self.coord_correction_history = []
        # 新增：临时存储当前步的理论/实际坐标（用于手动控制显示）
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

        # ========== Improved prompt for exploration and decision-making ==========
        prompt = f"""
        Your Role: Autonomous Google Street View Navigator & Visual Location Predictor
        CORE MISSION: Predict the GPS coordinates of your INITIAL starting point using ONLY VISUAL FEATURES.
        
        CONTEXT:
        - Action parameters: {self.data_handler.move_step}m per move | {TURN_ANGLE}° per turn | FOV step: {FOV_STEP}° | Pitch step: {PITCH_STEP}°
        - Current heading: {self.data_handler.heading:.1f}°
        - Taken actions: {action_history_str}
        - Progress: {self.current_step}/{self.max_steps}
        
        RULES:
        1. No GPS coordinates provided. 
        2. All decisions must be based SOLELY on visual clues from the street view image.
        3. All MOVE actions are RELATIVE to your CURRENT VIEW DIRECTION (not fixed geographic north/south).
        4. Only move ALONG THE ROAD. Do NOT walk into buildings, grass, walls, rivers, or off-road.
        
        ACTION OPTIONS:
        "move forward" = Move FORWARD {self.data_handler.move_step}m (current view front)
        "move left" = Move LEFT {self.data_handler.move_step}m (current view left side)
        "move right" = Move RIGHT {self.data_handler.move_step}m (current view right side)
        "move backward" = Move BACKWARD {self.data_handler.move_step}m (current view back)
        "turn left" = Turn LEFT {TURN_ANGLE}° (view left, no movement)
        "turn right" = Turn RIGHT {TURN_ANGLE}° (view right, no movement)
        "zoom in" = Zoom IN (FOV-{FOV_STEP}°)
        "zoom out" = Zoom OUT (FOV+{FOV_STEP}°)
        "look up" = Look UP (Pitch+{PITCH_STEP}°)
        "look down" = Look DOWN (Pitch-{PITCH_STEP}°)
        
        OUTPUT FORMAT:
        Please output in the following JSON format, including both your detailed reasoning and the action name:
        {{
          "reason": "[detailed_reasoning_for_your_decision]",
          "action": "[action_name]"
        }}
        
        You will be forced to make a final GPS prediction when max navigation steps are reached.
        """

        # 构建包含所有历史图像的内容
        content = [{"type": "text", "text": prompt}]
        
        # 添加之前所有的图像
        for step in sorted(self.image_b64_history):
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{self.image_b64_history[step]}", "detail": "high"}
            })
            content.append({"type": "text", "text": f"Image from Step {step}"})
        
        # 添加当前图像
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
        import re
        import json
        
        # 尝试解析JSON格式
        try:
            # 移除可能的代码块标记
            cleaned_response = gpt_response.strip()
            # 移除```json和```包围
            if cleaned_response.startswith('```json') and cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[7:-3].strip()
            # 移除```和```包围
            elif cleaned_response.startswith('```') and cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[3:-3].strip()
            
            response_data = json.loads(cleaned_response)
            action = response_data.get("action", "turn right")
            reason = response_data.get("reason", "No reason provided")
            # 确保动作在ACTION_MAP中
            if action not in self.ACTION_MAP:
                action = "turn right"
            return action, reason
        except (json.JSONDecodeError, ValueError):
            # 如果JSON解析失败，尝试匹配动作名称
            action_names = list(self.ACTION_MAP.keys())
            for action_name in action_names:
                if action_name in gpt_response.lower():
                    return action_name, "No reason provided"
            return "turn right", "Failed to parse response"

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

    def step(self, action: str) -> Tuple[Dict, float, bool, bool, Dict]:
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
        current_fov = self.data_handler.fov if hasattr(self.data_handler, 'fov') else FOV_DEFAULT
        current_pitch = self.data_handler.pitch

        # ========== 1. 执行对应动作 ==========
        if a_type == "move":
            # 移动动作：计算理论GPS并更新
            geo_ang = get_geo_angle(self.data_handler.heading, a_dir)
            ok, msg, actual_distance = self.data_handler.move(geo_ang, self.data_handler.move_step)
            info["msg"] = msg
            info["actual_distance"] = actual_distance
            # 只有移动成功才保存理论坐标
            if ok:
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
            current_fov = self.data_handler.fov if hasattr(self.data_handler, 'fov') else FOV_DEFAULT
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
            if a_type == "move" and ok and actual_lat and actual_lng and self.current_theoretical_lat:
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
            "fov": self.data_handler.fov if hasattr(self.data_handler, 'fov') else FOV_DEFAULT,
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
            full_hist = "\n".join(self.action_history) if self.action_history else "No actions"
            
            # 计算实际的最大步骤
            actual_max_step = max(self.image_b64_history.keys()) if self.image_b64_history else 0

            # ========== 你原版最终预测提示词，完全保留 ==========
            force_prompt = f"""
            MAX NAVIGATION STEPS ({self.max_steps}) REACHED! YOU MUST MAKE A FINAL PREDICTION OF THE INITIAL GPS COORDINATES.

            # CONTEXT (USE ALL INFORMATION BELOW FOR PREDICTION):
            1. Navigation History (all actions taken):
            {full_hist}
            2. Visual Data: You will receive ALL street view images from the entire navigation process (Step 0 to Step {actual_max_step}).
            3. Movement Rules: {self.data_handler.move_step}m per move | {TURN_ANGLE}° per turn | All moves are relative to current view direction.

            # STRICT RULES (VIOLATION = INVALID PREDICTION):
            1. NAVIGATION CONSTRAINT REVIEW: All movements were done **ALONG THE ROAD ONLY**, no off-road, no crossing buildings/walls.

            # OUTPUT FORMAT (EXACTLY THIS, NO OTHER TEXT, NO SYMBOLS, NO EXTRA LINES):
            "predicted_initial_coords: [LATITUDE],[LONGITUDE] | reasoning: [DETAILED VISUAL CLUES FROM ALL IMAGES]"
            - [LATITUDE/LONGITUDE]: Valid GPS float numbers (e.g., 40.7580,-73.9855)
            - [REASONING]: Specific visual features from MULTIPLE images that led to your prediction, clearly indicating which step each clue was observed in (e.g., "In Step 0, I saw...; In Step 2, I noticed...")
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
        return self.get_observation()

    def render(self, mode="human"):
        if mode == "human" and self.current_image:
            self.current_image.show(title=f"Step {self.current_step}")


# ====================== 辅助工具：计算两点GPS距离 ======================
def calculate_gps_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """
    用Haversine公式计算两点GPS坐标的实际距离（米）
    """
    R = 6378137.0  # 地球赤道半径（米）
    lat1_rad = math.radians(lat1)
    lng1_rad = math.radians(lng1)
    lat2_rad = math.radians(lat2)
    lng2_rad = math.radians(lng2)

    dlat = lat2_rad - lat1_rad
    dlng = lng2_rad - lng1_rad

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlng / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c  # 米
    return distance


# ====================== 生成可视化HTML页面 ======================
def generate_visualization(env, point_dir):
    """
    生成HTML页面，可视化轨迹和图像
    """
    # 收集轨迹数据
    trajectory = []
    # 添加初始位置
    trajectory.append({
        "step": 0,
        "lat": env.data_handler.initial_lat,
        "lng": env.data_handler.initial_lng,
        "heading": env.data_handler.initial_heading,
        "fov": env.data_handler.initial_fov,
        "pitch": env.data_handler.initial_pitch,
        "action": "Initial position"
    })
    
    # 添加所有操作的轨迹点
    if hasattr(env, 'trajectory_history'):
        for item in env.trajectory_history:
            trajectory.append(item)
    
    # 收集图像数据
    images = []
    for step, img_path in env.image_path_history.items():
        images.append({
            "step": step,
            "path": img_path
        })
    
    # 获取最终预测结果
    final_prediction = env.final_prediction if hasattr(env, 'final_prediction') else "No prediction available"
    
    # 计算预测偏差
    prediction_error = ""
    if final_prediction:
        try:
            # 尝试从预测结果中提取GPS坐标
            import re
            # 尝试匹配多种可能的坐标格式
            patterns = [
                r'predicted_initial_coords: ([\d.-]+),([\d.-]+)',  # 原始格式
                r'([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)',  # 简单的经纬度格式
                r'lat[\s:]*([-+]?\d*\.?\d+)[^,]*lng[\s:]*([-+]?\d*\.?\d+)',  # 带标签的格式
                r'latitude[\s:]*([-+]?\d*\.?\d+)[^,]*longitude[\s:]*([-+]?\d*\.?\d+)'  # 完整标签格式
            ]
            
            pred_lat = None
            pred_lng = None
            
            for pattern in patterns:
                match = re.search(pattern, final_prediction)
                if match:
                    pred_lat = float(match.group(1))
                    pred_lng = float(match.group(2))
                    break
            
            if pred_lat is not None and pred_lng is not None:
                # 计算与实际初始坐标的距离
                actual_lat = env.data_handler.initial_lat
                actual_lng = env.data_handler.initial_lng
                distance = calculate_gps_distance(pred_lat, pred_lng, actual_lat, actual_lng)
                prediction_error = f"Prediction Error: {distance:.2f} meters"
            else:
                prediction_error = "Could not extract GPS coordinates from prediction"
        except Exception as e:
            prediction_error = f"Error calculating prediction error: {str(e)}"
    else:
        prediction_error = "No prediction available"
    
    # 生成HTML内容
    html_content = f'''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Street View Navigation Visualization</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            text-align: center;
            color: #333;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        /* ========== 动画播放器样式 ========== */
        .animation-player {{
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
            margin-bottom: 30px;
            overflow: hidden;
        }}
        .player-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .player-title {{
            font-size: 18px;
            font-weight: bold;
        }}
        .player-controls {{
            display: flex;
            gap: 10px;
            align-items: center;
        }}
        .control-btn {{
            background: rgba(255,255,255,0.2);
            border: none;
            color: white;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            transition: background 0.2s;
        }}
        .control-btn:hover {{
            background: rgba(255,255,255,0.3);
        }}
        .control-btn.active {{
            background: rgba(255,255,255,0.4);
        }}
        .step-indicator {{
            background: rgba(255,255,255,0.2);
            padding: 8px 16px;
            border-radius: 6px;
            font-size: 14px;
        }}
        .player-content {{
            display: flex;
            height: 500px;
        }}
        .map-view {{
            flex: 1;
            position: relative;
            border-right: 2px solid #eee;
        }}
        #animationMap {{
            width: 100%;
            height: 100%;
        }}
        .map-overlay {{
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(255,255,255,0.95);
            padding: 10px 15px;
            border-radius: 8px;
            font-size: 13px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .map-overlay div {{
            margin: 3px 0;
        }}
        .streetview-view {{
            flex: 1;
            position: relative;
            background: #000;
        }}
        .streetview-view img {{
            width: 100%;
            height: 100%;
            object-fit: contain;
        }}
        .streetview-overlay {{
            position: absolute;
            bottom: 10px;
            left: 10px;
            background: rgba(0,0,0,0.8);
            color: white;
            padding: 10px 15px;
            border-radius: 8px;
            font-size: 13px;
        }}
        .player-footer {{
            padding: 15px 20px;
            background: #f8f9fa;
            border-top: 1px solid #eee;
        }}
        .timeline-slider {{
            width: 100%;
            height: 8px;
            -webkit-appearance: none;
            background: #ddd;
            border-radius: 4px;
            outline: none;
            cursor: pointer;
        }}
        .timeline-slider::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 20px;
            height: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 50%;
            cursor: pointer;
            box-shadow: 0 2px 6px rgba(0,0,0,0.2);
        }}
        .timeline-labels {{
            display: flex;
            justify-content: space-between;
            margin-top: 8px;
            font-size: 12px;
            color: #666;
        }}
        /* ========== 原有样式 ========== */
        .map-container {{
            width: 100%;
            height: 500px;
            margin-bottom: 20px;
            border: 1px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
        }}
        #map {{
            width: 100%;
            height: 100%;
        }}
        .timeline {{
            margin-top: 30px;
        }}
        .timeline h2 {{
            color: #333;
            margin-bottom: 20px;
        }}
        .step {{
            display: flex;
            margin-bottom: 30px;
            border-bottom: 1px solid #eee;
            padding-bottom: 20px;
        }}
        .step-info {{
            flex: 1;
            padding-right: 20px;
        }}
        .step-image {{
            flex: 1;
        }}
        .step-image img {{
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .step-number {{
            font-size: 24px;
            font-weight: bold;
            color: #4CAF50;
            margin-bottom: 10px;
        }}
        .step-details {{
            color: #666;
        }}
        .step-details p {{
            margin: 5px 0;
        }}
        .final-prediction {{
            margin-top: 30px;
            padding: 20px;
            background-color: #f9f9f9;
            border-radius: 8px;
            border: 1px solid #ddd;
        }}
        .final-prediction h2 {{
            color: #333;
            margin-bottom: 15px;
        }}
        .prediction-content {{
            background-color: white;
            padding: 15px;
            border-radius: 4px;
            border: 1px solid #eee;
        }}
        .prediction-content pre {{
            white-space: pre-wrap;
            font-family: Arial, sans-serif;
            color: #555;
            line-height: 1.5;
        }}
        .prediction-error {{
            margin-top: 15px;
            padding: 10px;
            background-color: #f0f8ff;
            border-left: 4px solid #1e90ff;
            border-radius: 4px;
        }}
        .prediction-error strong {{
            color: #1e90ff;
        }}
    </style>
    <script src="https://maps.googleapis.com/maps/api/js?key=AIzaSyAXmbPK8IXtGV9fjWVI8K4b5cw9Tshrvy0"></script>
</head>
<body>
    <div class="container">
        <h1>Street View Navigation Visualization</h1>
        
        <!-- ========== 动画播放器 ========== -->
        <div class="animation-player">
            <div class="player-header">
                <div class="player-title">🎬 Navigation Animation</div>
                <div class="player-controls">
                    <span class="step-indicator" id="stepIndicator">Step 0 / {{len(trajectory) - 1}}</span>
                    <button class="control-btn" id="rewindBtn" title="Rewind">⏮</button>
                    <button class="control-btn" id="prevBtn" title="Previous">◀</button>
                    <button class="control-btn active" id="playPauseBtn" title="Play/Pause">▶</button>
                    <button class="control-btn" id="nextBtn" title="Next">▶</button>
                    <button class="control-btn" id="fastForwardBtn" title="Fast Forward">⏭</button>
                </div>
            </div>
            <div class="player-content">
                <div class="map-view">
                    <div id="animationMap"></div>
                    <div class="map-overlay">
                        <div><strong>GPS:</strong> <span id="mapGps">-</span></div>
                        <div><strong>Heading:</strong> <span id="mapHeading">-</span></div>
                        <div><strong>FOV:</strong> <span id="mapFov">-</span></div>
                        <div><strong>Pitch:</strong> <span id="mapPitch">-</span></div>
                        <div><strong>Action:</strong> <span id="mapAction">-</span></div>
                    </div>
                </div>
                <div class="streetview-view">
                    <img id="streetviewImage" src="streetview_images/step_0_streetview.jpg" alt="Street View">
                    <div class="streetview-overlay">
                        <div id="streetviewStep">Step 0</div>
                        <div id="streetviewAction">Initial position</div>
                    </div>
                </div>
            </div>
            <div class="player-footer">
                <input type="range" class="timeline-slider" id="timelineSlider" min="0" max="{{len(trajectory) - 1}}" value="0">
                <div class="timeline-labels">
                    <span>Start</span>
                    <span>End (Prediction)</span>
                </div>
            </div>
        </div>
        
        <div class="map-container">
            <div id="map"></div>
        </div>
        
        <div class="timeline">
            <h2>Navigation Timeline</h2>
            {{timeline_html}}
        </div>
        
        <div class="final-prediction">
            <h2>Final Prediction</h2>
            <div class="prediction-content">
                <pre>{{final_prediction}}</pre>
                <div class="prediction-error">
                    <strong>{{prediction_error}}</strong>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // 轨迹数据
        const trajectory = {{trajectory_data}};
        
        // 动画播放器状态
        let currentStep = 0;
        let isPlaying = false;
        let playInterval = null;
        let animationMap = null;
        let positionMarker = null;
        let fovPolygon = null;
        let trajectoryPolyline = null;
        
        // 初始化动画地图
        function initAnimationMap() {{
            const center = new google.maps.LatLng(trajectory[0].lat, trajectory[0].lng);
            animationMap = new google.maps.Map(document.getElementById('animationMap'), {{
                zoom: 19,
                center: center,
                mapTypeId: google.maps.MapTypeId.SATELLITE
            }});
            
            // 绘制完整轨迹线
            const path = trajectory.map(p => new google.maps.LatLng(p.lat, p.lng));
            trajectoryPolyline = new google.maps.Polyline({{
                path: path,
                geodesic: true,
                strokeColor: '#FF0000',
                strokeOpacity: 0.6,
                strokeWeight: 3
            }});
            trajectoryPolyline.setMap(animationMap);
            
            // 初始化位置标记和FOV
            updateAnimationView(0);
        }}
        
        // 更新动画视图
        function updateAnimationView(step) {{
            currentStep = step;
            const point = trajectory[step];
            
            // 更新位置标记
            const pos = new google.maps.LatLng(point.lat, point.lng);
            
            if (!positionMarker) {{
                positionMarker = new google.maps.Marker({{
                    position: pos,
                    map: animationMap,
                    icon: {{
                        path: google.maps.SymbolPath.FORWARD_CLOSED_ARROW,
                        scale: 6,
                        fillColor: '#4CAF50',
                        fillOpacity: 1,
                        strokeColor: '#fff',
                        strokeWeight: 2,
                        rotation: point.heading
                    }},
                    title: `Step ${{point.step}}`
                }});
            }} else {{
                positionMarker.setPosition(pos);
                positionMarker.setIcon({{
                    path: google.maps.SymbolPath.FORWARD_CLOSED_ARROW,
                    scale: 6,
                    fillColor: '#4CAF50',
                    fillOpacity: 1,
                    strokeColor: '#fff',
                    strokeWeight: 2,
                    rotation: point.heading
                }});
            }}
            
            // 更新FOV锥形区域
            updateFovCone(point);
            
            // 更新地图overlay信息
            document.getElementById('mapGps').textContent = `${{point.lat.toFixed(6)}}, ${{point.lng.toFixed(6)}}`;
            document.getElementById('mapHeading').textContent = `${{point.heading.toFixed(1)}}°`;
            document.getElementById('mapFov').textContent = `${{point.fov.toFixed(1)}}°`;
            document.getElementById('mapPitch').textContent = `${{point.pitch.toFixed(1)}}°`;
            document.getElementById('mapAction').textContent = point.action;
            
            // 更新街景图像
            document.getElementById('streetviewImage').src = `streetview_images/step_${{point.step}}_streetview.jpg`;
            document.getElementById('streetviewStep').textContent = `Step ${{point.step}}`;
            document.getElementById('streetviewAction').textContent = point.action;
            
            // 更新步骤指示器
            document.getElementById('stepIndicator').textContent = `Step ${{step}} / ${{trajectory.length - 1}}`;
            
            // 更新滑块
            document.getElementById('timelineSlider').value = step;
            
            // 居中地图
            animationMap.setCenter(pos);
        }}
        
        // 更新FOV锥形区域
        function updateFovCone(point) {{
            const center = new google.maps.LatLng(point.lat, point.lng);
            const heading = point.heading;
            const fov = point.fov;
            const distance = 0.0002; // 约20米
            
            // 计算FOV左右边界角度
            const halfFov = fov / 2;
            const leftAngle = heading - halfFov;
            const rightAngle = heading + halfFov;
            
            // 转换为弧度并计算点
            const centerLat = point.lat;
            const centerLng = point.lng;
            
            function calcCoord(angle, dist) {{
                const R = 6378137;
                const d = dist * R;
                const lat1 = centerLat * Math.PI / 180;
                const lng1 = centerLng * Math.PI / 180;
                const ang = angle * Math.PI / 180;
                
                const lat2 = Math.asin(
                    Math.sin(lat1) * Math.cos(d / R) +
                    Math.cos(lat1) * Math.sin(d / R) * Math.cos(ang)
                );
                const lng2 = lng1 + Math.atan2(
                    Math.sin(ang) * Math.sin(d / R) * Math.cos(lat1),
                    Math.cos(d / R) - Math.sin(lat1) * Math.sin(lat2)
                );
                return [lat2 * 180 / Math.PI, lng2 * 180 / Math.PI];
            }}
            
            // 生成FOV多边形点
            const points = [center];
            for (let a = leftAngle; a <= rightAngle; a += 5) {{
                const coord = calcCoord(a, distance);
                points.push(new google.maps.LatLng(coord[0], coord[1]));
            }}
            points.push(center);
            
            // 删除旧的FOV多边形
            if (fovPolygon) {{
                fovPolygon.setMap(null);
            }}
            
            // 创建新的FOV多边形
            fovPolygon = new google.maps.Polygon({{
                paths: points,
                map: animationMap,
                fillColor: '#667eea',
                fillOpacity: 0.4,
                strokeColor: '#667eea',
                strokeWeight: 2
            }});
        }}
        
        // 播放动画
        function playAnimation() {{
            if (isPlaying) return;
            isPlaying = true;
            document.getElementById('playPauseBtn').textContent = '⏸';
            
            playInterval = setInterval(() => {{
                if (currentStep < trajectory.length - 1) {{
                    updateAnimationView(currentStep + 1);
                }} else {{
                    stopAnimation();
                }}
            }}, 1500);
        }}
        
        // 停止动画
        function stopAnimation() {{
            isPlaying = false;
            document.getElementById('playPauseBtn').textContent = '▶';
            if (playInterval) {{
                clearInterval(playInterval);
                playInterval = null;
            }}
        }}
        
        // 事件监听
        document.addEventListener('DOMContentLoaded', () => {{
            initAnimationMap();
            
            document.getElementById('playPauseBtn').addEventListener('click', () => {{
                if (isPlaying) {{
                    stopAnimation();
                }} else {{
                    if (currentStep >= trajectory.length - 1) {{
                        updateAnimationView(0);
                    }}
                    playAnimation();
                }}
            }});
            
            document.getElementById('prevBtn').addEventListener('click', () => {{
                stopAnimation();
                if (currentStep > 0) {{
                    updateAnimationView(currentStep - 1);
                }}
            }});
            
            document.getElementById('nextBtn').addEventListener('click', () => {{
                stopAnimation();
                if (currentStep < trajectory.length - 1) {{
                    updateAnimationView(currentStep + 1);
                }}
            }});
            
            document.getElementById('rewindBtn').addEventListener('click', () => {{
                stopAnimation();
                updateAnimationView(0);
            }});
            
            document.getElementById('fastForwardBtn').addEventListener('click', () => {{
                stopAnimation();
                updateAnimationView(trajectory.length - 1);
            }});
            
            document.getElementById('timelineSlider').addEventListener('input', (e) => {{
                stopAnimation();
                updateAnimationView(parseInt(e.target.value));
            }});
        }});
        
        // 初始化静态地图
        function initMap() {{
            const center = new google.maps.LatLng(trajectory[0].lat, trajectory[0].lng);
            
            const map = new google.maps.Map(document.getElementById('map'), {{
                zoom: 18,
                center: center,
                mapTypeId: google.maps.MapTypeId.SATELLITE
            }});
            
            // 添加标记
            trajectory.forEach((point, index) => {{
                const marker = new google.maps.Marker({{
                    position: new google.maps.LatLng(point.lat, point.lng),
                    map: map,
                    title: `Step ${{point.step}}: ${{point.action}}`
                }});
                
                // 添加信息窗口
                const infoWindow = new google.maps.InfoWindow({{
                    content: `<strong>Step ${{point.step}}</strong><br>
                             Lat: ${{point.lat.toFixed(6)}}<br>
                             Lng: ${{point.lng.toFixed(6)}}<br>
                             Heading: ${{point.heading.toFixed(1)}}°<br>
                             FOV: ${{point.fov.toFixed(1)}}°<br>
                             Pitch: ${{point.pitch.toFixed(1)}}°<br>
                             Action: ${{point.action}}`
                }});
                
                marker.addListener('click', function() {{
                    infoWindow.open(map, marker);
                }});
            }});
            
            // 绘制轨迹线
            const path = trajectory.map(point => new google.maps.LatLng(point.lat, point.lng));
            const polyline = new google.maps.Polyline({{
                path: path,
                geodesic: true,
                strokeColor: '#FF0000',
                strokeOpacity: 1.0,
                strokeWeight: 2
            }});
            
            polyline.setMap(map);
        }}
        
        // 页面加载完成后初始化地图
        window.onload = initMap;
    </script>
</body>
</html>
    '''
    
    # 生成时间线HTML
    timeline_html = ""
    for idx, item in enumerate(trajectory):
        # 获取对应步骤的原因（如果有的话）
        reason = "" 
        if hasattr(env, 'reason_history') and idx > 0 and idx <= len(env.reason_history):
            reason = env.reason_history[idx - 1]
        
        timeline_html += f"""
        <div class="step">
            <div class="step-info">
                <div class="step-number">Step {item['step']}</div>
                <div class="step-details">
                    <p><strong>Action:</strong> {item['action']}</p>
                    {f'<p><strong>Reason:</strong> {reason}</p>' if reason else ''}
                    <p><strong>GPS:</strong> {item['lat']:.6f}, {item['lng']:.6f}</p>
                    <p><strong>Heading:</strong> {item['heading']:.1f}°</p>
                    <p><strong>FOV:</strong> {item['fov']:.1f}°</p>
                    <p><strong>Pitch:</strong> {item['pitch']:.1f}°</p>
                </div>
            </div>
            <div class="step-image">
                <!-- 这里可以添加对应步骤的图像 -->
                <img src="streetview_images/step_{item['step']}_streetview.jpg" alt="Step {item['step']}">
            </div>
        </div>
        """
    
    # 格式化轨迹数据为JavaScript对象
    trajectory_data = str(trajectory).replace("'", "\"")
    
    # 替换HTML模板中的占位符
    html_content = html_content.replace("{timeline_html}", timeline_html)
    html_content = html_content.replace("{trajectory_data}", trajectory_data)
    html_content = html_content.replace("{final_prediction}", final_prediction)
    html_content = html_content.replace("{prediction_error}", prediction_error)
    
    # 保存HTML文件
    html_path = os.path.join(point_dir, "visualization.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"✅ 可视化HTML页面已生成: {html_path}")


# ====================== 手动控制（增强反馈版） ======================
def manual_control(env):
    # 新增FOV/Pitch的指令提示
    print("\n=== 手动操控模式 ===")
    print("移动：W=前 A=左移 D=右移 S=后")
    print("转向：Q=左转 E=右转")
    print("视角：Z=放大 X=缩小")  # 新增FOV指令
    print("俯仰：R=抬头 F=低头")  # 新增Pitch指令
    print("退出：exit")
    print(f"提示：目标步长 {env.data_handler.move_step} 米，仅移动动作会修正GPS坐标\n")

    # 初始化时显示初始图像
    if env.current_image is not None:
        env.render()

    while True:
        # 显示当前状态（仅显示实际GPS，非移动动作不修改）
        current_lat = env.data_handler.current_lat
        current_lng = env.data_handler.current_lng
        current_heading = env.data_handler.heading
        current_fov = env.data_handler.fov if hasattr(env.data_handler, 'fov') else 30
        current_pitch = env.data_handler.pitch
        print(f"📌 当前GPS坐标: {current_lat:.6f}, {current_lng:.6f}")
        print(f"🧭 朝向: {current_heading:.1f}° | 视角FOV: {current_fov}° | 俯仰Pitch: {current_pitch}°")

        # 获取用户指令
        cmd = input("> 请输入操作指令: ").strip().lower()
        if cmd == "exit":
            print("👋 退出手动操控模式")
            # 打印坐标修正历史（仅移动动作有记录）
            if env.coord_correction_history:
                print("\n📊 坐标修正历史（仅移动动作）：")
                for corr in env.coord_correction_history:
                    print(
                        f"Step {corr['step']}: 理论({corr['theoretical_lat']:.6f},{corr['theoretical_lng']:.6f}) → 实际({corr['actual_lat']:.6f},{corr['actual_lng']:.6f}) | 偏差 {corr['distance_diff']:.2f} 米")
            
            # 生成可视化HTML页面
            generate_visualization(env, pd)
            break

        # 扩展指令映射（新增z/x/r/f）
        cmd_map = {
            "w": "move forward", "a": "move left", "d": "move right", "s": "move backward",
            "q": "turn left", "e": "turn right",
            "z": "zoom in", "x": "zoom out",  # FOV缩放
            "r": "look up", "f": "look down"  # Pitch俯仰
        }
        if cmd not in cmd_map:
            print("❌ 无效指令！请输入 W/A/S/D/Q/E/Z/X/R/F 或 exit\n")
            continue

        # 执行操作前记录坐标（用于对比）
        pre_lat = current_lat
        pre_lng = current_lng

        # 执行动作并获取结果
        action = cmd_map[cmd]
        obs, reward, terminated, truncated, info = env.step(action)

        # 解析执行结果并反馈
        action_type = info["action_type"]
        relative_dir = info["relative_dir"]
        msg = info["msg"]

        print("\n--- 执行结果 ---")
        if action_type == "move":
            actual_distance = info.get("actual_distance", 0.0)
            if reward == 0.0:
                print(
                    f"✅ 移动成功！方向：{relative_dir} | 目标步长：{env.data_handler.move_step}米 | 实际移动：{actual_distance:.2f} 米")
                # 显示理论坐标和修正后的实际坐标
                if env.current_theoretical_lat and env.current_actual_lat:
                    print(f"📏 坐标修正过程：")
                    print(f"   理论计算坐标：{env.current_theoretical_lat:.6f}, {env.current_theoretical_lng:.6f}")
                    print(f"   实际采样坐标：{env.current_actual_lat:.6f}, {env.current_actual_lng:.6f}")
                    print(f"   坐标偏差：{env.current_correction_diff:.2f} 米")
            else:
                print(f"❌ 移动失败！原因：{msg} | 实际移动：{actual_distance:.2f} 米")
        else:
            # 转向/FOV/Pitch：仅反馈动作结果，无坐标变化
            print(f"✅ {msg}")

        # 显示执行后的最新状态
        print(f"📌 操作后GPS坐标: {env.data_handler.current_lat:.6f}, {env.data_handler.current_lng:.6f}")
        print(
            f"🧭 操作后朝向: {env.data_handler.heading:.1f}° | 视角FOV: {env.data_handler.fov}° | 俯仰Pitch: {env.data_handler.pitch}°\n")

        # 渲染最新街景
        env.render()


# ====================== 批处理运行（保留，无核心修改） ======================
def batch_run():
    # 初始化批处理目录
    batch_dir = init_batch_dir()
    
    # 初始化结果列表
    batch_results = []
    
    print(f"✅ 开始批量运行，共 {len(BATCH_GPS_TARGETS)} 个点位")

    for idx, point_info in enumerate(BATCH_GPS_TARGETS, 1):
        print(f"\n====== 第 {idx}/{len(BATCH_GPS_TARGETS)} 个点: {point_info['name']} ======")
        try:
            point_dir, img_dir = init_point_dir(point_info)
            env = StreetViewAutoLocEnv(
                initial_lat=point_info["lat"],
                initial_lng=point_info["lng"],
                move_step=point_info.get("move_step", GLOBAL_MOVE_STEP),
                max_steps=point_info.get("max_steps", GLOBAL_MAX_STEPS),
                point_dir=point_dir,
                img_save_dir=img_dir
            )
            env.reset()

            # 自动执行直到最大步数
            step = 0
            truncated = False
            
            # 处理max_steps为0的情况，直接基于初始图像进行预测
            if env.max_steps == 0:
                print("📌 max_steps为0，基于初始图像直接进行预测")
                if env.current_image is None:
                    print("❌ 无初始图像，跳过该点")
                else:
                    # 直接调用最终预测方法
                    env._force_final_prediction()
                    print(f"✅ 基于初始图像的预测完成: {env.final_prediction}")
            else:
                while step < env.max_steps and not truncated:
                    if env.current_image is None:
                        print("❌ 无图像，跳过该点")
                        break

                    # GPT 决策
                    action_str = env.gpt4o_decision(env.current_image, env.current_step + 1)
                    # 存储原始的大模型反馈
                    env.reason_history.append(action_str)
                    # 解析动作
                    action, _ = env.parse_gpt_decision(action_str)
                    # 执行动作
                    obs, reward, terminated, truncated, info = env.step(action)
                    # 添加原始反馈到info字典
                    info["reason"] = action_str
                    step += 1
                    
                    # 批量运行时也显示修正过程
                    action_type = info.get("action_type")
                    reason = info.get("reason", "No reason provided")
                    if action_type == "move" and "theoretical_lat" in info:
                        print(f"Step {env.current_step} | Action: {action} | {info['msg']}")
                        print(f"   Reason: {reason}")
                        print(
                            f"   理论坐标：{info['theoretical_lat']:.6f}, {info['theoretical_lng']:.6f} → 实际坐标：{info['actual_lat']:.6f}, {info['actual_lng']:.6f} (偏差{info['coord_correction_diff_m']:.2f}米) | reward: {reward}")
                    else:
                        print(f"Step {env.current_step} | Action: {action} | {info['msg']} | reward: {reward}")
                        print(f"   Reason: {reason}")
                    
                    # 如果移动失败，自动尝试转向动作
                    if info.get("action_type") == "move" and reward == -1.0 and step < env.max_steps:
                        # 尝试向左转
                        obs, reward, terminated, truncated, info = env.step("turn left")
                        step += 1
                        print(f"Step {env.current_step} | Action: turn left (自动左转) | {info['msg']} | reward: {reward}")
                        
                        # 再次获取图像
                        if env.current_image and step < env.max_steps:
                            # 基于新视角再次做决策
                            action_str = env.gpt4o_decision(env.current_image, env.current_step + 1)
                            action, _ = env.parse_gpt_decision(action_str)
                            obs, reward, terminated, truncated, info = env.step(action)
                            step += 1
                            
                            # 显示第二次决策的结果
                            action_type = info.get("action_type")
                            if action_type == "move" and "theoretical_lat" in info:
                                print(f"Step {env.current_step} | Action: {action} | {info['msg']}")
                                print(
                                    f"   理论坐标：{info['theoretical_lat']:.6f}, {info['theoretical_lng']:.6f} → 实际坐标：{info['actual_lat']:.6f}, {info['actual_lng']:.6f} (偏差{info['coord_correction_diff_m']:.2f}米) | reward: {reward}")
                            else:
                                print(f"Step {env.current_step} | Action: {action} | {info['msg']} | reward: {reward}")

                # 移除break语句，确保所有点位都能被处理

            # 保存最终预测结果
            result_path = os.path.join(point_dir, "final_prediction.txt")
            with open(result_path, "w", encoding="utf-8") as f:
                f.write(env.final_prediction)
            print(f"✅ 预测完成，结果已保存: {result_path}")

            # 保存坐标修正历史（仅移动动作）
            if env.coord_correction_history:
                corr_path = os.path.join(point_dir, "coord_correction_history.csv")
                with open(corr_path, "w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=env.coord_correction_history[0].keys())
                    writer.writeheader()
                    writer.writerows(env.coord_correction_history)
                print(f"✅ 坐标修正历史已保存: {corr_path}")
            
            # 生成可视化HTML页面
            generate_visualization(env, point_dir)
            
            # 收集点位结果信息
            point_result = {
                "name": point_info["name"],
                "actual_lat": point_info["lat"],
                "actual_lng": point_info["lng"],
                "prediction": env.final_prediction if hasattr(env, 'final_prediction') else "No prediction",
                "predicted_lat": None,
                "predicted_lng": None,
                "error": None
            }
            
            # 提取预测的GPS坐标
            if point_result["prediction"]:
                try:
                    import re
                    # 尝试匹配多种可能的坐标格式
                    patterns = [
                        r'predicted_initial_coords: ([\d.-]+),([\d.-]+)',  # 原始格式
                        r'([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)',  # 简单的经纬度格式
                        r'lat[\s:]*([-+]?\d*\.?\d+)[^,]*lng[\s:]*([-+]?\d*\.?\d+)',  # 带标签的格式
                        r'latitude[\s:]*([-+]?\d*\.?\d+)[^,]*longitude[\s:]*([-+]?\d*\.?\d+)'  # 完整标签格式
                    ]
                    
                    for pattern in patterns:
                        match = re.search(pattern, point_result["prediction"])
                        if match:
                            point_result["predicted_lat"] = float(match.group(1))
                            point_result["predicted_lng"] = float(match.group(2))
                            break
                    
                    # 计算偏差
                    if point_result["predicted_lat"] is not None and point_result["predicted_lng"] is not None:
                        point_result["error"] = calculate_gps_distance(
                            point_result["predicted_lat"], point_result["predicted_lng"],
                            point_result["actual_lat"], point_result["actual_lng"]
                        )
                except Exception as e:
                    print(f"❌ 计算偏差时出错: {str(e)}")
            
            # 添加到结果列表
            batch_results.append(point_result)

        except Exception as e:
            print(f"❌ 该点运行失败: {e}")
            traceback.print_exc()
    
    # 生成汇总HTML文件
    generate_batch_summary(batch_dir, batch_results)


def generate_batch_summary(batch_dir, batch_results):
    """
    生成批处理汇总HTML文件
    """
    # 计算平均偏差
    total_error = 0
    valid_count = 0
    for result in batch_results:
        if result["error"] is not None:
            total_error += result["error"]
            valid_count += 1
    
    average_error = total_error / valid_count if valid_count > 0 else 0
    
    # 计算不同误差范围的统计
    error_ranges = [1000, 25000, 125000, 250000]  # 1km, 25km, 125km, 250km
    range_counts = []
    for max_error in error_ranges:
        count = 0
        for result in batch_results:
            if result["error"] is not None and result["error"] <= max_error:
                count += 1
        range_counts.append(count)
    
    # 计算占比
    range_percentages = []
    for count in range_counts:
        percentage = (count / valid_count * 100) if valid_count > 0 else 0
        range_percentages.append(percentage)
    
    # 生成HTML内容
    html_content = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Batch Run Summary</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            text-align: center;
            color: #333;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .chart-container {
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
            padding: 30px;
            margin-bottom: 30px;
        }
        .chart-title {
            text-align: center;
            color: #333;
            margin-bottom: 20px;
            font-size: 20px;
            font-weight: bold;
        }
        .chart-wrapper {
            position: relative;
            height: 400px;
        }
        .summary {
            margin-bottom: 30px;
            padding: 20px;
            background-color: #f9f9f9;
            border-radius: 8px;
            border: 1px solid #ddd;
        }
        .summary h2 {
            color: #333;
            margin-bottom: 15px;
        }
        .summary-info {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
        }
        .summary-item {
            background-color: white;
            padding: 15px;
            border-radius: 4px;
            border: 1px solid #eee;
            flex: 1;
            min-width: 200px;
        }
        .summary-item strong {
            color: #1e90ff;
        }
        .results-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 30px;
        }
        .results-table th {
            background-color: #f2f2f2;
            padding: 12px;
            text-align: left;
            border-bottom: 2px solid #ddd;
        }
        .results-table td {
            padding: 12px;
            border-bottom: 1px solid #ddd;
        }
        .results-table tr:hover {
            background-color: #f5f5f5;
        }
        .error-cell {
            font-weight: bold;
            color: #1e90ff;
        }
        .average-error {
            margin-top: 30px;
            padding: 20px;
            background-color: #e6f7ff;
            border-radius: 8px;
            border: 1px solid #91d5ff;
        }
        .average-error h2 {
            color: #1890ff;
            margin-bottom: 15px;
        }
        .average-error p {
            font-size: 18px;
            font-weight: bold;
            color: #1890ff;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Batch Run Summary</h1>
        
        <!-- 误差分布折线图 -->
        <div class="chart-container">
            <div class="chart-title">Error Range Distribution</div>
            <div class="chart-wrapper">
                <canvas id="errorChart"></canvas>
            </div>
        </div>
        
        <div class="summary">
            <h2>Run Information</h2>
            <div class="summary-info">
                <div class="summary-item">
                    <strong>Total Points:</strong> {total_points}
                </div>
                <div class="summary-item">
                    <strong>Valid Predictions:</strong> {valid_count}
                </div>
                <div class="summary-item">
                    <strong>Invalid Predictions:</strong> {invalid_count}
                </div>
            </div>
        </div>
        
        <table class="results-table">
            <thead>
                <tr>
                    <th>Point Name</th>
                    <th>Actual GPS</th>
                    <th>Predicted GPS</th>
                    <th>Prediction Error (meters)</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
        
        <div class="average-error">
            <h2>Average Prediction Error</h2>
            <p>{average_error:.2f} meters</p>
        </div>
    </div>
    
    <script>
        // 误差范围数据
        const errorRanges = ['≤ 1km', '≤ 25km', '≤ 125km', '≤ 250km'];
        const percentages = {range_percentages};
        
        // 创建折线图
        const ctx = document.getElementById('errorChart').getContext('2d');
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: errorRanges,
                datasets: [{
                    label: 'Percentage of Points',
                    data: percentages,
                    borderColor: 'rgba(102, 126, 234, 1)',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    borderWidth: 3,
                    tension: 0,
                    pointBackgroundColor: 'rgba(102, 126, 234, 1)',
                    pointBorderColor: '#fff',
                    pointBorderWidth: 2,
                    pointRadius: 6,
                    pointHoverRadius: 8
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                    },
                    title: {
                        display: true,
                        text: 'Prediction Error Distribution',
                        font: {
                            size: 16
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return context.parsed.y.toFixed(1) + '%';
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        title: {
                            display: true,
                            text: 'Percentage (%)'
                        },
                        ticks: {
                            callback: function(value) {
                                return value + '%';
                            }
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Error Range'
                        }
                    }
                }
            }
        });
    </script>
</body>
</html>
    '''
    
    # 生成表格行
    table_rows = ""
    for result in batch_results:
        actual_gps = f"{result['actual_lat']:.6f}, {result['actual_lng']:.6f}"
        if result['predicted_lat'] is not None and result['predicted_lng'] is not None:
            predicted_gps = f"{result['predicted_lat']:.6f}, {result['predicted_lng']:.6f}"
            error = f"{result['error']:.2f}" if result['error'] is not None else "N/A"
        else:
            predicted_gps = "N/A"
            error = "N/A"
        
        table_rows += f"""
        <tr>
            <td>{result['name']}</td>
            <td>{actual_gps}</td>
            <td>{predicted_gps}</td>
            <td class="error-cell">{error}</td>
        </tr>
        """
    
    # 计算无效预测数
    invalid_count = len(batch_results) - valid_count
    
    # 替换占位符
    html_content = html_content.replace("{total_points}", str(len(batch_results)))
    html_content = html_content.replace("{valid_count}", str(valid_count))
    html_content = html_content.replace("{invalid_count}", str(invalid_count))
    html_content = html_content.replace("{table_rows}", table_rows)
    html_content = html_content.replace("{average_error:.2f}", f"{average_error:.2f}")
    html_content = html_content.replace("{range_percentages}", str(range_percentages))
    
    # 保存HTML文件
    summary_path = os.path.join(batch_dir, "batch_summary.html")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"✅ 批处理汇总HTML文件已生成: {summary_path}")

# ====================== 运行 ======================
if __name__ == "__main__":
    # 1. 手动控制（核心测试）
    # test_point = {"name": "ManualTest", "lat": 55.64641158091692, "lng":37.72502267722842}
    # pd, idir = init_point_dir(test_point)
    # env = StreetViewAutoLocEnv(
    #     initial_lat=test_point["lat"],
    #     initial_lng=test_point["lng"],
    #     point_dir=pd,
    #     img_save_dir=idir
    # )
    # env.reset()
    # manual_control(env)

    # 2. 批处理自动运行（如需启用，注释上面的手动控制代码）
    batch_run()