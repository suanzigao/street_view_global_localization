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

# ====================== 全局代理配置 ======================
PROXY_HOST = "127.0.0.1"
PROXY_PORT = 7897
os.environ["HTTP_PROXY"] = f"http://{PROXY_HOST}:{PROXY_PORT}"
os.environ["HTTPS_PROXY"] = f"http://{PROXY_HOST}:{PROXY_PORT}"
os.environ["NO_PROXY"] = "localhost,127.0.0.1,localdomain,.localdomain"

# ====================== 屏蔽警告 ==============================
warnings.filterwarnings("ignore", category=UserWarning, message="Gym has been unmaintained")
warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()

# ====================== 全局配置 ==============================
BATCH_ROOT_DIR = "batch_streetview_results"
GLOBAL_MAX_STEPS = 8
GLOBAL_MOVE_STEP = 15.0
GLOBAL_MAX_RETRIES = 3
# 新增：FOV/俯仰控制参数
FOV_STEP = 20
PITCH_STEP = 10
TURN_ANGLE = 10
# 新增：Google API参数
GOOGLE_API_RADIUS = 6
GPS_CSV_PATH = "minitest.csv"


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
    import shutil
    if os.path.exists(BATCH_ROOT_DIR):
        shutil.rmtree(BATCH_ROOT_DIR)
    os.makedirs(BATCH_ROOT_DIR, exist_ok=True)


def get_point_dir_name(point_info: Dict) -> str:
    lat_str = f"{point_info['lat']:.4f}".replace(".", "_")
    lng_str = f"{point_info['lng']:.4f}".replace(".", "_")
    return f"{point_info['name']}_lat{lat_str}_lng{lng_str}"


def init_point_dir(point_info: Dict) -> Tuple[str, str]:
    point_dir = os.path.join(BATCH_ROOT_DIR, get_point_dir_name(point_info))
    img_dir = os.path.join(point_dir, "streetview_images")
    os.makedirs(img_dir, exist_ok=True)
    return point_dir, img_dir


def image_to_base64(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format='JPEG', quality=100)  # 提升Base64编码的图片质量
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def save_streetview_image(image: Image.Image, step_num: int, img_save_dir: str) -> Tuple[str, str]:
    img_path = os.path.join(img_save_dir, f"step_{step_num - 1}_streetview.jpg")
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
        self.current_lat = self.current_lng = 0.0
        self.heading = 0.0
        self.pitch = 0.0
        self.move_step = move_step
        self.fov = 30

    def init_position(self, lat: float, lng: float, heading: float = 0.0, fov: float = 30.0, pitch: float = 0.0):
        self.initial_lat = lat
        self.initial_lng = lng
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
        return data if data.get("status") == "OK" else None

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
        "heading": env.data_handler.heading,
        "fov": env.data_handler.fov if hasattr(env.data_handler, 'fov') else 30,
        "pitch": env.data_handler.pitch,
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
    
    # 生成HTML内容
    html_content = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Street View Navigation Visualization</title>
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
        .map-container {
            width: 100%;
            height: 500px;
            margin-bottom: 20px;
            border: 1px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
        }
        #map {
            width: 100%;
            height: 100%;
        }
        .timeline {
            margin-top: 30px;
        }
        .timeline h2 {
            color: #333;
            margin-bottom: 20px;
        }
        .step {
            display: flex;
            margin-bottom: 30px;
            border-bottom: 1px solid #eee;
            padding-bottom: 20px;
        }
        .step-info {
            flex: 1;
            padding-right: 20px;
        }
        .step-image {
            flex: 1;
        }
        .step-image img {
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .step-number {
            font-size: 24px;
            font-weight: bold;
            color: #4CAF50;
            margin-bottom: 10px;
        }
        .step-details {
            color: #666;
        }
        .step-details p {
            margin: 5px 0;
        }
    </style>
    <script src="https://maps.googleapis.com/maps/api/js?key=AIzaSyAXmbPK8IXtGV9fjWVI8K4b5cw9Tshrvy0"></script>
</head>
<body>
    <div class="container">
        <h1>Street View Navigation Visualization</h1>
        
        <div class="map-container">
            <div id="map"></div>
        </div>
        
        <div class="timeline">
            <h2>Navigation Timeline</h2>
            {timeline_html}
        </div>
    </div>
    
    <script>
        // 轨迹数据
        const trajectory = {trajectory_data};
        
        // 初始化地图
        function initMap() {
            // 使用第一个点作为中心
            const center = new google.maps.LatLng(trajectory[0].lat, trajectory[0].lng);
            
            const map = new google.maps.Map(document.getElementById('map'), {
                zoom: 18,
                center: center,
                mapTypeId: google.maps.MapTypeId.SATELLITE
            });
            
            // 添加标记
            trajectory.forEach((point, index) => {
                const marker = new google.maps.Marker({
                    position: new google.maps.LatLng(point.lat, point.lng),
                    map: map,
                    title: `Step ${point.step}: ${point.action}`
                });
                
                // 添加信息窗口
                const infoWindow = new google.maps.InfoWindow({
                    content: `<strong>Step ${point.step}</strong><br>
                             Lat: ${point.lat.toFixed(6)}<br>
                             Lng: ${point.lng.toFixed(6)}<br>
                             Heading: ${point.heading.toFixed(1)}°<br>
                             FOV: ${point.fov.toFixed(1)}°<br>
                             Pitch: ${point.pitch.toFixed(1)}°<br>
                             Action: ${point.action}`
                });
                
                marker.addListener('click', function() {
                    infoWindow.open(map, marker);
                });
            });
            
            // 绘制轨迹线
            const path = trajectory.map(point => new google.maps.LatLng(point.lat, point.lng));
            const polyline = new google.maps.Polyline({
                path: path,
                geodesic: true,
                strokeColor: '#FF0000',
                strokeOpacity: 1.0,
                strokeWeight: 2
            });
            
            polyline.setMap(map);
        }
        
        // 页面加载完成后初始化地图
        window.onload = initMap;
    </script>
</body>
</html>
    '''
    
    # 生成时间线HTML
    timeline_html = ""
    for item in trajectory:
        timeline_html += f"""
        <div class="step">
            <div class="step-info">
                <div class="step-number">Step {item['step']}</div>
                <div class="step-details">
                    <p><strong>Action:</strong> {item['action']}</p>
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
            "w": 0, "a": 1, "d": 2, "s": 3,
            "q": 4, "e": 5,
            "z": 6, "x": 7,  # FOV缩放
            "r": 8, "f": 9  # Pitch俯仰
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
    init_batch_dir()
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
            for step in range(env.max_steps):
                if env.current_image is None:
                    print("❌ 无图像，跳过该点")
                    break

                # GPT 决策
                action_str = env.gpt4o_decision(env.current_image, env.current_step + 1)
                action = env.parse_gpt_decision(action_str)
                obs, reward, terminated, truncated, info = env.step(action)

                # 批量运行时也显示修正过程
                if action_type == "move" and "theoretical_lat" in info:
                    print(f"Step {env.current_step} | Action: {action} | {info['msg']}")
                    print(
                        f"   理论坐标：{info['theoretical_lat']:.6f}, {info['theoretical_lng']:.6f} → 实际坐标：{info['actual_lat']:.6f}, {info['actual_lng']:.6f} (偏差{info['coord_correction_diff_m']:.2f}米) | reward: {reward}")
                else:
                    print(f"Step {env.current_step} | Action: {action} | {info['msg']} | reward: {reward}")

                if truncated:
                    break

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

        except Exception as e:
            print(f"❌ 该点运行失败: {e}")
            traceback.print_exc()
            continue


# ====================== 运行 ======================
if __name__ == "__main__":
    # 1. 手动控制（核心测试）
    test_point = {"name": "ManualTest", "lat": 47.53097481458263, "lng": 19.06958490497365}
    pd, idir = init_point_dir(test_point)
    env = StreetViewAutoLocEnv(
        initial_lat=test_point["lat"],
        initial_lng=test_point["lng"],
        point_dir=pd,
        img_save_dir=idir
    )
    env.reset()
    manual_control(env)

    # 2. 批处理自动运行（如需启用，注释上面的手动控制代码）
    # batch_run()