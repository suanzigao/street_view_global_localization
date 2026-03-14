# ====================== 1. 全局代理配置（替换自己的代理端口）======================
import os

PROXY_HOST = "127.0.0.1"
PROXY_PORT = 7897  # 替换成你的代理端口（Clash/SSR/V2Ray等）
os.environ["HTTP_PROXY"] = f"http://{PROXY_HOST}:{PROXY_PORT}"
os.environ["HTTPS_PROXY"] = f"http://{PROXY_HOST}:{PROXY_PORT}"
os.environ["NO_PROXY"] = "localhost,127.0.0.1,localdomain,.localdomain"
# =================================================================================

# ====================== 2. 屏蔽警告 + 基础导入=============================
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message="Gym has been unmaintained")
warnings.filterwarnings("ignore", category=DeprecationWarning)
import gymnasium as gym
from gymnasium import spaces
import requests
import math
import numpy as np
import base64
import shutil
from typing import Tuple, Optional, Dict, Any, List
from PIL import Image
from io import BytesIO
import openai
from openai import OpenAI
from dotenv import load_dotenv
import time
import traceback
import re
import csv  # 新增：内置CSV模块，无需额外安装

# =================================================================================

load_dotenv()

# ------------------------------
# 全局配置：批量处理+文件存储（新增CSV文件路径）
# ------------------------------
BATCH_ROOT_DIR = "batch_streetview_results"
GLOBAL_MAX_STEPS = 8  # 全局最大导航步数
GLOBAL_MOVE_STEP = 5  # 全局每次移动距离（米）
GLOBAL_MAX_RETRIES = 3  # 全局API重试次数
TURN_ANGLE = 10  # 转向角度（左/右转均为15°，可按需修改）
GPS_CSV_PATH = "minitest.csv"  # 核心新增：GPS点位CSV文件路径（项目根目录）


# =================================================================================

# ------------------------------
# 新增核心工具函数：读取GPS点位CSV文件（带严格数据校验+容错）
# ------------------------------
def read_gps_targets_from_csv(csv_path: str) -> List[Dict]:
    """
    从CSV文件读取GPS点位信息，与原有硬编码格式完全兼容，带数据校验和容错处理
    :param csv_path: CSV文件路径
    :return: 校验后的GPS点位列表（格式：[{"name":xxx, "lat":xxx, "lng":xxx, "max_steps":xxx, "move_step":xxx}]）
    """
    gps_targets = []
    # 检查CSV文件是否存在
    if not os.path.exists(csv_path):
        print(f"❌ 关键错误：GPS点位CSV文件不存在！")
        print(f"📁 预期文件路径：{os.path.abspath(csv_path)}")
        print(f"📝 请在项目根目录创建【gps_targets.csv】，并按以下标准格式编写（表头必须一致）：")
        print("name,lat,lng,max_steps,move_step")
        print("NewYork_TimesSquare,40.7580,-73.9855,,")
        print("London_BigBen,51.5007,-0.1246,10,")
        print("Paris_EiffelTower,48.8584,2.2945,,6")
        raise FileNotFoundError(f"GPS CSV file not found: {csv_path}")

    # 读取CSV并校验数据
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # 校验必填表头
        required_headers = ["name", "lat", "lng"]
        if not all(h in reader.fieldnames for h in required_headers):
            raise ValueError(
                f"CSV表头错误！必须包含必填字段：{required_headers}，当前表头：{reader.fieldnames}\n"
                f"请检查表头是否为：name,lat,lng,max_steps,move_step（逗号分隔，无空格）"
            )

        # 逐行处理数据（line_num从2开始，表头为1）
        for line_num, row in enumerate(reader, start=2):
            try:
                # 提取并清洗必填字段
                name = row["name"].strip()
                lat_str = row["lat"].strip()
                lng_str = row["lng"].strip()

                # 校验必填字段有效性
                if not name:
                    print(f"⚠️ 第{line_num}行：点位名称为空，跳过该条数据")
                    continue
                if not lat_str or not lng_str:
                    print(f"⚠️ 第{line_num}行：纬度/经度为空，跳过该条数据（名称：{name}）")
                    continue

                # 转换数值类型并校验范围
                lat = float(lat_str)
                lng = float(lng_str)
                if not (-90 <= lat <= 90):
                    print(f"⚠️ 第{line_num}行：纬度超出范围（-90~90），跳过（名称：{name}，纬度：{lat}）")
                    continue
                if not (-180 <= lng <= 180):
                    print(f"⚠️ 第{line_num}行：经度超出范围（-180~180），跳过（名称：{name}，经度：{lng}）")
                    continue

                # 提取可选字段（留空则为None，后续用全局默认值）
                max_steps = int(row["max_steps"].strip()) if row["max_steps"].strip() else None
                move_step = float(row["move_step"].strip()) if row["move_step"].strip() else None

                # 构造标准点位字典（与原有硬编码格式一致）
                point_info = {
                    "name": name,
                    "lat": lat,
                    "lng": lng
                }
                if max_steps:
                    point_info["max_steps"] = max_steps
                if move_step:
                    point_info["move_step"] = move_step

                gps_targets.append(point_info)
                print(f"✅ 第{line_num}行：读取成功 | {name} | {lat:.6f},{lng:.6f}")

            except ValueError as e:
                print(f"⚠️ 第{line_num}行：数据类型错误，跳过 | 原因：{str(e)}（lat/lng为浮点数，max_steps为整数）")
            except Exception as e:
                print(f"⚠️ 第{line_num}行：未知错误，跳过 | 原因：{str(e)}")

    # 校验最终有效数据
    if not gps_targets:
        raise ValueError("CSV文件中无有效GPS点位！请检查数据格式是否正确，至少保留1行有效数据")

    print(f"\n📊 CSV文件读取完成 | 总有效GPS点位：{len(gps_targets)}")
    return gps_targets


# ------------------------------
# 批量GPS点位配置（从CSV读取，替代原有硬编码）
# ------------------------------
BATCH_GPS_TARGETS = read_gps_targets_from_csv(GPS_CSV_PATH)


# =================================================================================

# ------------------------------
# 原有工具函数：重试装饰器（无修改）
# ------------------------------
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
                        print(f"Failed after {max_retries} retries: {str(e)}")
                        traceback.print_exc()
                        raise
                    print(f"Execution failed, retrying in {delay}s (Attempt {retries}/{max_retries}): {str(e)}")
                    time.sleep(delay)
            return None

        return wrapper

    return decorator


# ------------------------------
# 原有工具函数：文件/目录操作（无修改）
# ------------------------------
def init_batch_dir():
    if os.path.exists(BATCH_ROOT_DIR):
        shutil.rmtree(BATCH_ROOT_DIR)
    os.makedirs(BATCH_ROOT_DIR, exist_ok=True)
    print(f"📦 Batch processing root dir created: {os.path.abspath(BATCH_ROOT_DIR)}")


def get_point_dir_name(point_info: Dict) -> str:
    lat_str = f"{point_info['lat']:.4f}".replace(".", "_")
    lng_str = f"{point_info['lng']:.4f}".replace(".", "_")
    return f"{point_info['name']}_lat{lat_str}_lng{lng_str}"


def init_point_dir(point_info: Dict) -> str:
    point_dir_name = get_point_dir_name(point_info)
    point_dir = os.path.join(BATCH_ROOT_DIR, point_dir_name)
    os.makedirs(point_dir, exist_ok=True)
    img_dir = os.path.join(point_dir, "streetview_images")
    os.makedirs(img_dir, exist_ok=True)
    return point_dir, img_dir


# ------------------------------
# 新增工具函数：图片转Base64编码
# ------------------------------
def image_to_base64(image: Image.Image) -> str:
    """
    将PIL图片转换为Base64编码字符串（JPEG格式）
    :param image: PIL图片对象
    :return: Base64编码字符串
    """
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='JPEG', quality=95)
    img_byte_arr.seek(0)
    return base64.b64encode(img_byte_arr.read()).decode('utf-8')


# ------------------------------
# 修改保存街景图片函数：新增Base64记录
# ------------------------------
def save_streetview_image(image: Image.Image, step_num: int, img_save_dir: str,
                          is_force_prediction: bool = False) -> Tuple[str, str]:
    """
    保存街景图片 + 生成Base64编码
    :param image: 街景图片对象
    :param step_num: 步骤数（从1开始）
    :param img_save_dir: 保存目录
    :param is_force_prediction: 是否为强制预测步骤（仅用于日志提示）
    :return: (图片绝对路径, Base64编码字符串)
    """
    # 步骤数-1转为图片编号（从0开始）
    img_num = step_num - 1
    img_name = f"step_{img_num}_streetview.jpg"
    img_path = os.path.join(img_save_dir, img_name)
    image.save(img_path, format='JPEG', quality=95)

    # 生成Base64编码
    img_b64 = image_to_base64(image)
    return os.path.abspath(img_path), img_b64
# ------------------------------
# 原有工具函数：相对视角转地理角度（适配6个动作，无修改）
# ------------------------------
def get_geo_angle(current_heading: float, relative_dir: str) -> float:
    """
    将相对当前视角的方向转为地理角度（0°=北，90°=东，180°=南，270°=西）
    :param current_heading: 当前朝向的地理角度
    :param relative_dir: 相对方向（forward/left/right/backward）
    :return: 对应的地理角度（0-360°）
    """
    angle_map = {
        "forward": current_heading,
        "left": current_heading + 90,
        "right": current_heading - 90,
        "backward": current_heading + 180
    }
    geo_angle = angle_map[relative_dir] % 360  # 取模保证角度在0-360°
    return geo_angle


# ------------------------------
# 原有数据层：Google街景数据处理（适配6个动作，无修改）
# ------------------------------
class GoogleStreetViewData:
    def __init__(self, api_key: str = None, max_retries: int = 3):
        self.google_api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.static_url = "https://maps.googleapis.com/maps/api/streetview"
        self.metadata_url = "https://maps.googleapis.com/maps/api/streetview/metadata"
        self.max_retries = max_retries
        self.initial_lat: float = 0.0
        self.initial_lng: float = 0.0
        self.current_lat: float = 0.0
        self.current_lng: float = 0.0
        self.heading: float = 0.0  # 当前朝向的地理角度（0°=北）
        self.pitch: float = 0.0  # 俯仰角固定为0°，移除相关操作
        self.move_step: float = 5.0

    def init_position(self, lat: float, lng: float, heading: float = 0.0, pitch: float = 0.0):
        self.initial_lat = lat
        self.initial_lng = lng
        self.current_lat = lat
        self.current_lng = lng
        self.heading = heading
        self.pitch = 0.0  # 强制固定俯仰角为0°

    @retry_on_failure(max_retries=3, delay=1.0)
    def get_metadata(self, lat: float, lng: float) -> Optional[Dict]:
        params = {"location": f"{lat},{lng}", "radius": 50, "key": self.google_api_key}
        response = requests.get(self.metadata_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data if data.get("status") == "OK" else None

    def calculate_new_coords(self, move_direction: int) -> Tuple[float, float]:
        R = 6378137.0
        lat_rad = math.radians(self.current_lat)
        lng_rad = math.radians(self.current_lng)
        move_angle = math.radians(move_direction)
        new_lat_rad = math.asin(
            math.sin(lat_rad) * math.cos(self.move_step / R) +
            math.cos(lat_rad) * math.sin(self.move_step / R) * math.cos(move_angle)
        )
        new_lng_rad = lng_rad + math.atan2(
            math.sin(move_angle) * math.sin(self.move_step / R) * math.cos(lat_rad),
            math.cos(self.move_step / R) - math.sin(lat_rad) * math.sin(new_lat_rad)
        )
        new_lat = math.degrees(new_lat_rad)
        new_lng = math.degrees(new_lng_rad)
        new_lng = (new_lng + 180) % 360 - 180
        return new_lat, new_lng

    def turn(self, delta_angle: float) -> None:
        """统一转向方法，左转为-，右转为+"""
        self.heading = (self.heading + delta_angle) % 360

    @retry_on_failure(max_retries=3, delay=1.0)
    def get_street_view_image(self) -> Optional[Image.Image]:
        params = {
            "location": f"{self.current_lat},{self.current_lng}",
            "heading": self.heading,
            "pitch": self.pitch,
            "fov":30,
            "radius":10,
            "size": "640x640",
            "key": self.google_api_key
        }
        response = requests.get(self.static_url, params=params, stream=True, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img_byte = BytesIO()
        img.save(img_byte, format='JPEG', quality=95)
        img_byte.seek(0)
        return Image.open(img_byte)

    def move(self, direction: float) -> Tuple[bool, str]:
        """direction为地理角度（浮点数）"""
        new_lat, new_lng = self.calculate_new_coords(direction)
        if not self.get_metadata(new_lat, new_lng):
            return False, "No street view available (move failed)"
        self.current_lat, self.current_lng = new_lat, new_lng
        return True, "Move successful"


# ------------------------------
# 原有核心逻辑层：GPT-4o视觉定位（6个动作适配，仅修复openai_client初始化）
# ------------------------------
class StreetViewAutoLocEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}
    # 6个动作映射（基于当前视角的相对方向）
    ACTION_MAP = {
        0: ("move", "forward"),  # 向前移动（当前视角正前方）
        1: ("move", "left"),  # 向左移动（当前视角左侧）
        2: ("move", "right"),  # 向右移动（当前视角右侧）
        3: ("move", "backward"),  # 向后移动（当前视角正后方）
        4: ("turn", "left"),  # 向左转
        5: ("turn", "right")  # 向右转
    }

    def __init__(
            self,
            initial_lat: float,
            initial_lng: float,
            move_step: float = 5.0,
            openai_api_key: str = None,
            max_retries: int = 3,
            max_steps: int = 10,
            point_dir: str = "./",
            img_save_dir: str = "./images"
    ):
        super().__init__()
        self.data_handler = GoogleStreetViewData(max_retries=max_retries)
        self.data_handler.move_step = move_step
        self.data_handler.init_position(initial_lat, initial_lng)
        self.max_retries = max_retries
        self.max_steps = max_steps
        self.current_step = 0
        self.move_step = move_step
        self.point_dir = point_dir
        self.img_save_dir = img_save_dir

        # 修复：初始化OpenAI客户端
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.openai_client = OpenAI(api_key=self.openai_api_key, base_url="https://api.laozhang.ai/v1")

        # 动作空间：6个离散动作
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=int),
            "lat": spaces.Box(low=-90, high=90, shape=(1,), dtype=float),
            "lng": spaces.Box(low=-180, high=180, shape=(1,), dtype=float),
            "heading": spaces.Box(low=0, high=360, shape=(1,), dtype=float),
            "pitch": spaces.Box(low=-90, high=90, shape=(1,), dtype=float)
        })

        self.current_image: Optional[Image.Image] = None
        self.final_prediction: str = ""
        self.action_history: List[str] = []
        self.image_path_history: Dict[int, str] = {}
        # 新增：保存每一步图片的Base64编码（用于最终预测时批量上传）
        self.image_b64_history: Dict[int, str] = {}
    @retry_on_failure(max_retries=3, delay=2.0)
    def gpt4o_decision(self, image: Image.Image, step_num: int) -> str:
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='JPEG', quality=95)
        img_byte_arr.seek(0)
        b64_encoded = base64.b64encode(img_byte_arr.read()).decode('utf-8')
        action_history_str = "\n- ".join(self.action_history) if self.action_history else "No navigation actions yet"

        # GPT-4o提示词：明确6个动作，基于当前视角相对方向
        prompt = f"""
        Your Role: Autonomous Google Street View Navigator & Visual Location Predictor
        CORE MISSION: Predict the GPS coordinates of your INITIAL starting point using ONLY VISUAL FEATURES.
        ACTION VALUES CONTEXT (FOR NAVIGATION ONLY - NO CALCULATION):
        - Exact action params: {self.move_step}m per move | {TURN_ANGLE}° per turn | Current heading: {self.data_handler.heading:.1f}°
        - Taken actions: {action_history_str}
        - Current steps / Max steps: {self.current_step}/{self.max_steps}

        ABSOLUTE RULES (VIOLATION = INVALID PREDICTION):
        1. NO GPS coordinates provided, NO geometric/spatial calculations allowed!
        2. All decisions/predictions MUST be based SOLELY on visual clues from the street view image.
        3. Navigate toward unique visual landmarks (street signs, buildings, landmarks) to find identifiable features.
        4. All MOVE actions are RELATIVE to your CURRENT VIEW DIRECTION (not fixed geographic north/south)!
        5. You MUST ONLY move ALONG THE ROAD. Do NOT walk into buildings, grass, walls, rivers, or off-road.

        # 仅有的6个Action Options（CHOOSE ONLY ONE NUMBER, NO EXTRA TEXT, NO OTHER WORDS）
        0 = Move FORWARD {self.move_step}m (current view front)
        1 = Move LEFT {self.move_step}m (current view left side)
        2 = Move RIGHT {self.move_step}m (current view right side)
        3 = Move BACKWARD {self.move_step}m (current view back)
        4 = Turn LEFT {TURN_ANGLE}° (view left, no movement)
        5 = Turn RIGHT {TURN_ANGLE}° (view right, no movement)

        # OUTPUT FORMAT (STRICTLY FOLLOW, NO DEVIATIONS, NO SYMBOLS, NO EXPLANATION)
        Output ONLY the number of your chosen action (0/1/2/3/4/5), e.g., 0, 4, 5 (nothing else)!
        You will be forced to make a final GPS prediction when max navigation steps are reached.
        """
        response = self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64_encoded}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            temperature=0.1,
            max_tokens=100,
            timeout=60
        )
        return response.choices[0].message.content.strip()

    # 解析GPT决策：仅返回0-5，异常默认向右转（5）
    def parse_gpt_decision(self, gpt_response: str) -> int:
        action_match = re.search(r'^(\d+)', gpt_response.strip())
        if not action_match:
            print(
                f"⚠️ GPT output format error (response: {gpt_response}) - default action: Turn Right {TURN_ANGLE}° (5)")
            return 5
        action = int(action_match.group(1))
        action = max(0, min(5, action))  # 强制钳位在0-5
        return action

    def get_observation(self) -> Dict[str, Any]:
        if self.current_image is None:
            image_arr = np.zeros((480, 640, 3), dtype=int)
        else:
            image_arr = np.array(self.current_image)
        return {
            "image": image_arr,
            "lat": np.array([self.data_handler.current_lat], dtype=float),
            "lng": np.array([self.data_handler.current_lng], dtype=float),
            "heading": np.array([self.data_handler.heading], dtype=float),
            "pitch": np.array([self.data_handler.pitch], dtype=float)
        }

    # 执行6个动作的核心step函数
    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        self.current_step += 1
        action_type, relative_dir = self.ACTION_MAP.get(action, ("turn", "right"))
        info = {
            "msg": "", "action_type": action_type, "relative_dir": relative_dir,
            "current_step": self.current_step, "max_steps": self.max_steps,
            "current_heading": self.data_handler.heading
        }
        reward = 0.0
        terminated = False
        truncated = False
        action_desc = ""
        current_heading = self.data_handler.heading

        # 执行移动动作（0-3）：相对方向→地理角度→移动
        if action_type == "move":
            geo_angle = get_geo_angle(current_heading, relative_dir)
            success, msg = self.data_handler.move(geo_angle)
            info["msg"] = msg
            info["move_geo_angle"] = geo_angle
            reward = -1.0 if not success else 0.0
            action_desc = f"Moved {relative_dir.upper()} {self.move_step}m (geo angle: {geo_angle:.1f}°)"

        # 执行转向动作（4-5）：左+15°，右-15°
        elif action_type == "turn":
            delta_angle = -TURN_ANGLE if relative_dir == "left" else TURN_ANGLE
            self.data_handler.turn(delta_angle)
            new_heading = self.data_handler.heading
            info[
                "msg"] = f"Turned {relative_dir.upper()} {abs(delta_angle)}° (heading: {current_heading:.1f}°→{new_heading:.1f}°)"
            info["delta_angle"] = delta_angle
            info["new_heading"] = new_heading
            action_desc = f"Turned {relative_dir.upper()} {abs(delta_angle)}° (heading: {new_heading:.1f}°)"

        # 记录动作历史
        if self.current_step in self.image_path_history:
            img_path = self.image_path_history[self.current_step]
            self.action_history.append(f"Step {self.current_step}: {action_desc} | Image: {img_path}")
        else:
            self.action_history.append(f"Step {self.current_step}: {action_desc} | Image: No image saved")

        # 获取更新后的街景图
        try:
            self.current_image = self.data_handler.get_street_view_image()
        except Exception as e:
            self.current_image = None
            info["msg"] += f" | Failed to get street view image: {str(e)[:50]}..."
            reward = -1.0

        # 达到最大步数：强制GPT-4o最终GPS预测（完全重构）
        if self.current_step >= self.max_steps and not terminated:
            truncated = True
            info["msg"] += " | Max steps reached - forcing final visual GPS prediction (using ALL history images)"
            if self.current_image:
                try:
                    # 保存最终预测图片（编号final_pred_step_num → 图片名step_{final_pred_step_num-1}）
                    final_pred_step_num = self.max_steps + 1
                    force_pred_img_path, force_pred_b64 = save_streetview_image(
                        self.current_image, final_pred_step_num, self.img_save_dir, is_force_prediction=True
                    )
                    self.image_path_history[final_pred_step_num] = force_pred_img_path
                    self.image_b64_history[final_pred_step_num] = force_pred_b64  # 记录最终图片的Base64
                    print(f"🖼️  Step {final_pred_step_num} (final prediction) image saved: {force_pred_img_path}")

                    # 构建完整的导航历史描述
                    full_action_history = "\n".join([
                        f"Step {i}: {action}" for i, action in enumerate(self.action_history, 1)
                    ]) if self.action_history else "No navigation actions executed"

                    # 优化后的提示词：明确要求结合所有历史图片和动作轨迹
                    force_prompt = f"""
                    MAX NAVIGATION STEPS ({self.max_steps}) REACHED! YOU MUST MAKE A FINAL PREDICTION OF THE INITIAL GPS COORDINATES.

                    # CONTEXT (USE ALL INFORMATION BELOW FOR PREDICTION):
                    1. Navigation History (all actions taken):
                    {full_action_history}

                    2. Visual Data: You will receive ALL street view images from the entire navigation process (Step 1 to Step {final_pred_step_num}).
                    3. Movement Rules: {self.move_step}m per move | {TURN_ANGLE}° per turn | All moves are relative to current view direction.

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

                    # 构建GPT消息：先文本，再批量上传所有图片
                    gpt_messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": force_prompt}
                            ]
                        }
                    ]

                    # 批量添加所有历史图片（按步骤顺序）
                    for step_num in sorted(self.image_b64_history.keys()):
                        img_b64 = self.image_b64_history[step_num]
                        gpt_messages[0]["content"].append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_b64}",
                                "detail": "high"
                            }
                        })
                        # 给每张图片添加步骤说明（帮助GPT理解顺序）
                        gpt_messages[0]["content"].append({
                            "type": "text",
                            "text": f"📸 Image from Step {step_num} (navigation process)"
                        })

                    # 调用GPT-4o做最终预测（增大max_tokens以容纳推理内容）
                    response = self.openai_client.chat.completions.create(
                        model="gpt-4o",
                        messages=gpt_messages,
                        temperature=0.1,
                        max_tokens=1000,  # 增大token数，容纳详细推理
                        timeout=120  # 延长超时时间（多图片处理需要更久）
                    )
                    self.final_prediction = response.choices[0].message.content.strip()
                    self.final_prediction += f"\n📸 Final prediction based on {len(self.image_b64_history)} images (Step 1 to {final_pred_step_num})"
                    self.final_prediction += f"\n📸 Final image path: {force_pred_img_path}"

                except Exception as e:
                    error_detail = traceback.format_exc()[:200]
                    self.final_prediction = f"Final prediction failed due to error: {str(e)[:100]}... | Error detail: {error_detail} | No final prediction image saved"
            else:
                self.final_prediction = "Final prediction failed: No valid street view image available for final step"
        # 构造返回值
        obs = self.get_observation()
        info.update({
            "initial_lat": self.data_handler.initial_lat,
            "initial_lng": self.data_handler.initial_lng,
            "action_history": self.action_history,
            "image_path_history": self.image_path_history,
            "reward": reward
        })
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Dict:
        super().reset(seed=seed)
        self.current_step = 0
        self.final_prediction = ""
        self.action_history = []
        self.image_path_history = {}
        # 新增：清空Base64历史
        self.image_b64_history = {}
        self.data_handler.init_position(
            self.data_handler.initial_lat, self.data_handler.initial_lng, heading=0.0, pitch=0.0
        )
        # 获取初始街景图
        try:
            self.current_image = self.data_handler.get_street_view_image()
        except Exception as e:
            self.current_image = None
            print(f"⚠️ Failed to get initial street view image: {str(e)}")
        return self.get_observation()

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        if mode == "human" and self.current_image:
            self.current_image.show(
                title=f"Street View - Step {self.current_step} | Heading: {self.data_handler.heading:.1f}°")
        elif mode == "rgb_array":
            return np.array(self.current_image) if self.current_image else np.zeros((480, 640, 3), dtype=np.uint8)
        return None

    def close(self):
        pass

    def auto_visual_positioning(self) -> Dict:
        point_name = os.path.basename(self.point_dir).split("_lat")[0]
        true_lat = self.data_handler.initial_lat
        true_lng = self.data_handler.initial_lng

        # 打印点位开始信息
        print("=" * 90)
        print(f"🚀 Starting Visual-Only Position Prediction - [{point_name}]")
        print("=" * 90)
        print(f"🔒 Critical: NO GPS data exposed to GPT (Pure visual reasoning)")
        print(f"⚙️  Navigation params: {self.move_step}m per move | {TURN_ANGLE}° per turn")
        print(f"📊 Max navigation steps: {self.max_steps} | Initial heading: 0.0° (North)")
        print(f"📍 True initial point (hidden): {true_lat:.6f}, {true_lng:.6f}")
        print(f"🖼️  Image save dir: {os.path.abspath(self.img_save_dir)}")
        print(f"📄 Result save dir: {os.path.abspath(self.point_dir)}")
        print("=" * 90 + "\n")

        self.reset()
        process_status = "success"
        error_reason = ""

        try:
            # 主导航循环
            while True:
                if not self.current_image:
                    print(f"⚠️ Step {self.current_step + 1}: No valid street view image - skip to next step")
                    self.current_step += 1
                    if self.current_step >= self.max_steps:
                        print("🔚 Max steps reached (no valid images) - forcing final prediction")
                        break
                    continue

                # 保存当前步骤图片 + 记录Base64
                current_step_num = self.current_step + 1
                img_save_path, img_b64 = save_streetview_image(self.current_image, current_step_num, self.img_save_dir)
                self.image_path_history[current_step_num] = img_save_path
                # 新增：保存Base64编码
                self.image_b64_history[current_step_num] = img_b64
                print(f"🖼️  Step {current_step_num} image saved: {img_save_path}")

                # 调用GPT-4o获取决策（这部分逻辑不变）
                print(
                    f"📡 Step {current_step_num}: Requesting GPT-4o visual decision (heading: {self.data_handler.heading:.1f}°)...")
                gpt_response = self.gpt4o_decision(self.current_image, current_step_num)
                print(f"💬 GPT Raw Response: {gpt_response}")
                print("-" * 60 + "\n")

                # 解析并执行动作（这部分逻辑不变）
                action = self.parse_gpt_decision(gpt_response)
                obs, reward, terminated, truncated, info = self.step(action)

                # 打印执行结果（这部分逻辑不变）
                action_info = self.action_history[-1] if self.action_history else "Initial step"
                print(f"✅ Executed Action {action}: {action_info}")
                print(f"📝 Status: {info['msg']}")
                print(f"💰 Reward: {reward} | Current Step: {self.current_step}/{self.max_steps}")
                print("=" * 60 + "\n")

                # 终止条件：达到最大步数
                if truncated:
                    break
        except Exception as e:
            process_status = "failed"
            error_reason = str(e)[:200] + "..." if len(str(e)) > 200 else str(e)
            print(f"❌ Point [{point_name}] processing failed: {error_reason}")
            traceback.print_exc()

        # 打印点位最终结果
        print("=" * 90)
        print(f"🎯 Position Prediction Complete - [{point_name}] | Status: [{process_status.upper()}]")
        print("=" * 90)
        print(f"📍 TRUE Initial GPS: {true_lat:.6f}, {true_lng:.6f}")
        if process_status == "success":
            print(f"🤖 GPT-4o Final Prediction:\n{self.final_prediction}")
        else:
            print(f"❌ Failure Reason: {error_reason}")
        print(f"🗂️  All files saved in: {os.path.abspath(self.point_dir)}")
        print("=" * 90 + "\n\n")

        # 保存点位详细结果文件
        result_file_name = f"{point_name}_prediction_result.txt"
        result_file_path = os.path.join(self.point_dir, result_file_name)
        try:
            with open(result_file_path, "w", encoding="utf-8") as f:
                f.write(f"GPT-4o Street View Visual-Only Position Prediction Result - [{point_name}]\n")
                f.write("=" * 90 + "\n")
                f.write(f"Processing Status: {process_status}\n")
                if process_status == "failed":
                    f.write(f"Failure Reason: {error_reason}\n")
                f.write(f"True Initial GPS Coordinates: {true_lat:.6f}, {true_lng:.6f}\n")
                f.write(f"Navigation Parameters: {self.move_step}m per move | {TURN_ANGLE}° per turn\n")
                f.write(f"Max Steps Allowed: {self.max_steps} | Steps Executed: {self.current_step}\n")
                f.write(f"Image Save Directory: {os.path.abspath(self.img_save_dir)}\n")
                f.write(f"Result File Path: {os.path.abspath(result_file_path)}\n")
                f.write(f"\n📜 Detailed Action History (with image paths):\n")
                if self.action_history:
                    for action in self.action_history:
                        f.write(f"  {action}\n")
                else:
                    f.write(f"  No navigation actions executed\n")
                f.write(f"\n📸 Step-Image Path Mapping:\n")
                if self.image_path_history:
                    for step_num, img_path in sorted(self.image_path_history.items()):
                        f.write(f"  Step {step_num}: {img_path}\n")
                else:
                    f.write(f"  No street view images saved\n")
                if process_status == "success":
                    f.write(f"\n🤖 GPT-4o Final Prediction:\n{self.final_prediction}\n")
            print(f"💾 Point result saved: {os.path.abspath(result_file_path)}")
        except Exception as e:
            print(f"⚠️ Failed to save point result file: {str(e)}")
            result_file_path = "Save failed"

        # 构造点位结果字典
        point_result = {
            "name": point_name,
            "true_lat": true_lat,
            "true_lng": true_lng,
            "max_steps": self.max_steps,
            "move_step": self.move_step,
            "process_status": process_status,
            "error_reason": error_reason,
            "gpt_prediction": self.final_prediction if process_status == "success" else "",
            "point_dir": os.path.abspath(self.point_dir),
            "img_save_dir": os.path.abspath(self.img_save_dir),
            "result_file_path": result_file_path,
            "executed_steps": self.current_step
        }

        self.close()
        return point_result


# ------------------------------
# 原有批量处理核心函数（无修改，兼容CSV读取的点位）
# ------------------------------
def batch_visual_positioning() -> List[Dict]:
    init_batch_dir()
    total_points = len(BATCH_GPS_TARGETS)
    batch_results = []

    print(f"📢 Starting batch processing - Total GPS points: {total_points} | 6 Navigation Actions")
    print(f"📦 Batch root dir: {os.path.abspath(BATCH_ROOT_DIR)}")
    print("=" * 100 + "\n\n")

    for idx, point_info in enumerate(BATCH_GPS_TARGETS, 1):
        print(f"📌 Processing Point {idx}/{total_points} - [{point_info['name']}]")
        print(f"📍 GPS: {point_info['lat']:.6f}, {point_info['lng']:.6f}")
        print("-" * 80 + "\n")

        point_dir, img_save_dir = init_point_dir(point_info)
        point_max_steps = point_info.get("max_steps", GLOBAL_MAX_STEPS)
        point_move_step = point_info.get("move_step", GLOBAL_MOVE_STEP)
        point_max_retries = GLOBAL_MAX_RETRIES

        try:
            env = StreetViewAutoLocEnv(
                initial_lat=point_info["lat"],
                initial_lng=point_info["lng"],
                move_step=point_move_step,
                max_retries=point_max_retries,
                max_steps=point_max_steps,
                point_dir=point_dir,
                img_save_dir=img_save_dir
            )
            point_result = env.auto_visual_positioning()
            batch_results.append(point_result)
        except Exception as e:
            err_reason = str(e)[:200] + "..." if len(str(e)) > 200 else str(e)
            batch_results.append({
                "name": point_info["name"],
                "true_lat": point_info["lat"],
                "true_lng": point_info["lng"],
                "max_steps": point_max_steps,
                "move_step": point_move_step,
                "process_status": "failed",
                "error_reason": f"Env init failed: {err_reason}",
                "gpt_prediction": "",
                "point_dir": os.path.abspath(point_dir),
                "img_save_dir": os.path.abspath(img_save_dir),
                "result_file_path": "Env init failed, no result file",
                "executed_steps": 0
            })
            print(f"❌ Point {idx}/{total_points} [{point_info['name']}] env init failed: {err_reason}\n\n")

    generate_batch_summary(batch_results)
    return batch_results


# ------------------------------
# 原有批量汇总报告函数（无修改）
# ------------------------------
def generate_batch_summary(batch_results: List[Dict]):
    summary_file = os.path.join(BATCH_ROOT_DIR, "batch_processing_summary_report.txt")
    total = len(batch_results)
    success = len([r for r in batch_results if r["process_status"] == "success"])
    failed = total - success
    success_rate = (success / total) * 100 if total > 0 else 0.0

    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(
            "GPT-4o Street View Visual-Only Positioning - BATCH PROCESSING SUMMARY REPORT (6 ACTIONS + CSV DRIVEN)\n")
        f.write("=" * 120 + "\n")
        f.write(
            f"Batch Processing Stats: Total={total} | Success={success} | Failed={failed} | Success Rate={success_rate:.1f}%\n")
        f.write(f"Batch Root Directory: {os.path.abspath(BATCH_ROOT_DIR)}\n")
        f.write(f"Processing Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
        f.write(
            f"Global Params: MaxSteps={GLOBAL_MAX_STEPS} | MoveStep={GLOBAL_MOVE_STEP}m | TurnAngle={TURN_ANGLE}° | MaxRetries={GLOBAL_MAX_RETRIES}\n")
        f.write(f"GPS CSV File: {os.path.abspath(GPS_CSV_PATH)}\n")
        f.write(f"6 Navigation Actions: 0=Forward |1=LeftMove |2=RightMove |3=Backward |4=LeftTurn |5=RightTurn\n")
        f.write("=" * 120 + "\n\n")

        for idx, result in enumerate(batch_results, 1):
            f.write(f"[{idx}/{total}] POINT: {result['name']}\n")
            f.write(f"  True GPS Coordinates: {result['true_lat']:.6f}, {result['true_lng']:.6f}\n")
            f.write(
                f"  Navigation Params: MaxSteps={result['max_steps']} | MoveStep={result['move_step']}m | TurnAngle={TURN_ANGLE}°\n")
            f.write(f"  Processing Status: {result['process_status'].upper()}\n")
            if result['process_status'] == "failed":
                f.write(f"  Failure Reason: {result['error_reason']}\n")
            f.write(f"  Point Directory: {result['point_dir']}\n")
            f.write(f"  Image Save Directory: {result['img_save_dir']}\n")
            f.write(f"  Result File Path: {result['result_file_path']}\n")
            f.write(f"  Executed Steps: {result['executed_steps']}\n")
            if result['process_status'] == "success" and result['gpt_prediction']:
                f.write(f"  GPT Prediction Preview: {result['gpt_prediction'][:150]}...\n")
            f.write("-" * 100 + "\n")

    print("=" * 100)
    print("📊 Batch Processing Complete - Summary Report Generated (CSV Driven + 6 Actions)")
    print("=" * 100)
    print(f"📈 Overall Stats: Total={total} | Success={success} | Failed={failed} | Success Rate={success_rate:.1f}%")
    print(f"📄 Summary Report: {os.path.abspath(summary_file)}")
    print(f"📦 All Results Saved In: {os.path.abspath(BATCH_ROOT_DIR)}")
    print("=" * 100)


# ------------------------------
# 程序运行入口（无修改）
# ------------------------------
if __name__ == "__main__":
    # 执行CSV驱动的批量视觉定位
    batch_results = batch_visual_positioning()

    # 单独运行单个点位（注释上面一行，取消下面注释并修改point_info）
    # point_info = {"name": "Test_NewYork", "lat": 40.7580, "lng": -73.9855}
    # point_dir, img_save_dir = init_point_dir(point_info)
    # env = StreetViewAutoLocEnv(
    #     initial_lat=point_info["lat"],
    #     initial_lng=point_info["lng"],
    #     point_dir=point_dir,
    #     img_save_dir=img_save_dir
    # )
    # point_result = env.auto_visual_positioning()