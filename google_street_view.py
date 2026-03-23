import os
import math
import requests
from typing import Tuple, Optional, Dict
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv

# ====================== 全局代理配置 ======================
PROXY_HOST = "127.0.0.1"
PROXY_PORT = 7897
os.environ["HTTP_PROXY"] = f"http://{PROXY_HOST}:{PROXY_PORT}"
os.environ["HTTPS_PROXY"] = f"http://{PROXY_HOST}:{PROXY_PORT}"
os.environ["NO_PROXY"] = "localhost,127.0.0.1,localdomain,.localdomain"

# 加载.env文件中的环境变量
load_dotenv()

from config import GOOGLE_API_RADIUS, FOV_DEFAULT, GLOBAL_MOVE_STEP
from utils import retry_on_failure, calculate_gps_distance

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
        self.fov = FOV_DEFAULT

    def init_position(self, lat: float, lng: float, heading: float = 0.0, fov: float = FOV_DEFAULT, pitch: float = 0.0):
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
