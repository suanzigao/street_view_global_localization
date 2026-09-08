"""
Google街景数据处理模块

封装Google Street View API的调用逻辑，提供街景图像获取和元数据查询功能
"""

import os
import math
import requests
from PIL import Image
from io import BytesIO
from typing import Tuple, Optional, Dict
from utils.decorators import retry_on_failure


class GoogleStreetViewData:
    """
    Google街景数据处理器
    
    负责与Google Street View API交互，获取街景图像和元数据
    """
    
    def __init__(self, 
                 api_key: str = None, 
                 max_retries: int = 3, 
                 move_step: float = 15.0,
                 radius: int = 10,
                 image_size: str = "640x640"):
        """
        初始化街景数据处理器
        
        Args:
            api_key: Google API密钥，默认为环境变量GOOGLE_API_KEY
            max_retries: 最大重试次数，默认3次
            move_step: 移动步长（米），默认15米
            radius: 搜索半径（米），默认10米
            image_size: 返回图像尺寸，默认640x640
        """
        self.google_api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.static_url = "https://maps.googleapis.com/maps/api/streetview"
        self.metadata_url = "https://maps.googleapis.com/maps/api/streetview/metadata"
        self.max_retries = max_retries
        self.move_step = move_step
        self.radius = radius
        self.image_size = image_size
        
        # 初始状态
        self.initial_lat = 0.0
        self.initial_lng = 0.0
        self.initial_heading = 0.0
        self.initial_pitch = 0.0
        self.initial_fov = 60
        
        # 当前状态
        self.current_lat = 0.0
        self.current_lng = 0.0
        self.heading = 0.0
        self.pitch = 0.0
        self.fov = 60
    
    def init_position(self, 
                      lat: float, 
                      lng: float, 
                      heading: float = 0.0, 
                      fov: int = 60, 
                      pitch: float = 0.0) -> None:
        """
        初始化位置和视角参数
        
        Args:
            lat: 纬度
            lng: 经度
            heading: 朝向角度（度）
            fov: 视角角度（度）
            pitch: 俯仰角度（度）
        """
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
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def get_metadata(self, lat: float, lng: float) -> Optional[Dict]:
        """
        获取指定位置的街景元数据
        
        Args:
            lat: 纬度
            lng: 经度
        
        Returns:
            元数据字典，包含实际采样点坐标等信息；若无街景数据返回None
        """
        params = {
            "location": f"{lat},{lng}",
            "radius": self.radius,
            "key": self.google_api_key
        }
        
        response = requests.get(self.metadata_url, params=params, timeout=10)
        data = response.json()
        
        if data.get("status") == "OK" and "location" in data:
            actual_lat = data["location"]["lat"]
            actual_lng = data["location"]["lng"]
            
            # 检查实际街景点距离是否在允许范围内
            distance = self._calculate_distance(lat, lng, actual_lat, actual_lng)
            if distance <= self.radius:
                return data
        
        return None
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def get_street_view_image(self) -> Tuple[Optional[Image.Image], Optional[float], Optional[float]]:
        """
        获取当前位置的街景图像
        
        Returns:
            (image: Image.Image, actual_lat: float, actual_lng: float)
            其中actual_lat和actual_lng是实际采样点的坐标
        """
        params = {
            "location": f"{self.current_lat},{self.current_lng}",
            "heading": self.heading,
            "pitch": self.pitch,
            "fov": self.fov,
            "radius": self.radius,
            "size": self.image_size,
            "key": self.google_api_key
        }
        
        # 先获取元数据（包含实际采样点坐标）
        metadata = self.get_metadata(self.current_lat, self.current_lng)
        actual_lat, actual_lng = None, None
        if metadata and "location" in metadata:
            actual_lat = metadata["location"]["lat"]
            actual_lng = metadata["location"]["lng"]
        
        # 获取街景图像
        response = requests.get(self.static_url, params=params, stream=True, timeout=10)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        
        return image, actual_lat, actual_lng
    
    def turn(self, delta_angle: float) -> None:
        """
        调整朝向
        
        Args:
            delta_angle: 角度变化量（度），正数为右转，负数为左转
        """
        self.heading = (self.heading + delta_angle) % 360
    
    def move(self, 
             direction: float, 
             target_distance: float = None) -> Tuple[bool, str, float]:
        """
        移动到新位置（带街景可用性检查）
        
        Args:
            direction: 移动方向（地理角度，度）
            target_distance: 目标移动距离（米），默认为move_step
        
        Returns:
            (success: bool, message: str, actual_distance: float)
        """
        target_distance = target_distance or self.move_step
        
        # 计算新坐标
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
        
        # 检查新位置是否有街景
        if not self.get_metadata(nlat, nlng):
            return False, "No street view available at target location", 0.0
        
        # 计算实际移动距离
        actual_distance = self._calculate_distance(
            self.current_lat, self.current_lng,
            nlat, nlng
        )
        
        # 更新坐标
        self.current_lat, self.current_lng = nlat, nlng
        
        return True, "Move successful", actual_distance
    
    def _calculate_distance(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """
        使用Haversine公式计算两点距离（米）
        
        Args:
            lat1, lng1: 起点坐标
            lat2, lng2: 终点坐标
        
        Returns:
            距离（米）
        """
        R = 6378137.0
        lat1_rad = math.radians(lat1)
        lng1_rad = math.radians(lng1)
        lat2_rad = math.radians(lat2)
        lng2_rad = math.radians(lng2)
        
        dlat = lat2_rad - lat1_rad
        dlng = lng2_rad - lng1_rad
        
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlng / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c
