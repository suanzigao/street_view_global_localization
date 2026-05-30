"""
导航API接口模块

提供标准化的导航操作接口，封装转向、移动、视角调整等操作
遵循单一职责原则，确保接口清晰、易于扩展
"""

from typing import Tuple, Optional, Dict
import math


class NavigationAPI:
    """
    导航API接口类
    
    提供标准化的导航操作：移动、转向、视角调整等
    所有操作均基于相对方向，便于上层调用
    """
    
    # 动作类型常量
    ACTION_MOVE = "move"
    ACTION_TURN = "turn"
    ACTION_FOV = "fov"
    ACTION_PITCH = "pitch"
    
    # 方向常量
    DIR_FORWARD = "forward"
    DIR_BACKWARD = "backward"
    DIR_LEFT = "left"
    DIR_RIGHT = "right"
    DIR_UP = "up"
    DIR_DOWN = "down"
    
    def __init__(self, 
                 move_step: float = 15.0, 
                 turn_angle: float = 45.0,
                 fov_step: int = 20,
                 pitch_step: int = 10,
                 fov_default: int = 60):
        """
        初始化导航API
        
        Args:
            move_step: 每次移动的步长（米），默认15米
            turn_angle: 每次转向的角度（度），默认45度
            fov_step: 每次FOV调整的步长（度），默认20度
            pitch_step: 每次俯仰调整的步长（度），默认10度
            fov_default: 默认FOV值，默认60度
        """
        self.move_step = move_step
        self.turn_angle = turn_angle
        self.fov_step = fov_step
        self.pitch_step = pitch_step
        self.fov_default = fov_default
        
        # 当前状态
        self._current_lat: float = 0.0
        self._current_lng: float = 0.0
        self._heading: float = 0.0
        self._pitch: float = 0.0
        self._fov: int = fov_default
    
    def init_position(self, 
                      lat: float, 
                      lng: float, 
                      heading: float = 0.0,
                      pitch: float = 0.0,
                      fov: int = None) -> None:
        """
        初始化位置和视角
        
        Args:
            lat: 纬度
            lng: 经度
            heading: 朝向角度（度），0度为正北
            pitch: 俯仰角度（度），0度为水平
            fov: 视角角度（度）
        """
        self._current_lat = lat
        self._current_lng = lng
        self._heading = heading % 360
        self._pitch = max(-90, min(90, pitch))
        self._fov = fov if fov is not None else self.fov_default
    
    def get_state(self) -> Dict[str, float]:
        """
        获取当前导航状态
        
        Returns:
            当前状态字典，包含lat, lng, heading, pitch, fov
        """
        return {
            "lat": self._current_lat,
            "lng": self._current_lng,
            "heading": self._heading,
            "pitch": self._pitch,
            "fov": self._fov
        }
    
    def get_geo_angle(self, relative_dir: str) -> float:
        """
        将相对方向转换为地理角度
        
        Args:
            relative_dir: 相对方向（forward/left/right/backward）
        
        Returns:
            地理角度（度），0度为正北
        
        Raises:
            ValueError: 无效的相对方向
        """
        angle_map = {
            self.DIR_FORWARD: self._heading,
            self.DIR_LEFT: self._heading - 90,
            self.DIR_RIGHT: self._heading + 90,
            self.DIR_BACKWARD: self._heading + 180
        }
        
        if relative_dir not in angle_map:
            raise ValueError(f"Invalid relative direction: {relative_dir}")
        
        return angle_map[relative_dir] % 360
    
    def turn(self, direction: str) -> Tuple[bool, str]:
        """
        转向操作
        
        Args:
            direction: 转向方向（left/right）
        
        Returns:
            (success: bool, message: str)
        
        Raises:
            ValueError: 无效的转向方向
        """
        if direction not in (self.DIR_LEFT, self.DIR_RIGHT):
            raise ValueError(f"Invalid turn direction: {direction}")
        
        delta = -self.turn_angle if direction == self.DIR_LEFT else self.turn_angle
        self._heading = (self._heading + delta) % 360
        
        return True, f"Turned {direction.upper()} {abs(delta):.1f}°, heading: {self._heading:.1f}°"
    
    def calculate_move_coords(self, relative_dir: str, distance: float = None) -> Tuple[float, float]:
        """
        计算移动后的坐标（基于Haversine公式）
        
        Args:
            relative_dir: 相对移动方向（forward/left/right/backward）
            distance: 移动距离（米），默认为move_step
        
        Returns:
            (new_lat, new_lng): 新的经纬度坐标
        
        Raises:
            ValueError: 无效的移动方向
        """
        distance = distance if distance is not None else self.move_step
        geo_angle = self.get_geo_angle(relative_dir)
        
        R = 6378137.0  # 地球赤道半径（米）
        lat_r = math.radians(self._current_lat)
        lng_r = math.radians(self._current_lng)
        ang_r = math.radians(geo_angle)
        d = distance
        
        new_lat_r = math.asin(
            math.sin(lat_r) * math.cos(d / R) +
            math.cos(lat_r) * math.sin(d / R) * math.cos(ang_r)
        )
        new_lng_r = lng_r + math.atan2(
            math.sin(ang_r) * math.sin(d / R) * math.cos(lat_r),
            math.cos(d / R) - math.sin(lat_r) * math.sin(new_lat_r)
        )
        
        return math.degrees(new_lat_r), math.degrees(new_lng_r)
    
    def move(self, 
             relative_dir: str, 
             distance: float = None) -> Tuple[bool, str, float]:
        """
        移动操作（仅更新内部状态，不调用外部API）
        
        Args:
            relative_dir: 相对移动方向（forward/left/right/backward）
            distance: 移动距离（米），默认为move_step
        
        Returns:
            (success: bool, message: str, actual_distance: float)
        
        Raises:
            ValueError: 无效的移动方向
        """
        new_lat, new_lng = self.calculate_move_coords(relative_dir, distance)
        
        # 计算实际移动距离
        actual_distance = self._calculate_distance(
            self._current_lat, self._current_lng,
            new_lat, new_lng
        )
        
        # 更新位置
        self._current_lat, self._current_lng = new_lat, new_lng
        
        return True, f"Moved {relative_dir} {actual_distance:.2f}m", actual_distance
    
    def set_position(self, lat: float, lng: float) -> None:
        """
        直接设置位置（用于外部坐标修正）
        
        Args:
            lat: 纬度
            lng: 经度
        """
        self._current_lat = lat
        self._current_lng = lng
    
    def adjust_fov(self, direction: str) -> Tuple[bool, str]:
        """
        调整视角FOV
        
        Args:
            direction: 调整方向（zoom_in/zoom_out）
        
        Returns:
            (success: bool, message: str)
        
        Raises:
            ValueError: 无效的调整方向
        """
        if direction == self.DIR_UP:  # zoom_in
            new_fov = max(10, self._fov - self.fov_step)
            action_desc = "ZOOM IN"
        elif direction == self.DIR_DOWN:  # zoom_out
            new_fov = min(120, self._fov + self.fov_step)
            action_desc = "ZOOM OUT"
        else:
            raise ValueError(f"Invalid FOV direction: {direction}")
        
        self._fov = new_fov
        return True, f"FOV {action_desc} to {self._fov}°"
    
    def adjust_pitch(self, direction: str) -> Tuple[bool, str]:
        """
        调整俯仰角度
        
        Args:
            direction: 调整方向（up/down）
        
        Returns:
            (success: bool, message: str)
        
        Raises:
            ValueError: 无效的调整方向
        """
        if direction == self.DIR_UP:  # look up
            new_pitch = min(90, self._pitch + self.pitch_step)
            action_desc = "UP"
        elif direction == self.DIR_DOWN:  # look down
            new_pitch = max(-90, self._pitch - self.pitch_step)
            action_desc = "DOWN"
        else:
            raise ValueError(f"Invalid pitch direction: {direction}")
        
        self._pitch = new_pitch
        return True, f"Pitch {action_desc} to {self._pitch}°"
    
    def _calculate_distance(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """
        使用Haversine公式计算两点GPS坐标的距离（米）
        
        Args:
            lat1, lng1: 起点坐标
            lat2, lng2: 终点坐标
        
        Returns:
            两点间距离（米）
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