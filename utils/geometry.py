"""
地理计算工具模块

提供GPS坐标计算相关的工具函数，包括角度转换、距离计算等
"""

import math


def get_geo_angle(current_heading: float, relative_dir: str) -> float:
    """
    将相对方向转换为地理角度
    
    Args:
        current_heading: 当前朝向（度）
        relative_dir: 相对方向（forward/left/right/backward）
    
    Returns:
        地理角度（度），0度为正北
    
    Raises:
        ValueError: 无效的相对方向
    """
    angle_map = {
        "forward": current_heading,
        "left": current_heading - 90,
        "right": current_heading + 90,
        "backward": current_heading + 180
    }
    
    if relative_dir not in angle_map:
        raise ValueError(f"Invalid relative direction: {relative_dir}")
    
    return angle_map[relative_dir] % 360


def calculate_gps_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """
    使用Haversine公式计算两点GPS坐标的距离（米）
    
    Args:
        lat1, lng1: 起点坐标
        lat2, lng2: 终点坐标
    
    Returns:
        两点间距离（米）
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
    
    return R * c


def calculate_relative_offset(initial_lat: float, initial_lng: float, 
                               current_lat: float, current_lng: float) -> tuple:
    """
    计算当前GPS坐标相对于初始GPS坐标的东向和北向偏移（米）
    
    Args:
        initial_lat, initial_lng: 初始坐标
        current_lat, current_lng: 当前坐标
    
    Returns:
        (relative_x, relative_y): 东向偏移（米），北向偏移（米）
    """
    R = 6378137.0  # 地球赤道半径（米）
    
    initial_lat_rad = math.radians(initial_lat)
    initial_lng_rad = math.radians(initial_lng)
    current_lat_rad = math.radians(current_lat)
    current_lng_rad = math.radians(current_lng)
    
    dlat = current_lat_rad - initial_lat_rad
    dlng = current_lng_rad - initial_lng_rad
    
    # 北向偏移 = 纬度差 × 地球半径
    relative_y = dlat * R
    
    # 东向偏移 = 经度差 × 地球半径 × cos(纬度)
    avg_lat_rad = (initial_lat_rad + current_lat_rad) / 2
    relative_x = dlng * R * math.cos(avg_lat_rad)
    
    return relative_x, relative_y