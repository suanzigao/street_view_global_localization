import os
import math
import time
import requests
from typing import List, Dict, Tuple
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 全局代理配置
PROXY_HOST = "127.0.0.1"
PROXY_PORT = 7897
os.environ["HTTP_PROXY"] = f"http://{PROXY_HOST}:{PROXY_PORT}"
os.environ["HTTPS_PROXY"] = f"http://{PROXY_HOST}:{PROXY_PORT}"
os.environ["NO_PROXY"] = "localhost,127.0.0.1,localdomain,.localdomain"

# 全局配置
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
METADATA_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"
MAX_RETRIES = 3
RETRY_DELAY = 2.0


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


def get_street_view_metadata(lat: float, lng: float, radius: int = 50) -> Dict:
    """
    获取指定坐标的Street View元数据
    """
    params = {
        "location": f"{lat},{lng}",
        "radius": radius,
        "key": GOOGLE_API_KEY
    }
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(METADATA_URL, params=params, timeout=10)
            return response.json()
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            if attempt < MAX_RETRIES - 1:
                print(f"连接超时，{RETRY_DELAY}秒后重试... (尝试 {attempt + 1}/{MAX_RETRIES})")
                time.sleep(RETRY_DELAY)
            else:
                print(f"获取元数据失败: {e}")
                return {"status": "ERROR"}
        except Exception as e:
            print(f"获取元数据时发生错误: {e}")
            return {"status": "ERROR"}


def generate_grid_points(center_lat: float, center_lng: float, radius: float, grid_step: float) -> List[Tuple[float, float]]:
    """
    在指定半径范围内生成网格点
    """
    points = []
    # 计算网格范围
    # 纬度每度约111320米，经度每度约111320*cos(lat)米
    lat_step = grid_step / 111320.0
    lng_step = grid_step / (111320.0 * math.cos(math.radians(center_lat)))
    
    # 计算网格大小
    grid_size = int(radius * 2 / grid_step) + 1
    
    # 生成网格点
    for i in range(grid_size):
        for j in range(grid_size):
            # 计算相对位置
            rel_x = (i - grid_size/2) * grid_step
            rel_y = (j - grid_size/2) * grid_step
            
            # 计算距离中心的距离
            distance = math.sqrt(rel_x**2 + rel_y**2)
            if distance <= radius:
                # 转换为GPS坐标
                lat = center_lat + (rel_y / 111320.0)
                lng = center_lng + (rel_x / (111320.0 * math.cos(math.radians(center_lat))))
                points.append((lat, lng))
    
    return points


def detect_street_view_samples(center_lat: float, center_lng: float, radius: float) -> List[Dict]:
    """
    检测指定范围内的Street View采样点
    """
    # 生成网格点，网格步长设为20米（加快速度）
    grid_points = generate_grid_points(center_lat, center_lng, radius, 20)
    
    # 存储唯一的采样点
    sample_points = []
    seen_locations = set()
    
    total_points = len(grid_points)
    print(f"开始检测 {total_points} 个网格点...")
    
    for i, (lat, lng) in enumerate(grid_points):
        if i % 10 == 0:
            print(f"检测进度: {i}/{total_points} ({i/total_points*100:.1f}%)")
        
        metadata = get_street_view_metadata(lat, lng, int(radius))
        if metadata.get("status") == "OK" and "location" in metadata:
            # 获取实际采样点坐标
            actual_lat = metadata["location"]["lat"]
            actual_lng = metadata["location"]["lng"]
            
            # 检查是否已经添加过这个位置（避免重复）
            location_key = f"{actual_lat:.6f},{actual_lng:.6f}"
            if location_key not in seen_locations:
                seen_locations.add(location_key)
                sample_points.append({
                    "lat": actual_lat,
                    "lng": actual_lng,
                    "distance_from_center": calculate_gps_distance(center_lat, center_lng, actual_lat, actual_lng)
                })
    
    # 按距离中心的距离排序
    sample_points.sort(key=lambda x: x["distance_from_center"])
    
    print(f"检测完成，找到 {len(sample_points)} 个采样点")
    return sample_points


def calculate_sample_spacing(sample_points: List[Dict]) -> List[Dict]:
    """
    计算采样点之间的间距（每个点与最近点的距离）
    """
    if len(sample_points) < 2:
        return sample_points
    
    # 计算每个点与其他所有点的距离，取最小值
    for i, curr_point in enumerate(sample_points):
        min_distance = float('inf')
        for j, other_point in enumerate(sample_points):
            if i != j:
                distance = calculate_gps_distance(
                    curr_point["lat"], curr_point["lng"],
                    other_point["lat"], other_point["lng"]
                )
                if distance < min_distance:
                    min_distance = distance
        curr_point["closest_distance"] = min_distance
    
    return sample_points


def analyze_street_view_samples(center_lat: float, center_lng: float, radius: float) -> Dict:
    """
    分析指定范围内的Street View采样点
    """
    # 检测采样点
    sample_points = detect_street_view_samples(center_lat, center_lng, radius)
    
    # 计算采样点间距
    sample_points = calculate_sample_spacing(sample_points)
    
    # 计算平均间距（基于最近点距离）
    total_distance = 0
    count = 0
    for point in sample_points:
        if "closest_distance" in point:
            total_distance += point["closest_distance"]
            count += 1
    
    average_spacing = total_distance / count if count > 0 else 0
    
    return {
        "center_lat": center_lat,
        "center_lng": center_lng,
        "radius": radius,
        "total_samples": len(sample_points),
        "average_spacing": average_spacing,
        "sample_points": sample_points
    }


if __name__ == "__main__":
    # 示例用法
    center_lat = 47.53097481458263
    center_lng = 19.06958490497365
    radius = 100  # 100米范围
    
    print(f"分析中心点 ({center_lat:.6f}, {center_lng:.6f}) 周围 {radius} 米范围内的Street View采样点...")
    
    result = analyze_street_view_samples(center_lat, center_lng, radius)
    
    print(f"\n分析结果:")
    print(f"总采样点数量: {result['total_samples']}")
    print(f"平均采样点间距: {result['average_spacing']:.2f} 米")
    
    print(f"\n采样点详情:")
    for i, point in enumerate(result['sample_points'], 1):
        distance_from_center = point['distance_from_center']
        closest_distance = point.get('closest_distance', 0)
        print(f"点 {i}: ({point['lat']:.6f}, {point['lng']:.6f}) - 距中心 {distance_from_center:.2f} 米")
        if closest_distance > 0:
            print(f"  与最近点距离: {closest_distance:.2f} 米")
