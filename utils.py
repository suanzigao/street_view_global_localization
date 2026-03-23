import time
import math
import csv
import os
import shutil
import base64
from typing import List, Dict, Optional, Tuple, Any
from PIL import Image
from io import BytesIO

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
def init_batch_dir(batch_root_dir: str):
    if os.path.exists(batch_root_dir):
        shutil.rmtree(batch_root_dir)
    os.makedirs(batch_root_dir, exist_ok=True)


def get_point_dir_name(point_info: Dict) -> str:
    lat_str = f"{point_info['lat']:.4f}".replace(".", "_")
    lng_str = f"{point_info['lng']:.4f}".replace(".", "_")
    return f"{point_info['name']}_lat{lat_str}_lng{lng_str}"


def init_point_dir(point_info: Dict, batch_root_dir: str) -> Tuple[str, str]:
    point_dir = os.path.join(batch_root_dir, get_point_dir_name(point_info))
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
