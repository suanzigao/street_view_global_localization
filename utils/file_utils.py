"""
文件工具模块

提供文件操作相关的工具函数，包括目录初始化、图像保存、CSV读取等
"""

import os
import csv
import datetime
from io import BytesIO
from typing import Tuple, Dict, List
from PIL import Image


def init_batch_dir(batch_root_dir: str = "batch_streetview_results") -> str:
    """
    初始化批处理目录
    
    Args:
        batch_root_dir: 批处理根目录
    
    Returns:
        当前批处理目录路径
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    batch_dir = os.path.join(batch_root_dir, f"{timestamp}_batch_run")
    os.makedirs(batch_dir, exist_ok=True)
    return batch_dir


def get_point_dir_name(point_info: Dict) -> str:
    """
    生成点位目录名称
    
    Args:
        point_info: 点位信息字典
    
    Returns:
        目录名称字符串
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    lat_str = f"{point_info['lat']:.4f}".replace(".", "_")
    lng_str = f"{point_info['lng']:.4f}".replace(".", "_")
    return f"{timestamp}_{point_info['name']}_lat{lat_str}_lng{lng_str}"


def init_point_dir(point_info: Dict, batch_dir: str = None, batch_root_dir: str = "batch_streetview_results") -> Tuple[str, str]:
    """
    初始化单个点位的目录结构
    
    Args:
        point_info: 点位信息字典
        batch_dir: 批处理目录（可选）
        batch_root_dir: 批处理根目录
    
    Returns:
        (point_dir: str, img_dir: str)
    """
    if batch_dir is not None:
        point_dir = os.path.join(batch_dir, get_point_dir_name(point_info))
    else:
        point_dir = os.path.join(batch_root_dir, get_point_dir_name(point_info))
    
    img_dir = os.path.join(point_dir, "streetview_images")
    os.makedirs(img_dir, exist_ok=True)
    
    return point_dir, img_dir


def image_to_base64(image: Image.Image) -> str:
    """
    将PIL图像转换为Base64编码字符串
    
    Args:
        image: PIL图像对象
    
    Returns:
        Base64编码字符串
    """
    import base64
    
    buf = BytesIO()
    image.save(buf, format='JPEG', quality=100)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def save_streetview_image(image: Image.Image, step_num: int, img_save_dir: str) -> Tuple[str, str]:
    """
    保存街景图像并返回路径和Base64编码
    
    Args:
        image: PIL图像对象
        step_num: 当前步骤编号
        img_save_dir: 图像保存目录
    
    Returns:
        (img_path: str, img_b64: str)
    """
    import base64
    
    # 确保目录存在
    os.makedirs(img_save_dir, exist_ok=True)
    
    img_path = os.path.join(img_save_dir, f"step_{step_num}_streetview.jpg")
    
    image.save(
        img_path,
        format='JPEG',
        quality=100,
        subsampling=0,
        optimize=False
    )
    
    buf = BytesIO()
    image.save(buf, format='JPEG', quality=100)
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    return img_path, img_b64


def read_gps_targets_from_csv(csv_path: str) -> List[Dict]:
    """
    从CSV文件读取GPS目标点位
    
    Args:
        csv_path: CSV文件路径
    
    Returns:
        GPS点位列表
    
    Raises:
        FileNotFoundError: CSV文件不存在
        ValueError: CSV格式不正确或无有效点位
    """
    gps_targets = []
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"GPS CSV file not found: {csv_path}")
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_headers = ["name", "lat", "lng"]
        
        if not all(h in reader.fieldnames for h in required_headers):
            raise ValueError(f"CSV must contain headers: {required_headers}")
        
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
            
            except Exception:
                continue
    
    if not gps_targets:
        raise ValueError("No valid GPS points found in CSV")
    
    return gps_targets