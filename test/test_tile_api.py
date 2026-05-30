import os
import math
import time
import csv
import base64
import requests
import numpy as np
from PIL import Image
from io import BytesIO
from typing import Tuple, Optional, Dict, Any, List
from dotenv import load_dotenv

# ====================== 全局代理配置 ======================
PROXY_HOST = "127.0.0.1"
PROXY_PORT = 7897
os.environ["HTTP_PROXY"] = f"http://{PROXY_HOST}:{PROXY_PORT}"
os.environ["HTTPS_PROXY"] = f"http://{PROXY_HOST}:{PROXY_PORT}"
os.environ["NO_PROXY"] = "localhost,127.0.0.1,localdomain,.localdomain"

# ====================== 屏蔽警告 ==============================
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Gym has been unmaintained")
warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()

# ====================== 全局配置 ==============================
GLOBAL_MAX_STEPS = 15
GLOBAL_MOVE_STEP = 15.0
GLOBAL_MAX_RETRIES = 3
# 新增：FOV/俯仰控制参数
FOV_STEP = 20
PITCH_STEP = 10
TURN_ANGLE = 30
FOV_DEFAULT = 50
# 新增：Google API参数
GOOGLE_API_RADIUS = 50

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

# ====================== Google Street View Tile API ======================
class GoogleStreetViewTileAPI:
    def __init__(self, api_key: str = None, max_retries: int = 3, move_step: float = GLOBAL_MOVE_STEP):
        self.google_api_key = api_key or os.getenv("GOOGLE_API_KEY")
        # 根据官方文档使用正确的API端点
        self.create_session_url = "https://tile.googleapis.com/v1/createSession"
        self.pano_ids_url = "https://tile.googleapis.com/v1/streetview/panoIds"
        self.metadata_url = "https://tile.googleapis.com/v1/streetview/metadata"
        self.tile_url = "https://tile.googleapis.com/v1/streetview/tiles"
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
        self.current_pano_id = None
        self.session_token = None

    def create_session(self) -> Optional[str]:
        """
        创建会话令牌
        """
        params = {
            "key": self.google_api_key
        }
        data = {
            "mapType": "streetview",
            "language": "zh-CN"
        }
        try:
            r = requests.post(self.create_session_url, params=params, json=data, timeout=10)
            if r.status_code == 200:
                response = r.json()
                session = response.get("session")
                if session:
                    print(f"✅ 会话令牌创建成功: {session}")
                    return session
                else:
                    print("❌ 会话令牌创建失败: 响应中无session字段")
            else:
                print(f"❌ 会话令牌创建失败，状态码: {r.status_code}")
                print(f"响应内容: {r.content[:200]}...")
        except Exception as e:
            print(f"❌ 会话令牌创建异常: {str(e)}")
        return None

    def get_pano_id(self, lat: float, lng: float) -> Optional[str]:
        """
        获取指定位置的Pano ID
        """
        if not self.session_token:
            self.session_token = self.create_session()
            if not self.session_token:
                return None

        params = {
            "session": self.session_token,
            "key": self.google_api_key
        }
        data = {
            "locations": [{
                "lat": lat,
                "lng": lng
            }],
            "radius": GOOGLE_API_RADIUS
        }
        try:
            r = requests.post(self.pano_ids_url, params=params, json=data, timeout=10)
            if r.status_code == 200:
                response = r.json()
                pano_ids = response.get("panoIds", [])
                if pano_ids and pano_ids[0]:
                    pano_id = pano_ids[0]
                    print(f"✅ Pano ID获取成功: {pano_id}")
                    return pano_id
                else:
                    print("❌ Pano ID获取失败: 无有效Pano ID")
            else:
                print(f"❌ Pano ID获取失败，状态码: {r.status_code}")
                print(f"响应内容: {r.content[:200]}...")
        except Exception as e:
            print(f"❌ Pano ID获取异常: {str(e)}")
        return None

    def get_streetview_metadata(self, pano_id: str) -> Optional[Dict]:
        """
        获取街景元数据
        """
        if not self.session_token:
            self.session_token = self.create_session()
            if not self.session_token:
                return None

        params = {
            "session": self.session_token,
            "key": self.google_api_key,
            "panoId": pano_id
        }
        try:
            r = requests.get(self.metadata_url, params=params, timeout=10)
            if r.status_code == 200:
                response = r.json()
                print("✅ 街景元数据获取成功")
                return response
            else:
                print(f"❌ 街景元数据获取失败，状态码: {r.status_code}")
                print(f"响应内容: {r.content[:200]}...")
        except Exception as e:
            print(f"❌ 街景元数据获取异常: {str(e)}")
        return None

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
        # 获取初始位置的pano ID
        self.current_pano_id = self.get_pano_id(lat, lng)

    @retry_on_failure(3, 1.0)
    def get_tile(self, pano_id: str, zoom: int, x: int, y: int) -> Optional[Image.Image]:
        """
        获取单个街景瓦片
        """
        if not self.session_token:
            self.session_token = self.create_session()
            if not self.session_token:
                return None

        # 构建正确的瓦片URL格式
        tile_url = f"{self.tile_url}/{zoom}/{x}/{y}"
        params = {
            "session": self.session_token,
            "key": self.google_api_key,
            "panoId": pano_id
        }
        
        r = requests.get(tile_url, params=params, stream=True, timeout=10)
        if r.status_code == 200:
            # 检查响应内容是否为图像
            content_type = r.headers.get('Content-Type', '')
            if 'image' in content_type:
                image = Image.open(BytesIO(r.content)).convert("RGB")
                return image
            else:
                print(f"⚠️  响应不是图像: {content_type}")
                print(f"响应内容: {r.content[:100]}...")
                raise Exception(f"响应不是图像: {content_type}")
        else:
            print(f"⚠️  获取瓦片失败，状态码: {r.status_code}")
            print(f"响应内容: {r.content[:100]}...")
            raise Exception(f"获取瓦片失败，状态码: {r.status_code}")

    def get_all_tiles(self, zoom: int = 0, save_dir: str = None) -> Tuple[Optional[Image.Image], int, List[Dict], float, Optional[Tuple[int, int]]]:
        """
        获取当前位置的所有瓦片并拼接成完整图像
        只使用Tile API，不使用其他API
        
        Args:
            zoom: 缩放级别
            save_dir: 保存瓦片图片的目录路径
            
        Returns:
            Tuple[Optional[Image.Image], int, List[Dict], float, Optional[Tuple[int, int]]]: 
            拼接后的图像（如果成功）、成功获取的瓦片数、每个瓦片的详细信息列表、总用时（秒）、图像尺寸
        """
        
        # 记录总开始时间
        total_start_time = time.time()
        
        if not self.current_pano_id:
            # 尝试获取当前位置的Pano ID
            self.current_pano_id = self.get_pano_id(self.current_lat, self.current_lng)
        
        if not self.current_pano_id:
            print("❌ 无法获取Pano ID")
            return None, 0, [], 0.0

        # 先获取元数据，从中读取瓦片尺寸
        tile_width, tile_height = 512, 512  # 默认值
        metadata = self.get_streetview_metadata(self.current_pano_id)
        if metadata:
            # 从元数据中提取瓦片尺寸
            if "tileWidth" in metadata and "tileHeight" in metadata:
                tile_width = metadata["tileWidth"]
                tile_height = metadata["tileHeight"]
                print(f"✅ 从元数据获取瓦片尺寸: {tile_width}x{tile_height}")
            else:
                print(f"⚠️ 元数据中未找到tileWidth/tileHeight，使用默认值 512x512")
                if metadata:
                    print(f"   元数据内容: {metadata}")

        # 根据zoom级别计算瓦片数量
        # zoom=0: 1x1瓦片
        # zoom=1: 2x2瓦片
        # zoom=2: 4x4瓦片
        # 以此类推
        tiles_per_side = 2 ** zoom
        total_width = tile_width * tiles_per_side
        total_height = tile_height * tiles_per_side

        # 创建空白画布（使用白色背景，便于观察）
        result = Image.new('RGB', (total_width, total_height), color=(255, 255, 255))

        # 获取瓦片并拼接
        tile_count = 0
        tile_details = []  # 存储每个瓦片的详细信息
        
        # 跟踪实际获取到的瓦片位置
        min_x = tiles_per_side
        max_x = -1
        min_y = tiles_per_side
        max_y = -1
        
        print(f"正在获取瓦片: zoom={zoom}, 共{tiles_per_side}x{tiles_per_side}个瓦片")
        print(f"Pano ID: {self.current_pano_id}")
        print(f"瓦片尺寸: {tile_width}x{tile_height}")
        
        # 循环获取所有瓦片
        for y in range(tiles_per_side):
            # 先获取 x=0 的瓦片
            x0_success = False
            if tiles_per_side > 0:
                tile_url = f"{self.tile_url}/{zoom}/0/{y}?session={self.session_token}&key={self.google_api_key}&panoId={self.current_pano_id}"
                print(f"瓦片URL: {tile_url}")
                
                try:
                    start_time = time.time()
                    tile = self.get_tile(self.current_pano_id, zoom, 0, y)
                    duration = time.time() - start_time
                    if tile:
                        result.paste(tile, (0 * tile_width, y * tile_height))
                        tile_count += 1
                        x0_success = True
                        
                        # 更新边界
                        min_x = min(min_x, 0)
                        max_x = max(max_x, 0)
                        min_y = min(min_y, y)
                        max_y = max(max_y, y)
                        
                        # 保存瓦片图片
                        if save_dir:
                            tile_filename = f"zoom{zoom}_x0_y{y}.jpg"
                            tile_path = os.path.join(save_dir, tile_filename)
                            tile.save(tile_path, quality=100)
                            print(f"✅ 瓦片 x=0, y={y} 保存至 {tile_path}")
                        
                        # 记录瓦片信息（使用元数据中的尺寸）
                        tile_info = {
                            "zoom": zoom,
                            "x": 0,
                            "y": y,
                            "success": True,
                            "width": tile_width,
                            "height": tile_height,
                            "duration": duration
                        }
                        tile_details.append(tile_info)
                        
                        print(f"✅ 瓦片 x=0, y={y} 获取成功，尺寸: {tile_width}x{tile_height}，耗时: {duration:.2f}秒")
                    else:
                        print(f"❌ 瓦片 x=0, y={y} 获取失败，耗时: {duration:.2f}秒")
                        tile_info = {
                            "zoom": zoom,
                            "x": 0,
                            "y": y,
                            "success": False,
                            "width": 0,
                            "height": 0,
                            "duration": duration
                        }
                        tile_details.append(tile_info)
                except Exception as e:
                    duration = time.time() - start_time
                    print(f"❌ 瓦片 x=0, y={y} 获取失败: {str(e)}，耗时: {duration:.2f}秒")
                    tile_info = {
                        "zoom": zoom,
                        "x": 0,
                        "y": y,
                        "success": False,
                        "width": 0,
                        "height": 0,
                        "duration": duration
                    }
                    tile_details.append(tile_info)
            
            # 如果 x=0 的瓦片获取失败，跳过这一行的所有瓦片
            if not x0_success:
                print(f"⚠️  跳过 y={y} 行的所有瓦片，因为 x=0 的瓦片获取失败")
                continue
            
            # 获取该行其他 x 坐标的瓦片
            for x in range(1, tiles_per_side):
                # 构建完整的URL以便调试
                tile_url = f"{self.tile_url}/{zoom}/{x}/{y}?session={self.session_token}&key={self.google_api_key}&panoId={self.current_pano_id}"
                print(f"瓦片URL: {tile_url}")
                
                tile_success = False
                for retry in range(3):
                    try:
                        start_time = time.time()
                        tile = self.get_tile(self.current_pano_id, zoom, x, y)
                        duration = time.time() - start_time
                        if tile:
                            result.paste(tile, (x * tile_width, y * tile_height))
                            tile_count += 1
                            tile_success = True
                            
                            # 更新边界
                            min_x = min(min_x, x)
                            max_x = max(max_x, x)
                            min_y = min(min_y, y)
                            max_y = max(max_y, y)
                            
                            # 保存瓦片图片
                            if save_dir:
                                tile_filename = f"zoom{zoom}_x{x}_y{y}.jpg"
                                tile_path = os.path.join(save_dir, tile_filename)
                                tile.save(tile_path, quality=100)
                                print(f"✅ 瓦片 x={x}, y={y} 保存至 {tile_path}")
                            
                            # 记录瓦片信息（使用元数据中的尺寸）
                            tile_info = {
                                "zoom": zoom,
                                "x": x,
                                "y": y,
                                "success": True,
                                "width": tile_width,
                                "height": tile_height,
                                "duration": duration
                            }
                            tile_details.append(tile_info)
                            
                            print(f"✅ 瓦片 x={x}, y={y} 获取成功，尺寸: {tile_width}x{tile_height}，耗时: {duration:.2f}秒")
                            break
                        else:
                            if retry < 2:
                                print(f"❌ 瓦片 x={x}, y={y} 获取失败，重试中 ({retry + 1}/3)")
                                time.sleep(1.0)
                            else:
                                print(f"❌ 瓦片 x={x}, y={y} 获取失败，已重试3次，耗时: {duration:.2f}秒")
                                tile_info = {
                                    "zoom": zoom,
                                    "x": x,
                                    "y": y,
                                    "success": False,
                                    "width": 0,
                                    "height": 0,
                                    "duration": duration
                                }
                                tile_details.append(tile_info)
                    except Exception as e:
                        duration = time.time() - start_time
                        if retry < 2:
                            print(f"❌ 瓦片 x={x}, y={y} 获取失败: {str(e)}，重试中 ({retry + 1}/3)")
                            time.sleep(1.0)
                        else:
                            print(f"❌ 瓦片 x={x}, y={y} 获取失败: {str(e)}，已重试3次，耗时: {duration:.2f}秒")
                            tile_info = {
                                "zoom": zoom,
                                "x": x,
                                "y": y,
                                "success": False,
                                "width": 0,
                                "height": 0,
                                "duration": duration
                            }
                            tile_details.append(tile_info)
                
                # 如果该瓦片重试三次后仍然失败，停止获取该行的剩余瓦片
                if not tile_success:
                    print(f"⚠️  停止获取 y={y} 行的剩余瓦片")
                    break

        # 计算总用时
        total_duration = time.time() - total_start_time
        
        print(f"瓦片获取完成，成功 {tile_count}/{tiles_per_side*tiles_per_side}，总用时: {total_duration:.2f}秒")
        
        # 检查是否有瓦片被成功获取
        if tile_count == 0:
            print("❌ 没有成功获取任何瓦片")
            return None, 0, [], total_duration, None
        
        # 裁剪图像，只保留有瓦片覆盖的区域
        if min_x <= max_x and min_y <= max_y:
            crop_left = min_x * tile_width
            crop_top = min_y * tile_height
            crop_right = (max_x + 1) * tile_width
            crop_bottom = (max_y + 1) * tile_height
            
            result = result.crop((crop_left, crop_top, crop_right, crop_bottom))
            print(f"✅ 图像已裁剪，保留范围: x[{min_x}-{max_x}], y[{min_y}-{max_y}]")
            print(f"   裁剪后尺寸: {result.size[0]}x{result.size[1]}")
        
        # 获取图像尺寸
        image_size = result.size
        
        return result, tile_count, tile_details, total_duration, image_size
    
    def convert_to_perspective(self, image: Image.Image, fov: float = 90.0) -> Optional[Image.Image]:
        """
        将全景图像转换为类似相机拍摄的透视图
        
        Args:
            image: 原始全景图像（equirectangular projection）
            fov: 视野角度 (默认90度)，值越大视角越宽
            
        Returns:
            透视图图像
        """
        try:
            import cv2
            import numpy as np
            
            # 转换为OpenCV格式
            img_array = np.array(image)
            height, width = img_array.shape[:2]
            
            # 计算输出图像的尺寸
            # FOV越小，输出图像越小（模拟长焦镜头）
            output_width = max(1, int(width * fov / 360))
            output_height = max(1, int(height * fov / 360))
            
            # 创建网格坐标
            u = np.linspace(-1, 1, output_width)
            v = np.linspace(1, -1, output_height)  # 反转v坐标以修正上下颠倒
            u_grid, v_grid = np.meshgrid(u, v)
            
            # 计算射线角度（等距投影）
            theta = u_grid * np.radians(fov / 2)
            phi = v_grid * np.radians(fov / 2)
            
            # 计算球面坐标
            x_sphere = np.sin(theta) * np.cos(phi)
            y_sphere = np.sin(phi)
            z_sphere = np.cos(theta) * np.cos(phi)
            
            # 转换为全景图像的纹理坐标
            lon = np.arctan2(x_sphere, z_sphere)
            lat = np.arcsin(np.clip(y_sphere, -1, 1))
            
            # 转换为像素坐标
            # 在equirectangular投影中，tex_y = 0 对应顶部（北极），tex_y = height 对应底部（南极）
            tex_x = ((lon + np.pi) / (2 * np.pi) * width).astype(np.int32) % width
            tex_y = ((np.pi / 2 - lat) / np.pi * height).astype(np.int32) % height
            
            # 创建有效坐标掩码
            valid_mask = (tex_x >= 0) & (tex_x < width) & (tex_y >= 0) & (tex_y < height)
            
            # 创建输出图像
            output = np.zeros((output_height, output_width, 3), dtype=np.uint8)
            
            # 使用向量化操作填充输出图像
            output[valid_mask] = img_array[tex_y[valid_mask], tex_x[valid_mask]]
            
            # 转换回PIL格式
            result = Image.fromarray(output)
            
            print(f"✅ 图像已转换为透视图 (FOV={fov}°)，尺寸: {result.size[0]}x{result.size[1]}")
            return result
        except Exception as e:
            print(f"❌ 转换透视图失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

# ====================== 主测试函数 ======================
def test_tile_api():
    # 初始化Tile API
    tile_api = GoogleStreetViewTileAPI()
    
    # 创建会话令牌
    session = tile_api.create_session()
    if not session:
        print("❌ 无法创建会话令牌，退出测试")
        return
    
    # 创建结果文件夹
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_dir = os.path.join("streetview_results", f"{timestamp}_test")
    tiles_dir = os.path.join(result_dir, "tiles")
    os.makedirs(tiles_dir, exist_ok=True)
    print(f"✅ 结果文件夹已创建: {result_dir}")
    
    # 存储所有zoom级别的瓦片详细信息
    all_tile_details = []
    
    # 存储每个zoom级别的总用时
    zoom_total_durations = {}
    
    # 设置初始位置（示例坐标：纽约时代广场）
    initial_lat = 40.7580
    initial_lng = -73.9855
    tile_api.init_position(initial_lat, initial_lng, heading=0, fov=FOV_DEFAULT, pitch=0)
    
    print(f"初始位置: {initial_lat}, {initial_lng}")
    print(f"初始Pano ID: {tile_api.current_pano_id}")
    
    # 测试1: 获取初始位置的所有瓦片并拼接（不同zoom级别）
    for zoom_level in [0, 1, 2, 3, 4, 5]:
        print(f"\n测试1: 获取初始位置的所有瓦片 (zoom={zoom_level})")
        zoom_dir = os.path.join(tiles_dir, f"zoom_{zoom_level}")
        os.makedirs(zoom_dir, exist_ok=True)
        initial_image, success_count, tile_details, total_duration, image_size = tile_api.get_all_tiles(zoom=zoom_level, save_dir=zoom_dir)
        if initial_image:
            initial_image.save(os.path.join(result_dir, f"initial_streetview_zoom{zoom_level}.jpg"))
            print(f"✅ 初始街景图像已保存为 {os.path.join(result_dir, f'initial_streetview_zoom{zoom_level}.jpg')}")
            print(f"✅ 成功获取 {success_count} 个瓦片，总用时: {total_duration:.2f}秒")
            print(f"✅ 图像尺寸: {image_size[0]}x{image_size[1]}")
        else:
            print(f"❌ 无法获取初始街景图像 (zoom={zoom_level})")
            print(f"✅ 成功获取 {success_count} 个瓦片，总用时: {total_duration:.2f}秒")
        
        # 记录该zoom级别的瓦片信息
        for tile_info in tile_details:
            tile_info["position"] = "initial"
            tile_info["pano_id"] = tile_api.current_pano_id
        all_tile_details.extend(tile_details)
        
        # 记录该zoom级别的总用时和图像尺寸
        zoom_total_durations[zoom_level] = {
            "total_duration": total_duration,
            "image_width": image_size[0] if image_size else 0,
            "image_height": image_size[1] if image_size else 0
        }
    
    # 保存CSV文件
    csv_path = os.path.join(result_dir, "tile_details.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["zoom", "x", "y", "success", "width", "height", "duration", "position", "pano_id"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for tile_info in all_tile_details:
            writer.writerow(tile_info)
    print(f"✅ 瓦片详细信息已保存为 {csv_path}")
    
    # 生成zoom级别的统计信息
    zoom_stats = {}
    for tile_info in all_tile_details:
        zoom = tile_info["zoom"]
        if zoom not in zoom_stats:
            # 根据zoom级别计算理论上的瓦片总数：(2^zoom) x (2^zoom)
            tiles_per_side = 2 ** zoom
            total_theoretical = tiles_per_side * tiles_per_side
            zoom_stats[zoom] = {
                "total": total_theoretical,  # 理论总数
                "success": 0,
                "width_sum": 0,
                "height_sum": 0,
                "duration_sum": 0.0
            }
        if tile_info["success"]:
            zoom_stats[zoom]["success"] += 1
            zoom_stats[zoom]["width_sum"] += tile_info["width"]
            zoom_stats[zoom]["height_sum"] += tile_info["height"]
            zoom_stats[zoom]["duration_sum"] += tile_info.get("duration", 0)
    
    # 保存zoom级别的统计信息
    zoom_csv_path = os.path.join(result_dir, "zoom_statistics.csv")
    with open(zoom_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["zoom", "total_tiles", "success_tiles", "success_rate", "avg_width", "avg_height", "avg_duration", "total_duration", "image_width", "image_height"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for zoom, stats in sorted(zoom_stats.items()):
            avg_width = stats["width_sum"] / stats["success"] if stats["success"] > 0 else 0
            avg_height = stats["height_sum"] / stats["success"] if stats["success"] > 0 else 0
            avg_duration = stats["duration_sum"] / stats["success"] if stats["success"] > 0 else 0
            zoom_info = zoom_total_durations.get(zoom, {"total_duration": 0, "image_width": 0, "image_height": 0})
            total_duration = zoom_info["total_duration"]
            image_width = zoom_info["image_width"]
            image_height = zoom_info["image_height"]
            writer.writerow({
                "zoom": zoom,
                "total_tiles": stats["total"],
                "success_tiles": stats["success"],
                "success_rate": f"{stats['success'] / stats['total'] * 100:.2f}%",
                "avg_width": f"{avg_width:.2f}",
                "avg_height": f"{avg_height:.2f}",
                "avg_duration": f"{avg_duration:.2f}",
                "total_duration": f"{total_duration:.2f}",
                "image_width": image_width,
                "image_height": image_height
            })
    print(f"✅ Zoom级别统计信息已保存为 {zoom_csv_path}")
    
    print(f"\n✅ 所有结果已保存至: {result_dir}")

# ====================== 批量处理函数 ======================
def process_single_location(tile_api, name, lat, lng, result_dir):
    """
    处理单个位置的街景瓦片
    """
    print(f"\n====== 处理点位: {name} ======")
    print(f"坐标: {lat}, {lng}")
    
    # 初始化位置
    tile_api.init_position(lat, lng, heading=0, fov=FOV_DEFAULT, pitch=0)
    if not tile_api.current_pano_id:
        print(f"❌ 无法获取Pano ID，跳过该点位")
        return False, None
    
    print(f"Pano ID: {tile_api.current_pano_id}")
    
    # 创建该点位的瓦片文件夹
    tiles_dir = os.path.join(result_dir, f"{name}_tiles")
    os.makedirs(tiles_dir, exist_ok=True)
    
    # 存储所有zoom级别的瓦片详细信息
    all_tile_details = []
    
    # 存储每个zoom级别的总用时
    zoom_total_durations = {}
    
    # 测试1: 获取初始位置的所有瓦片并拼接（不同zoom级别）
    for zoom_level in [0, 1, 2, 3, 4, 5]:
        print(f"\n处理 zoom={zoom_level}")
        zoom_dir = os.path.join(tiles_dir, f"zoom_{zoom_level}")
        os.makedirs(zoom_dir, exist_ok=True)
        initial_image, success_count, tile_details, total_duration, image_size = tile_api.get_all_tiles(zoom=zoom_level, save_dir=zoom_dir)
        if initial_image:
            initial_image.save(os.path.join(result_dir, f"{name}_streetview_zoom{zoom_level}.jpg"))
            print(f"✅ 街景图像已保存为 {os.path.join(result_dir, f'{name}_streetview_zoom{zoom_level}.jpg')}")
            print(f"✅ 成功获取 {success_count} 个瓦片，总用时: {total_duration:.2f}秒")
            print(f"✅ 图像尺寸: {image_size[0]}x{image_size[1]}")
            
            # 转换为透视图并保存（使用不同FOV值）
            for fov_value in [30, 60, 90, 120]:
                perspective_image = tile_api.convert_to_perspective(initial_image, fov=fov_value)
                if perspective_image:
                    perspective_image.save(os.path.join(result_dir, f"{name}_perspective_fov{fov_value}_zoom{zoom_level}.jpg"))
                    print(f"✅ 透视图 (FOV={fov_value}°) 已保存为 {os.path.join(result_dir, f'{name}_perspective_fov{fov_value}_zoom{zoom_level}.jpg')}")
        else:
            print(f"❌ 无法获取街景图像 (zoom={zoom_level})")
            print(f"✅ 成功获取 {success_count} 个瓦片，总用时: {total_duration:.2f}秒")
        
        # 记录该zoom级别的瓦片信息
        for tile_info in tile_details:
            tile_info["position"] = "initial"
            tile_info["pano_id"] = tile_api.current_pano_id
            tile_info["location_name"] = name
        all_tile_details.extend(tile_details)
        
        # 记录该zoom级别的总用时和图像尺寸
        zoom_total_durations[zoom_level] = {
            "total_duration": total_duration,
            "image_width": image_size[0] if image_size else 0,
            "image_height": image_size[1] if image_size else 0
        }
    
    # 保存该点位的CSV文件
    csv_path = os.path.join(result_dir, f"{name}_tile_details.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["zoom", "x", "y", "success", "width", "height", "duration", "position", "pano_id", "location_name"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for tile_info in all_tile_details:
            writer.writerow(tile_info)
    print(f"✅ 瓦片详细信息已保存为 {csv_path}")
    
    # 生成该点位的zoom级别统计信息
    zoom_stats = {}
    for tile_info in all_tile_details:
        zoom = tile_info["zoom"]
        if zoom not in zoom_stats:
            # 根据zoom级别计算理论上的瓦片总数：(2^zoom) x (2^zoom)
            tiles_per_side = 2 ** zoom
            total_theoretical = tiles_per_side * tiles_per_side
            zoom_stats[zoom] = {
                "total": total_theoretical,  # 理论总数
                "success": 0,
                "width_sum": 0,
                "height_sum": 0,
                "duration_sum": 0.0
            }
        if tile_info["success"]:
            zoom_stats[zoom]["success"] += 1
            zoom_stats[zoom]["width_sum"] += tile_info["width"]
            zoom_stats[zoom]["height_sum"] += tile_info["height"]
            zoom_stats[zoom]["duration_sum"] += tile_info.get("duration", 0)
    
    # 保存该点位的zoom级别统计信息
    zoom_csv_path = os.path.join(result_dir, f"{name}_zoom_statistics.csv")
    with open(zoom_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["zoom", "total_tiles", "success_tiles", "success_rate", "avg_width", "avg_height", "avg_duration", "total_duration", "image_width", "image_height"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for zoom, stats in sorted(zoom_stats.items()):
            avg_width = stats["width_sum"] / stats["success"] if stats["success"] > 0 else 0
            avg_height = stats["height_sum"] / stats["success"] if stats["success"] > 0 else 0
            avg_duration = stats["duration_sum"] / stats["success"] if stats["success"] > 0 else 0
            zoom_info = zoom_total_durations.get(zoom, {"total_duration": 0, "image_width": 0, "image_height": 0})
            total_duration = zoom_info["total_duration"]
            image_width = zoom_info["image_width"]
            image_height = zoom_info["image_height"]
            writer.writerow({
                "zoom": zoom,
                "total_tiles": stats["total"],
                "success_tiles": stats["success"],
                "success_rate": f"{stats['success'] / stats['total'] * 100:.2f}%",
                "avg_width": f"{avg_width:.2f}",
                "avg_height": f"{avg_height:.2f}",
                "avg_duration": f"{avg_duration:.2f}",
                "total_duration": f"{total_duration:.2f}",
                "image_width": image_width,
                "image_height": image_height
            })
    print(f"✅ Zoom级别统计信息已保存为 {zoom_csv_path}")
    
    return True, all_tile_details

def batch_process(csv_file="2.csv"):
    """
    批量处理CSV文件中的点位
    
    Args:
        csv_file: CSV文件路径
    """
    # 初始化Tile API
    tile_api = GoogleStreetViewTileAPI()
    
    # 创建会话令牌
    session = tile_api.create_session()
    if not session:
        print("❌ 无法创建会话令牌，退出测试")
        return
    
    # 创建结果文件夹
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_dir = os.path.join("streetview_results", f"{timestamp}_batch")
    os.makedirs(result_dir, exist_ok=True)
    print(f"✅ 批量结果文件夹已创建: {result_dir}")
    
    # 读取CSV文件
    if not os.path.exists(csv_file):
        print(f"❌ CSV文件不存在: {csv_file}")
        return
    
    # 存储所有点位的瓦片详细信息
    all_locations_details = []
    
    # 读取并处理每个点位
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        locations = list(reader)
        print(f"✅ 从CSV文件读取到 {len(locations)} 个点位")
        
        for i, location in enumerate(locations, 1):
            name = location.get("name", f"location_{i}")
            lat = float(location.get("lat", 0))
            lng = float(location.get("lng", 0))
            
            print(f"\n{'-' * 60}")
            print(f"处理第 {i}/{len(locations)} 个点位")
            
            success, details = process_single_location(tile_api, name, lat, lng, result_dir)
            if success and details:
                all_locations_details.extend(details)
    
    # 生成批量汇总统计信息
    if all_locations_details:
        batch_stats = {}
        for tile_info in all_locations_details:
            location_name = tile_info.get("location_name", "unknown")
            if location_name not in batch_stats:
                batch_stats[location_name] = {
                    "total_tiles": 0,
                    "success_tiles": 0,
                    "total_duration": 0.0
                }
            batch_stats[location_name]["total_tiles"] += 1
            if tile_info["success"]:
                batch_stats[location_name]["success_tiles"] += 1
                batch_stats[location_name]["total_duration"] += tile_info.get("duration", 0)
        
        # 保存批量汇总统计
        batch_summary_path = os.path.join(result_dir, "batch_summary.csv")
        with open(batch_summary_path, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["location_name", "total_tiles", "success_tiles", "success_rate", "total_duration"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for location_name, stats in batch_stats.items():
                success_rate = (stats["success_tiles"] / stats["total_tiles"] * 100) if stats["total_tiles"] > 0 else 0
                writer.writerow({
                    "location_name": location_name,
                    "total_tiles": stats["total_tiles"],
                    "success_tiles": stats["success_tiles"],
                    "success_rate": f"{success_rate:.2f}%",
                    "total_duration": f"{stats['total_duration']:.2f}"
                })
        print(f"✅ 批量汇总统计已保存为 {batch_summary_path}")
    
    print(f"\n{'-' * 60}")
    print(f"✅ 所有点位处理完成！")
    print(f"✅ 结果保存至: {result_dir}")

if __name__ == "__main__":
    import sys
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        # 有命令行参数，使用指定的CSV文件
        csv_file = sys.argv[1]
        print(f"✅ 使用命令行指定的CSV文件: {csv_file}")
        batch_process(csv_file)
    else:
        # 无命令行参数，使用默认CSV文件
        print(f"✅ 使用默认CSV文件: 1.csv")
        batch_process()
    
    # 单点位测试模式
    # test_tile_api()
