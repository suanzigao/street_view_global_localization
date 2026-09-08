"""
人工操作街景图像采集脚本

允许用户手动控制街景视角，收集图像数据用于后续投影和匹配

操作命令:
  f / forward   - 前进 (15米)
  b / backward  - 后退 (15米)
  l / left      - 左转 (45度)
  r / right     - 右转 (45度)
  u / up        - 抬头 (10度)
  d / down      - 低头 (10度)
  z / zoom_in   - 放大 (FOV减小20度)
  Z / zoom_out  - 缩小 (FOV增大20度)
  s / save      - 保存当前图像
  q / quit      - 退出
  h / help      - 显示帮助
"""

import os
import sys
import json
import math
from datetime import datetime
from PIL import Image
from io import BytesIO
from typing import Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.street_view import GoogleStreetViewData
from utils.file_utils import save_streetview_image
from utils.geometry import calculate_relative_offset


class ManualStreetViewCollector:
    def __init__(self):
        self.data_handler = GoogleStreetViewData(move_step=15.0, radius=10)
        
        self.current_step = 0
        self.images_saved = 0
        self.action_history = []
        
        self.relative_x = 0.0
        self.relative_y = 0.0
        self.relative_history = []
        
        self.actual_initial_lat = None
        self.actual_initial_lng = None
        
        self.point_dir = None
        self.img_save_dir = None
        
        self.current_image = None
        self.current_actual_lat = None
        self.current_actual_lng = None
    
    def init_position(self, lat: float, lng: float):
        self.data_handler.init_position(lat, lng, heading=0.0, fov=60, pitch=0.0)
        
        print(f"\n📍 初始化位置: ({lat:.6f}, {lng:.6f})")
        print(f"🎯 朝向: {self.data_handler.heading:.1f}°, FOV: {self.data_handler.fov}°, 俯仰角: {self.data_handler.pitch:.1f}°")
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        lat_str = f"{lat:.4f}".replace(".", "_")
        lng_str = f"{lng:.4f}".replace(".", "_")
        
        self.point_dir = os.path.join(
            "manual_streetview_results",
            f"{timestamp}_manual_lat{lat_str}_lng{lng_str}"
        )
        self.img_save_dir = os.path.join(self.point_dir, "streetview_images")
        os.makedirs(self.img_save_dir, exist_ok=True)
        
        print(f"\n📁 数据保存目录: {self.point_dir}")
        
        self.current_step = 0
        self.relative_x = 0.0
        self.relative_y = 0.0
        
        self._get_image()
    
    def _get_image(self):
        try:
            self.current_image, self.current_actual_lat, self.current_actual_lng = \
                self.data_handler.get_street_view_image()
            
            if self.current_actual_lat and self.current_actual_lng:
                if self.actual_initial_lat is None:
                    self.actual_initial_lat = self.current_actual_lat
                    self.actual_initial_lng = self.current_actual_lng
                
                ref_lat = self.actual_initial_lat
                ref_lng = self.actual_initial_lng
                self.relative_x, self.relative_y = calculate_relative_offset(
                    ref_lat, ref_lng,
                    self.current_actual_lat, self.current_actual_lng
                )
            
            if self.current_image:
                print(f"\n✅ 图像获取成功")
            else:
                print(f"\n❌ 无法获取图像")
        except Exception as e:
            print(f"\n❌ 获取图像失败: {e}")
            self.current_image = None
    
    def _save_image(self):
        if self.current_image is None:
            print("❌ 没有可保存的图像")
            return
        
        metadata = {
            'heading': self.data_handler.heading,
            'pitch': self.data_handler.pitch,
            'fov': self.data_handler.fov,
            'relative_x': self.relative_x,
            'relative_y': self.relative_y,
            'actual_lat': self.current_actual_lat if self.current_actual_lat else self.data_handler.current_lat,
            'actual_lng': self.current_actual_lng if self.current_actual_lng else self.data_handler.current_lng,
            'action': f"step_{self.current_step}"
        }
        
        img_path, img_b64 = save_streetview_image(
            self.current_image, self.current_step, self.img_save_dir, metadata=metadata
        )
        
        self.images_saved += 1
        self.relative_history.append({
            'step': self.current_step,
            'x': self.relative_x,
            'y': self.relative_y,
            'heading': self.data_handler.heading,
            'action': f"step_{self.current_step}"
        })
        
        print(f"\n✅ 图像已保存: step_{self.current_step}_streetview.jpg")
        print(f"   元数据: heading={metadata['heading']:.1f}°, pitch={metadata['pitch']:.1f}°, fov={metadata['fov']}°")
        print(f"   相对位置: X={self.relative_x:.2f}m, Y={self.relative_y:.2f}m")
    
    def move_forward(self):
        print(f"\n🚶 前进 15米...")
        geo_ang = self.data_handler.heading
        ok, msg, actual_distance = self.data_handler.move(geo_ang, 15.0)
        
        if ok:
            self.current_step += 1
            self.action_history.append(f"Step {self.current_step}: move forward")
            print(f"✅ {msg}")
            self._get_image()
            self._save_image()
        else:
            print(f"❌ {msg}")
    
    def move_backward(self):
        print(f"\n🚶 后退 15米...")
        geo_ang = (self.data_handler.heading + 180) % 360
        ok, msg, actual_distance = self.data_handler.move(geo_ang, 15.0)
        
        if ok:
            self.current_step += 1
            self.action_history.append(f"Step {self.current_step}: move backward")
            print(f"✅ {msg}")
            self._get_image()
            self._save_image()
        else:
            print(f"❌ {msg}")
    
    def turn_left(self):
        print(f"\n🔄 左转 45°...")
        self.data_handler.turn(-45)
        self.current_step += 1
        self.action_history.append(f"Step {self.current_step}: turn left")
        print(f"✅ 朝向: {self.data_handler.heading:.1f}°")
        self._get_image()
        self._save_image()
    
    def turn_right(self):
        print(f"\n🔄 右转 45°...")
        self.data_handler.turn(45)
        self.current_step += 1
        self.action_history.append(f"Step {self.current_step}: turn right")
        print(f"✅ 朝向: {self.data_handler.heading:.1f}°")
        self._get_image()
        self._save_image()
    
    def look_up(self):
        print(f"\n👆 抬头 10°...")
        self.data_handler.pitch = min(90, self.data_handler.pitch + 10)
        print(f"✅ 俯仰角: {self.data_handler.pitch:.1f}°")
        self._get_image()
    
    def look_down(self):
        print(f"\n👇 低头 10°...")
        self.data_handler.pitch = max(-90, self.data_handler.pitch - 10)
        print(f"✅ 俯仰角: {self.data_handler.pitch:.1f}°")
        self._get_image()
    
    def zoom_in(self):
        print(f"\n🔍 放大...")
        self.data_handler.fov = max(10, self.data_handler.fov - 20)
        print(f"✅ FOV: {self.data_handler.fov}°")
        self._get_image()
    
    def zoom_out(self):
        print(f"\n🔍 缩小...")
        self.data_handler.fov = min(120, self.data_handler.fov + 20)
        print(f"✅ FOV: {self.data_handler.fov}°")
        self._get_image()
    
    def show_status(self):
        print(f"\n" + "="*50)
        print(f"📍 当前位置: ({self.data_handler.current_lat:.6f}, {self.data_handler.current_lng:.6f})")
        print(f"🎯 朝向: {self.data_handler.heading:.1f}°, FOV: {self.data_handler.fov}°, 俯仰角: {self.data_handler.pitch:.1f}°")
        print(f"📊 相对初始位置: X={self.relative_x:.2f}m, Y={self.relative_y:.2f}m")
        print(f"📷 已保存图像: {self.images_saved}")
        print(f"📝 当前步骤: {self.current_step}")
        print(f"="*50)
    
    def show_help(self):
        print("\n" + "="*50)
        print("                    操作命令")
        print("="*50)
        print("  f / forward   - 前进 (15米)")
        print("  b / backward  - 后退 (15米)")
        print("  l / left      - 左转 (45度)")
        print("  r / right     - 右转 (45度)")
        print("  u / up        - 抬头 (10度)")
        print("  d / down      - 低头 (10度)")
        print("  z / zoom_in   - 放大 (FOV减小20度)")
        print("  Z / zoom_out  - 缩小 (FOV增大20度)")
        print("  s / save      - 保存当前图像")
        print("  q / quit      - 退出")
        print("  h / help      - 显示帮助")
        print("="*50)
    
    def run(self):
        print("\n" + "="*60)
        print("         人工街景图像采集工具")
        print("="*60)
        
        try:
            lat = float(input("请输入初始纬度: "))
            lng = float(input("请输入初始经度: "))
            
            if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
                print("❌ 坐标无效")
                return
            
            self.init_position(lat, lng)
            self._save_image()
            
            while True:
                self.show_status()
                
                cmd = input("\n请输入操作命令 (h查看帮助): ").strip().lower()
                
                if cmd in ['q', 'quit']:
                    print("\n👋 退出程序")
                    break
                
                elif cmd in ['h', 'help']:
                    self.show_help()
                
                elif cmd in ['f', 'forward']:
                    self.move_forward()
                
                elif cmd in ['b', 'backward']:
                    self.move_backward()
                
                elif cmd in ['l', 'left']:
                    self.turn_left()
                
                elif cmd in ['r', 'right']:
                    self.turn_right()
                
                elif cmd in ['u', 'up']:
                    self.look_up()
                
                elif cmd in ['d', 'down']:
                    self.look_down()
                
                elif cmd in ['z', 'zoom_in']:
                    self.zoom_in()
                
                elif cmd in ['Z', 'zoom_out']:
                    self.zoom_out()
                
                elif cmd in ['s', 'save']:
                    self._save_image()
                
                else:
                    print("❌ 未知命令，请输入 h 查看帮助")
        
        except KeyboardInterrupt:
            print("\n👋 用户中断")
        
        finally:
            if self.point_dir:
                summary = {
                    'initial_lat': self.data_handler.initial_lat,
                    'initial_lng': self.data_handler.initial_lng,
                    'actual_initial_lat': self.actual_initial_lat,
                    'actual_initial_lng': self.actual_initial_lng,
                    'total_images': self.images_saved,
                    'last_step': self.current_step,
                    'action_history': self.action_history,
                    'relative_history': self.relative_history
                }
                
                summary_path = os.path.join(self.point_dir, 'collection_summary.json')
                with open(summary_path, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, indent=2)
                
                print(f"\n📊 汇总文件已保存: {summary_path}")
                print(f"\n📁 所有数据保存在: {self.point_dir}")


if __name__ == "__main__":
    collector = ManualStreetViewCollector()
    collector.run()