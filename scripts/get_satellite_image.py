import os
import json
import requests
import math

def latlng_to_tile(lat, lng, zoom):
    """将经纬度转换为图块坐标"""
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    x = int((lng + 180.0) / 360.0 * n)
    y = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return x, y

def get_session_token(api_key):
    """创建会话令牌"""
    url = f"https://tile.googleapis.com/v1/createSession?key={api_key}"
    payload = {
        "mapType": "satellite",
        "language": "zh-CN",
        "region": "CN"
    }
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()["session"]

def get_satellite_tile(api_key, session, zoom, x, y, output_path):
    """获取单个卫星图块"""
    url = f"https://tile.googleapis.com/v1/2dtiles/{zoom}/{x}/{y}?session={session}&key={api_key}"
    response = requests.get(url)
    response.raise_for_status()
    with open(output_path, 'wb') as f:
        f.write(response.content)
    print(f"✅ 已保存: {output_path}")

def download_satellite_images(api_key, lat, lng, zoom_levels, output_dir):
    """下载多个缩放级别的卫星图像"""
    os.makedirs(output_dir, exist_ok=True)
    
    session = get_session_token(api_key)
    print(f"📋 会话令牌获取成功: {session[:20]}...")
    
    for zoom in zoom_levels:
        x, y = latlng_to_tile(lat, lng, zoom)
        output_path = os.path.join(output_dir, f"satellite_zoom{zoom}_x{x}_y{y}.png")
        get_satellite_tile(api_key, session, zoom, x, y, output_path)
    
    print("\n🎉 所有卫星图像下载完成!")

if __name__ == "__main__":
    env_path = "/home/suanzigao/PycharmProjects/street_view_global_localization/.env"
    with open(env_path, 'r') as f:
        for line in f:
            if line.startswith("GOOGLE_API_KEY="):
                api_key = line.strip().split("=")[1]
                break
    
    lat = 52.6003345
    lng = 13.4590728
    
    zoom_levels = [10, 12, 14, 16, 18, 20]
    
    output_dir = "/home/suanzigao/PycharmProjects/street_view_global_localization/satellite_images"
    
    print(f"📍 目标坐标: {lat}, {lng}")
    print(f"🔍 缩放级别: {zoom_levels}")
    print(f"📁 输出目录: {output_dir}\n")
    
    download_satellite_images(api_key, lat, lng, zoom_levels, output_dir)