import os
import csv
import requests
from PIL import Image
from io import BytesIO
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

class GoogleStreetViewDownloader:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.static_url = "https://maps.googleapis.com/maps/api/streetview"
        self.metadata_url = "https://maps.googleapis.com/maps/api/streetview/metadata"

    def get_street_view_image(self, lat: float, lng: float, heading: float = 0.0, 
                              pitch: float = 0.0, fov: int = 60, size: str = "640x640") -> Image.Image:
        params = {
            "location": f"{lat},{lng}",
            "heading": heading,
            "pitch": pitch,
            "fov": fov,
            "size": size,
            "key": self.api_key
        }
        response = requests.get(self.static_url, params=params, timeout=10)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        return image

    def has_street_view(self, lat: float, lng: float, radius: int = 10) -> bool:
        params = {
            "location": f"{lat},{lng}",
            "radius": radius,
            "key": self.api_key
        }
        response = requests.get(self.metadata_url, params=params, timeout=10)
        data = response.json()
        return data.get("status") == "OK"

def read_csv(csv_path: str) -> List[Dict]:
    targets = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                name = row.get("id", row.get("name", "")).strip()
                if not name:
                    continue
                lat = float(row["lat"].strip())
                lng = float(row["lng"].strip())
                targets.append({"id": name, "lat": lat, "lng": lng})
            except Exception as e:
                print(f"跳过无效行: {e}")
                continue
    return targets

def download_images(csv_path: str, output_folder: str = "streetview_images"):
    os.makedirs(output_folder, exist_ok=True)
    
    downloader = GoogleStreetViewDownloader()
    targets = read_csv(csv_path)
    
    print(f"发现 {len(targets)} 个点位")
    
    for idx, target in enumerate(targets, 1):
        try:
            print(f"[{idx}/{len(targets)}] 处理 {target['id']}...")
            
            if not downloader.has_street_view(target["lat"], target["lng"]):
                print(f"  ❌ 该位置无街景数据")
                continue
            
            image = downloader.get_street_view_image(target["lat"], target["lng"])
            image_path = os.path.join(output_folder, f"{target['id']}.jpg")
            image.save(image_path, format='JPEG', quality=100, subsampling=0)
            print(f"  ✅ 保存到 {image_path}")
            
        except Exception as e:
            print(f"  ❌ 下载失败: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="从CSV下载Google街景图像")
    parser.add_argument("csv_path", help="CSV文件路径")
    parser.add_argument("-o", "--output", default="streetview_images", help="输出文件夹")
    args = parser.parse_args()
    
    download_images(args.csv_path, args.output)
    print("下载完成！")