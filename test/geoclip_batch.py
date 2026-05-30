import os
import csv
import math
import time
import traceback
import warnings
import requests
from PIL import Image
from io import BytesIO
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv

load_dotenv()

# ====================== 全局代理配置 ======================
PROXY_HOST = "127.0.0.1"
PROXY_PORT = 7897
os.environ["HTTP_PROXY"] = f"http://{PROXY_HOST}:{PROXY_PORT}"
os.environ["HTTPS_PROXY"] = f"http://{PROXY_HOST}:{PROXY_PORT}"
os.environ["NO_PROXY"] = "localhost,127.0.0.1,localdomain,.localdomain"

# ====================== 屏蔽警告 ==============================
warnings.filterwarnings("ignore", category=UserWarning, message="Gym has been unmaintained")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ====================== 全局配置 ==============================
BATCH_ROOT_DIR = "batch_geoclip_results"
GOOGLE_API_RADIUS = 10
GPS_CSV_PATH = "1.csv"

# 当前批处理目录
CURRENT_BATCH_DIR = None


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

                point_info = {"name": name, "lat": lat, "lng": lng}
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
def init_batch_dir():
    global CURRENT_BATCH_DIR
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    CURRENT_BATCH_DIR = os.path.join(BATCH_ROOT_DIR, f"{timestamp}_batch_run")
    os.makedirs(CURRENT_BATCH_DIR, exist_ok=True)
    return CURRENT_BATCH_DIR


def get_point_dir_name(point_info: Dict) -> str:
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    lat_str = f"{point_info['lat']:.4f}".replace(".", "_")
    lng_str = f"{point_info['lng']:.4f}".replace(".", "_")
    return f"{timestamp}_{point_info['name']}_lat{lat_str}_lng{lng_str}"


def init_point_dir(point_info: Dict) -> Tuple[str, str]:
    global CURRENT_BATCH_DIR
    if CURRENT_BATCH_DIR is None:
        point_dir = os.path.join(BATCH_ROOT_DIR, get_point_dir_name(point_info))
    else:
        point_dir = os.path.join(CURRENT_BATCH_DIR, get_point_dir_name(point_info))
    img_dir = os.path.join(point_dir, "streetview_images")
    os.makedirs(img_dir, exist_ok=True)
    return point_dir, img_dir


def save_streetview_image(image: Image.Image, img_save_dir: str) -> str:
    img_path = os.path.join(img_save_dir, "streetview.jpg")
    image.save(
        img_path,
        format='JPEG',
        quality=100,
        subsampling=0,
        optimize=False
    )
    return img_path


# ====================== 地理角度工具 ======================
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


# ====================== Google Street View ======================
class GoogleStreetViewData:
    def __init__(self, api_key: str = None, max_retries: int = 3):
        self.google_api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.static_url = "https://maps.googleapis.com/maps/api/streetview"
        self.metadata_url = "https://maps.googleapis.com/maps/api/streetview/metadata"
        self.max_retries = max_retries

    @retry_on_failure(3, 1.0)
    def get_metadata(self, lat: float, lng: float) -> Optional[Dict]:
        params = {"location": f"{lat},{lng}", "radius": GOOGLE_API_RADIUS, "key": self.google_api_key}
        r = requests.get(self.metadata_url, params=params, timeout=10)
        data = r.json()
        if data.get("status") == "OK" and "location" in data:
            actual_lat = data["location"]["lat"]
            actual_lng = data["location"]["lng"]
            distance = calculate_gps_distance(lat, lng, actual_lat, actual_lng)
            if distance <= GOOGLE_API_RADIUS:
                return data
        return None

    @retry_on_failure(3, 1.0)
    def get_street_view_image(self, lat: float, lng: float, heading: float = 0.0, pitch: float = 0.0, fov: float = 50) -> Tuple[Optional[Image.Image], Optional[float], Optional[float]]:
        """
        获取街景图像和实际采样点的GPS坐标
        return: (图像, 实际lat, 实际lng)
        """
        params = {
            "location": f"{lat},{lng}",
            "heading": heading,
            "pitch": pitch,
            "fov": fov,
            "radius": GOOGLE_API_RADIUS,
            "size": "640x640",
            "key": self.google_api_key
        }
        # 1. 先获取元数据（包含实际采样点坐标）
        metadata = self.get_metadata(lat, lng)
        actual_lat, actual_lng = None, None
        if metadata and "location" in metadata:
            actual_lat = metadata["location"]["lat"]
            actual_lng = metadata["location"]["lng"]

        # 2. 获取街景图像
        r = requests.get(self.static_url, params=params, stream=True, timeout=10)
        
        # 检查响应状态码
        if r.status_code != 200:
            print(f"❌ API返回错误状态码: {r.status_code}")
            return None, actual_lat, actual_lng
        
        # 检查响应内容类型
        content_type = r.headers.get('Content-Type', '')
        if not content_type.startswith('image/'):
            print(f"❌ API返回非图像内容: {content_type}")
            return None, actual_lat, actual_lng
        
        # 尝试打开图像
        try:
            image = Image.open(BytesIO(r.content)).convert("RGB")
            return image, actual_lat, actual_lng
        except Exception as e:
            print(f"❌ 无法识别图像: {str(e)}")
            return None, actual_lat, actual_lng


# ====================== GeoCLIP 预测 ======================
class GeoCLIPPredictor:
    def __init__(self):
        from geoclip import GeoCLIP
        print("🔄 加载 GeoCLIP 模型...")
        self.model = GeoCLIP()
        print("✅ GeoCLIP 模型加载完成")

    def predict(self, image_path: str) -> Tuple[List[Tuple[float, float]], List[float]]:
        top_pred_gps, top_pred_prob = self.model.predict(image_path, top_k=5)
        return top_pred_gps, top_pred_prob

    def cleanup(self):
        import torch
        if hasattr(self, 'model'):
            del self.model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ====================== 生成批处理汇总 ======================
def generate_batch_summary(batch_dir, batch_results):
    """
    生成批处理汇总HTML文件
    """
    # 计算平均偏差
    total_error = 0
    valid_count = 0
    for result in batch_results:
        if result["error"] is not None:
            total_error += result["error"]
            valid_count += 1

    average_error = total_error / valid_count if valid_count > 0 else 0

    # 计算不同误差范围的统计
    error_ranges = [1000, 25000, 125000, 250000]  # 1km, 25km, 125km, 250km
    range_counts = []
    for max_error in error_ranges:
        count = 0
        for result in batch_results:
            if result["error"] is not None and result["error"] <= max_error:
                count += 1
        range_counts.append(count)

    # 计算占比
    range_percentages = []
    for count in range_counts:
        percentage = (count / valid_count * 100) if valid_count > 0 else 0
        range_percentages.append(percentage)

    # 生成HTML内容
    html_content = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GeoCLIP Batch Run Summary</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            text-align: center;
            color: #333;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .chart-container {
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
            padding: 30px;
            margin-bottom: 30px;
        }
        .chart-title {
            text-align: center;
            color: #333;
            margin-bottom: 20px;
            font-size: 20px;
            font-weight: bold;
        }
        .chart-wrapper {
            position: relative;
            height: 400px;
        }
        .summary {
            margin-bottom: 30px;
            padding: 20px;
            background-color: #f9f9f9;
            border-radius: 8px;
            border: 1px solid #ddd;
        }
        .summary h2 {
            color: #333;
            margin-bottom: 15px;
        }
        .summary-info {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
        }
        .summary-item {
            background-color: white;
            padding: 15px;
            border-radius: 4px;
            border: 1px solid #eee;
            flex: 1;
            min-width: 200px;
        }
        .summary-item strong {
            color: #1e90ff;
        }
        .results-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 30px;
        }
        .results-table th {
            background-color: #f2f2f2;
            padding: 12px;
            text-align: left;
            border-bottom: 2px solid #ddd;
        }
        .results-table td {
            padding: 12px;
            border-bottom: 1px solid #ddd;
        }
        .results-table tr:hover {
            background-color: #f5f5f5;
        }
        .error-cell {
            font-weight: bold;
            color: #1e90ff;
        }
        .average-error {
            margin-top: 30px;
            padding: 20px;
            background-color: #e6f7ff;
            border-radius: 8px;
            border: 1px solid #91d5ff;
        }
        .average-error h2 {
            color: #1890ff;
            margin-bottom: 15px;
        }
        .average-error p {
            font-size: 18px;
            font-weight: bold;
            color: #1890ff;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>GeoCLIP Batch Run Summary</h1>
        
        <!-- 误差分布折线图 -->
        <div class="chart-container">
            <div class="chart-title">Error Range Distribution</div>
            <div class="chart-wrapper">
                <canvas id="errorChart"></canvas>
            </div>
        </div>
        
        <div class="summary">
            <h2>Run Information</h2>
            <div class="summary-info">
                <div class="summary-item">
                    <strong>Total Points:</strong> {total_points}
                </div>
                <div class="summary-item">
                    <strong>Valid Predictions:</strong> {valid_count}
                </div>
                <div class="summary-item">
                    <strong>Invalid Predictions:</strong> {invalid_count}
                </div>
            </div>
        </div>
        
        <table class="results-table">
            <thead>
                <tr>
                    <th>Point Name</th>
                    <th>Actual GPS</th>
                    <th>Predicted GPS</th>
                    <th>Prediction Error (meters)</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
        
        <div class="average-error">
            <h2>Average Prediction Error</h2>
            <p>{average_error:.2f} meters</p>
        </div>
    </div>
    
    <script>
        // 误差范围数据
        const errorRanges = ['≤ 1km', '≤ 25km', '≤ 125km', '≤ 250km'];
        const percentages = {range_percentages};
        
        // 创建折线图
        const ctx = document.getElementById('errorChart').getContext('2d');
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: errorRanges,
                datasets: [{
                    label: 'Percentage of Points',
                    data: percentages,
                    borderColor: 'rgba(102, 126, 234, 1)',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    borderWidth: 3,
                    tension: 0,
                    pointBackgroundColor: 'rgba(102, 126, 234, 1)',
                    pointBorderColor: '#fff',
                    pointBorderWidth: 2,
                    pointRadius: 6,
                    pointHoverRadius: 8
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                    },
                    title: {
                        display: true,
                        text: 'Prediction Error Distribution',
                        font: {
                            size: 16
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return context.parsed.y.toFixed(1) + '%';
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        title: {
                            display: true,
                            text: 'Percentage (%)'
                        },
                        ticks: {
                            callback: function(value) {
                                return value + '%';
                            }
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Error Range'
                        }
                    }
                }
            }
        });
    </script>
</body>
</html>
    '''

    # 生成表格行
    table_rows = ""
    for result in batch_results:
        actual_gps = f"{result['actual_lat']:.6f}, {result['actual_lng']:.6f}"
        if result['predicted_lat'] is not None and result['predicted_lng'] is not None:
            predicted_gps = f"{result['predicted_lat']:.6f}, {result['predicted_lng']:.6f}"
            error = f"{result['error']:.2f}" if result['error'] is not None else "N/A"
        else:
            predicted_gps = "N/A"
            error = "N/A"

        table_rows += f""
        table_rows += f"        <tr>"
        table_rows += f"            <td>{result['name']}</td>"
        table_rows += f"            <td>{actual_gps}</td>"
        table_rows += f"            <td>{predicted_gps}</td>"
        table_rows += f"            <td class=\"error-cell\">{error}</td>"
        table_rows += f"        </tr>"
        table_rows += f""

    # 计算无效预测数
    invalid_count = len(batch_results) - valid_count

    # 替换占位符
    html_content = html_content.replace("{total_points}", str(len(batch_results)))
    html_content = html_content.replace("{valid_count}", str(valid_count))
    html_content = html_content.replace("{invalid_count}", str(invalid_count))
    html_content = html_content.replace("{table_rows}", table_rows)
    html_content = html_content.replace("{average_error:.2f}", f"{average_error:.2f}")
    html_content = html_content.replace("{range_percentages}", str(range_percentages))

    # 保存HTML文件
    summary_path = os.path.join(batch_dir, "batch_summary.html")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"✅ 批处理汇总HTML文件已生成: {summary_path}")


# ====================== 批处理运行 ======================
def batch_run():
    # 初始化批处理目录
    batch_dir = init_batch_dir()

    # 读取GPS目标点
    gps_targets = read_gps_targets_from_csv(GPS_CSV_PATH)

    # 初始化结果列表
    batch_results = []
    failed_points = []

    print(f"✅ 开始批量运行，共 {len(gps_targets)} 个点位")

    # 初始化Google Street View客户端
    street_view = GoogleStreetViewData()

    # 初始化GeoCLIP预测器（只加载一次）
    geoclip = GeoCLIPPredictor()

    for idx, point_info in enumerate(gps_targets, 1):
        print(f"\n====== 第 {idx}/{len(gps_targets)} 个点: {point_info['name']} ======")
        try:
            point_dir, img_dir = init_point_dir(point_info)

            # 获取街景图像
            print(f"📡 获取街景图像...")
            image, actual_lat, actual_lng = street_view.get_street_view_image(
                point_info["lat"],
                point_info["lng"]
            )

            if not image:
                print("❌ 无法获取街景图像，跳过该点")
                failed_points.append({"name": point_info["name"], "reason": "无法获取街景图像"})
                continue

            # 保存图像
            img_path = save_streetview_image(image, img_dir)
            print(f"✅ 街景图像已保存: {img_path}")

            # 使用GeoCLIP预测
            print(f"🤖 使用GeoCLIP预测地理位置...")
            top_pred_gps, top_pred_prob = geoclip.predict(img_path)

            # 保存预测结果
            result_path = os.path.join(point_dir, "geoclip_prediction.txt")
            with open(result_path, "w", encoding="utf-8") as f:
                f.write("GeoCLIP Prediction Results:\n")
                f.write("============================\n")
                for i, (gps, prob) in enumerate(zip(top_pred_gps, top_pred_prob)):
                    lat, lon = gps
                    f.write(f"Prediction {i+1}: ({lat:.6f}, {lon:.6f}) - Probability: {prob:.6f}\n")
            print(f"✅ 预测结果已保存: {result_path}")

            # 收集点位结果信息
            point_result = {
                "name": point_info["name"],
                "actual_lat": point_info["lat"],
                "actual_lng": point_info["lng"],
                "prediction": f"{top_pred_gps[0][0]:.6f}, {top_pred_gps[0][1]:.6f}",
                "predicted_lat": top_pred_gps[0][0],
                "predicted_lng": top_pred_gps[0][1],
                "error": None
            }

            # 计算偏差
            point_result["error"] = calculate_gps_distance(
                point_result["predicted_lat"], point_result["predicted_lng"],
                point_result["actual_lat"], point_result["actual_lng"]
            )
            print(f"📊 预测误差: {point_result['error']:.2f} 米")

            # 添加到结果列表
            batch_results.append(point_result)

        except Exception as e:
            print(f"❌ 该点运行失败: {e}")
            traceback.print_exc()
            failed_points.append({"name": point_info["name"], "reason": str(e)})

    # 清理GeoCLIP模型
    geoclip.cleanup()
    print("🧹 GeoCLIP 模型已清理")

    # 打印失败统计
    if failed_points:
        print(f"\n⚠️ 共有 {len(failed_points)} 个点位处理失败:")
        for fp in failed_points:
            print(f"   - {fp['name']}: {fp['reason']}")

    # 生成汇总HTML文件
    generate_batch_summary(batch_dir, batch_results)


# ====================== 运行 ======================
if __name__ == "__main__":
    batch_run()
