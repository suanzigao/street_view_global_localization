#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
处理已有的GeoAgent测试结果：将模型响应中的地址转换为GPS坐标
"""
import os
import sys
import json
import math
import csv
import re
import argparse
import logging
from typing import List, Dict, Tuple, Optional
import requests
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# 代理配置
PROXY_HOST = "127.0.0.1"
PROXY_PORT = 7897
os.environ["HTTP_PROXY"] = f"http://{PROXY_HOST}:{PROXY_PORT}"
os.environ["HTTPS_PROXY"] = f"http://{PROXY_HOST}:{PROXY_PORT}"
os.environ["NO_PROXY"] = "localhost,127.0.0.1,localdomain,.localdomain"


class GoogleGeocodingAPI:
    """Google Geocoding API 封装"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.geocode_url = "https://maps.googleapis.com/maps/api/geocode/json"

    def geocode(self, address: str) -> Optional[Tuple[float, float]]:
        """将地址转换为GPS坐标 (lat, lng)"""
        params = {
            "address": address,
            "key": self.api_key
        }
        try:
            r = requests.get(self.geocode_url, params=params, timeout=10)
            data = r.json()
            if data.get("status") == "OK" and data.get("results"):
                location = data["results"][0]["geometry"]["location"]
                return location["lat"], location["lng"]
            else:
                logger.warning(f"Geocode failed for '{address}': {data.get('status')}")
        except Exception as e:
            logger.warning(f"Geocode error for '{address}': {e}")
        return None


def calculate_gps_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """用Haversine公式计算两点GPS坐标的实际距离（米）"""
    R = 6378137.0
    lat1_rad, lng1_rad = math.radians(lat1), math.radians(lng1)
    lat2_rad, lng2_rad = math.radians(lat2), math.radians(lng2)

    dlat = lat2_rad - lat1_rad
    dlng = lng2_rad - lng1_rad

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlng / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def extract_final_answer(response: str) -> Optional[str]:
    """从模型响应中提取FinalAnswer"""
    try:
        # 移除代码块标记
        cleaned = response.strip()
        if cleaned.startswith('```json'):
            cleaned = cleaned[7:]
        if cleaned.endswith('```'):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        data = json.loads(cleaned)
        return data.get("FinalAnswer", "").strip()
    except Exception as e:
        logger.warning(f"Failed to extract FinalAnswer: {e}")
        return None


def read_gps_targets_from_csv(csv_path: str) -> Dict[str, Tuple[float, float]]:
    """从CSV文件读取GPS目标点，返回字典"""
    gps_dict = {}
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV文件不存在: {csv_path}")

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                name = row.get("name", "").strip() or row.get("id", "").strip()
                if not name:
                    continue
                lat = float(row["lat"].strip())
                lng = float(row["lng"].strip())
                gps_dict[name] = (lat, lng)
            except Exception as e:
                continue

    return gps_dict


def process_existing_results(batch_dir: str, csv_path: str):
    """处理已有的测试结果"""
    # 1. 读取CSV获取实际GPS坐标
    logger.info(f"读取CSV文件: {csv_path}")
    gps_dict = read_gps_targets_from_csv(csv_path)
    logger.info(f"读取到 {len(gps_dict)} 个点位")

    # 2. 初始化Geocoding API
    geocoding_api = GoogleGeocodingAPI()

    # 3. 遍历结果目录
    batch_results = []
    point_dirs = [d for d in os.listdir(batch_dir) if os.path.isdir(os.path.join(batch_dir, d))]
    point_dirs.sort()

    for idx, point_name in enumerate(point_dirs, 1):
        logger.info(f"\n==== [{idx}/{len(point_dirs)}] 处理: {point_name} ====")
        point_dir = os.path.join(batch_dir, point_name)
        response_path = os.path.join(point_dir, "model_response.json")

        # 获取实际GPS坐标
        actual_lat, actual_lng = gps_dict.get(point_name, (None, None))
        if actual_lat is None:
            logger.warning(f"未找到点位 {point_name} 的实际GPS坐标")
            continue

        result = {
            "name": point_name,
            "actual_lat": actual_lat,
            "actual_lng": actual_lng,
            "predicted_lat": None,
            "predicted_lng": None,
            "error": None,
            "address": None
        }

        # 读取模型响应
        if os.path.exists(response_path):
            try:
                with open(response_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    response = data.get("response", "")

                # 提取FinalAnswer
                address = extract_final_answer(response)
                if address:
                    result["address"] = address
                    logger.info(f"提取地址: {address}")

                    # 转换为GPS坐标
                    coords = geocoding_api.geocode(address)
                    if coords:
                        result["predicted_lat"], result["predicted_lng"] = coords
                        result["error"] = calculate_gps_distance(
                            result["predicted_lat"], result["predicted_lng"],
                            actual_lat, actual_lng
                        )
                        logger.info(f"转换GPS成功: {coords}")
                        logger.info(f"预测误差: {result['error']:.2f} 米")
                    else:
                        logger.warning("地址转换GPS失败")
                else:
                    logger.warning("未找到FinalAnswer")

            except Exception as e:
                logger.error(f"读取响应文件失败: {e}")
        else:
            logger.warning(f"响应文件不存在: {response_path}")

        batch_results.append(result)

        # 避免API限流
        if idx < len(point_dirs):
            import time
            time.sleep(0.5)

    # 4. 保存详细结果
    results_path = os.path.join(batch_dir, "batch_results_processed.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(batch_results, f, ensure_ascii=False, indent=2)
    logger.info(f"详细结果已保存: {results_path}")

    # 5. 生成汇总HTML
    generate_batch_summary(batch_dir, batch_results)


def generate_batch_summary(batch_dir: str, batch_results: List[Dict]):
    """生成批量汇总HTML文件"""
    total_points = len(batch_results)
    valid_predictions = sum(1 for r in batch_results if r.get("predicted_lat") is not None)
    invalid_predictions = total_points - valid_predictions

    total_error = sum(r.get("error", 0) for r in batch_results if r.get("error") is not None)
    average_error = total_error / valid_predictions if valid_predictions > 0 else 0

    error_ranges = [1000, 25000, 125000, 250000]
    range_counts = []
    for max_error in error_ranges:
        count = sum(1 for r in batch_results
                   if r.get("error") is not None and r["error"] <= max_error)
        range_counts.append(count)

    range_percentages = [(count / valid_predictions * 100) if valid_predictions > 0 else 0
                         for count in range_counts]

    table_rows = ""
    for result in batch_results:
        actual_gps = f"{result['actual_lat']:.6f}, {result['actual_lng']:.6f}"
        if result.get('predicted_lat') is not None:
            predicted_gps = f"{result['predicted_lat']:.6f}, {result['predicted_lng']:.6f}"
            error_str = f"{result['error']:.2f}" if result.get('error') is not None else "N/A"
        else:
            predicted_gps = "N/A"
            error_str = "N/A"

        table_rows += f"""
        <tr>
            <td>{result['name']}</td>
            <td>{actual_gps}</td>
            <td>{result.get('address', 'N/A')}</td>
            <td>{predicted_gps}</td>
            <td class="error-cell">{error_str}</td>
        </tr>
        """

    html_content = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GeoAgent Batch Run Summary (Processed)</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            text-align: center;
            color: #333;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        .chart-container {{
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
            padding: 30px;
            margin-bottom: 30px;
        }}
        .chart-title {{
            text-align: center;
            color: #333;
            margin-bottom: 20px;
            font-size: 20px;
            font-weight: bold;
        }}
        .chart-wrapper {{
            position: relative;
            height: 400px;
        }}
        .summary {{
            margin-bottom: 30px;
            padding: 20px;
            background-color: #f9f9f9;
            border-radius: 8px;
            border: 1px solid #ddd;
        }}
        .summary h2 {{
            color: #333;
            margin-bottom: 15px;
        }}
        .summary-info {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
        }}
        .summary-item {{
            background-color: white;
            padding: 15px;
            border-radius: 4px;
            border: 1px solid #eee;
            flex: 1;
            min-width: 200px;
        }}
        .summary-item strong {{
            color: #1e90ff;
        }}
        .results-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 30px;
        }}
        .results-table th {{
            background-color: #f2f2f2;
            padding: 12px;
            text-align: left;
            border-bottom: 2px solid #ddd;
            font-size: 12px;
        }}
        .results-table td {{
            padding: 12px;
            border-bottom: 1px solid #ddd;
            font-size: 12px;
        }}
        .results-table tr:hover {{
            background-color: #f5f5f5;
        }}
        .error-cell {{
            font-weight: bold;
            color: #1e90ff;
        }}
        .average-error {{
            margin-top: 30px;
            padding: 20px;
            background-color: #e6f7ff;
            border-radius: 8px;
            border: 1px solid #91d5ff;
        }}
        .average-error h2 {{
            color: #1890ff;
            margin-bottom: 15px;
        }}
        .average-error p {{
            font-size: 18px;
            font-weight: bold;
            color: #1890ff;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>GeoAgent Batch Run Summary (Processed)</h1>

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
                    <strong>Valid Predictions:</strong> {valid_predictions}
                </div>
                <div class="summary-item">
                    <strong>Invalid Predictions:</strong> {invalid_predictions}
                </div>
            </div>
        </div>

        <table class="results-table">
            <thead>
                <tr>
                    <th>Point Name</th>
                    <th>Actual GPS</th>
                    <th>Predicted Address</th>
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
        const errorRanges = ['≤ 1km', '≤ 25km', '≤ 125km', '≤ 250km'];
        const percentages = {range_percentages};

        const ctx = document.getElementById('errorChart').getContext('2d');
        new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: errorRanges,
                datasets: [{{
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
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        position: 'top',
                    }},
                    title: {{
                        display: true,
                        text: 'Prediction Error Distribution'
                    }},
                    tooltip: {{
                        callbacks: {{
                            label: function(context) {{
                                return context.parsed.y.toFixed(1) + '%';
                            }}
                        }}
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 100,
                        title: {{
                            display: true,
                            text: 'Percentage (%)'
                        }},
                        ticks: {{
                            callback: function(value) {{
                                return value + '%';
                            }}
                        }}
                    }},
                    x: {{
                        title: {{
                            display: true,
                            text: 'Error Range'
                        }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
    '''

    summary_path = os.path.join(batch_dir, "batch_summary_processed.html")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info(f"✅ 汇总HTML已生成: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='处理已有的GeoAgent测试结果')
    parser.add_argument('--batch_dir', required=True, help='已有的批处理结果目录')
    parser.add_argument('--csv_path', required=True, help='原始CSV文件路径（包含name, lat, lng列）')

    args = parser.parse_args()

    process_existing_results(args.batch_dir, args.csv_path)


if __name__ == '__main__':
    main()