"""
主入口模块

街景定位系统的主程序入口，包含批量运行和手动控制功能
"""

import os
import traceback
import warnings
from dotenv import load_dotenv

# 配置代理
PROXY_HOST = "127.0.0.1"
PROXY_PORT = 7897
os.environ["HTTP_PROXY"] = f"http://{PROXY_HOST}:{PROXY_PORT}"
os.environ["HTTPS_PROXY"] = f"http://{PROXY_HOST}:{PROXY_PORT}"
os.environ["NO_PROXY"] = "localhost,127.0.0.1,localdomain,.localdomain"

# 屏蔽警告
warnings.filterwarnings("ignore", category=UserWarning, message="Gym has been unmaintained")
warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()

# 全局配置
BATCH_ROOT_DIR = "batch_streetview_results"
GLOBAL_MAX_STEPS = 0
GLOBAL_MOVE_STEP = 15.0
GLOBAL_MAX_RETRIES = 3
FOV_STEP = 20
PITCH_STEP = 10
TURN_ANGLE = 45
FOV_DEFAULT = 60
GOOGLE_API_RADIUS = 10
GPS_CSV_PATH = "geochain_combined.csv"

# 导入模块
from utils import (
    init_batch_dir,
    init_point_dir,
    read_gps_targets_from_csv,
    calculate_gps_distance
)
from core import StreetViewAutoLocEnv


def generate_visualization(env, point_dir):
    """
    生成可视化HTML页面
    
    Args:
        env: StreetViewAutoLocEnv实例
        point_dir: 点位目录路径
    """
    trajectory = [{
        "step": 0,
        "lat": env.data_handler.initial_lat,
        "lng": env.data_handler.initial_lng,
        "heading": env.data_handler.initial_heading,
        "fov": env.data_handler.initial_fov,
        "pitch": env.data_handler.initial_pitch,
        "action": "Initial position"
    }]
    
    if hasattr(env, 'trajectory_history'):
        for item in env.trajectory_history:
            trajectory.append(item)
    
    images = []
    for step, img_path in env.image_path_history.items():
        images.append({"step": step, "path": img_path})
    
    final_prediction = env.final_prediction if hasattr(env, 'final_prediction') else "No prediction available"
    
    prediction_error = ""
    if final_prediction:
        try:
            import re
            patterns = [
                r'predicted_initial_coords: ([\d.-]+),([\d.-]+)',
                r'([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)',
                r'lat[\s:]*([-+]?\d*\.?\d+)[^,]*lng[\s:]*([-+]?\d*\.?\d+)',
                r'latitude[\s:]*([-+]?\d*\.?\d+)[^,]*longitude[\s:]*([-+]?\d*\.?\d+)'
            ]
            
            pred_lat = None
            pred_lng = None
            
            for pattern in patterns:
                match = re.search(pattern, final_prediction)
                if match:
                    pred_lat = float(match.group(1))
                    pred_lng = float(match.group(2))
                    break
            
            if pred_lat is not None and pred_lng is not None:
                actual_lat = env.data_handler.initial_lat
                actual_lng = env.data_handler.initial_lng
                distance = calculate_gps_distance(pred_lat, pred_lng, actual_lat, actual_lng)
                prediction_error = f"Prediction Error: {distance:.2f} meters"
            else:
                prediction_error = "Could not extract GPS coordinates from prediction"
        except Exception as e:
            prediction_error = f"Error calculating prediction error: {str(e)}"
    else:
        prediction_error = "No prediction available"
    
    # 生成timeline_html
    timeline_html = ""
    for idx, item in enumerate(trajectory):
        reason = ""
        if hasattr(env, 'reason_history') and idx > 0 and idx <= len(env.reason_history):
            reason = env.reason_history[idx - 1]
        
        timeline_html += f"""
        <div class="step">
            <div class="step-info">
                <div class="step-number">Step {item['step']}</div>
                <div class="step-details">
                    <p><strong>Action:</strong> {item['action']}</p>
                    {f'<p><strong>Reason:</strong> {reason}</p>' if reason else ''}
                    <p><strong>GPS:</strong> {item['lat']:.6f}, {item['lng']:.6f}</p>
                    <p><strong>Heading:</strong> {item['heading']:.1f}°</p>
                    <p><strong>FOV:</strong> {item['fov']:.1f}°</p>
                    <p><strong>Pitch:</strong> {item['pitch']:.1f}°</p>
                </div>
            </div>
            <div class="step-image">
                <img src="streetview_images/step_{item['step']}_streetview.jpg" alt="Step {item['step']}">
            </div>
        </div>
        """
    
    html_content = f'''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Street View Navigation Visualization</title>
    <script src="https://maps.googleapis.com/maps/api/js?key={os.getenv('GOOGLE_API_KEY', '')}"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
        h1 {{ text-align: center; color: #333; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        .map-container {{ width: 100%; height: 500px; margin-bottom: 20px; border: 1px solid #ddd; border-radius: 8px; overflow: hidden; }}
        #map {{ width: 100%; height: 100%; }}
        .timeline {{ margin-top: 30px; }}
        .timeline h2 {{ color: #333; margin-bottom: 20px; }}
        .step {{ display: flex; margin-bottom: 30px; border-bottom: 1px solid #eee; padding-bottom: 20px; }}
        .step-info {{ flex: 1; padding-right: 20px; }}
        .step-image {{ flex: 1; }}
        .step-image img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .step-number {{ font-size: 24px; font-weight: bold; color: #4CAF50; margin-bottom: 10px; }}
        .step-details {{ color: #666; }}
        .step-details p {{ margin: 5px 0; }}
        .final-prediction {{ margin-top: 30px; padding: 20px; background-color: #f9f9f9; border-radius: 8px; border: 1px solid #ddd; }}
        .final-prediction h2 {{ color: #333; margin-bottom: 15px; }}
        .prediction-content {{ background-color: white; padding: 15px; border-radius: 4px; border: 1px solid #eee; }}
        .prediction-content pre {{ white-space: pre-wrap; font-family: Arial, sans-serif; color: #555; line-height: 1.5; }}
        .prediction-error {{ margin-top: 15px; padding: 10px; background-color: #e6f7ff; border-left: 4px solid #1e90ff; border-radius: 4px; }}
        .prediction-error strong {{ color: #1e90ff; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Street View Navigation Visualization</h1>
        <div class="map-container"><div id="map"></div></div>
        <div class="timeline">
            <h2>Navigation Timeline</h2>
            {timeline_html}
        </div>
        <div class="final-prediction">
            <h2>Final Prediction</h2>
            <div class="prediction-content">
                <pre>{final_prediction}</pre>
                <div class="prediction-error"><strong>{prediction_error}</strong></div>
            </div>
        </div>
    </div>
    <script>
        const trajectory = {trajectory};
        function initMap() {{
            const center = new google.maps.LatLng(trajectory[0].lat, trajectory[0].lng);
            const map = new google.maps.Map(document.getElementById('map'), {{ zoom: 18, center: center, mapTypeId: google.maps.MapTypeId.SATELLITE }});
            trajectory.forEach((point, index) => {{
                const marker = new google.maps.Marker({{ position: new google.maps.LatLng(point.lat, point.lng), map: map, title: "Step " + point.step + ": " + point.action }});
                const infoWindow = new google.maps.InfoWindow({{ content: "<strong>Step " + point.step + "</strong><br>Lat: " + point.lat.toFixed(6) + "<br>Lng: " + point.lng.toFixed(6) + "<br>Heading: " + point.heading.toFixed(1) + "°<br>FOV: " + point.fov.toFixed(1) + "°<br>Pitch: " + point.pitch.toFixed(1) + "°<br>Action: " + point.action }});
                marker.addListener('click', function() {{ infoWindow.open(map, marker); }});
            }});
            const path = trajectory.map(point => new google.maps.LatLng(point.lat, point.lng));
            const polyline = new google.maps.Polyline({{ path: path, geodesic: true, strokeColor: '#FF0000', strokeOpacity: 1.0, strokeWeight: 2 }});
            polyline.setMap(map);
        }}
        window.onload = initMap;
    </script>
</body>
</html>
'''
    
    html_path = os.path.join(point_dir, "visualization.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"✅ 可视化HTML页面已生成: {html_path}")


def manual_control(env, point_dir):
    """
    手动控制模式
    
    Args:
        env: StreetViewAutoLocEnv实例
        point_dir: 点位目录路径
    """
    print("\n=== 手动操控模式 ===")
    print("移动：W=前 A=左移 D=右移 S=后")
    print("转向：Q=左转 E=右转")
    print("视角：Z=放大 X=缩小")
    print("俯仰：R=抬头 F=低头")
    print("退出：exit")
    print(f"提示：目标步长 {env.move_step} 米\n")
    
    if env.current_image is not None:
        env.render()
    
    while True:
        current_lat = env.data_handler.current_lat
        current_lng = env.data_handler.current_lng
        current_heading = env.data_handler.heading
        current_fov = env.data_handler.fov
        current_pitch = env.data_handler.pitch
        
        print(f"📌 当前GPS坐标: {current_lat:.6f}, {current_lng:.6f}")
        print(f"🧭 朝向: {current_heading:.1f}° | 视角FOV: {current_fov}° | 俯仰Pitch: {current_pitch}°")
        
        cmd = input("> 请输入操作指令: ").strip().lower()
        
        if cmd == "exit":
            print("👋 退出手动操控模式")
            
            if env.coord_correction_history:
                print("\n📊 坐标修正历史：")
                for corr in env.coord_correction_history:
                    print(f"Step {corr['step']}: 理论({corr['theoretical_lat']:.6f},{corr['theoretical_lng']:.6f}) → 实际({corr['actual_lat']:.6f},{corr['actual_lng']:.6f}) | 偏差 {corr['distance_diff']:.2f} 米")
            
            generate_visualization(env, point_dir)
            break
        
        cmd_map = {
            "w": "move forward", "a": "move left", "d": "move right", "s": "move backward",
            "q": "turn left", "e": "turn right",
            "z": "zoom in", "x": "zoom out",
            "r": "look up", "f": "look down"
        }
        
        if cmd not in cmd_map:
            print("❌ 无效指令！请输入 W/A/S/D/Q/E/Z/X/R/F 或 exit\n")
            continue
        
        action = cmd_map[cmd]
        obs, reward, terminated, truncated, info = env.step(action)
        
        print("\n--- 执行结果 ---")
        if info["action_type"] == "move":
            actual_distance = info.get("actual_distance", 0.0)
            if reward == 0.0:
                print(f"✅ 移动成功！方向：{info['relative_dir']} | 实际移动：{actual_distance:.2f} 米")
                if env.current_theoretical_lat and env.current_actual_lat:
                    print(f"📏 坐标修正：理论({env.current_theoretical_lat:.6f},{env.current_theoretical_lng:.6f}) → 实际({env.current_actual_lat:.6f},{env.current_actual_lng:.6f}) | 偏差 {env.current_correction_diff:.2f} 米")
            else:
                print(f"❌ 移动失败！原因：{info['msg']}")
        else:
            print(f"✅ {info['msg']}")
        
        print(f"📌 操作后GPS坐标: {env.data_handler.current_lat:.6f}, {env.data_handler.current_lng:.6f}")
        print(f"🧭 操作后朝向: {env.data_handler.heading:.1f}° | 视角FOV: {env.data_handler.fov}° | 俯仰Pitch: {env.data_handler.pitch}°\n")
        
        env.render()


def generate_batch_summary(batch_dir, batch_results):
    """
    生成批处理汇总HTML文件
    
    Args:
        batch_dir: 批处理目录
        batch_results: 批处理结果列表
    """
    total_error = 0
    valid_count = 0
    for result in batch_results:
        if result["error"] is not None:
            total_error += result["error"]
            valid_count += 1
    
    average_error = total_error / valid_count if valid_count > 0 else 0
    
    error_ranges = [1000, 25000, 125000, 250000]
    range_counts = []
    for max_error in error_ranges:
        count = 0
        for result in batch_results:
            if result["error"] is not None and result["error"] <= max_error:
                count += 1
        range_counts.append(count)
    
    range_percentages = [(count / valid_count * 100) if valid_count > 0 else 0 for count in range_counts]
    
    # 生成表格行
    def generate_table_rows(results):
        rows = ""
        for result in results:
            actual_gps = f"{result['actual_lat']:.6f}, {result['actual_lng']:.6f}"
            if result['predicted_lat'] is not None and result['predicted_lng'] is not None:
                predicted_gps = f"{result['predicted_lat']:.6f}, {result['predicted_lng']:.6f}"
                error = f"{result['error']:.2f}" if result['error'] is not None else "N/A"
            else:
                predicted_gps = "N/A"
                error = "N/A"
            rows += f"<tr><td>{result['name']}</td><td>{actual_gps}</td><td>{predicted_gps}</td><td class='error-cell'>{error}</td></tr>"
        return rows
    
    html_content = f'''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Batch Run Summary</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
        h1 {{ text-align: center; color: #333; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .chart-container {{ background: white; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.15); padding: 30px; margin-bottom: 30px; }}
        .chart-title {{ text-align: center; color: #333; margin-bottom: 20px; font-size: 20px; font-weight: bold; }}
        .chart-wrapper {{ position: relative; height: 400px; }}
        .summary {{ margin-bottom: 30px; padding: 20px; background-color: #f9f9f9; border-radius: 8px; border: 1px solid #ddd; }}
        .summary h2 {{ color: #333; margin-bottom: 15px; }}
        .summary-info {{ display: flex; flex-wrap: wrap; gap: 20px; }}
        .summary-item {{ background-color: white; padding: 15px; border-radius: 4px; border: 1px solid #eee; flex: 1; min-width: 200px; }}
        .summary-item strong {{ color: #1e90ff; }}
        .results-table {{ width: 100%; border-collapse: collapse; margin-top: 30px; }}
        .results-table th {{ background-color: #f2f2f2; padding: 12px; text-align: left; border-bottom: 2px solid #ddd; }}
        .results-table td {{ padding: 12px; border-bottom: 1px solid #ddd; }}
        .results-table tr:hover {{ background-color: #f5f5f5; }}
        .error-cell {{ font-weight: bold; color: #1e90ff; }}
        .average-error {{ margin-top: 30px; padding: 20px; background-color: #e6f7ff; border-radius: 8px; border: 1px solid #91d5ff; }}
        .average-error h2 {{ color: #1890ff; margin-bottom: 15px; }}
        .average-error p {{ font-size: 18px; font-weight: bold; color: #1890ff; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Batch Run Summary</h1>
        <div class="chart-container">
            <div class="chart-title">Error Range Distribution</div>
            <div class="chart-wrapper"><canvas id="errorChart"></canvas></div>
        </div>
        <div class="summary">
            <h2>Run Information</h2>
            <div class="summary-info">
                <div class="summary-item"><strong>Total Points:</strong> {len(batch_results)}</div>
                <div class="summary-item"><strong>Valid Predictions:</strong> {valid_count}</div>
                <div class="summary-item"><strong>Invalid Predictions:</strong> {len(batch_results) - valid_count}</div>
            </div>
        </div>
        <table class="results-table">
            <thead>
                <tr><th>Point Name</th><th>Actual GPS</th><th>Predicted GPS</th><th>Prediction Error (meters)</th></tr>
            </thead>
            <tbody>
                {generate_table_rows(batch_results)}
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
        new Chart(document.getElementById('errorChart'), {{
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
                        position: 'top'
                    }},
                    title: {{
                        display: true,
                        text: 'Prediction Error Distribution',
                        font: {{
                            size: 16
                        }}
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
    
    summary_path = os.path.join(batch_dir, "batch_summary.html")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"✅ 批处理汇总HTML文件已生成: {summary_path}")


def batch_run():
    """
    批处理运行模式
    """
    batch_dir = init_batch_dir()
    batch_results = []
    
    try:
        gps_targets = read_gps_targets_from_csv(GPS_CSV_PATH)
    except Exception as e:
        print(f"❌ 读取GPS文件失败: {e}")
        return
    
    print(f"✅ 开始批量运行，共 {len(gps_targets)} 个点位")
    
    for idx, point_info in enumerate(gps_targets, 1):
        print(f"\n====== 第 {idx}/{len(gps_targets)} 个点: {point_info['name']} ======")
        
        try:
            point_dir, img_dir = init_point_dir(point_info, batch_dir)
            
            env = StreetViewAutoLocEnv(
                initial_lat=point_info["lat"],
                initial_lng=point_info["lng"],
                move_step=point_info.get("move_step", GLOBAL_MOVE_STEP),
                max_steps=point_info.get("max_steps", GLOBAL_MAX_STEPS),
                point_dir=point_dir,
                img_save_dir=img_dir,
                turn_angle=TURN_ANGLE,
                fov_step=FOV_STEP,
                pitch_step=PITCH_STEP,
                fov_default=FOV_DEFAULT
            )
            
            env.reset()
            
            if env.max_steps == 0:
                print("📌 max_steps为0，基于初始图像直接进行预测")
                if env.current_image is not None:
                    env._force_final_prediction()
                    print(f"✅ 预测完成: {env.final_prediction}")
            else:
                step = 0
                truncated = False
                
                while step < env.max_steps and not truncated:
                    if env.current_image is None:
                        print("❌ 无图像，跳过该点")
                        break
                    
                    action_str = env.gpt4o_decision(env.current_image, env.current_step + 1)
                    action, reason = env.parse_gpt_decision(action_str)
                    env.reason_history.append(reason)
                    obs, reward, terminated, truncated, info = env.step(action)
                    info["reason"] = reason
                    step += 1
                    
                    print(f"Step {env.current_step} | Action: {action}")
                    print(f"   Reason: {reason}")
                    
                    action_type = info.get("action_type")
                    if action_type == "move" and "theoretical_lat" in info:
                        print(f"   {info['msg']}")
                        print(f"   理论坐标：{info['theoretical_lat']:.6f}, {info['theoretical_lng']:.6f} → 实际坐标：{info['actual_lat']:.6f}, {info['actual_lng']:.6f} (偏差{info['coord_correction_diff_m']:.2f}米)")
                    else:
                        print(f"   {info['msg']}")
            
            result_path = os.path.join(point_dir, "final_prediction.txt")
            with open(result_path, "w", encoding="utf-8") as f:
                f.write(env.final_prediction)
            print(f"✅ 预测结果已保存: {result_path}")
            
            if env.coord_correction_history:
                import csv
                corr_path = os.path.join(point_dir, "coord_correction_history.csv")
                with open(corr_path, "w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=env.coord_correction_history[0].keys())
                    writer.writeheader()
                    writer.writerows(env.coord_correction_history)
                print(f"✅ 坐标修正历史已保存: {corr_path}")
            
            generate_visualization(env, point_dir)
            
            point_result = {
                "name": point_info["name"],
                "actual_lat": point_info["lat"],
                "actual_lng": point_info["lng"],
                "prediction": env.final_prediction if hasattr(env, 'final_prediction') else "No prediction",
                "predicted_lat": None,
                "predicted_lng": None,
                "error": None
            }
            
            if point_result["prediction"]:
                try:
                    import re
                    patterns = [
                        r'predicted_initial_coords: ([\d.-]+),([\d.-]+)',
                        r'([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)',
                        r'lat[\s:]*([-+]?\d*\.?\d+)[^,]*lng[\s:]*([-+]?\d*\.?\d+)',
                        r'latitude[\s:]*([-+]?\d*\.?\d+)[^,]*longitude[\s:]*([-+]?\d*\.?\d+)'
                    ]
                    
                    for pattern in patterns:
                        match = re.search(pattern, point_result["prediction"])
                        if match:
                            point_result["predicted_lat"] = float(match.group(1))
                            point_result["predicted_lng"] = float(match.group(2))
                            break
                    
                    if point_result["predicted_lat"] is not None and point_result["predicted_lng"] is not None:
                        point_result["error"] = calculate_gps_distance(
                            point_result["predicted_lat"], point_result["predicted_lng"],
                            point_result["actual_lat"], point_result["actual_lng"]
                        )
                except Exception as e:
                    print(f"❌ 计算偏差时出错: {str(e)}")
            
            batch_results.append(point_result)
        
        except Exception as e:
            print(f"❌ 该点运行失败: {e}")
            traceback.print_exc()
    
    generate_batch_summary(batch_dir, batch_results)


if __name__ == "__main__":
    batch_run()
