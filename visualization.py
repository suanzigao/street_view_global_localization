import os
from typing import Dict, List

# ====================== 生成可视化HTML页面 ======================
def generate_visualization(env, point_dir):
    """
    生成HTML页面，可视化轨迹和图像
    """
    # 收集轨迹数据
    trajectory = []
    # 添加初始位置
    trajectory.append({
        "step": 0,
        "lat": env.data_handler.initial_lat,
        "lng": env.data_handler.initial_lng,
        "heading": env.data_handler.heading,
        "fov": env.data_handler.fov if hasattr(env.data_handler, 'fov') else 30,
        "pitch": env.data_handler.pitch,
        "action": "Initial position"
    })
    
    # 添加所有操作的轨迹点
    if hasattr(env, 'trajectory_history'):
        for item in env.trajectory_history:
            trajectory.append(item)
    
    # 收集图像数据
    images = []
    for step, img_path in env.image_path_history.items():
        images.append({
            "step": step,
            "path": img_path
        })
    
    # 生成HTML内容
    html_content = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Street View Navigation Visualization</title>
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
        .map-container {
            width: 100%;
            height: 500px;
            margin-bottom: 20px;
            border: 1px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
        }
        #map {
            width: 100%;
            height: 100%;
        }
        .timeline {
            margin-top: 30px;
        }
        .timeline h2 {
            color: #333;
            margin-bottom: 20px;
        }
        .step {
            display: flex;
            margin-bottom: 30px;
            border-bottom: 1px solid #eee;
            padding-bottom: 20px;
        }
        .step-info {
            flex: 1;
            padding-right: 20px;
        }
        .step-image {
            flex: 1;
        }
        .step-image img {
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .step-number {
            font-size: 24px;
            font-weight: bold;
            color: #4CAF50;
            margin-bottom: 10px;
        }
        .step-details {
            color: #666;
        }
        .step-details p {
            margin: 5px 0;
        }
    </style>
    <script src="https://maps.googleapis.com/maps/api/js?key=AIzaSyAXmbPK8IXtGV9fjWVI8K4b5cw9Tshrvy0"></script>
</head>
<body>
    <div class="container">
        <h1>Street View Navigation Visualization</h1>
        
        <div class="map-container">
            <div id="map"></div>
        </div>
        
        <div class="timeline">
            <h2>Navigation Timeline</h2>
            {timeline_html}
        </div>
    </div>
    
    <script>
        // 轨迹数据
        const trajectory = {trajectory_data};
        
        // 初始化地图
        function initMap() {
            // 使用第一个点作为中心
            const center = new google.maps.LatLng(trajectory[0].lat, trajectory[0].lng);
            
            const map = new google.maps.Map(document.getElementById('map'), {
                zoom: 18,
                center: center,
                mapTypeId: google.maps.MapTypeId.SATELLITE
            });
            
            // 添加标记
            trajectory.forEach((point, index) => {
                const marker = new google.maps.Marker({
                    position: new google.maps.LatLng(point.lat, point.lng),
                    map: map,
                    title: `Step ${point.step}: ${point.action}`
                });
                
                // 添加信息窗口
                const infoWindow = new google.maps.InfoWindow({
                    content: `<strong>Step ${point.step}</strong><br>
                             Lat: ${point.lat.toFixed(6)}<br>
                             Lng: ${point.lng.toFixed(6)}<br>
                             Heading: ${point.heading.toFixed(1)}°<br>
                             FOV: ${point.fov.toFixed(1)}°<br>
                             Pitch: ${point.pitch.toFixed(1)}°<br>
                             Action: ${point.action}`
                });
                
                marker.addListener('click', function() {
                    infoWindow.open(map, marker);
                });
            });
            
            // 绘制轨迹线
            const path = trajectory.map(point => new google.maps.LatLng(point.lat, point.lng));
            const polyline = new google.maps.Polyline({
                path: path,
                geodesic: true,
                strokeColor: '#FF0000',
                strokeOpacity: 1.0,
                strokeWeight: 2
            });
            
            polyline.setMap(map);
        }
        
        // 页面加载完成后初始化地图
        window.onload = initMap;
    </script>
</body>
</html>
    '''
    
    # 生成时间线HTML
    timeline_html = ""
    for item in trajectory:
        timeline_html += f"""
        <div class="step">
            <div class="step-info">
                <div class="step-number">Step {item['step']}</div>
                <div class="step-details">
                    <p><strong>Action:</strong> {item['action']}</p>
                    <p><strong>GPS:</strong> {item['lat']:.6f}, {item['lng']:.6f}</p>
                    <p><strong>Heading:</strong> {item['heading']:.1f}°</p>
                    <p><strong>FOV:</strong> {item['fov']:.1f}°</p>
                    <p><strong>Pitch:</strong> {item['pitch']:.1f}°</p>
                </div>
            </div>
            <div class="step-image">
                <!-- 这里可以添加对应步骤的图像 -->
                <img src="streetview_images/step_{item['step']}_streetview.jpg" alt="Step {item['step']}">
            </div>
        </div>
        """
    
    # 格式化轨迹数据为JavaScript对象
    trajectory_data = str(trajectory).replace("'", "\"")
    
    # 替换HTML模板中的占位符
    html_content = html_content.replace("{timeline_html}", timeline_html)
    html_content = html_content.replace("{trajectory_data}", trajectory_data)
    
    # 保存HTML文件
    html_path = os.path.join(point_dir, "visualization.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"✅ 可视化HTML页面已生成: {html_path}")
