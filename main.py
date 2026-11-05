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
GLOBAL_MAX_STEPS = 10
GLOBAL_MOVE_STEP = 15.0
GLOBAL_MAX_RETRIES = 3
FOV_STEP = 20
PITCH_STEP = 10
TURN_ANGLE = 45
FOV_DEFAULT = 60
GOOGLE_API_RADIUS = 10
GPS_CSV_PATH = "geochain_mini_10.csv"
USE_BEV = True

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

        # 该步的视觉观察（observation），与 reason 分开展示，作为历史记忆
        observation = ""
        if hasattr(env, 'image_meta_history') and item['step'] in env.image_meta_history:
            observation = env.image_meta_history[item['step']].get('observation', '')

        # 该步的当前计划（plan），展示意图如何跨步演进
        plan = ""
        if hasattr(env, 'image_meta_history') and item['step'] in env.image_meta_history:
            plan = env.image_meta_history[item['step']].get('plan', '')

        # 检查是否有分割图像
        has_segmented = hasattr(env, 'segmented_image_history') and item['step'] in env.segmented_image_history
        
        # 确定显示的图像路径
        if has_segmented:
            # 有分割图像时，显示分割后的图像
            seg_info = env.segmented_image_history[item['step']][0]
            img_src = "streetview_images/step{}_segmented.png".format(item['step'])
            img_alt = f"Step {item['step']} - Segmented: {seg_info.get('prompt', 'unknown')}"
            segment_label = f"<p><strong>Segmented Objects:</strong> {seg_info.get('prompt', 'unknown')}</p>"
        else:
            # 没有分割图像时，显示原始图像
            img_src = "streetview_images/step_{}_streetview.jpg".format(item['step'])
            img_alt = f"Step {item['step']}"
            segment_label = ""
        
        timeline_html += f"""
        <div class="step">
            <div class="step-info">
                <div class="step-number">Step {item['step']}</div>
                <div class="step-details">
                    <p><strong>Action:</strong> {item['action']}</p>
                    {f'<p><strong>Reason:</strong> {reason}</p>' if reason else ''}
                    {f'<p><strong>👁 Observation:</strong> {observation}</p>' if observation else ''}
                    {f'<p><strong>🧭 Plan:</strong> {plan}</p>' if plan else ''}
                    {segment_label}
                    <p><strong>GPS:</strong> {item['lat']:.6f}, {item['lng']:.6f}</p>
                    <p><strong>Heading:</strong> {item['heading']:.1f}°</p>
                    <p><strong>FOV:</strong> {item['fov']:.1f}°</p>
                    <p><strong>Pitch:</strong> {item['pitch']:.1f}°</p>
                </div>
            </div>
            <div class="step-image">
                <img src="{img_src}" alt="{img_alt}">
            </div>
        </div>
        """
    
    # 生成 Places API 搜索结果 HTML
    places_html = ""
    if hasattr(env, 'found_places') and env.found_places:
        places_html += '<div class="places-section"><h2>📍 Places Found via Google Places API</h2><div class="places-grid">'
        for place in env.found_places:
            places_html += f"""
            <div class="place-card">
                <div class="place-card-header">Step {place['step']}: {place['query']}</div>
                <div class="place-card-body">
                    <p><strong>🏢 Name:</strong> {place['name']}</p>
                    <p><strong>📍 Address:</strong> {place['address']}</p>
                    <p><strong>🌍 Coordinates:</strong> {place['lat']:.6f}, {place['lng']:.6f}</p>
                </div>
            </div>
            """
        places_html += "</div></div>"
    
    # 准备 Places 数据传给 JavaScript 地图（仅用于卡片展示，无需地图标记）
    found_places_list = env.found_places if hasattr(env, 'found_places') else []
    import json
    found_places_json = json.dumps(found_places_list, default=str)

    # 离散探索图数据（节点=采样点，边=move 方向及结果）
    graph_nodes = []
    graph_edges = []
    if hasattr(env, '_graph_nodes') and env._graph_nodes:
        node_id_by_key = {}
        for key, node in env._graph_nodes.items():
            node_id_by_key[key] = node["id"]
            graph_nodes.append({"id": node["id"], "lat": node["lat"], "lng": node["lng"], "step": node["step"]})
        for (from_key, dir_q), edge in env._graph_edges.items():
            if from_key not in node_id_by_key:
                continue
            to_id = node_id_by_key.get(edge["to"]) if edge["to"] else None
            graph_edges.append({
                "from": node_id_by_key[from_key],
                "to": to_id,
                "dir": dir_q,
                "step": edge["step"],
            })
    graph_nodes_json = json.dumps(graph_nodes)
    graph_edges_json = json.dumps(graph_edges)
    graph_has_data = "true" if graph_nodes else "false"
    
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
        
        /* ========== 动画播放器样式 ========== */
        .animation-player {{
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
            margin-bottom: 30px;
            overflow: hidden;
        }}
        .player-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .player-title {{ font-size: 18px; font-weight: bold; }}
        .player-controls {{ display: flex; gap: 10px; align-items: center; }}
        .control-btn {{
            background: rgba(255,255,255,0.2);
            border: none;
            color: white;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            transition: background 0.2s;
        }}
        .control-btn:hover {{ background: rgba(255,255,255,0.3); }}
        .control-btn.active {{ background: rgba(255,255,255,0.4); }}
        .step-indicator {{ background: rgba(255,255,255,0.2); padding: 8px 16px; border-radius: 6px; font-size: 14px; }}
        .player-content {{ display: flex; height: 500px; }}
        .map-view {{ flex: 1; position: relative; border-right: 2px solid #eee; }}
        #animationMap {{ width: 100%; height: 100%; }}
        .map-overlay {{
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(255,255,255,0.95);
            padding: 10px 15px;
            border-radius: 8px;
            font-size: 13px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .map-overlay div {{ margin: 3px 0; }}
        .streetview-view {{ flex: 1; position: relative; background: #000; }}
        .streetview-view img {{ width: 100%; height: 100%; object-fit: contain; }}
        .streetview-overlay {{
            position: absolute;
            bottom: 10px;
            left: 10px;
            background: rgba(0,0,0,0.8);
            color: white;
            padding: 10px 15px;
            border-radius: 8px;
            font-size: 13px;
        }}
        .player-footer {{ padding: 15px 20px; background: #f8f9fa; border-top: 1px solid #eee; }}
        .timeline-slider {{
            width: 100%;
            height: 8px;
            -webkit-appearance: none;
            background: #ddd;
            border-radius: 4px;
            outline: none;
            cursor: pointer;
        }}
        .timeline-slider::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 20px;
            height: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 50%;
            cursor: pointer;
            box-shadow: 0 2px 6px rgba(0,0,0,0.2);
        }}
        .timeline-labels {{ display: flex; justify-content: space-between; margin-top: 8px; font-size: 12px; color: #666; }}
        /* ========== 原有样式 ========== */
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
        .places-section {{ margin-top: 30px; }}
        .places-section h2 {{ color: #333; margin-bottom: 15px; }}
        .places-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 15px; }}
        .place-card {{ background: white; border-radius: 8px; border: 1px solid #e0e0e0; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
        .place-card-header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 10px 15px; font-weight: bold; font-size: 14px; }}
        .place-card-body {{ padding: 12px 15px; }}
        .place-card-body p {{ margin: 6px 0; color: #444; font-size: 13px; }}
        .place-card-body strong {{ color: #555; }}
        /* ========== 探索图可视化 ========== */
        .exploration-graph {{ margin-top: 30px; }}
        .exploration-graph h2 {{ color: #333; margin-bottom: 15px; }}
        #graphSvg {{ width: 100%; height: 560px; background: #fafafa; border: 1px solid #ddd; border-radius: 8px; }}
        .graph-legend {{ margin-top: 10px; font-size: 13px; color: #555; display: flex; gap: 20px; flex-wrap: wrap; }}
        .graph-legend span {{ display: inline-flex; align-items: center; gap: 5px; }}
        .legend-line {{ display: inline-block; width: 24px; height: 3px; border-radius: 2px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Street View Navigation Visualization</h1>
        
        <!-- ========== 动画播放器 ========== -->
        <div class="animation-player">
            <div class="player-header">
                <div class="player-title">🎬 Navigation Animation</div>
                <div class="player-controls">
                    <span class="step-indicator" id="stepIndicator">Step 0 / {len(trajectory) - 1}</span>
                    <button class="control-btn" id="rewindBtn" title="Rewind">⏮</button>
                    <button class="control-btn" id="prevBtn" title="Previous">◀</button>
                    <button class="control-btn active" id="playPauseBtn" title="Play/Pause">▶</button>
                    <button class="control-btn" id="nextBtn" title="Next">▶</button>
                    <button class="control-btn" id="fastForwardBtn" title="Fast Forward">⏭</button>
                </div>
            </div>
            <div class="player-content">
                <div class="map-view">
                    <div id="animationMap"></div>
                    <div class="map-overlay">
                        <div><strong>GPS:</strong> <span id="mapGps">-</span></div>
                        <div><strong>Heading:</strong> <span id="mapHeading">-</span></div>
                        <div><strong>FOV:</strong> <span id="mapFov">-</span></div>
                        <div><strong>Pitch:</strong> <span id="mapPitch">-</span></div>
                        <div><strong>Action:</strong> <span id="mapAction">-</span></div>
                    </div>
                </div>
                <div class="streetview-view">
                    <img id="streetviewImage" src="streetview_images/step_0_streetview.jpg" alt="Street View">
                    <div class="streetview-overlay">
                        <div id="streetviewStep">Step 0</div>
                        <div id="streetviewAction">Initial position</div>
                    </div>
                </div>
            </div>
            <div class="player-footer">
                <input type="range" class="timeline-slider" id="timelineSlider" min="0" max="{len(trajectory) - 1}" value="0">
                <div class="timeline-labels">
                    <span>Start</span>
                    <span>End (Prediction)</span>
                </div>
            </div>
        </div>
        
        <div class="map-container"><div id="map"></div></div>
        <div class="timeline">
            <h2>Navigation Timeline</h2>
            {timeline_html}
        </div>
        <div class="exploration-graph">
            <h2>🗺️ Exploration Graph (discrete nodes &amp; move directions)</h2>
            <svg id="graphSvg"></svg>
            <div class="graph-legend">
                <span><span class="legend-line" style="background:#4CAF50;"></span> move success (edge)</span>
                <span><span class="legend-line" style="background:#e74c3c;"></span> move blocked (dead end)</span>
                <span><span style="width:12px;height:12px;border-radius:50%;background:#2196F3;display:inline-block;"></span> visited node (sampling point)</span>
                <span><span style="width:12px;height:12px;border-radius:50%;background:#FF9800;display:inline-block;"></span> current node</span>
            </div>
        </div>
        {places_html}
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
        const graphNodes = {graph_nodes_json};
        const graphEdges = {graph_edges_json};
        const graphHasData = {graph_has_data};
        let currentStep = 0;
        let isPlaying = false;
        let playInterval = null;
        let animationMap = null;
        let positionMarker = null;
        let fovPolygon = null;
        let trajectoryPolyline = null;
        
        function initAnimationMap() {{
            const center = new google.maps.LatLng(trajectory[0].lat, trajectory[0].lng);
            animationMap = new google.maps.Map(document.getElementById('animationMap'), {{
                zoom: 19,
                center: center,
                mapTypeId: google.maps.MapTypeId.SATELLITE
            }});
            
            const path = trajectory.map(p => new google.maps.LatLng(p.lat, p.lng));
            trajectoryPolyline = new google.maps.Polyline({{
                path: path,
                geodesic: true,
                strokeColor: '#FF0000',
                strokeOpacity: 0.6,
                strokeWeight: 3
            }});
            trajectoryPolyline.setMap(animationMap);
            
            updateAnimationView(0);
        }}
        
        function updateAnimationView(step) {{
            currentStep = step;
            const point = trajectory[step];
            const pos = new google.maps.LatLng(point.lat, point.lng);
            
            if (!positionMarker) {{
                positionMarker = new google.maps.Marker({{
                    position: pos,
                    map: animationMap,
                    icon: {{
                        path: google.maps.SymbolPath.FORWARD_CLOSED_ARROW,
                        scale: 6,
                        fillColor: '#4CAF50',
                        fillOpacity: 1,
                        strokeColor: '#fff',
                        strokeWeight: 2,
                        rotation: point.heading
                    }},
                    title: `Step ${{point.step}}`
                }});
            }} else {{
                positionMarker.setPosition(pos);
                positionMarker.setIcon({{
                    path: google.maps.SymbolPath.FORWARD_CLOSED_ARROW,
                    scale: 6,
                    fillColor: '#4CAF50',
                    fillOpacity: 1,
                    strokeColor: '#fff',
                    strokeWeight: 2,
                    rotation: point.heading
                }});
            }}
            
            updateFovCone(point);
            
            document.getElementById('mapGps').textContent = `${{point.lat.toFixed(6)}}, ${{point.lng.toFixed(6)}}`;
            document.getElementById('mapHeading').textContent = `${{point.heading.toFixed(1)}}°`;
            document.getElementById('mapFov').textContent = `${{point.fov.toFixed(1)}}°`;
            document.getElementById('mapPitch').textContent = `${{point.pitch.toFixed(1)}}°`;
            document.getElementById('mapAction').textContent = point.action;
            
            document.getElementById('streetviewImage').src = `streetview_images/step_${{point.step}}_streetview.jpg`;
            document.getElementById('streetviewStep').textContent = `Step ${{point.step}}`;
            document.getElementById('streetviewAction').textContent = point.action;
            
            document.getElementById('stepIndicator').textContent = `Step ${{step}} / ${{trajectory.length - 1}}`;
            document.getElementById('timelineSlider').value = step;
            
            animationMap.setCenter(pos);
        }}
        
        function updateFovCone(point) {{
            const center = new google.maps.LatLng(point.lat, point.lng);
            const heading = point.heading;
            const fov = point.fov;
            const distance = 0.0002;
            
            const halfFov = fov / 2;
            const leftAngle = heading - halfFov;
            const rightAngle = heading + halfFov;
            
            const centerLat = point.lat;
            const centerLng = point.lng;
            
            function calcCoord(angle, dist) {{
                const R = 6378137;
                const d = dist * R;
                const lat1 = centerLat * Math.PI / 180;
                const lng1 = centerLng * Math.PI / 180;
                const ang = angle * Math.PI / 180;
                
                const lat2 = Math.asin(
                    Math.sin(lat1) * Math.cos(d / R) +
                    Math.cos(lat1) * Math.sin(d / R) * Math.cos(ang)
                );
                const lng2 = lng1 + Math.atan2(
                    Math.sin(ang) * Math.sin(d / R) * Math.cos(lat1),
                    Math.cos(d / R) - Math.sin(lat1) * Math.sin(lat2)
                );
                return [lat2 * 180 / Math.PI, lng2 * 180 / Math.PI];
            }}
            
            const points = [center];
            for (let a = leftAngle; a <= rightAngle; a += 5) {{
                const coord = calcCoord(a, distance);
                points.push(new google.maps.LatLng(coord[0], coord[1]));
            }}
            points.push(center);
            
            if (fovPolygon) {{
                fovPolygon.setMap(null);
            }}
            
            fovPolygon = new google.maps.Polygon({{
                paths: points,
                map: animationMap,
                fillColor: '#667eea',
                fillOpacity: 0.4,
                strokeColor: '#667eea',
                strokeWeight: 2
            }});
        }}
        
        function playAnimation() {{
            if (isPlaying) return;
            isPlaying = true;
            document.getElementById('playPauseBtn').textContent = '⏸';
            
            playInterval = setInterval(() => {{
                if (currentStep < trajectory.length - 1) {{
                    updateAnimationView(currentStep + 1);
                }} else {{
                    stopAnimation();
                }}
            }}, 1500);
        }}
        
        function stopAnimation() {{
            isPlaying = false;
            document.getElementById('playPauseBtn').textContent = '▶';
            if (playInterval) {{
                clearInterval(playInterval);
                playInterval = null;
            }}
        }}
        
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

        function drawGraph() {{
            if (!graphHasData || graphNodes.length === 0) return;
            const svg = document.getElementById('graphSvg');
            if (!svg) return;

            const width = svg.clientWidth || 800;
            const height = svg.clientHeight || 560;
            const padding = 70;

            // 等距圆柱投影：以第一个节点为原点，米为单位
            const refLat = graphNodes[0].lat;
            const refLng = graphNodes[0].lng;
            const cosLat = Math.cos(refLat * Math.PI / 180);
            const proj = (n) => {{
                const x = (n.lng - refLng) * 111320 * cosLat;
                const y = -(n.lat - refLat) * 110540;
                return [x, y];
            }};

            const pts = graphNodes.map(n => {{ const [x, y] = proj(n); return {{ n, x, y }}; }});
            let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
            pts.forEach(p => {{ minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x); minY = Math.min(minY, p.y); maxY = Math.max(maxY, p.y); }});
            const spanX = (maxX - minX) || 1;
            const spanY = (maxY - minY) || 1;
            const scale = Math.min((width - 2 * padding) / spanX, (height - 2 * padding) / spanY);
            const midX = (minX + maxX) / 2, midY = (minY + maxY) / 2;
            const toSvg = (x, y) => [width / 2 + (x - midX) * scale, height / 2 + (y - midY) * scale];

            const idToP = {{}};
            pts.forEach(p => {{ idToP[p.n.id] = p; }});

            const dirNames = {{0:'N',45:'NE',90:'E',135:'SE',180:'S',225:'SW',270:'W',315:'NW'}};
            const NS = 'http://www.w3.org/2000/svg';

            // 边
            graphEdges.forEach(e => {{
                const fromP = idToP[e.from];
                if (!fromP) return;
                const [x1, y1] = toSvg(fromP.x, fromP.y);
                let x2, y2;
                const blocked = (e.to === null || e.to === undefined);
                if (blocked) {{
                    const ang = (e.dir || 0) * Math.PI / 180;
                    x2 = x1 + Math.sin(ang) * 34;
                    y2 = y1 - Math.cos(ang) * 34;
                }} else {{
                    const toP = idToP[e.to];
                    if (!toP) return;
                    [x2, y2] = toSvg(toP.x, toP.y);
                }}
                const line = document.createElementNS(NS, 'line');
                line.setAttribute('x1', x1); line.setAttribute('y1', y1);
                line.setAttribute('x2', x2); line.setAttribute('y2', y2);
                line.setAttribute('stroke', blocked ? '#e74c3c' : '#4CAF50');
                line.setAttribute('stroke-width', blocked ? '2' : '2.5');
                if (blocked) line.setAttribute('stroke-dasharray', '6,4');
                svg.appendChild(line);
                if (!blocked) {{
                    const t = document.createElementNS(NS, 'text');
                    t.setAttribute('x', (x1 + x2) / 2);
                    t.setAttribute('y', (y1 + y2) / 2);
                    t.setAttribute('font-size', '10');
                    t.setAttribute('fill', '#2e7d32');
                    t.setAttribute('text-anchor', 'middle');
                    t.textContent = dirNames[e.dir] || e.dir;
                    svg.appendChild(t);
                }}
            }});

            // 节点（最后一个 visited 节点 = 当前节点）
            const currentNodeId = graphNodes[graphNodes.length - 1].id;
            pts.forEach(p => {{
                const [x, y] = toSvg(p.x, p.y);
                const isCur = (p.n.id === currentNodeId);
                const c = document.createElementNS(NS, 'circle');
                c.setAttribute('cx', x); c.setAttribute('cy', y);
                c.setAttribute('r', isCur ? 10 : 7);
                c.setAttribute('fill', isCur ? '#FF9800' : '#2196F3');
                c.setAttribute('stroke', '#fff'); c.setAttribute('stroke-width', '2');
                svg.appendChild(c);
                const lb = document.createElementNS(NS, 'text');
                lb.setAttribute('x', x + 13); lb.setAttribute('y', y + 4);
                lb.setAttribute('font-size', '12'); lb.setAttribute('fill', '#333');
                lb.setAttribute('font-weight', 'bold');
                lb.textContent = '#' + p.n.id + (isCur ? ' (now)' : '');
                svg.appendChild(lb);
            }});
        }}
        
        document.addEventListener('DOMContentLoaded', () => {{
            initAnimationMap();
            initMap();
            drawGraph();
            
            document.getElementById('playPauseBtn').addEventListener('click', () => {{
                if (isPlaying) {{
                    stopAnimation();
                }} else {{
                    if (currentStep >= trajectory.length - 1) {{
                        updateAnimationView(0);
                    }}
                    playAnimation();
                }}
            }});
            
            document.getElementById('prevBtn').addEventListener('click', () => {{
                stopAnimation();
                if (currentStep > 0) updateAnimationView(currentStep - 1);
            }});
            
            document.getElementById('nextBtn').addEventListener('click', () => {{
                stopAnimation();
                if (currentStep < trajectory.length - 1) updateAnimationView(currentStep + 1);
            }});
            
            document.getElementById('rewindBtn').addEventListener('click', () => {{
                stopAnimation();
                updateAnimationView(0);
            }});
            
            document.getElementById('fastForwardBtn').addEventListener('click', () => {{
                stopAnimation();
                updateAnimationView(trajectory.length - 1);
            }});
            
            document.getElementById('timelineSlider').addEventListener('input', (e) => {{
                stopAnimation();
                updateAnimationView(parseInt(e.target.value));
            }});
        }});
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
    
    error_ranges = [1000, 3000, 5000, 10000, 15000, 20000, 25000, 125000, 250000]
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
        const errorRanges = ['≤ 1km', '≤ 3km', '≤ 5km', '≤ 10km', '≤ 15km', '≤ 20km', '≤ 25km', '≤ 125km', '≤ 250km'];
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
    shared_sam3_tool = None
    
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
                fov_default=FOV_DEFAULT,
                use_bev=USE_BEV
            )
            
            if shared_sam3_tool is None:
                print("🔄 预加载SAM3模型...")
                env.preload_sam3()
                shared_sam3_tool = env.sam3_tool
            else:
                env.sam3_tool = shared_sam3_tool
                print("♻️ 复用已加载的SAM3模型")
            
            env.reset()
            
            if env.max_steps == 0:
                print("📌 max_steps为0，基于初始图像直接进行预测")
                if env.current_image is not None:
                    env._force_final_prediction()
                    print(f"✅ 预测完成: {env.final_prediction}")
            else:
                truncated = False
                
                while env.current_step < env.max_steps and not truncated:
                    if env.current_image is None:
                        print("❌ 无图像，跳过该点")
                        break
                    
                    action_str = env.gpt4o_decision(env.current_image, env.current_step + 1)
                    print(f"📝 GPT原始响应:\n{action_str}\n{'-'*80}")
                    action, reason, seg_prompts, search_query = env.parse_gpt_decision(action_str)
                    env.reason_history.append(reason)
                    if env.current_step in env.image_meta_history:
                        env.image_meta_history[env.current_step]['reason'] = reason
                    
                    # 执行动作
                    obs, reward, terminated, truncated, info = env.step(action)
                    info["reason"] = reason
                    
                    action_type = info.get("action_type")
                    
                    # 检查是否检测到死循环
                    if action_type == "invalid":
                        print(f"❌ {info.get('msg', 'Unknown error')}")
                        print(f"🔄 强制右转探索新区域...")
                        # 强制执行右转操作，帮助大模型打破死循环
                        move_action = "turn right"
                        obs, reward, terminated, truncated, move_info = env.step(move_action)
                        # 记录强制动作发生的步骤，重置死循环检测窗口
                        # 使用负数索引来标识强制动作，避免与正常步骤混淆
                        env.force_action_step = -env.current_step  # 使用负数标识强制动作
                        # 添加强制动作标记到动作历史，以便死循环检测时识别
                        env.action_history.append(f"Step {env.current_step}: force_action {move_action}")
                        print(f"   自动执行: {move_action}")
                        if move_info.get('msg'):
                            print(f"   {move_info['msg']}")
                        # 跳过本次循环，让大模型在新位置重新决策
                        continue
                    
                    print(f"Step {env.current_step} | Action: {action}")
                    print(f"   Reason: {reason}")
                    
                    # 检查是否是 search 操作
                    if action_type == "search":
                        print(f"   {info['msg']}")
                    elif action_type == "segment":
                        # 分割已在 env.step() 内部完成：文字证据 + 图像证据均存入 env，
                        # 这里只打印结果，避免重复调用 SAM3。
                        print(f"   {info.get('msg', '')}")
                    elif action_type == "move" and "theoretical_lat" in info:
                        print(f"   {info['msg']}")
                        print(f"   理论坐标：{info['theoretical_lat']:.6f}, {info['theoretical_lng']:.6f} → 实际坐标：{info['actual_lat']:.6f}, {info['actual_lng']:.6f} (偏差{info['coord_correction_diff_m']:.2f}米)")
                    else:
                        print(f"   {info.get('msg', 'No message')}")
            
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
        finally:
            if 'env' in locals():
                env.close(release_sam3=False)
                print("🔄 环境已重置，SAM3模型保留供下次使用")
    
    if shared_sam3_tool is not None:
        print("🔄 释放共享SAM3模型...")
        if hasattr(shared_sam3_tool, 'close'):
            shared_sam3_tool.close()
        print("✅ SAM3模型已释放")
    
    generate_batch_summary(batch_dir, batch_results)


if __name__ == "__main__":
    batch_run()
