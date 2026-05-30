import csv
from typing import List, Dict

def read_csv(csv_path: str) -> List[Dict]:
    """读取CSV文件"""
    points = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                points.append({
                    'name': row['name'],
                    'lat': float(row['lat']),
                    'lng': float(row['lng'])
                })
            except (KeyError, ValueError):
                continue
    return points

def get_continent(lat: float, lng: float) -> str:
    """根据经纬度判断所属大洲"""
    # 欧洲: 纬度35-70, 经度-10到40
    if 35 <= lat <= 70 and -10 <= lng <= 40:
        return 'Europe'
    
    # 亚洲: 纬度10-75, 经度25到180
    if 10 <= lat <= 75 and 25 <= lng <= 180:
        return 'Asia'
    
    # 北美洲: 纬度10-75, 经度-180到-50
    if 10 <= lat <= 75 and -180 <= lng <= -50:
        return 'North America'
    
    # 南美洲: 纬度-60到15, 经度-90到-35
    if -60 <= lat <= 15 and -90 <= lng <= -35:
        return 'South America'
    
    # 非洲: 纬度-35到37, 经度-20到55
    if -35 <= lat <= 37 and -20 <= lng <= 55:
        return 'Africa'
    
    # 大洋洲: 纬度-50到-10, 经度110到180
    if -50 <= lat <= -10 and 110 <= lng <= 180:
        return 'Oceania'
    
    return 'Unknown'

def analyze_points(points: List[Dict]) -> Dict:
    """分析点位分布"""
    continent_counts = {
        'Europe': 0,
        'Asia': 0,
        'North America': 0,
        'South America': 0,
        'Africa': 0,
        'Oceania': 0,
        'Unknown': 0
    }
    
    for point in points:
        continent = get_continent(point['lat'], point['lng'])
        continent_counts[continent] += 1
    
    return continent_counts

def generate_html(continent_counts: Dict, total_points: int) -> str:
    """生成HTML报告"""
    europe_pct = (continent_counts['Europe'] / total_points) * 100
    asia_pct = (continent_counts['Asia'] / total_points) * 100
    
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GPS点位大洲分布分析</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; padding: 40px 20px; }}
        .container {{ max-width: 900px; margin: 0 auto; background: white; border-radius: 20px; box-shadow: 0 20px 60px rgba(0,0,0,0.3); overflow: hidden; }}
        .header {{ background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); color: white; padding: 30px; text-align: center; }}
        .header h1 {{ font-size: 28px; margin-bottom: 10px; }}
        .header p {{ opacity: 0.8; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 20px; padding: 30px; }}
        .stat-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 25px; border-radius: 15px; text-align: center; }}
        .stat-card .value {{ font-size: 42px; font-weight: bold; }}
        .stat-card .label {{ font-size: 14px; opacity: 0.9; margin-top: 5px; }}
        .chart-container {{ padding: 30px; }}
        .chart-wrapper {{ height: 400px; }}
        .table-section {{ padding: 0 30px 30px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f8f9fa; font-weight: 600; }}
        tr:hover {{ background: #f8f9fa; }}
        .progress-bar {{ height: 20px; background: #e9ecef; border-radius: 10px; overflow: hidden; margin: 10px 0; }}
        .progress-fill {{ height: 100%; border-radius: 10px; transition: width 0.5s ease; }}
        .europe {{ background: linear-gradient(90deg, #3498db, #2980b9); }}
        .asia {{ background: linear-gradient(90deg, #e74c3c, #c0392b); }}
        .footer {{ background: #f8f9fa; padding: 20px; text-align: center; color: #666; font-size: 14px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌍 GPS点位大洲分布分析</h1>
            <p>数据集: geochain_mini_test.csv | 总点位: {total_points} 个</p>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <div class="value">{continent_counts['Europe']}</div>
                <div class="label">欧洲点位</div>
            </div>
            <div class="stat-card">
                <div class="value">{continent_counts['Asia']}</div>
                <div class="label">亚洲点位</div>
            </div>
            <div class="stat-card">
                <div class="value">{continent_counts['North America']}</div>
                <div class="label">北美洲点位</div>
            </div>
            <div class="stat-card">
                <div class="value">{continent_counts['South America']}</div>
                <div class="label">南美洲点位</div>
            </div>
        </div>
        
        <div class="chart-container">
            <h2 style="text-align: center; margin-bottom: 20px; color: #333;">大洲分布比例</h2>
            <div class="chart-wrapper">
                <canvas id="continentChart"></canvas>
            </div>
        </div>
        
        <div class="table-section">
            <h2 style="margin-bottom: 15px; color: #333;">详细统计</h2>
            <table>
                <tr>
                    <th>大洲</th>
                    <th>数量</th>
                    <th>占比</th>
                    <th>进度条</th>
                </tr>
                <tr>
                    <td>欧洲</td>
                    <td>{continent_counts['Europe']}</td>
                    <td>{europe_pct:.1f}%</td>
                    <td><div class="progress-bar"><div class="progress-fill europe" style="width: {europe_pct}%"></div></div></td>
                </tr>
                <tr>
                    <td>亚洲</td>
                    <td>{continent_counts['Asia']}</td>
                    <td>{asia_pct:.1f}%</td>
                    <td><div class="progress-bar"><div class="progress-fill asia" style="width: {asia_pct}%"></div></div></td>
                </tr>
                <tr>
                    <td>北美洲</td>
                    <td>{continent_counts['North America']}</td>
                    <td>{(continent_counts['North America']/total_points*100):.1f}%</td>
                    <td><div class="progress-bar"><div class="progress-fill" style="width: {(continent_counts['North America']/total_points*100):.1f}%; background: linear-gradient(90deg, #2ecc71, #27ae60);"></div></div></td>
                </tr>
                <tr>
                    <td>南美洲</td>
                    <td>{continent_counts['South America']}</td>
                    <td>{(continent_counts['South America']/total_points*100):.1f}%</td>
                    <td><div class="progress-bar"><div class="progress-fill" style="width: {(continent_counts['South America']/total_points*100):.1f}%; background: linear-gradient(90deg, #f39c12, #e67e22);"></div></div></td>
                </tr>
                <tr>
                    <td>大洋洲</td>
                    <td>{continent_counts['Oceania']}</td>
                    <td>{(continent_counts['Oceania']/total_points*100):.1f}%</td>
                    <td><div class="progress-bar"><div class="progress-fill" style="width: {(continent_counts['Oceania']/total_points*100):.1f}%; background: linear-gradient(90deg, #9b59b6, #8e44ad);"></div></div></td>
                </tr>
                <tr>
                    <td>非洲</td>
                    <td>{continent_counts['Africa']}</td>
                    <td>{(continent_counts['Africa']/total_points*100):.1f}%</td>
                    <td><div class="progress-bar"><div class="progress-fill" style="width: {(continent_counts['Africa']/total_points*100):.1f}%; background: linear-gradient(90deg, #1abc9c, #16a085);"></div></div></td>
                </tr>
            </table>
        </div>
        
        <div class="footer">
            数据来源: geochain_mini_test.csv | 生成时间: 2026-05-21
        </div>
    </div>
    
    <script>
        var ctx = document.getElementById('continentChart').getContext('2d');
        var chart = new Chart(ctx, {{
            type: 'doughnut',
            data: {{
                labels: ['欧洲', '亚洲', '北美洲', '南美洲', '大洋洲', '非洲'],
                datasets: [{{
                    data: [
                        {continent_counts['Europe']},
                        {continent_counts['Asia']},
                        {continent_counts['North America']},
                        {continent_counts['South America']},
                        {continent_counts['Oceania']},
                        {continent_counts['Africa']}
                    ],
                    backgroundColor: [
                        'rgba(52, 152, 219, 0.8)',
                        'rgba(231, 76, 60, 0.8)',
                        'rgba(46, 204, 113, 0.8)',
                        'rgba(243, 156, 18, 0.8)',
                        'rgba(155, 89, 182, 0.8)',
                        'rgba(26, 188, 156, 0.8)'
                    ],
                    borderColor: [
                        'rgba(52, 152, 219, 1)',
                        'rgba(231, 76, 60, 1)',
                        'rgba(46, 204, 113, 1)',
                        'rgba(243, 156, 18, 1)',
                        'rgba(155, 89, 182, 1)',
                        'rgba(26, 188, 156, 1)'
                    ],
                    borderWidth: 2
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        position: 'right',
                        labels: {{
                            padding: 20,
                            font: {{ size: 14 }}
                        }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>"""
    return html

def main():
    csv_path = '/home/suanzigao/PycharmProjects/street_view_global_localization/geochain_mini_test.csv'
    points = read_csv(csv_path)
    total_points = len(points)
    print(f"读取到 {total_points} 个点位")
    
    continent_counts = analyze_points(points)
    print("\n大洲分布统计:")
    for continent, count in continent_counts.items():
        if count > 0:
            print(f"  {continent}: {count} 个 ({(count/total_points*100):.1f}%)")
    
    # 生成HTML
    html = generate_html(continent_counts, total_points)
    output_path = '/home/suanzigao/PycharmProjects/street_view_global_localization/continent_analysis.html'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"\nHTML报告已生成: {output_path}")

if __name__ == '__main__':
    main()
