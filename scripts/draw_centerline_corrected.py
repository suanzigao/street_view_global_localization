"""修正道路中心线绘制"""
import cv2
import numpy as np
import os

INPUT_IMAGE = "/home/suanzigao/PycharmProjects/street_view_global_localization/batch_streetview_results/2026-06-01_14-36-04_batch_run/2026-06-01_14-36-04_Qc7f8UCkWQSdu_0Wc7BDtQ_lat55_7132_lng12_5929/streetview_images/vggt_remote_output/bev_color.png"
OUTPUT_DIR = os.path.dirname(INPUT_IMAGE)

def main():
    print("🚀 修正道路中心线")
    print("=" * 40)
    
    # 读取图像
    img = cv2.imread(INPUT_IMAGE)
    if img is None:
        print(f"❌ 无法读取图像")
        return
    
    # 根据实际图像重新分析道路走向
    # 主路：沿着图像中间的灰色道路，从左下角(150, 480)到右上角(512, 80)
    # 右侧延伸道路：从(350, 300)向右延伸到(512, 380)
    # 左侧支路：从左上角(80, 150)向下汇入主路(180, 400)
    
    main_road = [
        (150, 480),  # 起点：左下角道路
        (200, 420),  # 斑马线区域
        (260, 360),  # 中间区域
        (320, 300),  # 路口区域
        (380, 240),  # 继续延伸
        (450, 160),  # 接近右上角
        (510, 80)    # 终点：右上角
    ]
    
    right_extension = [
        (320, 300),  # 从主路分出
        (400, 330),  # 向右延伸
        (480, 360),  # 继续向右
        (512, 380)   # 图像边缘
    ]
    
    left_branch = [
        (80, 150),   # 起点：左上角
        (120, 250),  # 向下延伸
        (160, 350),  # 继续向下
        (180, 400)   # 汇入主路
    ]
    
    # 绘制中心线
    result = img.copy()
    
    # 主路（红色）
    for i in range(len(main_road) - 1):
        cv2.line(result, main_road[i], main_road[i+1], (0, 0, 255), 3)
    
    # 右侧延伸（橙色）
    for i in range(len(right_extension) - 1):
        cv2.line(result, right_extension[i], right_extension[i+1], (0, 165, 255), 3)
    
    # 左侧支路（蓝色）
    for i in range(len(left_branch) - 1):
        cv2.line(result, left_branch[i], left_branch[i+1], (255, 0, 0), 3)
    
    # 标记关键点
    for i, point in enumerate(main_road):
        cv2.circle(result, point, 5, (0, 255, 0), -1)
    
    # 保存结果
    output_path = os.path.join(OUTPUT_DIR, "road_centerline_corrected.png")
    cv2.imwrite(output_path, result)
    print(f"✅ 保存: {output_path}")
    
    print("\n🎉 修正完成!")
    print("道路中心线已重新绘制：")
    print("  - 🔴 红色：主路")
    print("  - 🟠 橙色：右侧延伸道路")
    print("  - 🔵 蓝色：左侧支路")

if __name__ == "__main__":
    main()