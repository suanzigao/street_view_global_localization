"""手动绘制道路中心线"""
import cv2
import numpy as np
import os

INPUT_IMAGE = "/home/suanzigao/PycharmProjects/street_view_global_localization/batch_streetview_results/2026-06-01_14-36-04_batch_run/2026-06-01_14-36-04_Qc7f8UCkWQSdu_0Wc7BDtQ_lat55_7132_lng12_5929/streetview_images/vggt_remote_output/bev_color.png"
OUTPUT_DIR = os.path.dirname(INPUT_IMAGE)

def main():
    print("🚀 手动绘制道路中心线")
    print("=" * 40)
    
    # 读取图像
    print("\n[1/2] 读取图像...")
    img = cv2.imread(INPUT_IMAGE)
    if img is None:
        print(f"❌ 无法读取图像: {INPUT_IMAGE}")
        return
    
    # 根据分析结果定义道路中心线坐标
    # 主路：从左下角(80, 480)到右上角(510, 50)
    # 支路：从左上角(60, 180)到交汇点(200, 380)
    main_road = [
        (80, 480),   # 起点：左下角
        (150, 420),  # 中间点1
        (220, 360),  # 中间点2（斑马线区域）
        (280, 300),  # 中间点3
        (350, 220),  # 中间点4
        (420, 140),  # 中间点5
        (510, 50)    # 终点：右上角
    ]
    
    branch_road = [
        (60, 180),   # 起点：左上角
        (120, 260),  # 中间点1
        (160, 320),  # 中间点2
        (200, 380)   # 终点：交汇点
    ]
    
    # 绘制中心线
    print("\n[2/2] 绘制中心线...")
    result = img.copy()
    
    # 绘制主路（红色）
    for i in range(len(main_road) - 1):
        cv2.line(result, main_road[i], main_road[i+1], (0, 0, 255), 3)
    
    # 绘制支路（蓝色）
    for i in range(len(branch_road) - 1):
        cv2.line(result, branch_road[i], branch_road[i+1], (255, 0, 0), 3)
    
    # 在关键点添加标记
    for i, point in enumerate(main_road):
        cv2.circle(result, point, 5, (0, 255, 0), -1)
        cv2.putText(result, str(i), (point[0]+10, point[1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 保存结果
    output_path = os.path.join(OUTPUT_DIR, "road_centerline_manual.png")
    cv2.imwrite(output_path, result)
    print(f"✅ 保存: {output_path}")
    
    print("\n🎉 完成!")
    print("道路中心线已绘制：")
    print("  - 🔴 红色：主路（西南→东北）")
    print("  - 🔵 蓝色：支路（西北→东南）")
    print("  - 🟢 绿色圆点：关键点")

if __name__ == "__main__":
    main()