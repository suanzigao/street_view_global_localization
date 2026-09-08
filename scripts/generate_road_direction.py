"""根据详细描述生成T型路口道路走向图"""
import cv2
import numpy as np
import os

OUTPUT_DIR = "/home/suanzigao/PycharmProjects/street_view_global_localization/batch_streetview_results/2026-06-01_14-36-04_batch_run/2026-06-01_14-36-04_Qc7f8UCkWQSdu_0Wc7BDtQ_lat55_7132_lng12_5929/streetview_images/vggt_remote_output"

def main():
    print("🚀 根据详细描述生成T型路口道路走向图")
    print("=" * 50)
    
    # 创建画布
    img = np.zeros((700, 800, 3), dtype=np.uint8)
    img[:] = (255, 255, 255)  # 白色背景
    
    # 交叉口中心点
    cx, cy = 420, 380
    
    # 绘制方向标记（正北向上）
    cv2.putText(img, 'N', (cx-8, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 2)
    cv2.putText(img, 'S', (cx-8, 670), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 2)
    cv2.putText(img, 'W', (10, cy+6), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 2)
    cv2.putText(img, 'E', (770, cy+6), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 2)
    
    # 绘制方位十字
    cv2.line(img, (cx, 20), (cx, 680), (200, 200, 200), 1)
    cv2.line(img, (20, cy), (780, cy), (200, 200, 200), 1)
    
    # ================================================
    # 主路（T字"竖"）- 红色
    # 绝对走向：北偏东12°（12°）↔ 南偏西12°（192°）
    # ================================================
    length_main = 280
    
    # 北偏东12°方向（画面顶部偏右）
    angle_north_east = 90 - 12  # 转换为图像坐标系角度
    x1 = cx + int(length_main * np.cos(np.radians(angle_north_east)))
    y1 = cy - int(length_main * np.sin(np.radians(angle_north_east)))
    
    # 南偏西12°方向（画面右下方）
    angle_south_west = 270 - 12  # 转换为图像坐标系角度
    x2 = cx + int(length_main * np.cos(np.radians(angle_south_west)))
    y2 = cy - int(length_main * np.sin(np.radians(angle_south_west)))
    
    # 绘制主路（最宽阔）
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 20)
    
    # 绘制主路车道线
    for offset in [-5, 0, 5]:
        px1 = x1 + int(offset * np.cos(np.radians(angle_north_east + 90)))
        py1 = y1 - int(offset * np.sin(np.radians(angle_north_east + 90)))
        px2 = x2 + int(offset * np.cos(np.radians(angle_south_west + 90)))
        py2 = y2 - int(offset * np.sin(np.radians(angle_south_west + 90)))
        cv2.line(img, (px1, py1), (px2, py2), (255, 255, 255), 3)
    
    # ================================================
    # 支路（T字"横"）- 蓝色
    # 绝对走向：南偏西52°（232°）↔ 北偏东52°（52°）
    # 从左下方延伸至交叉口并终止，单向汇入主路
    # ================================================
    length_branch = 220
    
    # 南偏西52°方向（画面左下方）
    angle_south_west_branch = 270 - 52  # 转换为图像坐标系角度
    x3 = cx + int(length_branch * np.cos(np.radians(angle_south_west_branch)))
    y3 = cy - int(length_branch * np.sin(np.radians(angle_south_west_branch)))
    
    # 支路终止于交叉口，不穿过
    cv2.line(img, (x3, y3), (cx, cy), (255, 0, 0), 15)
    
    # 绘制停止线（靠近交叉口）
    stop_line_dist = 40
    stop_x = cx + int(stop_line_dist * np.cos(np.radians(angle_south_west_branch)))
    stop_y = cy - int(stop_line_dist * np.sin(np.radians(angle_south_west_branch)))
    
    # 停止线垂直于支路方向
    perp_angle = angle_south_west_branch + 90
    cv2.line(img, 
             (int(stop_x - 30 * np.cos(np.radians(perp_angle))), 
              int(stop_y + 30 * np.sin(np.radians(perp_angle)))),
             (int(stop_x + 30 * np.cos(np.radians(perp_angle))), 
              int(stop_y - 30 * np.sin(np.radians(perp_angle)))),
             (255, 255, 255), 4)
    
    # 绘制斑马线（人行横道）
    for i in range(4):
        zebra_offset = stop_line_dist + 20 + i * 12
        zx = cx + int(zebra_offset * np.cos(np.radians(angle_south_west_branch)))
        zy = cy - int(zebra_offset * np.sin(np.radians(angle_south_west_branch)))
        cv2.line(img,
                 (int(zx - 25 * np.cos(np.radians(perp_angle))),
                  int(zy + 25 * np.sin(np.radians(perp_angle)))),
                 (int(zx + 25 * np.cos(np.radians(perp_angle))),
                  int(zy - 25 * np.sin(np.radians(perp_angle)))),
                 (255, 255, 255), 6)
    
    # 绘制绿色护栏（道路两侧）
    for side in [-12, 12]:
        hx1 = x3 + int(side * np.cos(np.radians(angle_south_west_branch + 90)))
        hy1 = y3 - int(side * np.sin(np.radians(angle_south_west_branch + 90)))
        hx2 = stop_x + int(side * np.cos(np.radians(angle_south_west_branch + 90)))
        hy2 = stop_y - int(side * np.sin(np.radians(angle_south_west_branch + 90)))
        cv2.line(img, (hx1, hy1), (hx2, hy2), (0, 150, 0), 3)
    
    # ================================================
    # 绘制方向箭头
    # ================================================
    def draw_arrow(start, end, color, label=""):
        angle = np.arctan2(end[1]-start[1], end[0]-start[0])
        tip = end
        wing_len = 18
        
        wing1 = (
            int(tip[0] - wing_len * np.cos(angle - np.pi/6)),
            int(tip[1] - wing_len * np.sin(angle - np.pi/6))
        )
        wing2 = (
            int(tip[0] - wing_len * np.cos(angle + np.pi/6)),
            int(tip[1] - wing_len * np.sin(angle + np.pi/6))
        )
        
        cv2.line(img, wing1, tip, color, 3)
        cv2.line(img, wing2, tip, color, 3)
        
        if label:
            text_x = int((start[0] + end[0]) / 2 + 20 * np.cos(angle + np.pi/2))
            text_y = int((start[1] + end[1]) / 2 + 20 * np.sin(angle + np.pi/2))
            cv2.putText(img, label, (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # 主路双向箭头
    mid_main_n = ((cx+x1)//2, (cy+y1)//2)
    mid_main_s = ((cx+x2)//2, (cy+y2)//2)
    draw_arrow((cx, cy), (x1, y1), (0, 0, 255), "12°")
    draw_arrow((cx, cy), (x2, y2), (0, 0, 255), "192°")
    
    # 支路单向箭头（汇入主路）
    mid_branch = ((cx+x3)//2, (cy+y3)//2)
    draw_arrow((x3, y3), (cx, cy), (255, 0, 0), "232°")
    
    # 交叉口标记
    cv2.circle(img, (cx, cy), 8, (0, 0, 0), -1)
    cv2.circle(img, (cx, cy), 12, (0, 0, 0), 2)
    
    # ================================================
    # 添加标注
    # ================================================
    cv2.putText(img, "T型平面交叉口道路走向图", (180, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3)
    
    # 图例
    cv2.rectangle(img, (50, 580), (180, 605), (0, 0, 255), -1)
    cv2.putText(img, "主路（北偏东12°↔南偏西12°）", (190, 600), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
    
    cv2.rectangle(img, (50, 615), (180, 640), (255, 0, 0), -1)
    cv2.putText(img, "支路（南偏西52°→交叉口，单向汇入）", (190, 630), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
    
    cv2.rectangle(img, (50, 650), (100, 675), (0, 150, 0), -1)
    cv2.putText(img, "绿色护栏", (110, 668), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
    
    # 路口特征说明
    cv2.putText(img, f"路口夹角: 88°", (600, 580), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "平面交叉 / 无环岛 / 无信号灯", (600, 610), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img, "主路南北贯通，支路西南汇入", (600, 640), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # 保存结果
    output_path = os.path.join(OUTPUT_DIR, "road_direction_t_intersection_corrected.png")
    cv2.imwrite(output_path, img)
    print(f"✅ 保存: {output_path}")
    
    print("\n🎉 完成!")
    print("\n路口特征：")
    print("  - 🔴 主路（T字'竖'）: 北偏东12° ↔ 南偏西12°，连续贯穿交叉口")
    print("  - 🔵 支路（T字'横'）: 南偏西52° → 交叉口，单向汇入")
    print("  - 道路夹角: 88°（几乎垂直）")
    print("  - 交叉口类型: 平面交叉，无环岛、立交桥或信号灯")
    print("  - 主路特征: 最宽阔，有多条平行白色车道线")
    print("  - 支路特征: 有停止线、斑马线、绿色护栏")

if __name__ == "__main__":
    main()