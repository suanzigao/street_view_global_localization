"""从BEV图像提取道路中心线"""
import cv2
import numpy as np
import os

INPUT_IMAGE = "/home/suanzigao/PycharmProjects/street_view_global_localization/batch_streetview_results/2026-06-01_14-36-04_batch_run/2026-06-01_14-36-04_Qc7f8UCkWQSdu_0Wc7BDtQ_lat55_7132_lng12_5929/streetview_images/vggt_remote_output/bev_color.png"
OUTPUT_DIR = os.path.dirname(INPUT_IMAGE)

def extract_road_region(image):
    """提取道路区域（基于颜色特征）"""
    # 转换到HSV空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 道路通常是低饱和度的灰色区域
    saturation = hsv[..., 1]
    value = hsv[..., 2]
    
    # 低饱和度 + 中等亮度 = 道路区域
    low_saturation = saturation < 100
    medium_value = (value > 30) & (value < 230)
    
    road_mask = (low_saturation & medium_value).astype(np.uint8) * 255
    
    # 形态学操作去除噪点
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel)
    
    return road_mask

def skeletonize(image):
    """使用距离变换提取骨架（中心线）"""
    # 距离变换
    dist = cv2.distanceTransform(image, cv2.DIST_L2, 3)
    
    # 归一化
    cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
    
    # 找到局部最大值作为骨架
    kernel = np.ones((3, 3), dtype=np.float32)
    local_max = cv2.filter2D(dist, -1, kernel)
    local_max = (dist == local_max).astype(np.uint8) * 255
    
    # 阈值化
    skeleton = np.where(dist > 0.05, 255, 0).astype(np.uint8)
    skeleton = cv2.bitwise_and(skeleton, local_max)
    
    return skeleton

def main():
    print("🚀 道路中心线提取")
    print("=" * 40)
    
    # 1. 读取图像
    print("\n[1/4] 读取图像...")
    img = cv2.imread(INPUT_IMAGE)
    if img is None:
        print(f"❌ 无法读取图像: {INPUT_IMAGE}")
        return
    print(f"  图像尺寸: {img.shape[0]} × {img.shape[1]}")
    
    # 2. 提取道路区域
    print("\n[2/4] 提取道路区域...")
    road_mask = extract_road_region(img)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "road_mask.png"), road_mask)
    print("✅ 保存: road_mask.png")
    
    # 3. 骨架化提取中心线
    print("\n[3/4] 提取中心线...")
    centerline = skeletonize(road_mask)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "road_centerline.png"), centerline)
    print("✅ 保存: road_centerline.png")
    
    # 4. 在原图上叠加中心线
    print("\n[4/4] 叠加可视化...")
    result = img.copy()
    result[centerline > 0] = [0, 0, 255]  # 红色中心线
    cv2.imwrite(os.path.join(OUTPUT_DIR, "road_centerline_overlay.png"), result)
    print("✅ 保存: road_centerline_overlay.png")
    
    print("\n🎉 完成!")
    print(f"输出目录: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()