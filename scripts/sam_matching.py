import cv2
import numpy as np
import os
import sys
from PIL import Image

sys.path.append('/home/suanzigao/PycharmProjects/street_view_global_localization')

from tools.sam3_tool import SAM3Tool

def segment_road_with_sam(image_path, sam_tool):
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 无法加载图像: {image_path}")
        return None, None
    
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    result = sam_tool.segment(pil_img, prompt="road")
    
    if result.get('success') and len(result.get('masks', [])) > 0:
        road_mask = result['masks'][0]
        road_mask = (road_mask * 255).astype(np.uint8)
        return img, road_mask
    else:
        print(f"⚠️ SAM3未检测到道路，使用Canny边缘检测")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        kernel = np.ones((5, 5), np.uint8)
        road_mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        return img, road_mask

def template_matching(bev_road, sat_road):
    result = cv2.matchTemplate(sat_road, bev_road, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    return max_loc, max_val

def main():
    bev_path = "/home/suanzigao/PycharmProjects/street_view_global_localization/streetview_images/semantic_pointcloud/conf_filtered_bev.png"
    satellite_path = "/home/suanzigao/PycharmProjects/street_view_global_localization/satellite_images/satellite_zoom16_x35218_y21468.png"
    
    output_dir = "/home/suanzigao/PycharmProjects/street_view_global_localization/matching_results"
    os.makedirs(output_dir, exist_ok=True)
    
    print("🔄 初始化SAM3工具...")
    sam_tool = SAM3Tool()
    sam_tool.initialize()
    
    print("🔍 使用SAM3分割BEV图像中的道路...")
    img_bev, bev_road = segment_road_with_sam(bev_path, sam_tool)
    
    print("🔍 使用SAM3分割卫星图像中的道路...")
    img_sat, sat_road = segment_road_with_sam(satellite_path, sam_tool)
    
    if bev_road is None or sat_road is None:
        print("❌ 道路分割失败")
        return
    
    cv2.imwrite(os.path.join(output_dir, "sam_bev_road.png"), bev_road)
    cv2.imwrite(os.path.join(output_dir, "sam_sat_road.png"), sat_road)
    
    if bev_road.shape != sat_road.shape:
        bev_road_resized = cv2.resize(bev_road, (sat_road.shape[1], sat_road.shape[0]))
    else:
        bev_road_resized = bev_road
    
    max_loc, max_val = template_matching(bev_road_resized, sat_road)
    print(f"📍 模板匹配最佳位置: ({max_loc[0]}, {max_loc[1]}), 相似度: {max_val:.4f}")
    
    bev_h, bev_w = bev_road_resized.shape[:2]
    if max_loc[1] + bev_h <= img_sat.shape[0] and max_loc[0] + bev_w <= img_sat.shape[1]:
        matched_region = img_sat[max_loc[1]:max_loc[1]+bev_h, max_loc[0]:max_loc[0]+bev_w]
        blended = cv2.addWeighted(cv2.resize(img_bev, (bev_w, bev_h)), 0.5, matched_region, 0.5, 0)
        cv2.imwrite(os.path.join(output_dir, "sam_matched_blended.png"), blended)
        
        full_result = img_sat.copy()
        cv2.rectangle(full_result, max_loc, 
                      (max_loc[0] + bev_w, max_loc[1] + bev_h), 
                      (0, 255, 0), 5)
        cv2.imwrite(os.path.join(output_dir, "sam_satellite_with_location.png"), full_result)
    
    sam_tool.close()
    
    print(f"\n🎉 SAM3道路分割匹配完成! 结果保存在: {output_dir}")

if __name__ == "__main__":
    main()