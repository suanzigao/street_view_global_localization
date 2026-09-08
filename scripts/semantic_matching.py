import cv2
import numpy as np
import os

def load_and_preprocess(image_path, target_size=None):
    img = cv2.imread(image_path)
    if target_size:
        img = cv2.resize(img, target_size)
    return img

def extract_road_mask(img, lower_red=(0, 0, 200), upper_red=(50, 50, 255)):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_red, upper_red)
    return mask

def extract_satellite_road(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.erode(edges, kernel, iterations=1)
    
    return edges

def phase_correlation(src, dst):
    src_fft = np.fft.fft2(src)
    dst_fft = np.fft.fft2(dst)
    
    product = src_fft * np.conj(dst_fft)
    cross_power = product / np.abs(product)
    
    correlation = np.fft.ifft2(cross_power)
    correlation = np.abs(correlation)
    
    y, x = np.unravel_index(np.argmax(correlation), correlation.shape)
    
    if x > src.shape[1] // 2:
        x -= src.shape[1]
    if y > src.shape[0] // 2:
        y -= src.shape[0]
    
    return (x, y), correlation

def template_matching(bev_road, sat_road):
    result = cv2.matchTemplate(sat_road, bev_road, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    
    return max_loc, max_val

def main():
    bev_path = "/home/suanzigao/PycharmProjects/street_view_global_localization/streetview_images/semantic_pointcloud/conf_filtered_bev.png"
    satellite_path = "/home/suanzigao/PycharmProjects/street_view_global_localization/satellite_images/satellite_zoom16_x35218_y21468.png"
    
    output_dir = "/home/suanzigao/PycharmProjects/street_view_global_localization/matching_results"
    os.makedirs(output_dir, exist_ok=True)
    
    img_bev = load_and_preprocess(bev_path)
    img_sat = load_and_preprocess(satellite_path, target_size=img_bev.shape[:2][::-1])
    
    bev_road = extract_road_mask(img_bev)
    sat_road = extract_satellite_road(img_sat)
    
    cv2.imwrite(os.path.join(output_dir, "bev_road_mask.png"), bev_road)
    cv2.imwrite(os.path.join(output_dir, "sat_road_mask.png"), sat_road)
    
    shift, corr = phase_correlation(bev_road, sat_road)
    print(f"📍 相位相关法检测到的偏移: ({shift[0]}, {shift[1]})")
    
    max_loc, max_val = template_matching(bev_road, sat_road)
    print(f"📍 模板匹配最佳位置: ({max_loc[0]}, {max_loc[1]}), 相似度: {max_val:.4f}")
    
    bev_h, bev_w = bev_road.shape
    matched_region = img_sat[max_loc[1]:max_loc[1]+bev_h, max_loc[0]:max_loc[0]+bev_w]
    
    blended = cv2.addWeighted(img_bev, 0.5, matched_region, 0.5, 0)
    cv2.imwrite(os.path.join(output_dir, "semantic_matched_blended.png"), blended)
    
    full_result = img_sat.copy()
    cv2.rectangle(full_result, max_loc, 
                  (max_loc[0] + bev_w, max_loc[1] + bev_h), 
                  (0, 255, 0), 5)
    cv2.imwrite(os.path.join(output_dir, "satellite_with_bev_location.png"), full_result)
    
    print(f"\n🎉 语义匹配完成! 结果保存在: {output_dir}")
    print(f"📌 BEV在卫星图上的匹配位置: ({max_loc[0]}, {max_loc[1]})")

if __name__ == "__main__":
    main()