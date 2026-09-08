import cv2
import numpy as np
import os

def load_and_preprocess(image_path, target_size=None):
    img = cv2.imread(image_path)
    if target_size:
        img = cv2.resize(img, target_size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray

def orb_feature_matching(img1, gray1, img2, gray2):
    orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    return matches, kp1, kp2

def estimate_homography(good_matches, kp1, kp2):
    if len(good_matches) < 4:
        return None, 0
    
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    inliers = sum(mask.flatten()) if mask is not None else 0
    
    return M, inliers

def match_and_evaluate(bev_path, satellite_path, output_dir):
    img_bev, gray_bev = load_and_preprocess(bev_path)
    img_sat, gray_sat = load_and_preprocess(satellite_path, target_size=img_bev.shape[:2][::-1])
    
    matches, kp_bev, kp_sat = orb_feature_matching(img_bev, gray_bev, img_sat, gray_sat)
    
    good_matches = matches[:100]
    
    if len(good_matches) >= 4:
        M, inliers = estimate_homography(good_matches, kp_bev, kp_sat)
    else:
        M, inliers = None, 0
    
    bev_name = os.path.basename(bev_path).replace('.png', '')
    if M is not None:
        warped = cv2.warpPerspective(img_bev, M, (img_sat.shape[1], img_sat.shape[0]))
        blended = cv2.addWeighted(warped, 0.5, img_sat, 0.5, 0)
        cv2.imwrite(os.path.join(output_dir, f"{bev_name}_blended.png"), blended)
    
    return {
        'name': bev_name,
        'matches': len(matches),
        'inliers': inliers,
        'inlier_ratio': inliers / len(good_matches) if good_matches else 0
    }

def main():
    bev_dir = "/home/suanzigao/PycharmProjects/street_view_global_localization/streetview_images/semantic_pointcloud"
    satellite_path = "/home/suanzigao/PycharmProjects/street_view_global_localization/satellite_images/satellite_zoom16_x35218_y21468.png"
    output_dir = "/home/suanzigao/PycharmProjects/street_view_global_localization/matching_results"
    os.makedirs(output_dir, exist_ok=True)
    
    bev_files = [f for f in os.listdir(bev_dir) if f.endswith('.png') and 'bev' in f.lower()]
    
    print(f"📁 发现 {len(bev_files)} 个BEV图像")
    print(f"📷 卫星图像: {satellite_path}")
    print("\n" + "="*60)
    
    results = []
    for bev_file in sorted(bev_files):
        bev_path = os.path.join(bev_dir, bev_file)
        result = match_and_evaluate(bev_path, satellite_path, output_dir)
        results.append(result)
        
        status = "✅" if result['inlier_ratio'] > 0.1 else "⚠️"
        print(f"{status} {result['name']:30s} | 匹配点: {result['matches']:4d} | 内点: {result['inliers']:3d} | 比率: {result['inlier_ratio']:.2f}")
    
    print("\n" + "="*60)
    print("📊 匹配结果排序（按内点比率）:")
    results.sort(key=lambda x: x['inlier_ratio'], reverse=True)
    
    for i, r in enumerate(results[:5], 1):
        print(f"{i}. {r['name']}: 内点比率 {r['inlier_ratio']:.2f}")
    
    print(f"\n🎉 所有匹配完成! 结果保存在: {output_dir}")

if __name__ == "__main__":
    main()