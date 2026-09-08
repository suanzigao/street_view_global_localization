"""
透视图像转正交投影（平行光）视图
使用仿射变换进行透视校正
"""
import os
import cv2
import numpy as np

def perspective_to_ortho_affine(image, output_size=(512, 512)):
    """
    使用仿射变换将透视图像转换为正交视图
    模拟从斜视角到鸟瞰视角的转换
    """
    h, w = image.shape[:2]
    
    src_points = np.float32([
        [w * 0.1, 0],       # 左上角
        [w * 0.9, 0],       # 右上角
        [0, h],             # 左下角
        [w, h]              # 右下角
    ])
    
    dst_points = np.float32([
        [w * 0.3, h * 0.1], # 左上角 -> 向内收缩
        [w * 0.7, h * 0.1], # 右上角 -> 向内收缩
        [0, h],             # 左下角 -> 保持
        [w, h]              # 右下角 -> 保持
    ])
    
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    ortho_img = cv2.warpPerspective(image, M, (w, h))
    
    return cv2.resize(ortho_img, output_size)

def main():
    INPUT_DIR = "/home/suanzigao/PycharmProjects/street_view_global_localization/test_image"
    OUTPUT_DIR = os.path.join(INPUT_DIR, "ortho_output")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    image_paths = sorted([
        os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR)
        if f.endswith("_streetview.jpg")
    ], key=lambda p: int(os.path.basename(p).split("step_")[1].split("_")[0]))
    
    print(f"\n📷 找到 {len(image_paths)} 张图像")
    
    print("\n🔄 正在转换为正交视图...")
    for i, p in enumerate(image_paths):
        image = cv2.imread(p)
        ortho_img = perspective_to_ortho_affine(image)
        
        base_name = os.path.basename(p).replace("_streetview.jpg", "")
        output_path = os.path.join(OUTPUT_DIR, f"{base_name}_ortho.png")
        cv2.imwrite(output_path, ortho_img)
        
        non_zero = np.count_nonzero(cv2.cvtColor(ortho_img, cv2.COLOR_BGR2GRAY))
        total = ortho_img.shape[0] * ortho_img.shape[1]
        print(f"  ✅ {output_path} (有效像素: {non_zero/total*100:.1f}%)")
    
    print(f"\n🎉 完成! 所有正交视图已保存到: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()