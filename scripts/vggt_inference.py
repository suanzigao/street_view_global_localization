"""
VGGT 推理脚本：以街景图像为输入，输出点云数据
"""
import os
import sys
import glob
import torch
import numpy as np
from PIL import Image
from torchvision import transforms as TF
from vggt.models.vggt import VGGT

IMAGES_DIR = "/home/suanzigao/PycharmProjects/street_view_global_localization/batch_streetview_results/2026-05-30_22-18-26_batch_run/2026-05-30_22-18-26_Qc7f8UCkWQSdu_0Wc7BDtQ_lat55_7132_lng12_5929/streetview_images"
OUTPUT_DIR = os.path.join(IMAGES_DIR, "vggt_output")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载所有街景 jpg 图像（按 step 排序）
    image_paths = sorted(
        glob.glob(os.path.join(IMAGES_DIR, "step_*_streetview.jpg")),
        key=lambda p: int(p.split("step_")[1].split("_")[0])
    )
    print(f"Found {len(image_paths)} images:")
    for p in image_paths:
        print(f"  {os.path.basename(p)}")

    # 预处理图像（缩小尺寸以适配 8GB 显存）
    TARGET_SIZE = 518  # VGGT 训练分辨率，必须是 14 的整数倍
    print(f"Loading and resizing to {TARGET_SIZE}px...")
    images_list = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.Resampling.BICUBIC)
        images_list.append(TF.ToTensor()(img))
    images = torch.stack(images_list)
    print(f"Preprocessed images shape: {images.shape}")

    # 加载模型（从本地权重文件）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16  # 用 FP16 省显存
    print(f"Device: {device}, Dtype: {dtype}")

    WEIGHTS_PATH = "/home/suanzigao/PycharmProjects/street_view_global_localization/models/vggt_weights/model.safetensors"
    print(f"Loading VGGT model from local weights: {WEIGHTS_PATH}")
    model = VGGT(enable_point=False, enable_depth=False, enable_track=False)
    from safetensors.torch import load_file
    state_dict = load_file(WEIGHTS_PATH)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    print("Model loaded successfully! (camera-only mode for <8GB GPU)")

    # 推理（分批处理）
    BATCH_SIZE = 1  # 518px 单张推理
    print(f"Running inference in batches of {BATCH_SIZE} images...")

    all_pose_enc = []

    for start in range(0, len(images), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(images))
        batch = images[start:end]
        print(f"  Processing images {start}-{end-1} ({len(batch)} images)...")
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=dtype):
                predictions = model(batch.to(device))

        all_pose_enc.append(predictions["pose_enc"].cpu().numpy()[0])

    # 拼接结果
    pose_enc = np.concatenate(all_pose_enc, axis=0)

    print(f"\n=== Results ===")
    print(f"pose_enc shape: {pose_enc.shape}")

    # 保存结果
    for i, img_path in enumerate(image_paths):
        step_name = os.path.basename(img_path).replace("_streetview.jpg", "")
        np.save(os.path.join(OUTPUT_DIR, f"{step_name}_pose_enc.npy"), pose_enc[i])

    # 保存合并数据
    np.save(os.path.join(OUTPUT_DIR, "all_pose_enc.npy"), pose_enc)

    # 输出相机位姿信息
    print(f"\n=== Camera Pose Encoding (per image) ===")
    for i, img_path in enumerate(image_paths):
        pe = pose_enc[i]  # [Tx, Ty, Tz, Qx, Qy, Qz, Qw, FoV_h, FoV_w]
        print(f"  {os.path.basename(img_path)}: "
              f"T=({pe[0]:.2f}, {pe[1]:.2f}, {pe[2]:.2f}), "
              f"FoV=({np.rad2deg(pe[7]):.1f}°, {np.rad2deg(pe[8]):.1f}°)")

    print(f"\n✅ All results saved to: {OUTPUT_DIR}")
    print(f"Files: {sorted(os.listdir(OUTPUT_DIR))}")


if __name__ == "__main__":
    main()
