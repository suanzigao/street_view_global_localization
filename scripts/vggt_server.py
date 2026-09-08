"""
VGGT Remote Server - FastAPI服务
在远程服务器上运行此文件，提供VGGT推理API接口

运行方式:
    cd /path/to/vggt
    python vggt_server.py

或使用uvicorn:
    uvicorn vggt_server:app --host 0.0.0.0 --port 18888 --workers 1
"""
import os
import io
import json
import base64
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms as TF

# 添加VGGT到路径
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../models/vggt'))

from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

app = FastAPI(title="VGGT Remote Inference Server", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局模型变量
model = None
device = None


def init_model():
    """初始化VGGT模型"""
    global model, device
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available!")
    
    print("Loading VGGT model...")
    model = VGGT()
    
    # 从HuggingFace下载权重
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    
    model.eval()
    model = model.to(device)
    print("✅ VGGT model loaded successfully!")


@app.get("/health")
async def health_check():
    """健康检查"""
    if model is None:
        return {"status": "uninitialized"}
    return {"status": "ok"}


@app.post("/vggt/infer")
async def vggt_infer(request: dict):
    """
    VGGT推理接口
    参数:
        images: List[str] - base64编码的图像列表
        output_types: List[str] - 输出类型，可选: camera, depth, points
    
    返回:
        pose_enc: 相机位姿编码 [S, 9]
        depth: 深度图 [S, H, W]
        depth_conf: 深度置信度 [S, H, W]
        world_points: 世界坐标系点云 [S, H, W, 3]
        world_points_conf: 点云置信度 [S, H, W]
    """
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Model not initialized")
        
        images_b64 = request.get("images", [])
        output_types = request.get("output_types", ["camera", "depth", "points"])
        
        if not images_b64:
            raise HTTPException(status_code=400, detail="No images provided")
        
        # 解码图像
        TARGET_SIZE = 518
        images_list = []
        
        for img_b64 in images_b64:
            img_data = base64.b64decode(img_b64)
            img = Image.open(io.BytesIO(img_data)).convert("RGB")
            img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.Resampling.BICUBIC)
            images_list.append(TF.ToTensor()(img))
        
        images = torch.stack(images_list).to(device)
        print(f"Processing {len(images)} images...")
        
        # 推理
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images)
        
        result = {"success": True}
        
        # 相机位姿
        if "camera" in output_types:
            pose_enc = predictions["pose_enc"].cpu().numpy()
            result["pose_enc"] = pose_enc.tolist()
        
        # 深度图
        if "depth" in output_types:
            depth = predictions["depth"].cpu().numpy()
            depth_conf = predictions["depth_conf"].cpu().numpy()
            result["depth"] = depth.tolist()
            result["depth_conf"] = depth_conf.tolist()
        
        # 点云
        if "points" in output_types:
            world_points = predictions["world_points"].cpu().numpy()
            world_points_conf = predictions["world_points_conf"].cpu().numpy()
            result["world_points"] = world_points.tolist()
            result["world_points_conf"] = world_points_conf.tolist()
        
        torch.cuda.empty_cache()
        print("✅ Inference completed successfully")
        
        return result
    
    except Exception as e:
        print(f"❌ Error during inference: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/vggt/info")
async def vggt_info():
    """获取服务器信息"""
    return {
        "model": "VGGT-1B",
        "device": device,
        "status": "ready" if model else "not initialized"
    }


if __name__ == "__main__":
    init_model()
    
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=18888,
        workers=1,
        timeout_keep_alive=300
    )