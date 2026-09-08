"""
VGGT 远程推理客户端
通过 SSH 隧道调用远程服务器上的 VGGT 服务，获取 3D 场景结构信息
"""
import base64
import io
import os
import json
import requests
from typing import List, Dict, Optional, Tuple
from PIL import Image


class VGGTClient:
    """VGGT 远程推理客户端"""

    def __init__(self, base_url: str = "http://127.0.0.1:18888", timeout: int = 300):
        self.base_url = base_url
        self.timeout = timeout

    def health_check(self) -> Dict:
        resp = requests.get(f"{self.base_url}/health", timeout=10)
        return resp.json()

    def infer(
        self,
        images: List[Image.Image],
        output_types: List[str] = None,
    ) -> Dict:
        if output_types is None:
            output_types = ["camera", "depth", "points"]

        image_b64_list = []
        for img in images:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=90)
            image_b64_list.append(base64.b64encode(buf.getvalue()).decode("utf-8"))

        resp = requests.post(
            f"{self.base_url}/vggt/infer",
            json={"images": image_b64_list, "output_types": output_types},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def get_camera_motion_summary(
        self, images: List[Image.Image]
    ) -> str:
        """
        获取相机运动摘要，供大模型决策使用
        输入多张连续的街景图像，返回相机运动分析
        """
        result = self.infer(images, output_types=["camera"])

        if not result.get("success"):
            return "VGGT inference failed"

        pose_enc = result["pose_enc"]  # [S, 9]

        summary_parts = [f"VGGT Analysis ({len(images)} views):"]

        for i, pe in enumerate(pose_enc):
            tx, ty, tz = pe[0], pe[1], pe[2]
            qx, qy, qz, qw = pe[3], pe[4], pe[5], pe[6]

            summary_parts.append(f"  View {i}: T=({tx:.2f}, {ty:.2f}, {tz:.2f})")