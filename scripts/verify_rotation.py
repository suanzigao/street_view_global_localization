"""
验证旋转矩阵与Google Maps heading的对应关系

Google Maps heading定义：
- heading=0: 正北
- heading=90: 正东（顺时针）
- heading=180: 正南
- heading=270: 正西

数学旋转：逆时针为正
所以需要使用 -heading 作为旋转角度
"""

import math

def verify_rotation(heading_deg, pitch_deg):
    yaw = -math.radians(heading_deg)
    pitch = math.radians(pitch_deg)
    
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    cos_pitch = math.cos(pitch)
    sin_pitch = math.sin(pitch)
    
    right_x = cos_yaw
    right_y = sin_yaw
    right_z = 0
    
    forward_x = sin_yaw * cos_pitch
    forward_y = cos_yaw * cos_pitch
    forward_z = sin_pitch
    
    down_x = -sin_yaw * sin_pitch
    down_y = cos_yaw * sin_pitch
    down_z = -cos_pitch
    
    print(f"\n=== heading={heading_deg}° (yaw={-heading_deg}°), pitch={pitch_deg}° ===")
    print(f"right:   ({right_x:.4f}, {right_y:.4f}, {right_z:.4f})")
    print(f"forward: ({forward_x:.4f}, {forward_y:.4f}, {forward_z:.4f})")
    print(f"down:    ({down_x:.4f}, {down_y:.4f}, {down_z:.4f})")
    
    u = 0
    v = 1
    
    ray_x = u * right_x + v * down_x + 1 * forward_x
    ray_y = u * right_y + v * down_y + 1 * forward_y
    ray_z = u * right_z + v * down_z + 1 * forward_z
    
    print(f"\n图像下方像素(u=0, v=1)的射线方向:")
    print(f"ray_world: ({ray_x:.4f}, {ray_y:.4f}, {ray_z:.4f})")
    print(f"→ 是否指向地面(Z<0): {'YES' if ray_z < 0 else 'NO'}")


verify_rotation(0, 0)
verify_rotation(90, 0)
verify_rotation(180, 0)
verify_rotation(45, 0)
