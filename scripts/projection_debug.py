"""
投影调试脚本 - 验证旋转矩阵的正确性
"""

import math

def debug_rotation(heading_deg, pitch_deg):
    yaw = math.radians(heading_deg)
    pitch = math.radians(pitch_deg)
    
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    cos_pitch = math.cos(pitch)
    sin_pitch = math.sin(pitch)
    
    print(f"\n=== heading={heading_deg}°, pitch={pitch_deg}° ===")
    
    right_x = cos_yaw
    right_y = sin_yaw
    right_z = 0
    
    forward_x = sin_yaw * cos_pitch
    forward_y = cos_yaw * cos_pitch
    forward_z = sin_pitch
    
    up_x = -sin_yaw * sin_pitch
    up_y = cos_yaw * sin_pitch
    up_z = cos_pitch
    
    down_x = -up_x
    down_y = -up_y
    down_z = -up_z
    
    print(f"right: ({right_x:.4f}, {right_y:.4f}, {right_z:.4f})")
    print(f"down:  ({down_x:.4f}, {down_y:.4f}, {down_z:.4f})")
    print(f"forward: ({forward_x:.4f}, {forward_y:.4f}, {forward_z:.4f})")
    print(f"up:    ({up_x:.4f}, {up_y:.4f}, {up_z:.4f})")
    
    u = 0
    v = 1
    
    ray_x = u * right_x + v * down_x + 1 * forward_x
    ray_y = u * right_y + v * down_y + 1 * forward_y
    ray_z = u * right_z + v * down_z + 1 * forward_z
    
    print(f"\n图像下方像素(u=0, v=1)的射线方向:")
    print(f"ray_world: ({ray_x:.4f}, {ray_y:.4f}, {ray_z:.4f})")
    print(f"→ 是否指向地面(Z<0): {'YES' if ray_z < 0 else 'NO'}")
    
    if ray_z < 0:
        t = -2.5 / ray_z
        print(f"→ 地面交点距离: t={t:.2f}米")
        print(f"→ 投影方向: X方向偏移={t*ray_x:.2f}米, Y方向偏移={t*ray_y:.2f}米")


debug_rotation(0, 0)
debug_rotation(0, -10)
debug_rotation(90, 0)
debug_rotation(45, 0)
