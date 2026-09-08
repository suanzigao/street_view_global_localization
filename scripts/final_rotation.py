"""
最终旋转矩阵验证

旋转顺序：先绕世界Z轴旋转yaw（-heading，因为Google Maps heading是顺时针），再绕相机X轴旋转pitch

旋转矩阵R:
[cos(yaw)  sin(yaw)  0]
[-sin(yaw) cos(yaw)  0]
[0         0         1]

验证：
- heading=0° (yaw=0): forward=(0,1,0)朝北, right=(1,0,0)朝东
- heading=90° (yaw=-90°): forward=(1,0,0)朝东, right=(0,-1,0)朝南
"""

import math

def verify_rotation():
    test_cases = [
        (0, 0),
        (90, 0),
        (45, 0),
    ]
    
    print("=== 旋转矩阵验证 ===")
    print()
    
    for heading_deg, pitch_deg in test_cases:
        yaw = -math.radians(heading_deg)
        pitch = math.radians(pitch_deg)
        
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        cos_pitch = math.cos(pitch)
        sin_pitch = math.sin(pitch)
        
        R = np.array([
            [cos_yaw, sin_yaw, 0],
            [-sin_yaw, cos_yaw, 0],
            [0, 0, 1]
        ], dtype=np.float64)
        
        right_cam = np.array([1, 0, 0])
        down_cam = np.array([0, 1, 0])
        forward_cam = np.array([0, 0, 1])
        
        right_world = R @ right_cam
        down_world = R @ down_cam
        forward_world = R @ forward_cam
        
        print(f"heading={heading_deg}°, pitch={pitch_deg}°:")
        print(f"  right: ({right_world[0]:.4f}, {right_world[1]:.4f}, {right_world[2]:.4f})")
        print(f"  down:  ({down_world[0]:.4f}, {down_world[1]:.4f}, {down_world[2]:.4f})")
        print(f"  forward: ({forward_world[0]:.4f}, {forward_world[1]:.4f}, {forward_world[2]:.4f})")
        print()


import numpy as np
verify_rotation()
