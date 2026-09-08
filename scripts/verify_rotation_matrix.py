"""
验证旋转矩阵的正确性
"""

import math

def get_rotation_vectors(heading_deg, pitch_deg):
    yaw = math.radians(heading_deg)
    pitch = math.radians(pitch_deg)
    
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    cos_pitch = math.cos(pitch)
    sin_pitch = math.sin(pitch)
    
    right_x = cos_yaw
    right_y = sin_yaw
    right_z = 0
    
    forward_x = -sin_yaw * cos_pitch
    forward_y = cos_yaw * cos_pitch
    forward_z = -sin_pitch
    
    down_x = sin_yaw * sin_pitch
    down_y = -cos_yaw * sin_pitch
    down_z = -cos_pitch
    
    return right_x, right_y, right_z, forward_x, forward_y, forward_z, down_x, down_y, down_z


print("=== 旋转矩阵验证 ===")
print()

test_cases = [
    (0, 0, "朝北, 水平"),
    (90, 0, "朝东, 水平"),
    (180, 0, "朝南, 水平"),
    (45, 0, "朝东北, 水平"),
    (0, -10, "朝北, 向下10°"),
]

for heading, pitch, desc in test_cases:
    rx, ry, rz, fx, fy, fz, dx, dy, dz = get_rotation_vectors(heading, pitch)
    print(f"heading={heading}°, pitch={pitch}° ({desc}):")
    print(f"  right: ({rx:.4f}, {ry:.4f}, {rz:.4f})")
    print(f"  forward: ({fx:.4f}, {fy:.4f}, {fz:.4f})")
    print(f"  down: ({dx:.4f}, {dy:.4f}, {dz:.4f})")
    print()

print("=== 射线方向验证 ===")
print()

def get_ray(heading_deg, pitch_deg, u, v):
    yaw = math.radians(heading_deg)
    pitch = math.radians(pitch_deg)
    
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    cos_pitch = math.cos(pitch)
    sin_pitch = math.sin(pitch)
    
    right_x = cos_yaw
    right_y = sin_yaw
    
    forward_x = -sin_yaw * cos_pitch
    forward_y = cos_yaw * cos_pitch
    forward_z = -sin_pitch
    
    down_x = sin_yaw * sin_pitch
    down_y = -cos_yaw * sin_pitch
    down_z = -cos_pitch
    
    ray_x = u * right_x + v * down_x + forward_x
    ray_y = u * right_y + v * down_y + forward_y
    ray_z = u * 0 + v * down_z + forward_z
    
    return ray_x, ray_y, ray_z


print("heading=0°, pitch=0°:")
ray_x, ray_y, ray_z = get_ray(0, 0, 0, 0)
print(f"  中心(u=0,v=0): ray=({ray_x:.4f}, {ray_y:.4f}, {ray_z:.4f}) → {'朝北 ✓' if abs(ray_x)<0.01 and ray_y>0.99 and abs(ray_z)<0.01 else '错误'}")
ray_x, ray_y, ray_z = get_ray(0, 0, 0, 1)
print(f"  下方(u=0,v=1): ray=({ray_x:.4f}, {ray_y:.4f}, {ray_z:.4f}) → {'朝北偏下 ✓' if abs(ray_x)<0.01 and ray_y>0.9 and ray_z<0 else '错误'}")
ray_x, ray_y, ray_z = get_ray(0, 0, 1, 0)
print(f"  右方(u=1,v=0): ray=({ray_x:.4f}, {ray_y:.4f}, {ray_z:.4f}) → {'朝东偏北 ✓' if ray_x>0.9 and ray_y>0.9 else '错误'}")

print()
print("heading=90°, pitch=0°:")
ray_x, ray_y, ray_z = get_ray(90, 0, 0, 0)
print(f"  中心(u=0,v=0): ray=({ray_x:.4f}, {ray_y:.4f}, {ray_z:.4f}) → {'朝东 ✓' if ray_x>0.99 and abs(ray_y)<0.01 and abs(ray_z)<0.01 else '错误'}")
ray_x, ray_y, ray_z = get_ray(90, 0, 0, 1)
print(f"  下方(u=0,v=1): ray=({ray_x:.4f}, {ray_y:.4f}, {ray_z:.4f}) → {'朝东偏下 ✓' if ray_x>0.9 and abs(ray_y)<0.1 and ray_z<0 else '错误'}")

print()
print("heading=45°, pitch=0°:")
ray_x, ray_y, ray_z = get_ray(45, 0, 0, 0)
print(f"  中心(u=0,v=0): ray=({ray_x:.4f}, {ray_y:.4f}, {ray_z:.4f}) → {'朝东北 ✓' if abs(ray_x-ray_y)<0.01 and ray_x>0.7 else '错误'}")
