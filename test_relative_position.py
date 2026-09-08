"""
测试相对位置计算功能

验证：
1. 重置时relative_x=0, relative_y=0
2. 移动后relative_x, relative_y使用真实经纬度计算
3. 元数据JSON文件正确保存
"""

import os
import json
import sys

# 配置代理
PROXY_HOST = "127.0.0.1"
PROXY_PORT = 7897
os.environ["HTTP_PROXY"] = f"http://{PROXY_HOST}:{PROXY_PORT}"
os.environ["HTTPS_PROXY"] = f"http://{PROXY_HOST}:{PROXY_PORT}"
os.environ["NO_PROXY"] = "localhost,127.0.0.1,localdomain,.localdomain"

from dotenv import load_dotenv
load_dotenv()

from core import StreetViewAutoLocEnv


def test_relative_position():
    test_dir = "test_relative_position_output"
    img_dir = os.path.join(test_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    
    initial_lat = 55.71320186378624
    initial_lng = 12.59293370993324
    
    print(f"🚀 开始测试相对位置计算")
    print(f"初始坐标: ({initial_lat}, {initial_lng})")
    
    env = StreetViewAutoLocEnv(
        initial_lat=initial_lat,
        initial_lng=initial_lng,
        move_step=15.0,
        max_steps=3,
        point_dir=test_dir,
        img_save_dir=img_dir,
        fov_default=60,
        use_sam3=False
    )
    
    print(f"\n📋 重置后状态:")
    print(f"  relative_x: {env.relative_x:.4f}")
    print(f"  relative_y: {env.relative_y:.4f}")
    
    # 检查step 0的元数据
    step0_metadata_path = os.path.join(img_dir, "step_0_metadata.json")
    if os.path.exists(step0_metadata_path):
        with open(step0_metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"\n📄 step_0_metadata.json:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
    
    # 执行移动操作
    print(f"\n🔄 执行 move forward")
    obs, reward, terminated, truncated, info = env.step("move forward")
    
    print(f"\n📋 移动后状态:")
    print(f"  relative_x: {env.relative_x:.4f}")
    print(f"  relative_y: {env.relative_y:.4f}")
    print(f"  当前坐标: ({env.data_handler.current_lat:.10f}, {env.data_handler.current_lng:.10f})")
    
    # 检查step 1的元数据
    step1_metadata_path = os.path.join(img_dir, "step_1_metadata.json")
    if os.path.exists(step1_metadata_path):
        with open(step1_metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"\n📄 step_1_metadata.json:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
    
    # 执行转向操作
    print(f"\n🔄 执行 turn right")
    obs, reward, terminated, truncated, info = env.step("turn right")
    
    print(f"\n📋 转向后状态:")
    print(f"  relative_x: {env.relative_x:.4f}")
    print(f"  relative_y: {env.relative_y:.4f}")
    print(f"  朝向: {env.data_handler.heading:.1f}°")
    
    # 检查step 2的元数据
    step2_metadata_path = os.path.join(img_dir, "step_2_metadata.json")
    if os.path.exists(step2_metadata_path):
        with open(step2_metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"\n📄 step_2_metadata.json:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
    
    # 执行第二次移动
    print(f"\n🔄 执行 move forward")
    obs, reward, terminated, truncated, info = env.step("move forward")
    
    print(f"\n📋 第二次移动后状态:")
    print(f"  relative_x: {env.relative_x:.4f}")
    print(f"  relative_y: {env.relative_y:.4f}")
    print(f"  当前坐标: ({env.data_handler.current_lat:.10f}, {env.data_handler.current_lng:.10f})")
    
    # 检查step 3的元数据
    step3_metadata_path = os.path.join(img_dir, "step_3_metadata.json")
    if os.path.exists(step3_metadata_path):
        with open(step3_metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"\n📄 step_3_metadata.json:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
    
    # 验证相对位置计算是否正确
    from utils.geometry import calculate_relative_offset
    expected_x, expected_y = calculate_relative_offset(
        initial_lat, initial_lng,
        env.data_handler.current_lat, env.data_handler.current_lng
    )
    
    print(f"\n✅ 验证结果:")
    print(f"  expected_relative_x: {expected_x:.4f}")
    print(f"  actual_relative_x: {env.relative_x:.4f}")
    print(f"  expected_relative_y: {expected_y:.4f}")
    print(f"  actual_relative_y: {env.relative_y:.4f}")
    
    if abs(expected_x - env.relative_x) < 0.01 and abs(expected_y - env.relative_y) < 0.01:
        print(f"\n🎉 测试通过！相对位置计算正确！")
    else:
        print(f"\n❌ 测试失败！相对位置计算有误差！")
        sys.exit(1)
    
    env.close(release_sam3=False)
    print(f"\n📁 测试输出目录: {test_dir}")


if __name__ == "__main__":
    test_relative_position()
