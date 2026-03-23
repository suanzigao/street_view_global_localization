import os
import traceback
import warnings

# ====================== 全局代理配置 ======================
PROXY_HOST = "127.0.0.1"
PROXY_PORT = 7897
os.environ["HTTP_PROXY"] = f"http://{PROXY_HOST}:{PROXY_PORT}"
os.environ["HTTPS_PROXY"] = f"http://{PROXY_HOST}:{PROXY_PORT}"
os.environ["NO_PROXY"] = "localhost,127.0.0.1,localdomain,.localdomain"

from config import BATCH_ROOT_DIR, GLOBAL_MAX_STEPS, GLOBAL_MOVE_STEP, GPS_CSV_PATH
from utils import init_batch_dir, init_point_dir, read_gps_targets_from_csv
from env import StreetViewAutoLocEnv
from visualization import generate_visualization

# ====================== 屏蔽警告 ==============================
warnings.filterwarnings("ignore", category=UserWarning, message="Gym has been unmaintained")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ====================== 读取GPS CSV ======================
BATCH_GPS_TARGETS = read_gps_targets_from_csv(GPS_CSV_PATH)

# ====================== 手动控制（增强反馈版） ======================
def manual_control(env):
    # 新增FOV/Pitch的指令提示
    print("\n=== 手动操控模式 ===")
    print("移动：W=前 A=左移 D=右移 S=后")
    print("转向：Q=左转 E=右转")
    print("视角：Z=放大 X=缩小")  # 新增FOV指令
    print("俯仰：R=抬头 F=低头")  # 新增Pitch指令
    print("退出：exit")
    print(f"提示：目标步长 {env.data_handler.move_step} 米，仅移动动作会修正GPS坐标\n")

    # 初始化时显示初始图像
    if env.current_image is not None:
        env.render()

    while True:
        # 显示当前状态（仅显示实际GPS，非移动动作不修改）
        current_lat = env.data_handler.current_lat
        current_lng = env.data_handler.current_lng
        current_heading = env.data_handler.heading
        current_fov = env.data_handler.fov if hasattr(env.data_handler, 'fov') else 30
        current_pitch = env.data_handler.pitch
        print(f"📌 当前GPS坐标: {current_lat:.6f}, {current_lng:.6f}")
        print(f"🧭 朝向: {current_heading:.1f}° | 视角FOV: {current_fov}° | 俯仰Pitch: {current_pitch}°")

        # 获取用户指令
        cmd = input("> 请输入操作指令: ").strip().lower()
        if cmd == "exit":
            print("👋 退出手动操控模式")
            # 打印坐标修正历史（仅移动动作有记录）
            if env.coord_correction_history:
                print("\n📊 坐标修正历史（仅移动动作）：")
                for corr in env.coord_correction_history:
                    print(
                        f"Step {corr['step']}: 理论({corr['theoretical_lat']:.6f},{corr['theoretical_lng']:.6f}) → 实际({corr['actual_lat']:.6f},{corr['actual_lng']:.6f}) | 偏差 {corr['distance_diff']:.2f} 米"
                    )
            
            # 生成可视化HTML页面
            generate_visualization(env, point_dir)
            break

        # 扩展指令映射（新增z/x/r/f）
        cmd_map = {
            "w": 0, "a": 1, "d": 2, "s": 3,
            "q": 4, "e": 5,
            "z": 6, "x": 7,  # FOV缩放
            "r": 8, "f": 9  # Pitch俯仰
        }
        if cmd not in cmd_map:
            print("❌ 无效指令！请输入 W/A/S/D/Q/E/Z/X/R/F 或 exit\n")
            continue

        # 执行操作前记录坐标（用于对比）
        pre_lat = current_lat
        pre_lng = current_lng

        # 执行动作并获取结果
        action = cmd_map[cmd]
        obs, reward, terminated, truncated, info = env.step(action)

        # 解析执行结果并反馈
        action_type = info["action_type"]
        relative_dir = info["relative_dir"]
        msg = info["msg"]

        print("\n--- 执行结果 ---")
        if action_type == "move":
            actual_distance = info.get("actual_distance", 0.0)
            if reward == 0.0:
                print(
                    f"✅ 移动成功！方向：{relative_dir} | 目标步长：{env.data_handler.move_step}米 | 实际移动：{actual_distance:.2f} 米"
                )
                # 显示理论坐标和修正后的实际坐标
                if env.current_theoretical_lat and env.current_actual_lat:
                    print(f"📏 坐标修正过程：")
                    print(f"   理论计算坐标：{env.current_theoretical_lat:.6f}, {env.current_theoretical_lng:.6f}")
                    print(f"   实际采样坐标：{env.current_actual_lat:.6f}, {env.current_actual_lng:.6f}")
                    print(f"   坐标偏差：{env.current_correction_diff:.2f} 米")
            else:
                print(f"❌ 移动失败！原因：{msg} | 实际移动：{actual_distance:.2f} 米")
        else:
            # 转向/FOV/Pitch：仅反馈动作结果，无坐标变化
            print(f"✅ {msg}")

        # 显示执行后的最新状态
        print(f"📌 操作后GPS坐标: {env.data_handler.current_lat:.6f}, {env.data_handler.current_lng:.6f}")
        print(
            f"🧭 操作后朝向: {env.data_handler.heading:.1f}° | 视角FOV: {env.data_handler.fov}° | 俯仰Pitch: {env.data_handler.pitch}°\n"
        )

        # 渲染最新街景
        env.render()


# ====================== 批处理运行（保留，无核心修改） ======================
def batch_run():
    init_batch_dir(BATCH_ROOT_DIR)
    print(f"✅ 开始批量运行，共 {len(BATCH_GPS_TARGETS)} 个点位")

    for idx, point_info in enumerate(BATCH_GPS_TARGETS, 1):
        print(f"\n====== 第 {idx}/{len(BATCH_GPS_TARGETS)} 个点: {point_info['name']} ======")
        try:
            point_dir, img_dir = init_point_dir(point_info, BATCH_ROOT_DIR)
            env = StreetViewAutoLocEnv(
                initial_lat=point_info["lat"],
                initial_lng=point_info["lng"],
                move_step=point_info.get("move_step", GLOBAL_MOVE_STEP),
                max_steps=point_info.get("max_steps", GLOBAL_MAX_STEPS),
                point_dir=point_dir,
                img_save_dir=img_dir
            )
            env.reset()

            # 自动执行直到最大步数
            for step in range(env.max_steps):
                if env.current_image is None:
                    print("❌ 无图像，跳过该点")
                    break

                # GPT 决策
                action_str = env.gpt4o_decision(env.current_image, env.current_step + 1)
                action = env.parse_gpt_decision(action_str)
                obs, reward, terminated, truncated, info = env.step(action)

                # 批量运行时也显示修正过程
                action_type = info["action_type"]
                if action_type == "move" and "theoretical_lat" in info:
                    print(f"Step {env.current_step} | Action: {action} | {info['msg']}")
                    print(
                        f"   理论坐标：{info['theoretical_lat']:.6f}, {info['theoretical_lng']:.6f} → 实际坐标：{info['actual_lat']:.6f}, {info['actual_lng']:.6f} (偏差{info['coord_correction_diff_m']:.2f}米) | reward: {reward}"
                    )
                else:
                    print(f"Step {env.current_step} | Action: {action} | {info['msg']} | reward: {reward}")

                if truncated:
                    break

            # 保存最终预测结果
            result_path = os.path.join(point_dir, "final_prediction.txt")
            with open(result_path, "w", encoding="utf-8") as f:
                f.write(env.final_prediction)
            print(f"✅ 预测完成，结果已保存: {result_path}")

            # 保存坐标修正历史（仅移动动作）
            if env.coord_correction_history:
                import csv
                corr_path = os.path.join(point_dir, "coord_correction_history.csv")
                with open(corr_path, "w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=env.coord_correction_history[0].keys())
                    writer.writeheader()
                    writer.writerows(env.coord_correction_history)
                print(f"✅ 坐标修正历史已保存: {corr_path}")

            # 生成可视化HTML页面
            generate_visualization(env, point_dir)

        except Exception as e:
            print(f"❌ 该点运行失败: {e}")
            traceback.print_exc()
            continue


# ====================== 运行 ======================
if __name__ == "__main__":
    # 1. 手动控制（核心测试）
    test_point = {"name": "ManualTest", "lat": 47.53097481458263, "lng": 19.06958490497365}
    point_dir, img_dir = init_point_dir(test_point, BATCH_ROOT_DIR)
    env = StreetViewAutoLocEnv(
        initial_lat=test_point["lat"],
        initial_lng=test_point["lng"],
        point_dir=point_dir,
        img_save_dir=img_dir
    )
    env.reset()
    manual_control(env)

    # 2. 批处理自动运行（如需启用，注释上面的手动控制代码）
    # batch_run()
