# 导入核心依赖
from datasets import load_dataset
import csv
import os
from tqdm import tqdm
import requests

# ===================== 代理配置（必改：替换为你的实际代理地址+端口）=====================
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'   # 替换成自己的代理
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'  # 和HTTP代理一致
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'
requests.packages.urllib3.disable_warnings()
# ================================================================================

def main():
    # 固定配置（无需修改）
    CSV_SAVE_PATH = "geochain_mini_test.csv"  # 生成的CSV文件名（区分mini_test）
    DATASET_NAME = "sahitiy51/geochain"       # 数据集完整标识
    TARGET_SPLIT = "mini_test"                # 仅加载mini_test拆分
    CSV_HEADERS = ["name", "lat", "lng", "max_steps", "move_step"]  # 要求的列名

    try:
        print(f"开始加载的{TARGET_SPLIT}拆分...")
        # 极简兼容加载：直接指定拆分，无多余参数
        dataset = load_dataset("sahitiy51/geochain", split="mini_test")
        print(f"✅ 加载成功！{TARGET_SPLIT}拆分共{len(dataset)}条样本")

        # 确保CSV保存目录存在
        save_dir = os.path.dirname(CSV_SAVE_PATH)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        # 写入CSV文件
        print(f"\n开始生成CSV文件：{os.path.abspath(CSV_SAVE_PATH)}")
        with open(CSV_SAVE_PATH, "w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_HEADERS)
            writer.writeheader()  # 写入表头

            # 逐行写入，显示进度
            for sample in tqdm(dataset, desc=f"{TARGET_SPLIT}写入CSV", unit="条"):
                writer.writerow({
                    "name": sample["key"],    # key → name
                    "lat": sample["lat"],     # 原lat字段
                    "lng": sample["lon"],     # 原lon → lng
                    "max_steps": "",          # 置空
                    "move_step": ""           # 置空
                })

        # 成功提示
        print(f"\n🎉 操作完成！")
        print(f"📂 CSV文件路径：{os.path.abspath(CSV_SAVE_PATH)}")
        print(f"📋 列名：{CSV_HEADERS} | 数据量：1行表头 + {len(dataset)}行数据")

    # 字段缺失错误
    except KeyError as e:
        print(f"\n❌ 错误：数据集无【{e}】字段！")
    # 其他错误（网络/代理/拆分不存在）
    except Exception as e:
        print(f"\n❌ 执行失败：{str(e)}")
        print("\n📌 排查建议：")
        print(f"1. 确认代理地址正确且服务开启")
        print(f"2. 确认数据集有{TARGET_SPLIT}拆分：https://huggingface.co/datasets/sahitiy51/geochain")
        print(f"3. 激活虚拟环境升级依赖：source /home/suanzigao/PycharmProjects/PythonProject/.venv/bin/activate && pip install --upgrade datasets tqdm requests")

if __name__ == "__main__":
    print("📦 虚拟环境依赖升级命令（复制执行）：")
    print("source /home/suanzigao/PycharmProjects/PythonProject/.venv/bin/activate && pip install --upgrade datasets tqdm requests\n")
    main()