import pandas as pd
import requests
import time
import os
# ====================== 全局代理配置 ======================
PROXY_HOST = "127.0.0.1"
PROXY_PORT = 7897
os.environ["HTTP_PROXY"] = f"http://{PROXY_HOST}:{PROXY_PORT}"
os.environ["HTTPS_PROXY"] = f"http://{PROXY_HOST}:{PROXY_PORT}"
os.environ["NO_PROXY"] = "localhost,127.0.0.1,localdomain,.localdomain"

# ---------------- 核心配置（仅需替换API Key） ----------------
API_KEY = "AIzaSyAXmbPK8IXtGV9fjWVI8K4b5cw9Tshrvy0"  # 替换为你的有效API Key
INPUT_CSV = "geochain_mini.csv"  # 你的原始数据文件路径
OUTPUT_CSV = "geochain_mini1.csv"  # 校正后输出文件
SEARCH_RADIUS = 50  # 搜索半径（米），城市区域建议50-100米


# ---------------- 核心函数：获取最近街景采样点坐标 ----------------
def get_nearest_streetview_coords(lat, lng):
    """
    调用Google街景元数据API，获取最近街景采样点的经纬度
    参数：lat(原始纬度), lng(原始经度)
    返回：(是否成功, 街景纬度, 街景经度, 街景ID/错误信息)
    """
    # Metadata API请求地址（无需Session ID）
    api_url = "https://maps.googleapis.com/maps/api/streetview/metadata"

    # 请求参数（核心：location=经纬度，radius=搜索半径，key=API密钥）
    params = {
        "location": f"{lat},{lng}",
        "radius": SEARCH_RADIUS,
        "key": API_KEY,
        "source": "outdoor"  # 仅搜索户外街景（过滤室内/私人区域）
    }

    try:
        # 发送GET请求（无需POST，更简单）
        response = requests.get(api_url, params=params, timeout=10)
        response.raise_for_status()  # 捕获HTTP错误（如403/429）
        data = response.json()

        # 解析结果：status为OK表示找到街景
        if data.get("status") == "OK":
            # 提取街景采样点的精准经纬度和Panoid（唯一标识）
            sv_lat = data["location"]["lat"]
            sv_lng = data["location"]["lng"]
            sv_pano_id = data["pano_id"]
            return (True, sv_lat, sv_lng, sv_pano_id)
        else:
            # 常见错误：ZERO_RESULTS（无街景）、OVER_QUERY_LIMIT（API超限）
            return (False, None, None, f"API返回状态: {data.get('status')}")

    except requests.exceptions.HTTPError as e:
        return (False, None, None, f"HTTP错误: {e.response.status_code}")
    except requests.exceptions.Timeout:
        return (False, None, None, "请求超时")
    except Exception as e:
        return (False, None, None, f"未知错误: {str(e)}")


# ---------------- 批量处理主函数（适配新列名） ----------------
def batch_correct_coords():
    # 1. 读取原始CSV数据
    try:
        df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
        print(f"✅ 成功读取CSV，共{len(df)}行数据")
    except Exception as e:
        print(f"❌ 读取CSV失败: {e}")
        return

    # 2. 检查必要列（适配你的列名：name, lat, lng, max_steps, move_step）
    required_cols = ["name", "lat", "lng", "max_steps", "move_step"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ CSV缺少核心列：{', '.join(missing_cols)}")
        return

    # 3. 逐行处理（直接修改原始lat、lng列）
    total_rows = len(df)
    success_count = 0
    fail_count = 0
    
    for idx, row in df.iterrows():
        try:
            # 直接读取独立的lat、lng列（无需解析拼接字段）
            original_lat = row["lat"]
            original_lng = row["lng"]

            # 跳过空值
            if pd.isna(original_lat) or pd.isna(original_lng):
                print(f"⚠️  第{idx + 1}/{total_rows}行：{row['name']} - 经纬度为空")
                fail_count += 1
                continue

            # 转换为浮点数（防止字符串格式）
            original_lat = float(original_lat)
            original_lng = float(original_lng)

            # 调用API获取最近街景坐标
            success, sv_lat, sv_lng, sv_pano_id = get_nearest_streetview_coords(original_lat, original_lng)

            if success:
                # 校正成功，直接覆盖原始经纬度
                df.at[idx, "lat"] = sv_lat
                df.at[idx, "lng"] = sv_lng
                print(f"✅ 第{idx + 1}/{total_rows}行：{row['name']} → {original_lat},{original_lng} → {sv_lat},{sv_lng}")
                success_count += 1
            else:
                # 校正失败，仅打印信息
                print(f"❌ 第{idx + 1}/{total_rows}行：{row['name']} - {sv_pano_id}")
                fail_count += 1

            # 限流：每请求间隔0.5秒，避免触发API频率限制
            time.sleep(0.5)

        except ValueError:
            print(f"⚠️  第{idx + 1}/{total_rows}行：{row['name']} - 经纬度非数字")
            fail_count += 1
        except Exception as e:
            print(f"⚠️  第{idx + 1}/{total_rows}行：{row['name']} - {e}")
            fail_count += 1

    # 5. 保存校正后的数据（仅保留原始列）
    try:
        df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
        print(f"\n🎉 批量处理完成！结果已保存至：{OUTPUT_CSV}")
        # 输出统计信息
        print(f"📊 处理统计：成功{success_count}行，失败{fail_count}行")
    except Exception as e:
        print(f"❌ 保存CSV失败: {e}")


# ---------------- 执行主程序 ----------------
if __name__ == "__main__":
    batch_correct_coords()