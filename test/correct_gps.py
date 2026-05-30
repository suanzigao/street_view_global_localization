import pandas as pd
import requests
import time

# ---------------- 核心配置 ----------------
API_KEY = "AIzaSyAXmbPK8IXtGV9fjWVI8K4b5cw9Tshrvy0"
INPUT_CSV = "/home/suanzigao/PycharmProjects/street_view_global_localization/geochain_10000_formatted.csv"
OUTPUT_CSV = "/home/suanzigao/PycharmProjects/street_view_global_localization/geochain_10000_corrected.csv"
SEARCH_RADIUS = 50

def get_nearest_streetview_coords(lat, lng):
    api_url = "https://maps.googleapis.com/maps/api/streetview/metadata"
    params = {
        "location": f"{lat},{lng}",
        "radius": SEARCH_RADIUS,
        "key": API_KEY,
        "source": "outdoor"
    }
    try:
        response = requests.get(api_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get("status") == "OK":
            sv_lat = data["location"]["lat"]
            sv_lng = data["location"]["lng"]
            sv_pano_id = data["pano_id"]
            return (True, sv_lat, sv_lng, sv_pano_id)
        else:
            return (False, None, None, f"API返回状态: {data.get('status')}")
    except requests.exceptions.HTTPError as e:
        return (False, None, None, f"HTTP错误: {e.response.status_code}")
    except requests.exceptions.Timeout:
        return (False, None, None, "请求超时")
    except Exception as e:
        return (False, None, None, f"未知错误: {str(e)}")

def batch_correct_coords():
    try:
        df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
        print(f"✅ 成功读取CSV，共{len(df)}行数据")
    except Exception as e:
        print(f"❌ 读取CSV失败: {e}")
        return

    required_cols = ["name", "lat", "lng", "max_steps", "move_step"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ CSV缺少核心列：{', '.join(missing_cols)}")
        return

    corrected_rows = []
    total_rows = len(df)
    success_count = 0
    fail_count = 0

    for idx, row in df.iterrows():
        try:
            original_lat = row["lat"]
            original_lng = row["lng"]

            if pd.isna(original_lat) or pd.isna(original_lng):
                print(f"⚠️  第{idx + 1}/{total_rows}行：{row['name']} - 经纬度为空，跳过")
                fail_count += 1
                continue

            original_lat = float(original_lat)
            original_lng = float(original_lng)

            success, sv_lat, sv_lng, sv_pano_id = get_nearest_streetview_coords(original_lat, original_lng)

            if success:
                new_row = row.copy()
                new_row["lat"] = sv_lat
                new_row["lng"] = sv_lng
                corrected_rows.append(new_row)
                print(f"✅ 第{idx + 1}/{total_rows}行：{row['name']} 校正成功")
                success_count += 1
            else:
                print(f"❌ 第{idx + 1}/{total_rows}行：{row['name']} - {sv_pano_id}，已删除")
                fail_count += 1

            time.sleep(0.5)

        except ValueError:
            print(f"⚠️  第{idx + 1}/{total_rows}行：{row['name']} - 经纬度非数字，跳过")
            fail_count += 1
        except Exception as e:
            print(f"⚠️  第{idx + 1}/{total_rows}行：{row['name']} - {e}，跳过")
            fail_count += 1

    corrected_df = pd.DataFrame(corrected_rows)

    try:
        corrected_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
        print(f"\n🎉 批量处理完成！结果已保存至：{OUTPUT_CSV}")
        print(f"📊 处理统计：成功{success_count}行，失败{fail_count}行")
        print(f"📁 输出文件包含{len(corrected_df)}行数据")
    except Exception as e:
        print(f"❌ 保存CSV失败: {e}")

if __name__ == "__main__":
    batch_correct_coords()