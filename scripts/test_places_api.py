"""测试 Google Places API 文本搜索"""
import os, json, requests
from dotenv import load_dotenv

load_dotenv("/home/suanzigao/PycharmProjects/street_view_global_localization/.env")
API_KEY = os.getenv("GOOGLE_API_KEY")

def test_findplacefromtext(query: str):
    """旧版: Find Place From Text (最可能直接用)"""
    url = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"
    params = {
        "input": query,
        "inputtype": "textquery",
        "fields": "place_id,name,formatted_address,geometry,types,rating",
        "key": API_KEY,
    }
    r = requests.get(url, params=params, timeout=10)
    print(f"\n=== findplacefromtext: '{query}' ===")
    print(f"Status: {r.status_code}")
    data = r.json()
    print(json.dumps(data, indent=2, ensure_ascii=False))
    return data

def test_textsearch_new(query: str):
    """新版: places:searchText (POST, 新版 key 格式)"""
    url = "https://places.googleapis.com/v1/places:searchText"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": API_KEY,
        "X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.location,places.types,places.rating",
    }
    body = {"textQuery": query, "languageCode": "en"}
    r = requests.post(url, headers=headers, json=body, timeout=10)
    print(f"\n=== places:searchText (new): '{query}' ===")
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        print(json.dumps(r.json(), indent=2, ensure_ascii=False))
    else:
        print(r.text[:500])
    return r

def test_nearbysearch(lat, lng, radius=1000, keyword=None):
    """旧版: Nearby Search (按坐标周边搜)"""
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "location": f"{lat},{lng}",
        "radius": radius,
        "key": API_KEY,
    }
    if keyword:
        params["keyword"] = keyword
    r = requests.get(url, params=params, timeout=10)
    print(f"\n=== nearbysearch: lat={lat}, lng={lng}, keyword={keyword} ===")
    print(f"Status: {r.status_code}")
    data = r.json()
    results = data.get("results", [])
    print(f"Found {len(results)} places")
    for place in results[:3]:
        print(f"  - {place.get('name')}: {place.get('vicinity')} (rating: {place.get('rating', 'N/A')})")
    return data


if __name__ == "__main__":
    # 1. 测试旧版 findplacefromtext（最可能支持）
    test_findplacefromtext("Harbour House,Denmark")

    # 2. 测试旧版 findplacefromtext 另一个查询
    test_findplacefromtext("Nykredit Copenhagen")

    # 3. 测试新版 places:searchText
    test_textsearch_new("Nykredit")

    # 4. 用已知坐标测试 nearby search（验证 key 是否支持 Places API）
    test_nearbysearch(55.7132, 12.5929, radius=500, keyword="Nykredit")

    print("\n✅ Done!")
