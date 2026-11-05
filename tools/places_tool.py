"""
Google Places API 工具模块 - 为 Agent 提供地点搜索能力

功能：
1. 通过 text query 搜索地点，获取 GPS 坐标和地址
2. 辅助大模型在探索街景时识别看到的店铺/建筑名称
"""
import os, json, requests
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))
API_KEY = os.getenv("GOOGLE_API_KEY", "")


class PlacesAPITool:
    """Google Places API 文本搜索工具"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or API_KEY

    def find_place(self, query: str) -> dict:
        """
        使用 Find Place From Text API 搜索地点

        Args:
            query: 搜索文本，如 "Nykredit"、"Starbucks Copenhagen"

        Returns:
            dict: {
                'success': bool,
                'candidates': list of {
                    'name': str,
                    'address': str,
                    'lat': float,
                    'lng': float,
                    'place_id': str,
                    'rating': float,
                    'types': list
                },
                'error': str or None
            }
        """
        if not self.api_key:
            return {"success": False, "error": "No API key configured", "candidates": []}

        try:
            url = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"
            params = {
                "input": query,
                "inputtype": "textquery",
                "fields": "place_id,name,formatted_address,geometry,types,rating",
                "key": self.api_key,
            }
            r = requests.get(url, params=params, timeout=10)
            data = r.json()

            if data.get("status") != "OK":
                return {"success": False, "error": data.get("status", "Unknown error"), "candidates": []}

            candidates = []
            for c in data.get("candidates", []):
                loc = c.get("geometry", {}).get("location", {})
                candidates.append({
                    "name": c.get("name", ""),
                    "address": c.get("formatted_address", ""),
                    "lat": loc.get("lat"),
                    "lng": loc.get("lng"),
                    "place_id": c.get("place_id", ""),
                    "rating": c.get("rating"),
                    "types": c.get("types", []),
                })

            return {"success": True, "candidates": candidates, "error": None}

        except Exception as e:
            return {"success": False, "error": str(e), "candidates": []}

    def find_place_top_k(self, query: str, k: int = 20) -> list:
        """
        使用新版 Places Text Search API (places:searchText) 搜索地点，返回最多 k 个候选。

        Args:
            query: 搜索文本，如 "Starbucks, Zurich"
            k: 期望返回的候选数（上限 20）

        Returns:
            list of {
                'place_id': str,
                'name': str,
                'address': str,
                'lat': float,
                'lng': float,
                'rating': float,
                'types': list
            }
        """
        if not self.api_key:
            return []

        try:
            url = "https://places.googleapis.com/v1/places:searchText"
            headers = {
                "Content-Type": "application/json",
                "X-Goog-Api-Key": self.api_key,
                "X-Goog-FieldMask": "places.id,places.displayName,places.formattedAddress,places.location,places.rating,places.types",
            }
            body = {
                "textQuery": query,
                "languageCode": "en",
                "pageSize": min(max(int(k), 1), 20),
            }
            r = requests.post(url, headers=headers, json=body, timeout=15)
            data = r.json()

            candidates = []
            for p in data.get("places", []):
                loc = p.get("location", {})
                candidates.append({
                    "place_id": p.get("id", ""),
                    "name": (p.get("displayName") or {}).get("text", ""),
                    "address": p.get("formattedAddress", ""),
                    "lat": loc.get("latitude"),
                    "lng": loc.get("longitude"),
                    "rating": p.get("rating"),
                    "types": p.get("types", []),
                })
            return candidates

        except Exception as e:
            print(f"      ⚠️ 地点搜索失败: {e}")
            return []


def tool_call_search(query: str) -> dict:
    """工具调用接口"""
    tool = PlacesAPITool()
    return tool.find_place(query)


TOOL_DESCRIPTION = {
    "name": "places_search",
    "description": "使用 Google Places API 文本搜索地点。当在街景中看到商店名、品牌、建筑名称等文本时，可以通过此工具查询该地点的 GPS 坐标。",
    "parameters": {
        "query": {
            "type": "string",
            "description": "要搜索的文本，如 'Nykredit'、'Harbour House Denmark'"
        }
    },
    "returns": {
        "type": "object",
        "description": "包含候选地点列表，每项含 name、address、lat、lng、rating、types"
    }
}
