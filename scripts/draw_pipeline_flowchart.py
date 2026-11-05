"""绘制三段式级联定位流程图（从左到右排列）。

Stage 1 采用 ReAct Agent 样式（Thought -> Action -> Observation 循环），
Stage 2 / Stage 3 仅概述大致流程。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

for fp in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"]:
    try:
        fm.fontManager.addfont(fp)
    except Exception:
        pass

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Droid Sans Fallback"]
plt.rcParams["axes.unicode_minus"] = False

# 阶段配色
C_S1 = "#D6E4F0"
C_S2 = "#D9EAD3"
C_S3 = "#FCE5CD"
C_OUT = "#FFF2CC"
A_S1 = "#1a3c5e"
A_S2 = "#2d5a27"
A_S3 = "#8a5a1a"

fig, ax = plt.subplots(figsize=(27, 17))
ax.set_xlim(0, 310)
ax.set_ylim(0, 210)
ax.axis("off")


def box(x, y, w, h, fc, ec="#666666", lw=1.5, r=1.5, ls="-"):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                       fc=fc, ec=ec, lw=lw, linestyle=ls)
    ax.add_patch(p)
    return p


def text(x, y, s, size=10, weight="normal", ha="center", va="center", color="#222222"):
    ax.text(x, y, s, size=size, weight=weight, ha=ha, va=va, color=color)


def arrow(x1, y1, x2, y2, color="#444444", lw=1.8, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=16, color=color, lw=lw))


def column_header(cx_c, top, num, name, sub, accent):
    text(cx_c, top - 8, num, size=18, weight="bold", color=accent)
    text(cx_c, top - 18, name, size=16, weight="bold")
    text(cx_c, top - 25, sub, size=13, color="#555555")


def sub_box(cx, cx_c, y, main, note, dashed=False):
    if dashed:
        box(cx + 5, y, 76, 14, "white", ec=A_S3, lw=1.2, r=0.8, ls=(0, (3, 2)))
        text(cx_c, y + 9, main, size=14)
        text(cx_c, y + 4.5, note, size=11, color=A_S3)
    else:
        box(cx + 5, y, 76, 14, "white", ec="#999999", lw=1.0, r=0.8)
        text(cx_c, y + 9, main, size=14)
        text(cx_c, y + 4.5, note, size=11, color="#666666")


def out_box(cx, cx_c, y, label):
    box(cx + 5, y, 76, 14, C_OUT, ec="#aa8833", lw=1.2, r=0.8)
    text(cx_c, y + 7, label, size=14, weight="bold", color="#8a6d1a")


# ===== 标题 =====
text(155, 202, "三段式级联定位流程", size=26, weight="bold")

CY_TOP = 195
CY_BOTTOM = 66
CX1, CX2, CX3 = 4, 112, 220
CC1, CC2, CC3 = 47, 155, 263

# ============================================================
# Stage 1 —— ReAct Agent
# ============================================================
box(CX1, CY_BOTTOM, 86, CY_TOP - CY_BOTTOM, C_S1, ec="#999999", lw=1.5)
column_header(CC1, CY_TOP, "Stage 1", "ReAct Agent 自主导航",
              "城市级 · 输出 城市名 + 城市经纬度", A_S1)

sub_box(CX1, CC1, 158, "输入", "geochain CSV (name, lat, lng)")

text(CC1, 146, "ReAct 循环 · 最多 25 步", size=14, weight="bold", color=A_S1)

# 三个 ReAct 节点
def react_node(y, title, note):
    box(CX1 + 12, y, 62, 14, "white", ec="#999999", lw=1.0, r=0.8)
    text(CC1, y + 9, title, size=14, weight="bold")
    text(CC1, y + 4.5, note, size=11, color="#666666")

react_node(130, "Thought", "GPT-4o 推理下一步")
arrow(CC1, 130, CC1, 118.5, lw=1.2)
react_node(115, "Action", "turn / move / zoom / search")
arrow(CC1, 115, CC1, 103.5, lw=1.2)
react_node(100, "Observation", "新视角 / 分割 / 搜索结果")

# 回环箭头（Observation -> Thought）
ax.plot([78, 88, 88], [107, 107, 137], color=A_S1, lw=1.6)
arrow(88, 137, 78.5, 137, color=A_S1, lw=1.6)

out_box(CX1, CC1, 74, "输出: 城市名 + 城市经纬度")

# ============================================================
# Stage 2
# ============================================================
box(CX2, CY_BOTTOM, 86, CY_TOP - CY_BOTTOM, C_S2, ec="#999999", lw=1.5)
column_header(CC2, CY_TOP, "Stage 2", "招牌语义定位",
              "街区级 · 输出 预测经纬度", A_S2)

sub_box(CX2, CC2, 158, "招牌识别与空间锚定", "VGGT 深度 · SAM3 · OCR → 招牌 (x,y)")
text(CC2, 149, "↓", size=16, weight="bold", color="#444444")
sub_box(CX2, CC2, 130, "招牌文字空间投票", "Places top-20 + DBSCAN → 投票坐标")
text(CC2, 121, "↓", size=16, weight="bold", color="#444444")
sub_box(CX2, CC2, 102, "高斯核招牌配准", "BEV(x,y) ↔ GPS(lat,lng) ±300m")
out_box(CX2, CC2, 74, "输出: 预测经纬度")

# ============================================================
# Stage 3
# ============================================================
box(CX3, CY_BOTTOM, 86, CY_TOP - CY_BOTTOM, C_S3, ec="#999999", lw=1.5)
column_header(CC3, CY_TOP, "Stage 3", "BEV ↔ OSM 路网匹配",
              "细粒度 · 输出 坐标 + 置信度", A_S3)

sub_box(CX3, CC3, 158, "Stage 1.5 BEV 投影（桥接）", "路面分割 → 反投影 → BEV", dashed=True)
text(CC3, 149, "↓", size=16, weight="bold", color="#444444")
sub_box(CX3, CC3, 130, "下载并渲染 OSM 路网", "半径 1500m → 二值图")
text(CC3, 121, "↓", size=16, weight="bold", color="#444444")
sub_box(CX3, CC3, 102, "BEV ↔ OSM 卷积匹配", "matchTemplate 模板滑动")
out_box(CX3, CC3, 74, "输出: 坐标 + 置信度")

# ===== 段间箭头（从左到右） =====
arrow(90, 112, 112, 112, color="#444444", lw=2.0)
text(101, 118, "城市名 + 城市经纬度", size=13, color="#444444")
arrow(198, 112, 220, 112, color="#444444", lw=2.0)
text(209, 118, "预测经纬度", size=13, color="#444444")

plt.tight_layout()
out = "/home/suanzigao/PycharmProjects/street_view_global_localization/pipeline_flowchart.png"
plt.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
print("已生成:", out)
