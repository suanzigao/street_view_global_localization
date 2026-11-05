# 街景全局定位（Street View Global Localization）

基于三阶段级联的街景定位系统：Agent 自主导航 + 多模态推理 → 招牌空间投票 + 高斯核配准 → BEV↔OSM 路网卷积匹配。

## 项目结构

```
street_view_global_localization/
├── core/                     # 核心模块（导航环境、街景数据处理）
├── localization/             # 三阶段级联定位（正式模块）
│   ├── three_stage_localization.py   # 三阶段编排入口
│   ├── vggt_depth_to_bev.py          # Stage 2：VGGT 深度 → BEV
│   └── gaussian_corr_localization.py # Stage 2：高斯核配准
├── prompts/                  # 提示词模块
├── tools/                    # SAM3 分割、Places 搜索、VGGT 客户端
├── utils/                    # 工具模块（几何计算、文件操作等）
├── scripts/                  # 临时/实验脚本（暂存区）
├── test/                     # 测试脚本
├── models/                   # 第三方模型（SAM3 等）
└── main.py                   # 主入口（批量运行、手动控制）
```

## 启动前准备

### 1. 克隆仓库并初始化子模块

`models/sam3` 是一个独立的 git 仓库（submodule），必须单独初始化：

```bash
git clone https://github.com/suanzigao/street_view_global_localization.git
cd street_view_global_localization

# 初始化 SAM3 子模块（若 .gitmodules 缺失则手动 clone）
git submodule update --init --recursive
# 若上述命令失败，手动 clone 到 models/sam3
# git clone <sam3-repo-url> models/sam3
```

### 2. 创建 Python 3.12 虚拟环境

```bash
cd street_view_global_localization
python3.12 -m venv .venv
source .venv/bin/activate
```

本项目统一使用 `.venv/bin/python`（Python 3.12）运行脚本。

### 3. 安装依赖

项目当前未提供 `requirements.txt`，需手动安装核心依赖：

```bash
.venv/bin/pip install torch torchvision numpy pillow openai gymnasium requests python-dotenv opencv-python huggingface_hub
```

依赖说明：

| 包 | 用途 |
|----|------|
| `torch` / `torchvision` | SAM3 模型推理 |
| `numpy` / `opencv-python` | BEV 投影、图像处理 |
| `openai` | 调用 GPT-4o 决策（导航 + 最终预测） |
| `gymnasium` | 强化学习环境封装 |
| `requests` | Google Street View / Places / VGGT API |
| `python-dotenv` | 加载 `.env` 环境变量 |
| `huggingface_hub` | SAM3 权重自动下载 |
| `pillow` | 图像处理 |

### 4. 配置 `.env` 环境变量

在项目根目录创建 `.env` 文件（此文件已被 `.gitignore` 忽略，不会提交）：

```env
GOOGLE_API_KEY=你的_google_api_key
OPENAI_API_KEY=你的_openai_api_key
```

- `GOOGLE_API_KEY`：Google Street View / Places API 密钥。
- `OPENAI_API_KEY`：GPT-4o 密钥（当前代码通过 `https://api.laozhang.ai/v1` 中转调用）。

### 5. 配置本地代理

多个脚本在文件头部硬编码了本地代理 `127.0.0.1:7897`（Clash 等）：

```python
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"
```

**需确保本地代理运行在 7897 端口**（用于访问 Google API）。若代理端口不同，需修改 `main.py`、`test/*.py`、`localization/*.py` 中的 `PROXY_PORT` / `HTTP_PROXY` 配置。

### 6. SAM3 权重下载

SAM3 模型权重会在首次运行时自动从 HuggingFace 下载（`models/sam3/sam3/model_builder.py` 中的 `download_ckpt_from_hf()`）。首次运行 `preload_sam3()` 时可能需要下载约数百 MB 权重，请保证网络畅通。

### 7. 启动 VGGT 服务（仅 Stage 2 / Stage 3 需要）

Stage 1（Agent 导航）不依赖 VGGT；但**完整三阶段流程需要 VGGT 服务**提供深度图：

- VGGT 服务地址：`http://127.0.0.1:18888`（`/vggt/infer` 与 `/health`）
- 通过 SSH 隧道连接到远程服务器：

```bash
ssh -CNg -L 18888:127.0.0.1:18888 -p 37859 root@connect.westb.seetacloud.com
```

验证服务可用：

```bash
curl http://127.0.0.1:18888/health
```

> 只跑 Stage 1 不需要 VGGT；跑完整三阶段必须确保 VGGT 服务可达，否则 Stage 2 无法生成 `sign_markers.json`。

### 8. Google Places API

`tools/places_tool.py` 使用 Google Places API 做地标搜索（`search:` 动作）。需在 `GOOGLE_API_KEY` 对应的 Google Cloud 项目中启用 **Places API**。

## 运行

### 单点三阶段完整流程

```bash
.venv/bin/python localization/three_stage_localization.py \
  --lat 55.6463583 \
  --lng 37.7249981 \
  --name E-fIB3K6E-Z-GSW_xI-rSw \
  --max-steps 20 \
  --move-step 15.0
```

### 仅 Stage 1（Agent 导航，无需 VGGT）

```bash
.venv/bin/python test/single_stage1.py 20
```

### 批量 Stage 1

```bash
# 前 N 个点，每个点 max_steps 步
.venv/bin/python test/batch_stage1.py 20 50
```

## 结果产物

每个点位目录下会生成：

| 文件 | 说明 |
|------|------|
| `stage1_coarse.json` | Stage 1：城市名 + 粗 GPS + 探索证据 |
| `stage2_spatial_vote.json` | Stage 2：招牌投票 + 高斯核配准结果 |
| `stage3_osm_match.json` | Stage 3：BEV↔OSM 卷积匹配结果 |
| `three_stage_result.json` | 三阶段最终汇总 |
| `visualization.html` | 导航轨迹 + 探索图可视化 |
| `streetview_images/` | 街景图 + BEV 图 |

## 说明

- 代码存放遵循项目约定：正式模块原位修改，新功能放 `scripts/`，测试放 `test/`（详见 `.trae/rules/project_rules.md`）。
- 代理端口、API 密钥等敏感配置不要提交到仓库。
