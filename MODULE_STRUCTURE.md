# 街景定位系统 - 模块化重构文档

## 概述

本项目参考VADAR架构设计模式，对原有单文件`test.py`进行了模块化重构，将代码按照功能职责拆分为多个独立模块，遵循单一职责原则，提高代码的可维护性和可扩展性。

***

## 项目结构

```
street_view_global_localization/
├── core/                     # 核心模块
│   ├── __init__.py           # 模块导出
│   ├── navigation_api.py     # 导航API接口（转向、移动、视角调整）
│   ├── street_view.py        # Google街景数据处理
│   └── env.py                # Gym环境封装
├── prompts/                  # 提示词模块
│   ├── __init__.py           # 模块导出
│   ├── navigation_prompt.py  # 导航决策提示词
│   └── prediction_prompt.py  # 最终位置预测提示词
├── utils/                    # 工具模块
│   ├── __init__.py           # 模块导出
│   ├── decorators.py         # 装饰器工具（重试装饰器）
│   ├── file_utils.py         # 文件操作工具
│   └── geometry.py           # 地理计算工具
├── main.py                   # 主入口（批量运行、手动控制）
└── MODULE_STRUCTURE.md       # 模块结构文档
```

***

## 模块功能说明

### 1. prompts/navigation_prompt.py - 导航决策提示词

**功能定位**：定义大模型导航决策时使用的提示词，包含角色定义、规则约束和动作选项

**导出常量**：
- `NAVIGATION_PROMPT` - 导航决策提示词模板

**使用方式**：
```python
from prompts import NAVIGATION_PROMPT
# 格式化使用，传入 step_num 和 max_steps 参数
prompt = NAVIGATION_PROMPT.format(step_num=1, max_steps=10)
```

***

### 2. prompts/prediction_prompt.py - 最终位置预测提示词

**功能定位**：定义大模型进行最终GPS位置预测时使用的提示词

**导出常量**：
- `FINAL_PREDICTION_PROMPT` - 最终位置预测提示词模板

**使用方式**：
```python
from prompts import FINAL_PREDICTION_PROMPT
# 格式化使用，传入 max_steps 参数
prompt = FINAL_PREDICTION_PROMPT.format(max_steps=10)
```

***

### 3. core/navigation_api.py - 导航API接口

**功能定位**：提供标准化的导航操作接口，封装转向、移动、视角调整等操作

**核心类**：`NavigationAPI`

**API接口列表**：

| 方法名                       | 功能说明      | 参数                                               | 返回值                       |
| ------------------------- | --------- | ------------------------------------------------ | ------------------------- |
| `init_position()`         | 初始化位置和视角  | `lat`, `lng`, `heading=0`, `pitch=0`, `fov=None` | `None`                    |
| `get_state()`             | 获取当前导航状态  | 无                                                | `Dict[str, float]`        |
| `get_geo_angle()`         | 相对方向转地理角度 | `relative_dir: str`                              | `float`                   |
| `turn()`                  | 转向操作      | `direction: str` (left/right)                    | `Tuple[bool, str]`        |
| `calculate_move_coords()` | 计算移动后坐标   | `relative_dir: str`, `distance=None`             | `Tuple[float, float]`     |
| `move()`                  | 移动操作      | `relative_dir: str`, `distance=None`             | `Tuple[bool, str, float]` |
| `set_position()`          | 直接设置位置    | `lat: float`, `lng: float`                       | `None`                    |
| `adjust_fov()`            | 调整视角FOV   | `direction: str` (up/down)                       | `Tuple[bool, str]`        |
| `adjust_pitch()`          | 调整俯仰角度    | `direction: str` (up/down)                       | `Tuple[bool, str]`        |

**使用示例**：

```python
from core.navigation_api import NavigationAPI

# 创建导航API实例
nav = NavigationAPI(
    move_step=15.0,      # 移动步长15米
    turn_angle=45.0,     # 转向角度45度
    fov_step=20,         # FOV调整步长20度
    pitch_step=10        # 俯仰调整步长10度
)

# 初始化位置
nav.init_position(lat=37.7749, lng=-122.4194, heading=90.0)

# 执行移动
success, msg, distance = nav.move("forward")
print(f"移动结果: {msg}, 距离: {distance:.2f}m")

# 执行转向
success, msg = nav.turn("left")
print(f"转向结果: {msg}")

# 获取当前状态
state = nav.get_state()
print(f"当前状态: {state}")
```

***

### 4. core/street_view.py - Google街景数据处理

**功能定位**：封装Google Street View API的调用逻辑，提供街景图像获取和元数据查询

**核心类**：`GoogleStreetViewData`

**API接口列表**：

| 方法名                       | 功能说明       | 参数                                             | 返回值                          |
| ------------------------- | ---------- | ---------------------------------------------- | ---------------------------- |
| `init_position()`         | 初始化位置和视角参数 | `lat`, `lng`, `heading=0`, `fov=60`, `pitch=0` | `None`                       |
| `get_metadata()`          | 获取街景元数据    | `lat: float`, `lng: float`                     | `Optional[Dict]`             |
| `get_street_view_image()` | 获取街景图像     | 无                                              | `Tuple[Image, float, float]` |
| `turn()`                  | 调整朝向       | `delta_angle: float`                           | `None`                       |
| `move()`                  | 移动到新位置     | `direction: float`, `target_distance=None`     | `Tuple[bool, str, float]`    |

**使用示例**：

```python
from core.street_view import GoogleStreetViewData

# 创建街景数据处理器
sv = GoogleStreetViewData(
    api_key="your_api_key",
    move_step=15.0,
    radius=10
)

# 初始化位置
sv.init_position(lat=37.7749, lng=-122.4194)

# 获取街景图像
image, actual_lat, actual_lng = sv.get_street_view_image()
print(f"实际采样点: ({actual_lat:.6f}, {actual_lng:.6f})")

# 移动
success, msg, distance = sv.move(direction=90.0)
```

***

### 5. core/env.py - Gym环境封装

**功能定位**：将街景定位任务封装为Gym风格的强化学习环境

**核心类**：`StreetViewAutoLocEnv`

**API接口列表**：

| 方法名                    | 功能说明       | 参数                                                  | 返回值                                          |
| ---------------------- | ---------- | --------------------------------------------------- | -------------------------------------------- |
| `__init__()`           | 初始化环境      | `initial_lat`, `initial_lng`, `move_step=15.0`, ... | -                                            |
| `reset()`              | 重置环境       | `seed=None`, `options=None`                         | `observation`                                |
| `step()`               | 执行一步动作     | `action: str`                                       | `(obs, reward, terminated, truncated, info)` |
| `render()`             | 渲染图像       | `mode="human"`                                      | `None`                                       |
| `gpt4o_decision()`     | 调用GPT-4o决策 | `image: Image`, `step_num: int`                     | `str`                                        |
| `parse_gpt_decision()` | 解析决策响应     | `gpt_response: str`                                 | `Tuple[str, str]`                            |
| `get_observation()`    | 获取观察       | 无                                                   | `Dict[str, Any]`                             |

**使用示例**：

```python
from core.env import StreetViewAutoLocEnv

# 创建环境
env = StreetViewAutoLocEnv(
    initial_lat=37.7749,
    initial_lng=-122.4194,
    max_steps=10,
    move_step=15.0
)

# 重置环境
obs = env.reset()

# 执行动作
obs, reward, terminated, truncated, info = env.step("move forward")
```

***

### 6. utils/decorators.py - 装饰器工具

**功能定位**：提供通用装饰器

**函数列表**：

| 函数名                  | 功能说明  | 参数                           | 返回值         |
| -------------------- | ----- | ---------------------------- | ----------- |
| `retry_on_failure()` | 重试装饰器 | `max_retries=3`, `delay=1.0` | `decorator` |

***

### 7. utils/file_utils.py - 文件操作工具

**功能定位**：提供文件操作相关工具函数

**函数列表**：

| 函数名                           | 功能说明       | 参数                                                   | 返回值               |
| ----------------------------- | ---------- | ---------------------------------------------------- | ----------------- |
| `init_batch_dir()`            | 初始化批处理目录   | `batch_root_dir="batch_streetview_results"`          | `str`             |
| `init_point_dir()`            | 初始化点位目录    | `point_info: Dict`, `batch_dir=None`, ...            | `Tuple[str, str]` |
| `image_to_base64()`           | 图像转Base64  | `image: Image`                                       | `str`             |
| `save_streetview_image()`     | 保存街景图像     | `image: Image`, `step_num: int`, `img_save_dir: str` | `Tuple[str, str]` |
| `read_gps_targets_from_csv()` | 读取GPS点位CSV | `csv_path: str`                                      | `List[Dict]`      |

***

### 8. utils/geometry.py - 地理计算工具

**功能定位**：提供GPS坐标计算相关工具函数

**函数列表**：

| 函数名                        | 功能说明      | 参数                                            | 返回值     |
| -------------------------- | --------- | --------------------------------------------- | ------- |
| `get_geo_angle()`          | 相对方向转地理角度 | `current_heading: float`, `relative_dir: str` | `float` |
| `calculate_gps_distance()` | 计算GPS距离   | `lat1`, `lng1`, `lat2`, `lng2`                | `float` |

***

### 9. main.py - 主入口

**功能定位**：项目主入口，提供批量运行和手动控制功能

**函数列表**：

| 函数名                        | 功能说明      | 参数                           | 返回值    |
| -------------------------- | --------- | ---------------------------- | ------ |
| `batch_run()`              | 批量运行模式    | 无                            | `None` |
| `manual_control()`         | 手动控制模式    | `env`, `point_dir`           | `None` |
| `generate_visualization()` | 生成可视化HTML | `env`, `point_dir`           | `None` |
| `generate_batch_summary()` | 生成批处理汇总   | `batch_dir`, `batch_results` | `None` |

***

## 模块依赖关系

```
main.py
    ├── core.env (StreetViewAutoLocEnv)
    │       ├── core.street_view (GoogleStreetViewData)
    │       │       └── utils.decorators (retry_on_failure)
    │       ├── core.navigation_api (NavigationAPI)
    │       ├── utils.file_utils (save_streetview_image, image_to_base64)
    │       ├── utils.geometry (get_geo_angle, calculate_gps_distance)
    │       ├── utils.decorators (retry_on_failure)
    │       └── prompts (NAVIGATION_PROMPT, FINAL_PREDICTION_PROMPT)
    └── utils (init_batch_dir, init_point_dir, read_gps_targets_from_csv, calculate_gps_distance)
```

***

## 运行方式

```bash
# 批量运行模式
python main.py

# 使用虚拟环境
/home/suanzigao/PycharmProjects/street_view_global_localization/.venv/bin/python main.py
```

***

## 模型配置

### 1. 本地 Qwen2.5-VL-7B 模型

系统支持使用本地部署的 Qwen2.5-VL-7B 模型进行决策：

```python
# 配置示例 (core/env.py)
self.openai_client = OpenAI(
    api_key="dummy",
    base_url="http://localhost:8000/v1",
    http_client=httpx.Client(transport=httpx.HTTPTransport(proxy="http://127.0.0.1:7897"))
)

response = self.openai_client.chat.completions.create(
    model="Qwen2.5-VL-7B-Instruct",
    messages=[{"role": "user", "content": ...}]
)
```

### 2. 环境变量配置 (.env)

```env
GOOGLE_API_KEY=your_google_api_key
OPENAI_API_KEY=your_openai_api_key
```

***

## 决策输出

每次大模型决策时，系统会输出决策原因：

```
Step 1 | Action: turn right
   Reason: The current view shows a blue bus, a pedestrian crossing, and a mix of urban and green spaces. To gather more visual information about the surroundings, I will turn right to explore the area further.
   Turned RIGHT 45°
```

***

## 已知问题与修复

| 问题                   | 位置                    | 原因                                    | 修复方案                                                  |
| -------------------- | --------------------- | ------------------------------------- | ----------------------------------------------------- |
| f-string 大括号转义错误     | `main.py`             | HTML模板中JavaScript代码的大括号需要转义           | 使用 `str.replace()` 替换占位符                              |
| httpx 代理配置错误         | `core/env.py`         | `httpx.Client()` 不接受 `proxies` 参数     | 使用 `HTTPTransport(proxy="...")`                       |
| Python 3 base64 编码错误 | `utils/file_utils.py` | Python 3 不支持 `bytes.encode('base64')` | 使用 `base64.b64encode(buf.getvalue()).decode('utf-8')` |
| 目录不存在错误              | `utils/file_utils.py` | 保存图像时目录可能不存在                          | 添加 `os.makedirs(img_save_dir, exist_ok=True)`         |
| Google API 代理配置      | `core/street_view.py` | requests 调用没有使用代理                     | 为 requests.get() 添加 `proxies=PROXIES`                 |

***

## 设计原则

1. **单一职责原则**：每个模块只负责一个功能领域
2. **接口清晰**：API接口具有明确的命名规范和参数定义
3. **依赖明确**：模块间依赖关系清晰，通过导入语句明确声明
4. **可扩展性**：模块设计考虑未来扩展需求
5. **可维护性**：添加必要的代码注释，提高代码可读性
