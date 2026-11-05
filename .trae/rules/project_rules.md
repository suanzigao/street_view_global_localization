# 项目规则

## 代码存放约定

本项目的文件存放遵循以下约定，后续所有调整/开发都必须遵守：

1. **改动原有代码** → 直接修改原文件，保持单一事实来源，不做复制。
   - 正式功能模块（如 `localization/`、`core/`、`tools/`、`utils/`、`prompts/`）直接原位修改。

2. **新功能代码** → 保存到 `scripts/` 目录。
   - `scripts/` 作为临时/暂存区，最终不保留，仅用于承载新功能、实验脚本、调试脚本。

3. **代码测试** → 保存到 `test/` 目录。
   - 所有测试脚本、验证脚本统一放到 `test/`。

## 其他约定

- 临时生成的脚本不得散落到项目根目录，须归入 `scripts/` 或 `test/`。
- 三阶段级联定位的核心代码位于 `localization/`，属已定型的正式模块，保留原地。

## 虚拟环境

- 本项目使用虚拟环境 `.venv`（Python 3.12），路径为 `/home/suanzigao/PycharmProjects/street_view_global_localization/.venv`。
- 运行 Python 脚本时，统一使用该虚拟环境解释器：`/home/suanzigao/PycharmProjects/street_view_global_localization/.venv/bin/python`（在项目根目录下可简写为 `.venv/bin/python`）。
