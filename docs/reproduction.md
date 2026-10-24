# 复现范围与文件使用

## 已运行的部分

环境为 AMD Radeon PRO W7900D，PyTorch 2.9.1 / ROCm 7.2 系列，官方 robotwin 镜像及 Devzone 数据挂载。实际运行的版本和诊断结果见 `evidence/official-run-report.txt`。

`scripts/` 中收录实际使用过的脚本，保留当时的路径约定。它们是实验记录，不是可在任意电脑一条命令执行的安装包。云端脚本使用 `/RoboTwin`、`/workspace`；本机审查脚本使用项目根目录下的 `outputs/data-screening/`。复现时需恢复相同数据布局，或显式调整脚本顶部路径。

Python 数据处理主要依赖 numpy、pyarrow、Pillow 和 PyAV；LeRobot 读取验证应使用原 ROCm 镜像中的训练环境。不要用重新安装的不同版本统计函数替换原聚合方式，却继续声称与历史结果相同。

诊断评测入口未整份转载。恢复时先从匹配版本的 `/RoboTwin/scripts/eval_policy_xpolicylab.py` 复制到 `/workspace/eval_policy_smoke_retry.py`，再用 `patch /workspace/eval_policy_smoke_retry.py scripts/official/scene_retry.patch` 应用记录的差异；随后才能使用重试脚本。这一步仅用于重现历史诊断，不用于替代正式评测协议。

## 数据选择如何复现

1. 从赛事官方来源获得 50 任务的 clean 数据，使用官方转换流程生成 LeRobot。
2. 运行完整性/统计审查，生成各任务指标。
3. 应用第一轮选择规则，再读取本仓库的视频及语义决策记录。
4. 通过 `evidence/selection_final.json` 中原始 episode 编号重建最终选择；该文件不包含图像、动作或状态数据。
5. 保持 episode 映射、连续索引与原始数值一致，重算 LeRobot 聚合统计。
6. 对最终数据运行云端 loader 检查，并单独计算 LingBot 统一特征空间 norm_stats。

第 6 步针对最终 2455 条版本尚未在云端完成；不能用早期 2479/2478 条版本通过的 loader 检查代替。

## 尚未运行的训练

`configs/` 中是准备好的原生 YAML、续训模板、预算与独立强增广规格。它们不触发训练、云实例创建或付费。阅读该目录说明后再决定执行。

尚缺：完整强增广接入、最终 norm_stats、足够的持久化存储、外部时间截止与实例释放流程。当前自己的训练尚未开始，续训模板使用明确占位符；只允许填入本项目未来产出的 checkpoint。

## 公开内容处理

只发布选择编号、必要的指令案例、方法脚本和摘要；不发布原始 parquet、视频、模型权重、登录资料、机器连接信息或云账户截图。历史文本中的本机绝对路径做了可移植化处理。`evidence/file-manifest.json` 记录来源文件及发布副本的校验值，便于区分原文与脱敏副本。
