# 实验记录与证据索引

记录日期：2026-10-24。以下按工作阶段排序，不把尚未发生的训练填入历史。

| 编号 | 状态 | 输入与操作 | 输出/证据 |
|---|---|---|---|
| E00 | 已执行 | 官方基座推理、场景初始化诊断与单回合闭环 | [运行报告](../evidence/official-run-report.txt)、[诊断差异](../scripts/official/scene_retry.patch) |
| E01 | 已执行 | 原始 2500 条 clean 的完整性及任务内运动统计 | [初筛策略](../evidence/initial_selection_policy.json)、[初筛汇总](../evidence/initial_selection_summary.json) |
| E02 | 已执行 | 2479 条数值、对齐、重复及候选复查 | [汇总](../evidence/second_pass_summary.json) |
| E03 | 已执行 | 10 条候选的三视角视频复核 | [完成记录](../evidence/video10_summary.json)、[逐条决定](../evidence/video10_decisions.json) |
| E04 | 已执行 | 语义与左右臂动作一致性审查 | [汇总](../evidence/semantic_summary.json)、[23 条排除依据](../evidence/semantic_exclusions.json) |
| E05 | 已执行 | 剩余 33 条视频候选复核 | [汇总](../evidence/video33_summary.json)、[逐条决定](../evidence/video33_decisions.json) |
| E06 | 已执行 | 122 条多臂指令角色/事件核对 | [汇总](../evidence/multi_arm_summary.json)、[4 条软问题](../evidence/multi_arm_soft_flags.json) |
| P01 | 仅准备 | 8 卡基座全参数训练，预算段 250 步 | [原生 YAML](../configs/full_sft_8gpu_base.yaml) |
| P02 | 仅模板 | 自己已训练的合规模型的 2e-5 续训 | [续训模板](../configs/full_sft_8gpu_continue.template.yaml) |
| P03 | 本地接入和检查完成，未云训 | 强图像增广、teacher 对齐 | [配方](../configs/augmentation_recipe.json)、[集成检查](../evidence/training_integration_checks.json) |
| E07 | CPU 已计算 | 筛后归一化统计，含动作块末尾填充 | [统计报告](../configs/norm_stats/norm_report.json) |
| P04 | 离线模拟通过，未调用云 API | DCP 保存、恢复校验与 RCC 释放控制 | [操作流程](training-readiness.md)、[测试记录](../evidence/training_preparation_tests.txt) |

## 当前冻结的数据选择

- 50 个任务，2455 条轨迹，540182 帧。
- 累计排除 45/2500 = 1.8%，包括 21 条统计启发式、1 条视频指令矛盾、23 条源码与动作记录条件推断。
- [最终原始 episode 选择编号](../evidence/selection_final.json) 和 [汇总](../evidence/final_selection_summary.json)。
- 最终数据归档 SHA-256：`a6172b446ed7d5563ef9937d3721130e43174502e06ebb1b422fe41aa7d92548`。归档本身不公开，仍依赖官方原视频挂载。

## 成绩与训练状态

目前只有官方基座的诊断闭环结果 0/1。没有执行 P01/P02，没有 self-trained checkpoint、训练 loss 曲线、全任务成功率或可归因的提分结论。正式训练后应追加实际起点、配置哈希、完成步数、运行时长、费用、保存文件与评测协议，不覆盖历史记录。

## 费用口径

本次 32 credits 后续训练授权中，10 条视频轮预留 1 credit、33 条视频轮实际页面显示 1 credit，保守剩余 30。更早的环境/初筛费用发生在这一授权之前，不能把“剩余 30”写成全部项目只用了 2 credits，也不能写成账户余额。

所有已启动的历史实例均已记录释放；本次配置准备没有启动实例。
