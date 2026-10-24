# 来源与采用方式

阅读记录日期：2026-10-24。

## 官方与平台

- [天池比赛入口](https://tianchi.aliyun.com/competition/entrance/532514)
- [LingBot-VLA-v2 官方仓库](https://github.com/Robbyant/lingbot-vla-v2)，本次镜像中的提交 `951475ae1b1d87553e7dc47c97b53a3d695c0d13`。
- [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin)，实际运行版本 `266f3aadf505a4f7fe9af0faa41a20f5f47cd123`。
- [AMD 赛事算力说明](https://github.com/AMD-DEV-CONTEST/Embodied-AI-Challenge-AMD-Platform-2026-09)。
- [AMD 参考仓库](https://github.com/ZiguanWang/Robotwin-radeon-cloud/tree/27b2f6fcb6c715e2cbb6db4431381f87c94b86ed)，用于 Full-SFT 参数、teacher、容量和速度估算。

中国站实际页面与泛用文档存在差异：本账号当前只显示 1/8 卡；页面给出按开始小时计费，Dev 池不自动销毁。本项目预算采用账户页面的保守口径。

## 五篇参赛者笔记

以下数值均为作者自报，直接作为本项目路线选择的参考；没有花本项目算力复现实验，也不是本项目成绩。

| 来源 | 作者报告/建议 | 本项目采用方式 |
|---|---|---|
| [星光碎片](https://xhslink.cn/o/Ae0iUrhx9dr) | 20k+2k，完整评测 Clean 83.12%、Random 48.28%；基座阶段使用正常学习率，已有模型小学习率加丰富图像增广 | 采用分阶段学习率与强化图像增广方向；batch 256；不照搬 80GB 卡的速度 |
| [Cur5es / TEAM EVA](https://xhslink.cn/o/6l5AxmAGUgN) | 完整万回合 Clean 81.18%、Random 59.74%；评论建议扩大训练量并避免 LoRA；更好的快速评测模型尚未完成同规模正式评测 | 采用全参数主线；不把快速与完整评测的数值混为一谈，未公开的核心方法不补写 |
| [hiyo](https://xhslink.cn/o/8LvdRdAChxx) | 完整评测均值 67.21%；列出采样、增强、优化、推理等探索方向，报告大规模训练和评测投入 | 用于理解时间成本；罗列“尝试过”不等于公开了可用参数 |
| [好好睡觉：阶段记录](https://xhslink.cn/o/8A5auYCYAeJ) | 图片给出 Clean 85.92%、Random 47.25%；评论约 24k 步、八卡 | 用于参考训练量，不当作我们的预算可达目标 |
| [好好睡觉：day11 增广配方](https://xhslink.cn/o/1C2onRFuj3Q) | Clean 85.12%、Random 50.98%，2500 条示教、24k 步；公开 10% 原图/90% 组合增广及部分参数 | 公开参数转为 `augmentation_recipe.json`；缺失参数显式留空 |

正文与全部 18 张配图已经阅读。可见评论也已阅读，但部分折叠回复受登录拦截，第五篇评论区未完整加载。**没有宣称全部评论读完。** 本仓库不转载原帖截图或完整正文。
