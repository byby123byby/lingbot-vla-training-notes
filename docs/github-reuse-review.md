# GitHub 复用调查：先接能用的工程，效果记录按原口径保留

检索日期：2026-10-24。目标是 LingBot-VLA v2、RoboTwin 50 任务、clean-only 独立训练。查阅 GitHub 搜索结果、8 个候选 README、关联 Hugging Face 模型卡、官方训练源码及 Radeon Cloud CLI 文档。本次没有下载他人模型、做消融、开云实例或复测成功率。

**本轮直接复用官方训练器、特征映射、DCP 保存路径和 AMD 官方 CLI／SDK。没有找到一份同时满足“公开强增广代码、可核对同比赛协议成绩、与本项目全参数路线兼容”的现成高分仓库，因此没有把未公开方案说成已直接接入。** 公开配方的图像操作用现有 NumPy/Pillow 算子实现，缺少的参数明确标为工程默认值。

## 按本项目可用性排序

### 1. Robbyant LingBot-VLA v2 + AMD 适配仓库：已采用的实现基础

- 来源：[官方模型代码](https://github.com/Robbyant/lingbot-vla-v2/tree/951475ae1b1d87553e7dc47c97b53a3d695c0d13)、[AMD 训练实现](https://github.com/ZiguanWang/Robotwin-radeon-cloud/tree/27b2f6fcb6c715e2cbb6db4431381f87c94b86ed)。角色：直接实现；这是已有基础，不算本次新发现。
- 两个具体匹配点：相同 6B 基座与 55 维特征空间；AMD W7900D 的 FSDP2 全参数／完整 teacher 路线。省掉重写训练器、机器人映射和 checkpoint 格式。
- 实际采用：保留原训练主循环，替换两个增广函数，增加步边界保存截止、完整性标记和恢复失败拦截。CPU 检查通过；未验证真实 GPU 恢复。
- 不采用官方 RoboTwin 后训练权重，其 clean+randomized 数据和成绩不适用于本项目 clean-only 独立训练。
- 阅读深度：相关 transform、feature mapping、norm、checkpoint、训练循环、AMD 启动脚本。推荐投入：当前主线。
- 版本：上述固定提交；首次发布：本次未重建；观测：2026-10-24；热度事件：未核验。后继：发现 [LeRobot 的 LingBot-VLA v2 端口](https://huggingface.co/docs/lerobot/main/en/lingbot_vla_v2)，但本轮不迁移到另一套训练框架。

### 2. AMD Radeon Cloud 官方 CLI：本轮已采用的释放控制实现

- 来源：[官方文档仓库](https://github.com/AMD-AIM/radeon-cloud-docs/tree/e442835d9ae11d811feffd4257d7931b58f72112)、[实例命令](https://amd-aim.github.io/radeon-cloud-docs/cli/instances/)、[自动化约定](https://amd-aim.github.io/radeon-cloud-docs/cli/automation/)。角色：直接工程复用。
- 匹配点：中国站实例资源；训练结束后必须销毁才能停止计费。复用 `amd-oneclick-sdk==1.0.2`，不抓取浏览器 cookie，不自造销毁 API。
- 已实现：命名账号绑定、中国站限定、当前实例身份比对、到期销毁、状态确认、超时告警。离线模拟覆盖错实例、SSH 失联、API 失败及 terminating 状态。
- 边界：设备授权尚未进行；没有调用真实云 API。SDK 没有按租约版本条件删除的接口，期间必须独占该账号当前实例，禁止与队友交替创建实例。
- 阅读深度：官方认证／实例／CLI 文档及已安装包的 client、instance commands、JSON 输出代码。推荐投入：开训前必备。
- 版本：文档固定提交；SDK 1.0.2；首次发布日期：未核验；观测：2026-10-24；热度事件：无判断。现成实现与当前版本已找到。

### 3. Owlsnet clean50 20k：同比赛长训记录，能参考参数，代码尚不可直接复用

- 来源：[模型卡](https://huggingface.co/Owlsnet/lingbot-vla-v2-robotwin-clean50-20k/blob/main/README.md)。角色：效果与训练量参考。作者关联 GitHub 为 `HAOTianGa03/lingbot-robotwin-clean50`，模型卡明确说是私有仓库；本次 GitHub 访问也未成功。
- 匹配点：官方 6B 基座起训；50×50 clean 示教、相同比赛。作者报告累计 20k 步，在 50×100 回合口径下 clean **77.32%**、randomized **40.24%**。
- 可用内容：clean 统计、颜色增广、分段续训和训练量记录。作者用 Muon 与长学习率计划；不能把它等价为本项目几百步 AdamW 的预期成绩。不下载作者权重，也不改为接其模型。
- 阅读深度：完整模型卡；没有访问私有实现。推荐投入：参考，不作为本轮可复制代码。
- 版本：模型卡显示 `d41864d`；首次发布日期：未核验；观测：2026-10-24；热度：未核验。作者代码后继检查：访问受阻；不能宣称不存在公开后继。

### 4. hggforget Residual Megaflow：小开销提升的线索，留给自己的模型训成之后

- 来源：[残差模型卡](https://huggingface.co/hggforget/residual_megaflow)、[对应 SFT 模型卡](https://huggingface.co/hggforget/lingbotvla-robotwin-clean-sft)。角色：后续机制参考；模型卡链接的实现是 `lingbot-ai/lingbot-vla-v2`，本次无法取得代码。
- 匹配点：同一 LingBot v2／RoboTwin；在已训 SFT 上增加小型残差头。作者报告 8 卡约 50 分钟，randomized unseen 的 50×10 回合由 **34.8% → 38.4%**，提高 3.6 个百分点。
- 采用边界：依赖已训练到 10000 步的冻结 SFT，且绑定基座和 norm 指纹。它不能直接替代从官方基座开始的第一段训练，不能拿作者残差头套到不同的自训权重上。暂不接入，等自己的基座可用且实现可访问后再考虑。
- 阅读深度：完整残差模型卡、SFT 模型卡；代码未取得。推荐投入：保留线索，不消耗当前预算。
- 效果日期：2026-10-18；模型卡版本：本次未取得固定提交号，保存了读取内容的哈希；首次发布日期：未知；观测：2026-10-24；热度：未知。作者实现检查：受阻。

### 5. 工程相似仓库：可以借鉴记录方式，不能包装成“已有高分方案”

- [RobustTwin-VLA](https://github.com/CangLing-Cerulean/RobustTwin-VLA)：也是 clean-only、完整 teacher 与全参数路线，但 README 明确尚未训练；本次仓库 API 没有显示代码许可证，未复制其代码。初建 2026-10-12；阅读 README；推荐轻参考。
- [Coola666 pipeline](https://github.com/Coola666/lingbot-vla-robotwin-pipeline)：Apache-2.0；有完整公开的负结果记录。其自训模型单任务最终物理成功仍为 0/20，19/20 属于官方后训练模型。可参考区分动作误差与真实任务成功，不当作高分训练配方。初建 2026-07-29；阅读 README；推荐轻参考。
- [sxh-kk/LingBot](https://github.com/sxh-kk/LingBot)、[EclipseaHime017](https://github.com/EclipseaHime017/lingbot-vla-robotwin)、[AlexDrives](https://github.com/AlexDrives/Lingbot-RoboTwin)、[zhao-zetong](https://github.com/zhao-zetong/robotwin-training)、[MI300X practice](https://github.com/zyiguo/lingbot-vla-mi300x-practice)：读到工程、短训或其他硬件记录，没有在本次读取范围内找到可直接继承的同协议高分强增广实现。未复制代码；版本／首次时间未进一步固定；统一观测日期 2026-10-24，热度未核验。

## 迁移与验证范围

| 来源机制 | 本项目行为 | 验证证据 | 失败情形 |
|---|---|---|---|
| 官方图像参数重放 | 共用参数及 seed，保留当前／未来与 teacher 路径 | `training_integration_checks.json` | 非 RGB／错误浮点范围拒绝；不做生成式改图 |
| 官方特征映射与动作块 | CPU 加权统计，episode 末尾重复动作 | `norm_report.json`；与显式窗口及官方单批 RunningStats 对照 | 轨迹 ID、parquet 哈希或数量不符拒绝 |
| 分布式 DCP 的协同保存 | 优化器边界统一停止，完整状态和校验清单 | 本地结构／损坏测试；固定源码补丁检查 | 部分保存不得续训；GPU 集体通信未测试 |
| 独立控制器结束资源租约 | 中国站 RCC 删除并轮询消失 | 本地假客户端状态演练 | 网络断开不能保证按时释放，明确告警 |

分布式作业的抢占保存是从高性能计算调度借用的工程结构：调度截止对应计费截止，协同保存对应 DCP，外部调度器对应 RCC 控制器。成立前提是所有 rank 能抵达保存边界、PVC 可写和外部控制器联网；单步卡死或控制器失联时不保证保存完成。参考 [PyTorch DCP](https://docs.pytorch.org/docs/main/distributed/checkpoint.html)，没有引入 Ray 等额外调度系统。

检索覆盖：直接仓库搜索、图像增广机制、作者模型卡与关联实现、负结果、当前 CLI／LeRobot 版本、相邻的分布式抢占保存。停止原因：本轮候选已足够做工程决策，继续检索的收益低于完成实现；不是宣称已穷尽所有仓库。私有／访问失败的实现仍是缺口。
