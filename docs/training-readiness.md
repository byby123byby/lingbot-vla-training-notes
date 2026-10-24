# 独立训练准备与保存释放流程

2026-10-24：代码、统计文件与离线检查已完成；未创建云实例、未运行模型训练、未调用云资源 API，新增 cloud credits 为 0。下面的执行命令是将来开训时的操作说明，本轮没有执行。

## 已完成的内容

| 项目 | 产物与边界 |
|---|---|
| 强增广 | `scripts/training/strong_augment.py`；10% 原图分支，90% 组合增广；当前／未来帧及多视角共用参数与随机种子，teacher 使用对应增广图像 |
| 接入 | `install_runtime.py` 核验官方源码 SHA-256 后安装两个小范围补丁；固定 LingBot `951475ae1b1d87553e7dc47c97b53a3d695c0d13`；不接受未知版本静默替换 |
| 数据统计 | `configs/norm_stats/clean_2455.json`；原始 episode 身份、已存档 parquet 哈希、连续帧、元数据数量与最终筛选清单均检查 |
| 保存 | 到截止时间后在下一次完整优化器步结束处同步停止；三条保存路径均写 DCP、全文件校验和与原子 COMPLETE 标记 |
| 恢复 | `run_budgeted.py --resume` 验证本项目标记、rank 数、统计哈希和全部文件；加载失败直接报错，不重新开始训练 |
| 释放 | `release_watch.py` 使用 AMD 官方 SDK 1.0.2；完成即申请释放，最迟计费起点后 165 分钟申请；确认无实例才写 RELEASED |

强增广未公开的幅度已写成显式工程默认值，见 [配方](../configs/augmentation_recipe.json)。遮挡保护采用中心区域、饱和度和边缘检测的保守启发式：不确定的区域跳过遮挡；它不是语义分割，不能保证识别每个边缘物体。没有添加作者未披露的色温扰动。所有实现只改变图像，不修改动作或指令。

统计使用官方 **填充前** 的四个特征：状态／动作各 12 维手臂和 2 维夹爪，再由官方模型映射进 55 维。动作按 50 步窗口和末尾重复填充加权，共 27009100 个动作项；状态为 540182 项。采用固定全局 5000-bin 直方图，避免自适应重分桶的顺序误差；与官方单批次统计的小样本对照通过，但不宣称与官方多进程重分桶输出逐位一致。

## 本地重算与检查（不消耗 cloud credits）

在仓库根目录的独立 Python 环境中安装 `configs/requirements-local.txt`。其中 CPU 验证用 Torch 2.8.0；**不要拿本机依赖文件覆盖 AMD 实例自带的 ROCm 环境**。

```bash
python scripts/training/compute_filtered_norm.py \
  --data-root /path/to/clean_screen_semantic_20261024 \
  --output configs/norm_stats
python -m unittest discover -s tests -p 'test_*.py' -v
python scripts/training/release_watch.py \
  --plan configs/release_plan.template.json --output /path/to/offline-check
```

最后一条默认只输出计划，不登录、不访问云 API。训练入口也只有显式 `--execute` 才会执行；本次没有运行任何 `--execute` 命令。检查报告见 [集成证据](../evidence/training_integration_checks.json) 与 [本地测试](../evidence/training_preparation_tests.txt)。

## 将来开训前，先准备实例外的控制器

控制器必须放在不会随实例一起销毁、能持续联网的机器上。Mac 应保持唤醒；关机、断网、授权失效或平台 API 故障都可能让销毁无法确认。流程会显式报错，不能把预算上限当成平台的硬性扣费保证。

1. 由账号持有人进行一次官方设备授权，保存到中国站命名账号。此授权尚未进行：

   ```bash
   rcc account register lingbot-china \
     --base-url https://developer.amd.com.cn/radeon --no-browser
   ```

2. 在未来获准开训、实例已创建后，读取实际状态：

   ```bash
   rcc -A lingbot-china --output json instance status
   ```

   复制模板为私有 `release_plan.private.json`，填入当前 `instance_id`、SSH host/port、SSH config alias 和**平台实际计费起点**。实例 ID 可能跨次复用，因此同时绑定 SSH 信息，并要求此段期间不与队友交替创建／销毁实例。保留整小时计费的 15 分钟销毁余量。模板的 `0/null/false` 是故意的启动阻断，不是可直接执行的参数。

3. 确认 `/workspace` 是 PVC，并且有至少 **85 GiB 可用空间**。此前观察到的 25GB 持久盘不够，启动器会拒绝。100GB 是否对当前账号开放，需要未来从实际资源页面确认；本轮没有扩容或购买。PVC 中保存模型和优化器，控制器下载小型保存回执与校验清单；不要求在 30 分钟内通过公网下载整个约 70GB DCP。

4. 启动独立释放控制器，它不会创建实例或启动训练：

   ```bash
   python scripts/training/release_watch.py \
     --plan release_plan.private.json --output runtime-private --execute
   ```

   它核对实例身份后写 `runtime-private/WATCHDOG_ARMED.json`。将此回执复制到实例；训练入口只接受五分钟内、属于本次目录与统计文件的回执。若准备过久，重新确认并启动控制器，不修改旧回执时间。

## 实例内安装与未来训练入口

先恢复筛后数据压缩包、原视频挂载和官方模型；数据列表必须指向最新的 2455 条版本。下面命令应在未来实例的官方 Python 环境中执行：

```bash
python scripts/training/install_runtime.py \
  --source /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2
mkdir -p /workspace/runtime/norm_stats
cp configs/norm_stats/clean_2455.json /workspace/runtime/norm_stats/clean_2455.json
```

先运行 `run_budgeted.py` 不带 `--execute` 查看绝对截止时刻；将 `BILLING_START_EPOCH` 设置为实际计费起点的 Unix 秒数。将来获准开训后才添加以下两个执行确认参数：

```bash
python scripts/training/run_budgeted.py \
  --config configs/full_sft_8gpu_base.yaml \
  --source /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2 \
  --run-dir /workspace/runtime/outputs/full_sft_clean2455_8gpu_base \
  --billing-start-epoch "$BILLING_START_EPOCH" \
  --watchdog-armed /workspace/WATCHDOG_ARMED.json \
  --pvc-confirmed --execute
```

同一官方源码只能安装补丁一次；重复安装会因哈希变化拒绝。保留 `.before-training-prep` 原始备份；切换官方版本时重新审查补丁，不能跳过哈希检查。

## 截止与异常行为

| 触发条件 | 行为 |
|---|---|
| 到 250 步或计费起点后 135 分钟 | 下一优化器步边界同步停止，写完整 DCP 与校验清单；也可在运行目录放 `STOP_REQUESTED` 请求提前保存 |
| 正常退出且完整保存 | 外部控制器取回回执，立即请求销毁，不等满三小时 |
| 训练异常退出 | 保留 PVC 中现有文件，控制器申请销毁；结果标为失败，不冒充训练成功 |
| 到计费起点后 165 分钟 | 不依赖 SSH 能否连接，申请销毁；即使保存尚未完成也不无限延长计费。未完成的 DCP 没有 COMPLETE 标记，将来不得续训 |
| API 暂时失败 | 有限超时后重查和重试；平台返回 terminating 不算释放完成 |
| 到 180 分钟仍未确认释放 | 写 RELEASE_UNCONFIRMED 并报错，必须立即通过中国站控制台处理；不声称费用已停止 |

下次接自己的完整 DCP 时用新的输出目录、相同 8 卡拓扑和统计文件，将 `max_steps` 设为**累计目标步数**，再给 `run_budgeted.py` 添加 `--resume /workspace/.../checkpoints/global_step_N`。保存了主进程 RNG 与上游 loader 状态，但没有证明多 worker 预取和随机增广逐位复现；训练可恢复与逐位复现不是同一承诺。

## 验证范围

本地通过：图像重放／范围／前景保护、官方 prepare_images 当前未来及 teacher 对齐、官方归一化读写与逆变换、动作窗口加权对照、原始轨迹 ID 与文件哈希、检查点缺失／损坏拒绝、错实例保护、SSH 失联时截止释放、API 错误重试和未释放状态识别。

未执行：真实 8 卡 forward/backward、约 70GB DCP 的云盘写入与恢复、真实云端销毁、GPU 吞吐与成功率。不会为这些检查开实例。

来源：[AMD CLI](https://amd-aim.github.io/radeon-cloud-docs/cli/instances/)、[AMD 自动化](https://amd-aim.github.io/radeon-cloud-docs/cli/automation/)、[官方 LingBot 源码](https://github.com/Robbyant/lingbot-vla-v2/tree/951475ae1b1d87553e7dc47c97b53a3d695c0d13)。
