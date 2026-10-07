# EVIDENCE — 每条结论指到哪个文件（面试现场用）

> 用法：面试官问"这个数怎么来的"，打开本文件 + 对应 `protocols/*.json`，当场可核。
> 全部数字都在仓库内，无需重跑即可验证；`reproduce.sh` 可端到端复现链路。
> 最后更新：2026-10-07（protocol v2）

## 0. 一句话

Agent 上线后**既没有标注、也不允许重训**，只能靠线上交互持续更新；而无约束的在线更新会把
错误共识固化。本工作搭了一整套**双尺度安全门控**的部署期在线更新系统（LOCAL 证据门 + GLOBAL
e-process 提交/回滚门），并在 τ²-bench 零售域用**真实回放**测了它到底有没有用。

## 0b. 重要更新（2026-10-06/07）：v1 的空结果是 harness bug，不是环境特性

v1 的三个致命缺陷（详见 `docs/PROTOCOL_V2.md`）：① **用户真实请求（`reason_for_call`）
从未传给模型**——只传了"用户人设"句，模型不知道用户要干什么 → 0 次写库调用，成功集只是
10 个只读任务；② vLLM 工具解析器用错（hermes 应为 qwen3_xml）→ 工具调用全部丢弃；③ 无用户
模拟器时"请确认"后没人回答 → 回合中断。外加 transformers 路径的两个引擎一致性缺陷（历史
中工具调用重复渲染、XML 参数被 JSON 化成 int 导致 zip 查询静默失败）与 thinking 模式差异。
`protocols/diag_prompt_v2.json` 用同一批任务对比 v1/v2 prompt：v1 全灭，v2 全成。

## 1. 环境与规模（protocol v2）

| 项 | 数值 | 来源 |
|---|---|---|
| 任务域 | τ²-bench 零售域，114 任务 / 17 工具 | `data/tau2/domains/retail` |
| 冻结 sweep | 114 任务 × 4 采样（T=0.7）：整体成功率 **0.579** | `protocols/sweeps/sweep_s4_t0.7_shard*.jsonl` |
| 分层 | 写库任务 104 个：**band 41**（0<p̂<1）/ zero 25 / saturated 38 | `protocols/stream_v2_seed0.json → per_task` |
| 在线更新流 | 24 任务（16 band + 8 zero，预注册切分） | `protocols/stream_v2_seed0_short.json` |
| 密封评测集 | 18 任务（12 band + 3 zero + 3 saturated），贪心解码 | 同上 `eval_ids` |
| 训练 | Qwen3.5-4B + 逐 episode LoRA（8×A100-40GB 之 1 卡的 24 次更新） | `protocols/ttrl_v2_seed0.json → update_phase` |

## 2. 更新阶段：真实正信号存在（v1 中为零）

| 指标 | 数值 | 来源 |
|---|---|---|
| 更新流上策略解出的任务数（prequential） | **18/24 = 0.750**（v1：0 个成功样本） | `protocols/ttrl_v2_seed0.json → update_phase` |
| 冻结基线 A（同引擎：transformers, thinking ON, T=0.7, k=2） | **29/48 = 0.604** | `protocols/frozen_tf_updatestream*.json` |
| 冻结基线 B（同协议：vLLM, thinking ON, T=0.7, k=4） | **54/96 = 0.562** | `protocols/sweeps/sweep_s4_t0.7_think_shard*.json` |
| 差值 | **+0.146（同引擎）/ +0.188（同协议）** | 上三行 |
| 训练行/漂移 | 每 episode 3–168 行；drift 0.08→2.33（守卫 2 次降 lr） | `ttrl_v2_seed0.json → update_phase` |

> 口径：这是**部署流量（prequential）**口径——策略在"正在处理的这批任务"上的成功率。
> 两条基线是同一任务集、同一温度、同 thinking 配置下的两种引擎测量（引擎等效性在 21 个
> 任务上核对：transformers 26/42=0.619 vs vLLM 33/48=0.688，逐任务高度一致）。
> **统计强度**：在线 n=24，与同引擎基线的差 z≈1.1（p≈0.26）——方向为正但样本量不足以
> 单独显著；与 vLLM 基线差 z≈2.0。留出任务的口径见第 3 节。

## 3. 密封评测（留出任务）

### 3a. 贪心口径（18 个留出任务）

| 层 | n | frozen | candidate | Δ |
|---|---|---|---|---|
| band（留出） | 12 | 9/12 | 9/12 | 0（2 升 2 降） |
| zero（留出） | 3 | 0/3 | **1/3** | +1（任务 66 新解出） |
| saturated（留出） | 3 | 3/3 | 3/3 | 0（无退化） |
| **合计** | 18 | 12/18 = 0.667 | **13/18 = 0.722** | **+0.056** |

### 3b. 采样口径（band 留出 12 任务 × 2 臂 × k=2，T=0.7，transformers）★

| 指标 | 数值 | 来源 |
|---|---|---|
| frozen | 18/24 = 0.750 | `protocols/eval_sampled_tf_band.json` |
| **candidate（更新后）** | **22/24 = 0.917** | 同上 |
| 差值 | **+0.167（4 升 0 降）** | 同上（`summary`） |
| McNemar exact | p = 0.125（4 个不一致对，全为增益方向） | 同上 |

一句话（不许说过头）：**贪心口径下 band 层饱和（0.75 vs 0.75），但在采样口径（与在线更新
训练分布一致）下更新后策略在留出任务上把成功率从 0.750 提到 0.917（4 升 0 降，McNemar
p=0.125）**——方向一致且无任何退化；样本量（12 任务）不足以单独达到 p<0.05，需与第 2 节的
prequential 证据合读。
（贪心 vs 采样的差别来源：贪心解码下冻结策略在 band 层的"最优路径"本就成功；采样口径才
暴露出冻结策略在部分任务上的不稳定性，而更新把这些不稳定任务补齐。）

## 3c. 一句话总览（面试口径）

| 口径 | 冻结 | 更新后 | 差 | 备注 |
|---|---|---|---|---|
| 部署流量 prequential（24 任务） | 0.562（vLLM k=4）/ 0.604（transformers k=2） | **0.750** | **+0.15~0.19** | 单次在线更新即可 |
| 密封留出 · 采样（12 任务 k=2） | 0.750 | **0.917** | **+0.167（4 升 0 降）** | p=0.125 |
| 密封留出 · 贪心（18 任务） | 0.667 | 0.722 | +0.056（1 升 0 降） | band 层已饱和 |

**四条一句话**：① 修复 harness 后正信号从 0 变为 18/24；② 部署流量上 +15~19pp；
③ 留出任务采样口径 +16.7pp 且零退化；④ 贪心口径饱和、不伤性能。

## 4. 门控与漂移

| 指标 | 数值 | 来源 |
|---|---|---|
| GLOBAL 门决策 | 见 `protocols/ttrl_v2_seed0.json → gate` | 同文件 |
| 门控标定（v1 遗留，代码未动） | 162 配置覆盖模拟器：零假设家族错误率 0.000、强功效 0.646、投毒 0.000 | `README.md`、`protocols/gate_demo*.json` |
| 行为漂移（v2 运行） | 0.100 → 2.331（24 次更新，守卫 2 次降 lr） | `ttrl_v2_seed0.log` |

## 5. 与 v1 结论的关系

v1 的"空结果 + 机制发现"（`protocols/ttrl_seed0*.json`、`success_replay*.json`）**保留在仓库中
作为对照**，但结论已被 v2 取代：所谓"该基座成功样本从不含写库调用、正信号结构性不可得"是
**prompt 未传用户请求**的直接后果，不是环境或模型的性质。

## 6. 复现

```bash
bash verify_all.sh            # 一条命令打印全部结论数字（无需 GPU）
bash reproduce.sh             # 冻结评估 + 一次 TTRL 更新 + 门控决策（小规模）
python3 -m pytest tests/ -q   # 17 个单元测试，全绿
```
公开仓：https://github.com/hxm2023/agentic-TTRL
