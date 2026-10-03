# EVIDENCE — 每条结论指到哪个文件（面试现场用）

> 用法：面试官问"这个数怎么来的"，打开本文件 + 对应 `protocols/*.json`，当场可核。
> 全部数字都在仓库内，无需重跑即可验证；`reproduce.sh` 可端到端复现链路。
> 最后更新：2026-10-01

## 0. 一句话

Agent 上线后**既没有标注、也不允许重训**，只能靠线上交互持续更新；而无约束的在线更新会把
错误共识固化。本工作搭了一整套**双尺度安全门控**的部署期在线更新系统（LOCAL 证据门 + GLOBAL
e-process 提交/回滚门），并在 τ²-bench 零售域用**真实回放**测了它到底有没有用。

## 1. 环境与规模

| 项 | 数值 | 来源 |
|---|---|---|
| 任务域 | τ²-bench 零售域，**114 个任务 / 17 个工具** | `README.md`、`TECH_REPORT.md` |
| 冻结评估集 | **46 个任务**（密封集，所有对比都用它） | `protocols/frozen_seed0.json`（`n = 46`） |
| 模型/更新 | Qwen3.5-4B + 部署期 LoRA（逐 episode 边界更新，68 次更新） | `protocols/ttrl_seed*.json → update_phase`（长度 68） |
| 工程 | vLLM 服务 + LoRA 加载/切换/回滚；**17 个单元测试 + CI**（`pytest tests/ -q` → 17 passed） | `src/`、`tests/`（3 个文件，pytest 收集 17 项）、`.github/workflows/ci.yml` |

## 2. 主结果：部署期在线更新**没有提升任务成功率**（空结果，如实报告）

| 指标 | 数值 | 来源 |
|---|---|---|
| 冻结基线成功率 | **0.1087（5/46）** | `protocols/frozen_seed0.json → success_rate` |
| 更新后（候选策略）成功率 | **0.1087**（零翻转） | `protocols/ttrl_seed0.json → eval.frozen_rate / candidate_rate`；`ttrl_seed1_v3.json`、`success_replay.json` 同 |
| 行为确实变了（不是没更新） | 行为差异 25/44 任务；logit 漂移可测 0.08 → 4.0 | `protocols/success_replay.json → eval.behavior_diff`；下表 |
| 失败模式分类 | early_stop 50% / wrong_tool 20% / wrong_args 15% | `protocols/failure_taxonomy.json` |

## 3. 门控：证据不足时正确回滚

| 指标 | 数值 | 来源 |
|---|---|---|
| GLOBAL 门决策 | **ROLLBACK**（lcb_gain < 0） | `protocols/ttrl_seed0.json → gate`、`success_replay.json → gate` |
| 门控标定 | 162 配置覆盖模拟器：零假设家族错误率 0.000、SESOI 功效 0.111、强功效 0.646、投毒 0.000 | `README.md`（`protocols/gate_demo*.json` 为演示） |

## 4. 行为漂移（逐次更新的可核查序列）

| run | 首 → 末（68 次更新） | 峰值 | 源文件 |
|---|---|---|---|
| seed 0（v3，主链） | **0.084 → 4.001** | 6.86 | `protocols/ttrl_seed0_v3.json → update_phase[*].drift` |
| seed 0（v4） | 0.080 → 3.801 | 7.92 | `protocols/ttrl_seed0_v4.json` |
| seed 1（v3） | 0.108 → 2.048 | 5.25 | `protocols/ttrl_seed1_v3.json` |
| seed 0（greedy 模式） | 0.241 → 0.598 | 0.69 | `protocols/ttrl_seed0_greedy.json` |

一句话：**更新确实在改变行为**（drift 单调累积），但成功率不动 → 说明缺的不是"更新强度"，
而是**正信号本身拿不到**（见下）。

## 5. 机制发现（这条比成功率更有价值）

该基座在该环境里的「成功」样本**从不包含状态变更调用（modify）**，因此 TTRL 需要的"做对了"
的正信号在结构上不可得——这解释了第 2 节的空结果，并给出正结果配方（更强基座 / 软评估器 /
密集正信号）。产物：`protocols/success_replay.json`、`protocols/success_replay_strong.json`、
`protocols/fewshot_probe*.json`、`TECH_REPORT.md §机制`。

## 6. 复现

```bash
bash reproduce.sh                 # 冻结评估 + 一次 TTRL 更新 + 门控决策（小规模）
python3 -m pytest tests/ -q       # 17 个单元测试，全绿
```
公开仓：https://github.com/hxm2023/agentic-TTRL
