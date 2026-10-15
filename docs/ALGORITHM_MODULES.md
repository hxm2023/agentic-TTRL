# agent-ttrl2 核心算法模块详解（架构图设计输入）

> 用途：完整拆解核心算法的各个模块——职责、输入、输出、内部逻辑、文件位置、
> 数据契约，并附「画图建议」。画架构图时以第 8 节为准。

## 0. 总览：六个核心模块 + 一条主循环

```
环境 env ──► Rollout agent ──► 证据与信用 credit ──► 训练 trainer ──┐
   ▲                                                              │
   └──────────────── 下一回合（策略已更新） ◄───────────────────────┘
                                  │
                          全局门 gates（部署判决）
                                  │
                          密封评估（46 题）
```

| # | 模块 | 文件 | 一句话职责 |
|---|------|------|-----------|
| 1 | 环境 env | `src/ttrl2/env/tau2_env.py`, `receipts.py` | 可重放任务 + 隐藏 DB 状态判分 |
| 2 | Rollout agent | `src/ttrl2/agent/loop.py`, `transformers_loop.py` | 策略↔环境多轮工具调用循环 |
| 3 | 证据与信用 credit | `src/ttrl2/trainer/lora_update.py`（上） | E_hard 回执 → 冲突检测 → 分组信用 → 训练行 |
| 4 | 训练 trainer | `src/ttrl2/trainer/lora_update.py`（下） | LoRA 生命周期 + 策略梯度 + 漂移探针 |
| 5 | 门控 gates | `src/ttrl2/gates/local_gate.py`, `global_gate.py` | 局部门（信用过滤）/ 全局门（提交回滚） |
| 6 | 驱动 driver | `scripts/run_stream.py` | 流编排、三臂对比、产物输出 |

外部依赖：冻结基座（Qwen3.5-4B / Llama-3.1-8B）、τ²-bench 零售数据、vLLM 服务
（基线用）、GPU0 训练 / GPU1 推理的双卡布局。

---

## 1. 环境模块（env）

**职责**：把 τ²-bench 零售域包装成"可重放、可判分"的 episode 抽象。

**核心类 `Tau2Episode`**：
- 构造：加载干净 RetailDB（1000 订单 / 500 用户 / 50 商品），应用任务初始状态
- `step(tool_name, arguments) → ToolReceipt`：执行 16 个零售工具之一，返回 E_hard 回执
- `evaluate() → bool`：**隐藏判分**——在另一份干净 DB 上重放参考答案动作序列得到
  目标状态，与当前 DB 全量比对（状态相等 = 1 分）
- `snapshot_state()`：诊断用状态指纹

**数据契约**：`ToolReceipt(tool_name, arguments, ok, output, error)`、
`EpisodeRecord(task_id, receipts[], success, turns)`

**关键属性**：确定性重放；tool-only 协议（无 user simulator）；评估器隐藏
（判分标准不进 prompt）。

---

## 2. Rollout 模块（agent）

**职责**：驱动"策略 ↔ 环境"的多轮工具调用循环。

**两个实现**：`loop.py`（经 vLLM OpenAI 接口，用于基线/探针）；
`transformers_loop.py`（进程内生成，**TTRL 主实验用**——适配器必须实时生效）。

**循环逻辑**：
- 输入拼装：system = 身份声明 + 136 行政策文档；16 个工具 schema 经 chat template
  注入；user = 任务指令 + 已知信息
- 每轮：贪心生成 → 解析工具调用（`qwen3_xml` / `llama3_json` 双解析器）→ 环境执行
  → 回执追加对话 → 继续
- 终止：无工具调用 或 达 20 轮上限
- 输出：完整 transcript（assistant 轮 + tool 回执轮）+ 回执明细

**约束**：贪心解码（可复现、行为与评估一致）；单轮 ≤384 token；Llama 模板限制
"每条消息单工具调用"，需逐调用拆分。

---

## 3. 证据与信用模块（credit）

这是"学什么"的核心，四个子功能：

**3.1 结构化动作分组**：16 个工具 → 4 组——`identify`（查用户）、`read`
（查订单/商品/详情）、`modify`（改地址/换货/退货/取消）、`other`（计算器/转人工）；
`stop`（终止回合）作为伪组。

**3.2 冲突检测**：`detect_conflicts(receipts, identified_user) → {group: [reason]}`
- `REPEATED_FAIL_SAME_ARGS`：失败调用被同参数原样重试
- `MISSING_IDENTITY_BEFORE_MODIFY`：出现修改类调用但全程无成功的身份识别
- `USER_MISMATCH_ON_MODIFY`：修改对象（订单 user_id）≠ 已识别用户

**3.3 分组信用（失败感知）**：`assign_group_credit(...) → [CreditRow]`

| 情形 | identify/read/other | modify | stop |
|------|--------------------|--------|------|
| 成功 | +0.5 | +0.5 | +0.5 |
| 失败 | 0（正确前置步骤不罚） | −0.3（调用成功但任务失败）/ −0.1（调用本身报错） | −0.3 |
| 冲突组 | 0（abstain，`EVIDENCE_CONFLICT:*`） | 同左 | — |

**3.4 训练行构造**：`build_training_rows(...) → rows`
- 每个带工具调用的 assistant 回合生成一行，优势 = 该回合所有调用的平均信用
- 最后回答回合生成一行（终止奖励）
- **优势为 0 的行过滤掉**——"弃权"的机械含义：不进梯度

---

## 4. 训练模块（trainer）

**4.1 LoRA 生命周期**：`make_lora_model(base, r=16, alpha=32)`，7 个投影模块；
基座全冻结，可训练参数 2123 万（0.5%）；适配器 = 版本单元（提交/回滚只换它）。

**4.2 参考模型**：独立加载的冻结基座副本（no_grad 算 ref_logp）。为什么独立：
共享对象的前向会污染后续带梯度前向（混合架构实测 OOM 29GB vs 独立 20.5GB）。
8B 模型装不下两份 → KL-free 模式（护栏代替 KL）。

**4.3 策略梯度更新**：`grpo_update(model, ref_model, tokenizer, rows, ...)`
- 单行流程：chat template 渲染 → 定位动作 span（字符偏移 + offset mapping）→
  **分块 logsumexp** 算 token 对数概率 → 损失 = `−(优势 × logp).mean() + 0.1 × KL`
- KL 项：`clamp(logp_policy − logp_ref, min=0)`，只作用动作 span、单边惩罚
- 超参：AdamW、lr 5e-6、**每回合 4 步（成功 8 步）**、梯度裁剪 1.0
- 显存设计：ref 先算先释放、分块 logsumexp、梯度检查点；峰值 20.5GB

**4.4 漂移探针**：`logit_drift(...) → 标量`——固定 probe prompt，比较策略与基座
top-50 候选的对数概率平均差；实测轨迹 0.08→4.0（68 回合）；
**护栏：>2.0 学习率减半（下限 1e-6）**。

---

## 5. 门控模块（gates）

**5.1 局部门**（`local_gate.py` + credit 内联）
- 位置：信用分配**之前**
- 机制：三条冲突规则 → 冲突组信用置零
- 熔断器：`DriftMonitor(window=5, threshold=3)`——5 回合窗口内 3 次冲突 →
  暂停更新（fail-closed）

**5.2 全局门**（`global_gate.py`）
- 位置：更新**之后**、部署**之前**
- 输入：配对影子评估差值（`gain_diffs` = 候选−父版本；`harm_diffs` = 父版本−候选）
- 判定：经验-伯恩斯坦 e-process

```
α_k = 6α/(π²k²)                     # 跨候选误差预算，可加和
r = √(2σ²·ln(1/α_side)/n) + 7·ln(1/α_side)/(3(n−1))   # 方差为 0 时退化：λ/8 + ln(1/α)/(λn)
LCB_gain = mean(gain) − r    UCB_harm = mean(harm) + r
COMMIT ⟺ LCB ≥ 0.01 且 UCB ≤ 0.10；样本 <2 → INCONCLUSIVE
```

- 冻结常量：α=0.05、ε_gain=0.01、ε_harm=0.10、λ=0.5、n=512
- 参数来源：162 组配置覆盖模拟器（null 0.000 / sesoi 0.111 / strong 0.646 /
  poisoned 0.000）
- 输出：`GateOutcome(decision, lcb_gain, ucb_harm, alpha_k, n_gain, n_anchor,
  reason_codes)`

---

## 6. 驱动模块（run_stream.py）

**6.1 流构建**：114 题按 seed 打乱；前 68 = 更新流；后 46 = 密封评估（零接触）。

**6.2 三种模式**：`frozen`（基线 0.109）；`bon`（N 采样 + **E_hard 可见启发式**
选优，不用隐藏评估器）；`ttrl`（主实验）：
- 更新阶段（68 回合循环）：rollout（当前适配器）→ 判分 → 冲突检测 → 信用 →
  构建行 → `grpo_update`（4/8 步）→ 漂移探针 → 护栏
- 评估阶段：冻结 vs 候选各跑 46 题（贪心）
- 影子阶段：更新流抽 20 题配对重跑 → 全局门判决

**6.3 产物**：`protocols/ttrl_seed*.json`——`update_phase`（每回合 task_id /
success / turns / calls / conflicts / halt / rows / tokens / loss / drift）、
`eval`（逐题 frozen vs candidate）、`gate`（decision / lcb_gain / n_shadow /
reason_codes）、`config`；全量带 SHA256SUMS。

---

## 7. 辅助模块（scripts/）

`calibrate.py`（难度校准，初始成功率准入）；`gate_demo.py`（投毒适配器对抗测试）；
`success_replay.py` / `train_from_trajectories.py`（正样本回放）；
`failure_taxonomy.py`（失败模式分类）；`diagnostics.py` / `drift_analysis.py`
（诊断聚合）；`make_figures.py`（图表）。

---

## 8. 画图建议（架构图直接用）

**推荐布局：横向模块流水 + 三行分区**

**第一行（顶部）**：`τ²-bench 环境`（框内小字：Tau2Episode / step→回执 /
evaluate→隐藏 DB 判分）

**第二行（中部，从左到右）**：
1. `Rollout Loop`（贪心 / ≤20 轮 / 工具 schema 注入 / XML 解析）
2. `E_hard Receipts`（ok / error / output）
3. `Local Gate`（三条冲突规则 → credit 置零 / DriftMonitor 熔断）
4. `Group Credit`（identify / read / modify / stop；失败感知 ±0.5 / −0.3）
5. `Training Rows`（按回合聚合，优势=组信用均值，0 优势丢弃）
6. `LoRA Update`（GRPO 变体 / KL β=0.1 / AdamW lr 5e-6 / 4~8 步）
7. `Drift Probe`（top-50 logp 差 / 护栏 >2.0 减半）

**第三行（底部）**：`Shadow Eval`（20 题配对）→ `Global Gate`（EB e-process，
α=0.05）→ `COMMIT / ROLLBACK` 菱形 → `Sealed Eval`（46 题，frozen vs candidate）

**箭头标注（直接抄）**：
- 环境 → Rollout：任务指令 + 政策 + 工具 schema
- Rollout → Receipts：每次工具调用
- Receipts → Local Gate：E_hard 证据 vs 行为
- Local Gate → Credit：冲突组置零（abstain）
- Credit → Rows：回合聚合
- Rows → Update：优势加权策略梯度
- Update → Drift：行为漂移测量（循环回 Rollout）
- Update → Shadow：候选 vs 父版本
- Shadow → Gate：gain / harm 差值
- Gate → Sealed Eval：COMMIT 部署候选 / ROLLBACK 保持冻结

**可选泳道版（时序图）**：四条泳道——策略 / 环境 / 门控 / 训练器；时序：生成 →
执行 → 回执 → 冲突检测 → 信用 → 训练行 → 前向反向 → 漂移 → 下一回合。

**建议配色**：安全门控用暖色（橙/琥珀）与训练流程（蓝灰）区分；环境与评估用
绿色/红色；GPU 布局用小标签挂在 Rollout（GPU1 推理）和 LoRA Update（GPU0 训练）
两个框上。

---

## 9. 关键参数总表

| 参数 | 值 | 说明 |
|------|-----|------|
| LoRA | r=16, α=32, dropout 0.05 | 7 个投影模块 |
| lr | 5e-6（护栏减半，下限 1e-6） | AdamW |
| 步数 | 4 / 成功 8 | 每回合 |
| KL β | 0.1 | 单边、动作 span |
| 梯度裁剪 | 1.0 | clip_grad_norm |
| 漂移护栏阈值 | 2.0 | >2.0 减半 |
| 冲突熔断 | window 5 / threshold 3 | DriftMonitor |
| 全局门 | α=0.05, ε_gain=0.01, ε_harm=0.10, λ=0.5, n=512 | 覆盖模拟器冻结 |
| 回合上限 / 单轮 token | 20 轮 / 384 | rollout |
| 流划分 | 68 更新 / 46 密封 | seed 打乱 |
| 影子样本 | 20 题配对 | 主实验实配 |
| 显存峰值 | 20.5GB / 32GB | 双模型 + 优化后 |
