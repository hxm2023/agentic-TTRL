# Resume Bullets (大模型后训练算法工程师)

## CORE PROJECT — Test-Time RL for Tool-Using Agents (Two-Scale Safety-Gated)

**One-liner**: 开源实现并测度了首个双尺度安全门控的 Agent 测试时强化学习系统
（τ²-bench 零售域 × Qwen3.5-4B，LoRA 更新，经验-Bernstein e-process 提交门），
端到端可复现（reproduce.sh + 决策日志 + SHA256SUMS）。

**Bullets（中英混排，按需选取）**:

1. **工程**: 从零搭建 agent 测试时 RL 全栈 —— vLLM 服务 + 动态 LoRA 适配器生命周期
   (load/swap/rollback)、逐回合边界 LoRA 更新 (peft)、隐藏评估器隔离的
   replayable 环境 (τ²-bench retail, 114 tasks)、2×RTX 5090 锁定的双卡架构
   (GPU0 训练 / GPU1 服务)，单任务全链路 smoke 门禁后放量。

2. **算法**: 提出双尺度安全门控 —— 局部门（E_hard/E_soft 证据冲突检测 →
   结构化动作组（identify/read/modify）信用置零，fail-closed）+ 全局门
   （empirical-Bernstein e-process 提交/回滚，α=0.05，由 162 配置覆盖模拟器
   冻结：零假设家族错误率 0.000、SESOI 功效 0.111、强功效 0.646、投毒 0.000）。

3. **测量与诊断（诚实实验）**: 冻结基线 0.109（46 任务密封集）→ BoN-4
   （temp 0.7 与 1.2）与强提示探针均无提升（0.109，失败为确定性早停，非
   采样噪声）→ TTRL 主对比 6 配置 × seeds：行为漂移可测（logp 漂移
   0.08→4.0）、eval 行为差异 25/44 任务、但未来任务成功零翻转；失败模式
   分类学（early_stop 50% / wrong_tool 20% / wrong_args 15%）；全局门
   在 n≈20 正确 fail-closed（ROLLBACK）。

4. **机制分析（结构性发现）**: 强成功回放实验逐条检查所有正样本轨迹——
   4B 模型在该环境上的"成功"样本从不包含状态变更（modify）调用（~400
   集测量、零证据冲突佐证），TTRL 更新对缺失行为的正信号结构性不可得
   （探索缺口绝对化），据此给出正结果配方（更强的基座/软评估器/密集正
   信号）。few-shot 能力探测（已构建，待跑）检验该能力是否提示可寻址。

4. **调试能力**: 解决前沿模型服务链上的硬问题 —— Blackwell/flashinfer 不兼容
   (SM12.x 需要 CUDA≥12.9 → triton_attn)、Qwen3.5 XML 工具解析器、混合架构
   LoRA 反向传播 in-place 崩溃 (flash-linear-attention)、共享基座参考模型
   毒化梯度前向、vLLM 0.26 LoRA 服务对混合模型无效的验证与绕行（transformers
   直接 rollout）、训练模式 dropout 污染生成、统一负向信用导致工具调用坍缩的
   修复（failure-aware credit）。

5. **工程规范**: 决策日志（每次决策: 证据/替代方案/拒绝原因/可证伪条件）、
   兼容性档案冻结、诚实标签（exploratory, single-config）、reproduce.sh、
   SHA256SUMS、GitHub 持续 code-first 推送。

## 支撑项目（如需）

- **agent-ttrl**（前序）: 证据分层 (E_hard/E_soft) + 预序协议 + 三通道预算账本 +
  安全提交 (SafeCommit)；三环境 (CTS/AppWorld/τ²-bench) × 两模型族 (Qwen3-4B,
  Mistral-7B) 的诚实空结果综合与复现框架。
