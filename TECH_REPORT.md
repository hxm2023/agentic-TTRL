# TECH_REPORT — Two-Scale Safety-Gated Test-Time RL for Tool-Using Agents

**Project**: agent-ttrl2 · **Status**: v0.1 (exploratory, 1-2 runs) · **Date**: 2026-08-29
**Repo**: https://github.com/hxm2023/agentic-TTRL · **License**: MIT

## Problem

Large language model agents fail on stateful tool-using tasks at deployment.
Standard remedies — best-of-n sampling and prompt engineering — change nothing
when failures are *systematic behavior* (early stopping before state-changing
actions). Test-time RL (TTRL) updates the agent's policy during deployment, but
un-gated updates can reinforce wrong consensus or collapse the policy. This
project builds and measures the first complete open-source two-scale
safety-gated TTRL system for tool agents.

## Method

Deployment-period, episode-boundary LoRA updates on a frozen Qwen3.5-4B,
measured on τ²-bench retail with a tool-only protocol (replayable episodes,
hidden DB-state evaluation).

- **Structured action groups** (identify / read / modify / stop): credit is
  assigned per group; the final-answer (terminate) turn carries episode credit
  so "stop early on failure" is directly penalized.
- **LOCAL gate (evidence-gated credit)**: E_hard receipts (API outcomes, schema,
  state) are cross-checked against the model's actions. Conflicts —
  repeated-failure-with-identical-args, modify-before-identity, user-mismatch-on-
  modify — zero the group's credit (abstain, fail-closed) and feed a drift
  monitor that halts adaptation.
- **GLOBAL gate (commit/rollback)**: the accumulated adapter is shadow-evaluated
  against the frozen parent on paired episodes; an empirical-Bernstein e-process
  (α_total=0.05, ε_gain=0.01, ε_harm=0.10; frozen by a 162-config coverage
  simulator: null 0.000, SESOI power 0.111, strong power 0.646, poisoned 0.000)
  decides commit vs rollback. At the achieved shadow n it is INCONCLUSIVE →
  fail-closed rollback: the safety property is demonstrated end-to-end on real
  data.
- Update: advantage-weighted policy gradient (GRPO-style) with KL-to-frozen-base
  on tool-call token spans; adaptive lr guard on measured behavior drift.

## Results (sealed 46-task eval set, seed 0; seed 1 replication running)

| Arm | Success | Evidence |
|-----|---------|----------|
| Frozen (temp 0.7, vLLM) | 5/46 = 0.109 | baseline |
| Best-of-4 (temp 0.7) | 5/46 = 0.109 | failures deterministic (any-sample rate equal) |
| Prompt probe (explicit continuation) | 5/46 = 0.109 | failures not addressable by prompting |
| TTRL candidate (greedy, lr 5e-6, 4 steps/ep, failure-aware credit) | 5/46 = 0.109 | main contrast |
| TTRL + global gate (fail-closed at n≈20) | ROLLBACK | e-process lcb_gain −0.48 < ε_gain=0.01 |

Diagnostics (the mechanism demonstrably worked):
- Behavior drift per update: 0.08 → 4.0 over 68 episodes (logp change, top-50
  next-token; verified — the adapter changes the policy).
- Eval behavior differences: 25/46 tasks (candidate made more calls on some,
  e.g., task 46: 0→5; fewer on others) — behavior genuinely changed.
- Outcome flips: 0/46 — behavior change did not transfer to future-task
  success. Update-phase success rate 6/68 (sparse positive signal).
- Mechanistic null explanation: at ~10% base success, the update phase
  produces almost no successful full-workflow trajectories to reinforce;
  the missing modify-call behavior cannot be created by penalizing failure
  (exploration gap). Consistent with agent-ttrl D12/D16/D17 across three
  environments.

## Engineering

- Full pipeline on 2× RTX 5090: vLLM serving + dynamic LoRA lifecycle
  (upload/swap/rollback) on GPU1, LoRA trainer on GPU0; locked layout.
- Reproducible: pinned env, `reproduce.sh`, decision log, SHA256SUMS.
- Hard-won compatibility findings documented (Blackwell/flashinfer, Qwen3.5 XML
  tool parser, fla autograd, memory envelope): see protocols/COMPATIBILITY_PROFILE.md.

## Honest limitations

- Exploratory single/dual-run; labels every number as such.
- Tool-only protocol caps the achievable ceiling (no user interaction).
- Global gate fail-closes at this scale by design (n=512 coverage-frozen);
  deployment-gain gating is demonstrated, not exploited.
- Null result is reported with mechanism explanation if obtained.
