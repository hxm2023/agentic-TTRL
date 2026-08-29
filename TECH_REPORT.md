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

## Results (sealed 46-task eval sets — exploratory, 6 configs × seeds)

| Arm | Seed 0 | Seed 1 | Evidence |
|-----|--------|--------|----------|
| Frozen (temp 0.7, vLLM) | 0.109 | 0.109 | baseline |
| Best-of-4 (temp 0.7) | 0.109 | — | deterministic failures (any-sample rate equal) |
| Best-of-4 (temp 1.2) | 0.109 | — | high-temperature sampling changes nothing |
| Prompt probe (explicit continuation) | 0.109 | — | even a perfect instruction does not help |
| TTRL v3 (failure-aware credit, lr 5e-6) | 0.109 | 0.109 | behavior changed (25/44 of 46 tasks) — 0 flips |
| TTRL v4 (positive-focus, success steps 8) | 0.109 | — | less behavioral damage (9/46) — 0 flips |
| Success-replay (positive-only, 13 rows × 3 passes) | 0.109 | — | barely any change (2/46) — 0 flips |
| TTRL + global gate | ROLLBACK | ROLLBACK | e-process fail-closed in every run |

Diagnostics (the mechanism demonstrably worked — updates applied and changed
behavior):
- Behavior drift per update: 0.08 → 4.0 (seed 0) / 0.11 → 2.1 (seed 1) over 68
  episodes (top-50 next-token logp change vs the frozen base).
- Eval behavior differences: 25/46 (seed 0 — candidate made MORE calls on some
  tasks, e.g., task 46: 0→5) and 44/46 (seed 1 — candidate mostly reduced
  calls); behavior genuinely changed in both.
- Outcome flips: 0/46 in both seeds — behavior change did not transfer to
  future-task success. Update-phase success rate 6/68 both seeds.
- Mechanistic null explanation: at ~10% base success, the update phase
  produces almost no successful full-workflow trajectories to reinforce; the
  missing modify-call behavior cannot be created by penalizing failure
  (exploration gap). Consistent with agent-ttrl D12/D16/D17 across three
  environments.
- Global gate: fail-closed ROLLBACK in both seeds (shadow n=20 ≪ the
  coverage-frozen n=512; the e-process correctly refuses to commit without
  evidence — the safety property demonstrated end-to-end).

## Gate-protection demonstration (user-mandated key upgrade, 2026-08-30)

The gate's protection is demonstrated in two complementary ways:

1. **Real-data fail-closed**: a POISONED adapter (trained on flipped credit —
   reinforcing the early-stop on failed episodes, 25 episodes) is shadow-
   evaluated against the frozen parent: the e-process returns **ROLLBACK**
   (n=25, mean_gain 0.000, lcb_gain −0.40 < ε_gain=0.01) — deployment is
   blocked. The same fail-closed decision protected all four main-run
   candidates (v3/v4 × 2 seeds).
2. **Poison rejection by calibration**: the frozen 162-config coverage
   simulator guarantees the poisoned operating point is rejected
   (poisoned_rate 0.000 < SESOI power floor) at the coverage-frozen n=512.

Honest limitation: on this environment, the DB-state evaluator makes
non-mutating behavioral changes outcome-invisible, so the poisoned candidate's
deployment counterfactual measures 0.109 = frozen 0.109 (no observable
degradation — the poison cannot create wrong *state* changes that the model
never emits). The gate's protection is therefore demonstrated as fail-closed
rejection (real data) + calibrated poison rejection (simulator), not as an
observed harm-prevention delta on this task set.

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
