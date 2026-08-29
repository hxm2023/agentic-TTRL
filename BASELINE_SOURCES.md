# Baseline & Data Sources

## τ²-bench (environment)
- Repo: `sierra-research/tau2-bench` (https://github.com/sierra-research/tau2-bench),
  MIT license, cloned 2026-08-29 (depth 1, main).
- Used: `data/tau2/domains/retail/{tasks.json, db.json, split_tasks.json,
  policy.md}` (114 tasks: train 74 / test 40) and `src/tau2/` (environment,
  tools, evaluator).
- Protocol modification: tool-only episodes (no user simulator). Documented in
  protocols/COMPATIBILITY_PROFILE.md.

## Model
- `Qwen/Qwen3.5-4B` (HuggingFace, 2026-03-02). Frozen base + LoRA adapters
  trained in this project. Fallback: `Qwen/Qwen3-4B`.

## Coverage simulator (global gate freeze)
- Parameters α_total=0.05, ε_gain=0.01, ε_harm=0.10, n=512 inherited from
  `agent-ttrl` (sibling project) `protocols/sweep_coverage_results.json`
  (decision D6, 2026-08-22, 162-config sweep). See
  `src/ttrl2/gates/global_gate.py` for the numbers and the source reference.
