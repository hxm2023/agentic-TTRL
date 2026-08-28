# Compatibility Profile (FROZEN 2026-08-29, D1)

Everything that can change between runs must be pinned here. Change = new entry
in docs/DECISION_LOG.md + re-verification.

## Model
- **Primary**: `Qwen/Qwen3.5-4B` (HF: 7,268,750 downloads, lastModified 2026-03-02,
  chat template + tool calling). Frozen base + LoRA adapters.
- **Fallback**: `Qwen/Qwen3-4B` (4,561,556 downloads, 2025-07-26).
- Server path: `/root/autodl-tmp/models/Qwen3.5-4B` (HF_ENDPOINT=https://hf-mirror.com,
  HF_HUB_DISABLE_XET=1).

## Environment
- **τ²-bench retail** (`sierra-research/tau2-bench` @ main, cloned 2026-08-29 depth 1).
- Data: `data/tau2/domains/retail/{tasks.json, db.json, split_tasks.json, policy.md}`
  (114 tasks: train 74 / test 40).
- **Protocol**: tool-only (agent → `apis.*` calls → hidden `evaluation_criteria` check
  on final DB state; NO user simulator in the loop). Calibrated task subset, target
  frozen initial success ∈ [0.10, 0.30].
- Max turns per episode: 20 (agent-ttrl saw 4-21 calls/task; cap at 20).

## Server / GPU layout (locked)
- autodl3: 2× RTX 5090 32GB (idle), 754GB RAM, CUDA 13.2, /root/autodl-tmp 450G.
- **GPU0 = LoRA training** (~20GB), **GPU1 = vLLM rollout** (~16GB). NEVER co-locate.

## Python env (autodl3, conda `ttrl2`, py 3.11)
- torch 2.11.0+cu130, vllm 0.26.0, trl 1.10.0, peft 0.20.0, transformers (trl 1.10
  compatible), pydantic>=2.7, numpy. Reuse agent-ttrl D10 profile.

## Two-scale gates (FROZEN — reuse, not re-derive)
- **GLOBAL** (commit/rollback): empirical-Bernstein e-process,
  α_total=0.05, ε_gain=0.01, ε_harm=0.10, n=512, λ=0.5
  (source: agent-ttrl protocols/sweep_coverage_results.json, decision D6).
- **LOCAL** (action credit): E_hard/E_soft conflict abstain + structured action groups
  (source: agent-ttrl credit/conflict_gate.py, R003).

## Network
- Server: HF/GitHub blocked. HF via hf-mirror.com; code/data via local clone + rsync.

## Rollout/eval constants
- temperature 0.7 (rollout), 0.0 (eval), max_tokens 256 (agent-ttrl D11: 128 truncates
  tool JSON; 256 verified), max episodes per task = 1 (first-attempt success).
