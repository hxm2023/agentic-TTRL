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
- torch 2.11.0+cu130, vllm 0.26.0, trl 1.10.0, peft 0.20.0, transformers 5.16.1,
  openai 3.5.0, pydantic 2.13.4, numpy 2.3.5, gymnasium 1.3.0. Reuse agent-ttrl D10
  profile. tau2-bench via PYTHONPATH=/root/autodl-tmp/tau2-bench/src (its pyproject
  requires py>=3.12; editable install rejected, PYTHONPATH used instead).

## vLLM server flags (FROZEN 2026-08-29 D1 — 5090/Blackwell + Qwen3.5 findings)
- `--attention-backend triton_attn` — REQUIRED: flashinfer wheel (0.6.14 from
  aliyun mirror) is compiled for CUDA < 12.9; its JIT check rejects SM 12.x.
- `VLLM_USE_FLASHINFER_SAMPLER=0` — REQUIRED for the same reason (sampler path).
- `--tool-call-parser qwen3_xml` — REQUIRED: Qwen3.5-4B chat template emits
  `<tool_call>` XML blocks (NOT Hermes JSON). Verified: hermes produces no calls.
- `--enable-auto-tool-choice`, `--max-model-len 32768`, `--gpu-memory-utilization 0.92`.

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
