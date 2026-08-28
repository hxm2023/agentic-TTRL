# DECISION_LOG — agent-ttrl2 (test-time RL for tool agents, fast-value 10-day plan)

Format: Decision · Evidence · Alternative · Why rejected · Compute spent · Falsification.
Mandatory pre-read: `C:\Users\w1828\repos\agent-ttrl\phase01\EXPERIMENT_DECISION_LOG.md`
(D6 global-gate freeze, D10-D17 nulls) + agent-ttrl CLAUDE.md lessons table.

## D1 — Env + model + compatibility profile (2026-08-29)

- **Decision (env)**: τ²-bench **retail** domain (`sierra-research/tau2-bench`, 1904★,
  industry-recognized tool-agent benchmark) with a **tool-only protocol** (agent → API
  calls → hidden criteria evaluation; NO user simulator in the loop), on a
  **difficulty-calibrated task subset**. Calibration target: frozen Qwen3.5-4B initial
  success ∈ [0.10, 0.30] on the subset (constitution gate: ≤0.30 with verified headroom).
- **Evidence**: agent-ttrl D14/D15/D16/D17 — full τ²-bench retail with Qwen3-4B /
  Mistral-7B: 0-8% success (too hard, floor); CTS stream 62.5% (no headroom). Real-world
  envs sit at the floor for small models; a calibrated subset with tool-only protocol is
  the only operating point with real learning headroom for a 4B class model.
- **Decision (model)**: **Qwen/Qwen3.5-4B** (verified on HF 2026-08-29: 7.27M downloads,
  lastModified 2026-03-02, 14 siblings incl. chat_template.jinja) — frozen base + LoRA.
  Fallback: **Qwen/Qwen3-4B** (4.56M downloads, 2025-07-26).
- **Decision (server/profile)**: autodl3 (2×RTX 5090 32GB idle, 450G on /root/autodl-tmp,
  CUDA 13.2). Conda env `ttrl2` (python 3.11), deps pinned to agent-ttrl D10 validated
  profile: torch 2.11.0+cu130, trl 1.10.0, vllm 0.26.0, peft 0.20.0. Card layout locked:
  GPU0 = LoRA training (~20GB), GPU1 = vLLM rollout (~16GB); NEVER co-locate.
- **Decision (gates, reused not re-derived)**: GLOBAL gate = empirical-Bernstein
  e-process, **frozen at agent-ttrl D6 coverage-simulator values** (162-config sweep,
  protocols/sweep_coverage_results.json): α_total=0.05, ε_gain=0.01, ε_harm=0.10,
  n=512, λ=0.5 → null_rate 0.000, sesoi(0.08) 0.111 ≥ 0.10 floor, strong(0.15) 0.646,
  poisoned 0.000. LOCAL gate = E_hard/E_soft evidence-conflict abstain (port of
  agent-ttrl `credit/conflict_gate.py`) + **structured action groups** (R003 lesson:
  never free-form model branches).
- **Decision (network)**: server has no HF/GitHub access; HF via `HF_ENDPOINT=https://hf-mirror.com`,
  code/data via local clone + rsync.
- **Why rejected**: AppWorld 0.2.0 (4B floor, D14), full τ²-bench retail (floor, D15-D17),
  CTS (no headroom, D12), custom closed env (loses "industry-recognized" value carrier;
  τ²-bench subset keeps it).
- **Compute spent**: 0 GPUh (D1 so far — model download + install only).
- **Falsification**: if no task subset puts frozen success in [0.10, 0.30] with BoN-4
  headroom ≥0.10 → pivot to mock domain or custom controlled env, re-record here.
  If Qwen3.5-4B fails tool-format smoke on vLLM → fall back to Qwen3-4B, re-record.

## D1b — Calibration result (2026-08-29, autodl3 GPU1, ~0.03 GPUh)

- **Result**: frozen Qwen3.5-4B, τ²-bench retail tool-only, 24 tasks (seed-0 sample
  of base split): **success_rate = 0.167 (4/24)** — inside the [0.10, 0.30] gate.
  107s total (avg 4.5s/task). All tasks replay; hidden evaluator verified against
  reference trajectories (task 0/5/20/40 replay → True).
- **Failure-mode analysis**: dominant failure = early stopping (1-3 tool calls then
  final answer; reference trajectories use 4-10 calls). Task 5: 0 calls. Errors are
  rare (4 receipt errors across 24 tasks). → The learning signal is clear: teach the
  model to CONTINUE the identify→read→modify workflow. BoN-4 headroom estimate:
  1-(0.833)^4 ≈ 0.52 (verified at D5-6).
- **Server stack frozen**: conda ttrl2 (py3.11): torch 2.11.0+cu130, vllm 0.26.0,
  trl 1.10.0, peft 0.20.0, transformers 5.16.1, openai 3.5.0. vLLM flags (5090
  findings): `--attention-backend triton_attn`, `VLLM_USE_FLASHINFER_SAMPLER=0`
  (flashinfer wheel compiled for CUDA<12.9, rejects SM 12.x), `--tool-call-parser
  qwen3_xml` (Qwen3.5-4B emits `<tool_call>` XML, NOT hermes JSON).
- **Decision**: D1 GATE PASSED. Main-run stream: full 114-task base (60% update /
  40% eval); frozen baseline = the calibrated 0.167 (will re-measure on the full
  eval set at D5-6). BoN-4 baseline for headroom at D5-6.
- **Compute spent**: ~0.03 GPUh calibration + installs. Falsification: if BoN-4
  headroom < 0.10 at D5-6 → re-calibrate subset or add difficulty.

## D2 — (pending) framework smoke gate

- Gate: full single-task pipeline works end-to-end on GPU1+GPU0 (format → rollout →
  update → eval) before any batch.
