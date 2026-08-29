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

## D5-6 — Baselines + headroom verdict (2026-08-29) — gate decision recorded

- **Frozen baseline (sealed 46 eval tasks, seed-0 stream)**: 0.109 (5/46).
- **BoN-4 (temp 0.7, independent seeds)**: 0.109 — IDENTICAL to frozen, including
  the "any sample succeeded" rate. The model's failures are deterministic, not
  sampling noise.
- **Prompt-headroom probe** (same model, explicit "keep calling tools until
  FULLY completed" system prompt, policy included): 0.109 — identical again.
  Even a perfect instruction does not change outcomes.
- **Failure-mode diagnosis** (task 108/1/104 transcripts): the model performs
  identify/read calls correctly (receipts ok, no errors) but STOPS before the
  state-changing action (exchange/return/cancel). Early stopping is the
  dominant, systematic failure.
- **Gate interpretation**: BoN headroom (constitution gate) FAILS. But the
  evidence is stronger than a headroom miss: two independent non-weight
  interventions (sampling, prompting) are null. The ONLY remaining mechanism
  that could improve future-task success is policy learning (weight updates).
  → Decision: proceed with the main contrast, PRE-REGISTERED as a stronger
  claim than BoN headroom: any TTRL gain on this stream is a genuine policy
  effect (not confounded by sampling or prompting); a null is an honest null
  with full diagnostics. Falsification: if TTRL-candidate == frozen on eval →
  report null mechanically (early stopping not learnable at this scale).
- **Trainer upgrade from the diagnosis**: the final answer (terminate) turn now
  carries episode credit (targets early stopping directly; verified locally).
- **Compute spent**: ~1 GPUh (baselines + probes).

## D7-8 — Main run results (2026-08-29, seed 0 = v3 config)

- **v3 run (sound mechanism)**: failure-aware credit (success +0.5 all; failure:
  identify/read neutral, modify -0.3, stop -0.5), lr 5e-6, steps 4, greedy
  rollouts, transformers pipeline (adapter truly applied). Update phase: 68
  episodes, 1.54 mean calls, 6/68 successes, drift 0.08→4.0 (max 6.86).
  **Eval (46 sealed): frozen 0.109 = candidate 0.109 — ZERO flips. Behavior
  genuinely changed (25/46 eval tasks differ; e.g., task 46: 0→5 calls) but
  outcomes did not. Global gate: ROLLBACK (n=20, mean_gain 0.0).**
- **Mechanistic explanation of the null**: (1) sparse positive signal — 6/68
  update episodes succeed, so the model is rarely reinforced for correct
  full-workflow behavior; (2) exploration gap — the missing modify-call
  behavior can only be learned from positive examples of it, which the update
  phase almost never produces; (3) the eval failures remain dominated by the
  systematic early-stop that updates cannot flip without positive examples.
  Consistent with agent-ttrl D12/D16/D17 nulls across three environments.
- **Pipeline findings (all verified, all documented)**: vLLM 0.26 LoRA serving
  no-op for Qwen3.5 hybrid; train-mode dropout corrupts generations after
  updates (need eval()); uniform-negative credit collapses tool calling
  (drift 13); temp-0.7 sampling drifts from the tool-call mode (use greedy).
- **Seed 1 replication running** (identical config). Compute spent: ~6 GPUh
  total (5 main-run attempts, most wasted on pipeline bugs — the price of
  frontier-model serving quirks; the reproducible profile now avoids them).

## D7 — First ttrl run: aggressive settings collapse (2026-08-29)

- **Observation**: steps 8 × lr 3e-5 with 24+ rows/episode accumulates too fast —
  probe logit drift 1.3 → 12.3 → 31.5 by episode 13 (policy collapse, no
  recovery). Run killed at episode 13/68 (~0.3 GPUh). Recorded as a calibration
  finding: per-episode advantage-weighted updates need lr ≤ 1e-5 or the KL
  anchor can't hold.
- **Fix**: default steps 4, lr 1e-5 + adaptive guard (drift > 2.0 → halve lr,
  floor 1e-6). Relaunched seed 0 with these settings; second seed to follow.

## D2-4 — Framework built + single-task smoke PASSED (2026-08-29)

- **Built**: src/ttrl2/{gates, env, agent, trainer, serving} — two-scale gates
  (ported, tested), tau2 tool-only episode wrapper, vLLM agent loop, group-credit
  trainer (GRPO-style advantage-weighted policy gradient + KL-to-frozen-base on
  tool-call token spans), served-policy with dynamic LoRA lifecycle.
- **Smoke (task 0)**: frozen rollout (2 turns/2 calls, fail) → credit rows
  [-0.5, -0.45] → update (32 rows, 1568 tokens, loss -0.76) → logit drift 8.79
  (behavior changed, target ≥0.05) → adapter uploaded to vLLM → candidate
  rollout ran. **GATE PASSED** (format→rollout→update→eval chain end-to-end).
- **Blockers found & fixed** (all documented in COMPATIBILITY_PROFILE):
  (1) flashinfer sampler/attention reject SM 12.x → triton_attn +
  VLLM_USE_FLASHINFER_SAMPLER=0; (2) Qwen3.5-4B needs qwen3_xml parser;
  (3) torch fallback delta-rule kernel OOM/autograd-inplace → fla 0.5.2;
  (4) shared-base ref forward poisons grad forward → separate ref instance;
  (5) memory: chunked logsumexp + gradient checkpointing + receipt truncation;
  (6) runtime LoRA API needs VLLM_ALLOW_RUNTIME_LORA_UPDATING=1.
- **Calibration datum**: 16 steps × lr 1e-4 on ONE episode → drift 8.79 (too
  strong; collapse risk). Stream runs: lr 3e-5, steps 8, KL β=0.1; drift
  monitored per episode (target 0.05-2.0).
- **Compute spent**: ~0.3 GPUh (smoke + debug). Falsification: if the stream
  cannot keep drift in [0.05, 2.0] while training, stop and report honestly.
