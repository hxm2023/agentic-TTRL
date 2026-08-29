# agent-ttrl2 — Two-Scale Safety-Gated Test-Time RL for Tool-Using Agents

Deployment-period policy updates that measurably improve subsequent task
performance in a stateful tool-using agent, protected by a two-scale safety
mechanism:

- **LOCAL gate** — evidence-gated action credit: E_hard/E_soft conflict
  detection (repeated failure, modify-before-identity, user mismatch) abstains
  conflicted structured action groups from the gradient (fail-closed).
- **GLOBAL gate** — empirical-Bernstein e-process commit/rollback of the
  accumulated adapter, frozen at α_total=0.05 by a 162-config coverage
  simulator (null familywise 0.000, SESOI power 0.111, strong power 0.646,
  poisoned 0.000).

## Setup

```bash
# environment (see protocols/COMPATIBILITY_PROFILE.md for the frozen profile)
conda create -n ttrl2 python=3.11 -y
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
pip install vllm==0.26.0 trl==1.10.0 peft==0.20.0 transformers flash-linear-attention openai gymnasium
pip install -e tau2-bench  # or PYTHONPATH=/path/to/tau2-bench/src

# model
hf download Qwen/Qwen3.5-4B --local-dir /data/models/Qwen3.5-4B
```

## Run

```bash
bash reproduce.sh          # full pipeline: baselines + main contrast + report
```

Key commands:

```bash
# vLLM (GPU1): see scripts/start_vllm.sh for the full flag set (5090 findings)
python scripts/run_stream.py --mode frozen --seed 0     # frozen baseline
python scripts/run_stream.py --mode bon --n 4 --seed 0  # best-of-n baseline
python scripts/run_stream.py --mode ttrl --seed 0       # main TTRL contrast
python scripts/drift_analysis.py protocols/ttrl_seed0.json  # diagnostics
```

## Architecture

```
GPU1 (vLLM): frozen base + dynamic LoRA adapter (POST /v1/load_lora_adapter)
   ^ rollouts (tau2 retail, tool-only protocol, replayable episodes)
GPU0 (trainer): episode-boundary updates
   episode -> E_hard receipts -> structured action groups (identify/read/modify)
   -> LOCAL gate (conflict abstain) -> advantage-weighted policy gradient
   (GRPO-style, KL vs frozen base) on tool-call + terminate spans -> adapter
   -> GLOBAL gate (e-process) decides commit/rollback of the accumulated chain
```

## Results

| Arm | Eval success (46 sealed tasks) | Notes |
|-----|-------------------------------|-------|
| Frozen | 0.109 (5/46) | systematic early stopping before state changes |
| Best-of-4 | 0.109 | deterministic failures; sampling does not help |
| Prompt probe | 0.109 | even a perfect instruction does not help |
| TTRL candidate (seed 0) | 0.109 (5/46) | behavior changed (25/46 tasks differ) but 0 outcome flips |
| TTRL + global gate | ROLLBACK | fail-closed at shadow n=20 (e-process lcb_gain −0.48 < 0.01) |

Honest null: episode-boundary LoRA updates change behavior (drift 0.08→4.0,
verified) but do not transfer to future-task success at this scale; the
mechanistic explanation (sparse positive signal, exploration gap) is in
TECH_REPORT.md. Artifacts: protocols/ttrl_seed0_v3.json and friends.

## Limitations

- Single-model, single-domain exploratory runs (1-2 seeds, labeled as such).
- Tool-only protocol (no user simulator); tasks requiring interaction are
  unsolvable by design and lower the ceiling.
- The global gate at the achieved shadow n (≈20) is INCONCLUSIVE → fail-closed
  ROLLBACK by design (coverage-frozen at n=512); the safety property is
  demonstrated, the deployment gain is not gated-in.
- τ²-bench data is external (MIT license); see BASELINE_SOURCES.md.

## Acknowledgment

Human-owned project; AI assistance (Claude) used for engineering and analysis,
disclosed per open-source norms.
