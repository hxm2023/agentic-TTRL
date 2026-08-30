#!/usr/bin/env bash
# v6 upgrade pipeline (2026-08-30): few-shot capability probe -> (conditional)
# TTRL with the few-shot rollout policy. The probe decides whether the
# state-changing capability is prompt-addressable (positive signal exists).
set -e
source /root/miniconda3/bin/activate ttrl2
export PYTHONPATH=/root/autodl-tmp/tau2-bench/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/autodl-tmp/agent-ttrl2
mkdir -p protocols

# 1. few-shot capability probe (GPU1 vLLM; ~8 min)
python scripts/fewshot_probe.py --out protocols/fewshot_probe.json

# 2. decision: did the example elicit state-changing calls with any success?
python - << 'PY'
import json
d = json.load(open("protocols/fewshot_probe.json"))
n_mod = d["modify_calls_total"]
ok = d["n_success"]
print(f"probe: success={d['success_rate']:.3f} modify_calls={n_mod} "
      f"tasks_with_modify={d['tasks_with_modify']}")
if n_mod > 0 and ok > 5:
    print("CAPABILITY_PRESENT")
else:
    print("CAPABILITY_ABSENT")
PY
