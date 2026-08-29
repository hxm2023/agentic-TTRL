#!/usr/bin/env bash
# Reproduce agent-ttrl2: baselines -> main contrast -> diagnostics.
# Prereqs: ttrl2 env (see README), vLLM serving qwen3.5-4b on :8001 with the
# frozen flag set (scripts/start_vllm.sh), tau2-bench on PYTHONPATH.
set -e
cd "$(dirname "$0")"
source /root/miniconda3/bin/activate ttrl2
export PYTHONPATH=/root/autodl-tmp/tau2-bench/src:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p protocols
SEED=${SEED:-0}
GPU_TRAIN=${GPU_TRAIN:-0}

echo "== D5 frozen baseline (sealed eval set) =="
python scripts/run_stream.py --mode frozen --seed "$SEED" --out "protocols/frozen_seed${SEED}.json"

echo "== D5 best-of-4 baseline =="
python scripts/run_stream.py --mode bon --n 4 --seed "$SEED" --out "protocols/bon4_seed${SEED}.json"

echo "== D7-8 main TTRL contrast =="
CUDA_VISIBLE_DEVICES="$GPU_TRAIN" python scripts/run_stream.py \
  --mode ttrl --seed "$SEED" --steps 4 --lr 1e-5 \
  --out "protocols/ttrl_seed${SEED}.json"

echo "== diagnostics =="
python scripts/drift_analysis.py "protocols/ttrl_seed${SEED}.json"

echo "== artifacts =="
ls -la protocols/ | grep -E "seed${SEED}"
echo "DONE"
