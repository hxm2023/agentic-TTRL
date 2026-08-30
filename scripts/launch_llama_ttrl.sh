#!/usr/bin/env bash
# Launch the Llama-3.1-8B TTRL main run — v7 (few-shot + temp-0.5 rollouts).
# Plain-prompt Llama invents IDs -> local-gate conflicts abstain everything
# (verified 2026-08-31); the few-shot probe showed the example fixes the
# identify step (4 modify-successes). The example is a documented protocol
# choice, applied to BOTH arms; the hidden evaluator stays clean.
set -e
source /root/miniconda3/bin/activate ttrl2
export PYTHONPATH=/root/autodl-tmp/tau2-bench/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/autodl-tmp/agent-ttrl2
nohup env CUDA_VISIBLE_DEVICES=0 python scripts/run_stream.py \
  --mode ttrl --seed 0 \
  --n-update 40 --n-eval 26 \
  --steps 4 --success-steps 8 --lr 5e-6 \
  --format llama3_json --no-kl \
  --update-temp 0.5 --drift-every 5 \
  --fewshot \
  --model-dir /root/autodl-tmp/models/Llama-3.1-8B-Instruct \
  --out protocols/ttrl_llama_fewshot_seed0.json \
  > /root/autodl-tmp/logs/ttrl_llama_fewshot_seed0.log 2>&1 &
echo "TTRL llama few-shot pid $!"
