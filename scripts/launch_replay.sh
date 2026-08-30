#!/usr/bin/env bash
# Launch the Llama-3.1-8B success-replay (valid positive modify examples).
set -e
source /root/miniconda3/bin/activate ttrl2
export PYTHONPATH=/root/autodl-tmp/tau2-bench/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/autodl-tmp/agent-ttrl2
nohup env CUDA_VISIBLE_DEVICES=0 python scripts/success_replay.py \
  --seed 0 --ids 62,10,105,67 \
  --passes 12 --steps 12 --lr 3e-5 \
  --format llama3_json --no-kl --fewshot --update-temp 0.7 \
  --model-dir /root/autodl-tmp/models/Llama-3.1-8B-Instruct \
  --out protocols/success_replay_llama.json \
  > /root/autodl-tmp/logs/success_replay_llama.log 2>&1 &
echo "llama replay pid $!"
