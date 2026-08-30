#!/usr/bin/env bash
# Train the Llama LoRA candidate on the SAVED successful trajectories.
set -e
source /root/miniconda3/bin/activate ttrl2
export PYTHONPATH=/root/autodl-tmp/tau2-bench/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/autodl-tmp/agent-ttrl2
nohup env CUDA_VISIBLE_DEVICES=0 python scripts/train_from_trajectories.py \
  --trajectories protocols/fewshot_probe_llama_update.json \
  --passes 12 --steps 12 --lr 3e-5 \
  --format llama3_json --no-kl --fewshot \
  --model-dir /root/autodl-tmp/models/Llama-3.1-8B-Instruct \
  --out protocols/success_replay_llama.json \
  > /root/autodl-tmp/logs/success_replay_llama.log 2>&1 &
echo "trajectory training pid $!"
