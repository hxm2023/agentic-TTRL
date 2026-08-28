#!/usr/bin/env bash
# D1 calibration run on autodl3: vLLM on GPU1 + frozen rollout on tau2 retail.
set -e
source /root/miniconda3/bin/activate ttrl2
export PYTHONPATH=/root/autodl-tmp/tau2-bench/src:${PYTHONPATH:-}
cd /root/autodl-tmp/agent-ttrl2
mkdir -p /root/autodl-tmp/logs protocols

# 1. vLLM on GPU1 (locked layout). Flags frozen 2026-08-29 D1: 5090 needs
# triton_attn + flashinfer sampler off; Qwen3.5-4B needs qwen3_xml tool parser.
export CUDA_VISIBLE_DEVICES=1
export VLLM_USE_FLASHINFER_SAMPLER=0
if ! curl -s http://localhost:8001/v1/models > /dev/null 2>&1; then
  nohup vllm serve /root/autodl-tmp/models/Qwen3.5-4B \
    --port 8001 --served-model-name qwen3.5-4b \
    --max-model-len 32768 --gpu-memory-utilization 0.92 \
    --attention-backend triton_attn \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    > /root/autodl-tmp/logs/vllm.log 2>&1 &
fi
echo "waiting for vLLM..."
for i in $(seq 1 120); do
  if curl -s http://localhost:8001/v1/models > /dev/null 2>&1; then
    echo "vLLM ready after ${i}x5s"; break
  fi
  sleep 5
done

# 2. frozen calibration rollout
N=${N:-24}
SEED=${SEED:-0}
python scripts/calibrate.py --endpoint http://localhost:8001/v1 \
  --model qwen3.5-4b --n "$N" --seed "$SEED" --split base \
  --out "protocols/calib_run_seed${SEED}.json"
