#!/usr/bin/env bash
# Launch frozen Qwen3.5-4B on GPU1 (vLLM, OpenAI-compatible, tool calling).
# GPU layout locked: GPU1 = rollout (~16GB), GPU0 = LoRA training. NEVER co-locate.
set -e
source /root/miniconda3/bin/activate ttrl2
export CUDA_VISIBLE_DEVICES=1
MODEL=${MODEL:-/root/autodl-tmp/models/Qwen3.5-4B}
PORT=${PORT:-8001}
nohup vllm serve "$MODEL" \
  --port "$PORT" \
  --served-model-name qwen3.5-4b \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.92 \
  --enable-lora \
  --lora-modules base=/root/autodl-tmp/models/Qwen3.5-4B \
  > /root/autodl-tmp/logs/vllm.log 2>&1 &
echo "vLLM starting on :$PORT (pid $!)"
