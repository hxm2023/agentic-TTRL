#!/usr/bin/env bash
# TTRL protocol-v2 driver (wxh-server): serve -> diag -> frozen sweep ->
# band selection -> TTRL run. Each stage is idempotent and resumable.
#
#   bash scripts/ttrl_v2_driver.sh serve         # vLLM on GPU1 :8002
#   bash scripts/ttrl_v2_driver.sh diag          # v1-vs-v2 prompt probe
#   bash scripts/ttrl_v2_driver.sh sweep         # 4-shard frozen sweep (K=4)
#   bash scripts/ttrl_v2_driver.sh band 0        # select band, seed 0
#   bash scripts/ttrl_v2_driver.sh ttrl 0        # main TTRL run, seed 0
set -u
cd /datac/ttrl2/agent-ttrl2
export PYTHONPATH=/datac/ttrl2/src:$PWD/src
export TAU2_DATA_DIR=/datac/ttrl2/data
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODEL=/datac/posttrain-paper/models/Qwen/Qwen3.5-4B
ENDPOINT=http://localhost:8002/v1
PY=python3
VLLM=$HOME/.local/bin/vllm

case "${1:-}" in
  serve)
    pkill -f "vllm serve [/]datac/posttrain-paper/models/Qwen/Qwen3.5-4B" 2>/dev/null || true
    sleep 5
    setsid nohup "$VLLM" serve "$MODEL" --served-model-name qwen3.5-4b \
      --port 8002 --max-model-len 32768 --gpu-memory-utilization 0.85 \
      --enable-auto-tool-choice --tool-call-parser qwen3_xml \
      > /datac/ttrl2/vllm_8002.log 2>&1 < /dev/null &
    echo "vllm starting (log /datac/ttrl2/vllm_8002.log)"
    ;;
  diag)
    $PY scripts/diag_prompt_v2.py --tasks 1,108,25,30,44,65 \
      --samples 2 --endpoint "$ENDPOINT" --out protocols/diag_prompt_v2.json
    ;;
  sweep)
    for s in 0 1 2 3; do
      nohup $PY scripts/sweep_frozen.py --shard $s/4 --samples "${2:-4}" \
        --endpoint "$ENDPOINT" --out-dir protocols/sweeps \
        > /datac/ttrl2/sweep_shard$s.log 2>&1 &
    done
    echo "4 sweep shards started"
    ;;
  band)
    $PY scripts/select_band.py --sweep 'protocols/sweeps/sweep_s*_t0.7_shard*.json' \
      --seed "${2:-0}" --n-update 24 --n-eval 12 --n-zero-update 16 \
      --n-zero-eval 6 --n-sat-eval 6 \
      --out "protocols/stream_v2_seed${2:-0}.json"
    ;;
  ttrl)
    S="${2:-0}"
    STREAM="${3:-protocols/stream_v2_seed${S}.json}"
    setsid nohup $PY scripts/run_stream.py --mode ttrl --seed "$S" \
      --stream-file "$STREAM" \
      --model-dir "$MODEL" --out "protocols/ttrl_v2_seed${S}.json" \
      --steps 3 --lr 1e-5 --kl-beta 0.1 --update-temp 0.7 \
      --eval-samples 1 --gate-shadow 8 \
      > "/datac/ttrl2/ttrl_v2_seed${S}.log" 2>&1 < /dev/null &
    echo "ttrl seed $S started (log /datac/ttrl2/ttrl_v2_seed${S}.log)"
    ;;
  evalfrozen)
    S="${2:-0}"
    $PY scripts/run_stream.py --mode frozen --seed "$S" \
      --stream-file "protocols/stream_v2_seed${S}.json" \
      --endpoint "$ENDPOINT" --out "protocols/frozen_v2_seed${S}.json"
    ;;
  *)
    echo "usage: $0 {serve|diag|sweep [K]|band [seed]|ttrl [seed]|evalfrozen [seed]}"
    ;;
esac
