#!/usr/bin/env bash
# After seed 0 finishes, run an independent replicate: a fresh band split
# (--seed 1) and a fresh TTRL chain on GPU1.
set -u
cd /datac/ttrl2/agent-ttrl2
for i in $(seq 1 480); do
  if [ -f protocols/ttrl_v2_seed0.json ]; then
    echo "[seed1] seed 0 done $(date)"
    break
  fi
  if ! pgrep -f 'run_strea[m].py' > /dev/null; then
    echo "[seed1] seed0 process gone without output — aborting $(date)"
    exit 1
  fi
  sleep 120
done
[ -f protocols/ttrl_v2_seed0.json ] || { echo "[seed1] timeout"; exit 1; }
bash scripts/ttrl_v2_driver.sh band 1
bash scripts/ttrl_v2_driver.sh ttrl 1
echo "[seed1] launched $(date)"
