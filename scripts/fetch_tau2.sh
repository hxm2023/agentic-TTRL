#!/usr/bin/env bash
# Fetch the pinned tau2-bench checkout (exact commit from BASELINE_SOURCES.md).
set -e
DEST=${1:-/root/autodl-tmp/tau2-bench}
PIN=a2c024725189473d2d7cea3a5cfdbcc67478e41f
if [ ! -d "$DEST/.git" ]; then
  git clone https://github.com/sierra-research/tau2-bench.git "$DEST"
fi
cd "$DEST"
git fetch --depth 1 origin "$PIN"
git checkout "$PIN"
echo "tau2-bench pinned at $PIN -> $DEST"
echo "usage: export PYTHONPATH=$DEST/src"
