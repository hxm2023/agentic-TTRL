#!/usr/bin/env bash
# One-command evidence replay for interviews: prints every headline number
# straight from the committed protocol JSONs (no GPU, ~30 s including tests).
set -eu
cd "$(dirname "$0")"
PY=$(command -v python3 || command -v python)   # Windows/Git-Bash 兼容（路径含空格，调用加引号）

echo "=== 1. 冻结基线（46 任务密封集） ==="
"$PY" - <<'PY'
import json
d = json.load(open("protocols/frozen_seed0.json"))
print(f"  n = {d['n']}   success_rate = {d['success_rate']:.4f}  ({d['n_success']}/{d['n']})")
PY

echo
echo "=== 2. 主对比：部署期在线更新后，成功率有没有变？ ==="
"$PY" - <<'PY'
import json, glob
for f in ["protocols/ttrl_seed0.json", "protocols/ttrl_seed1_v3.json", "protocols/success_replay.json"]:
    d = json.load(open(f))
    ev = d.get("eval", {})
    g = d.get("gate", {})
    print(f"  {f:<34} frozen={ev.get('frozen_rate', 0):.4f}  candidate={ev.get('candidate_rate', 0):.4f}"
          f"  行为差异={ev.get('behavior_diff', '-')}  门控={g.get('decision', '-')}")
PY

echo
echo "=== 3. 行为漂移（逐次更新，证明更新真的在改行为） ==="
"$PY" - <<'PY'
import json, glob
for f in sorted(glob.glob("protocols/ttrl_seed*_v*.json")) + ["protocols/ttrl_seed0_greedy.json"]:
    d = json.load(open(f))
    up = d.get("update_phase") or []
    dr = [u["drift"] for u in up if isinstance(u, dict) and "drift" in u]
    if dr:
        print(f"  {f.split('/')[-1]:<26} 更新 {len(dr)} 次: {dr[0]:.3f} -> {dr[-1]:.3f}  (峰值 {max(dr):.2f})")
PY

echo
echo "=== 4. 机制发现 + 其它探针 ==="
"$PY" - <<'PY'
import json, os
for f, keys in [("protocols/success_replay_strong.json", ("mode", "passes", "n_rows")),
                ("protocols/failure_taxonomy.json", ("counts", "n")),
                ("protocols/fewshot_probe.json", ("mode", "success_rate"))]:
    if os.path.exists(f):
        d = json.load(open(f))
        print(f"  {f:<40}", {k: d.get(k) for k in keys if k in d})
PY

echo
echo "=== 5. 单元测试 ==="
"$PY" -m pytest tests/ -q 2>&1 | tail -3

echo
echo "（端到端链路复现：bash reproduce.sh）"
