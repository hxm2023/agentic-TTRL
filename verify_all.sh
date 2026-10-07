#!/usr/bin/env bash
# One-command evidence replay for interviews: prints every headline number
# straight from the committed protocol JSONs (no GPU, ~30 s including tests).
set -eu
cd "$(dirname "$0")"
PY=$(command -v python3 || command -v python)   # Windows/Git-Bash 兼容（路径含空格，调用加引号）

echo "=== 1. protocol v2：冻结 sweep（114 任务 × 4 采样, T=0.7, thinking off） ==="
"$PY" - <<'PY'
import json, glob
rows = []
for f in sorted(glob.glob("protocols/sweeps/sweep_s4_t0.7_shard*.jsonl")):
    for line in open(f, encoding="utf-8"):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
full = [r for r in rows if len(r["runs"]) == 4]
write = [r for r in full if r["n_write_actions"] > 0]
band = [r for r in write if 0 < sum(x["success"] for x in r["runs"]) < 4]
zero = [r for r in write if sum(x["success"] for x in r["runs"]) == 0]
sat = [r for r in write if sum(x["success"] for x in r["runs"]) == 4]
ok = sum(1 for r in full for x in r["runs"] if x["success"])
n = sum(len(r["runs"]) for r in full)
print(f"  tasks={len(full)}  frozen_rate={ok/n:.3f}  band={len(band)} zero={len(zero)} sat={len(sat)}")
PY

echo
echo "=== 2. 同配置冻结基线（thinking ON, 42 个流/评测任务, k=4） vs 在线更新 ==="
"$PY" - <<'PY'
import json, glob
agg = {}
for f in glob.glob("protocols/sweeps/sweep_s4_t0.7_think_shard*.json"):
    for tid, t in json.load(open(f))["tasks"].items():
        agg.setdefault(tid, []).extend(t["runs"])
spec = json.load(open("protocols/stream_v2_seed0_short.json"))
runs = [r for t in spec["update_ids"] for r in agg.get(t, [])]
frozen = sum(1 for r in runs if r["success"]) / max(len(runs), 1)
d = json.load(open("protocols/ttrl_v2_seed0.json"))
up = d["update_phase"]
solved = sum(1 for u in up if u["success"])
print(f"  update stream (n={len(up)}): frozen={frozen:.3f}  ->  online(updated)={solved/len(up):.3f}"
      f"   delta={solved/len(up)-frozen:+.3f}")
ev = d["eval"]
print(f"  sealed eval (n={len(ev['frozen'])} greedy): frozen={ev['frozen_rate']:.3f}"
      f"  candidate={ev['candidate_rate']:.3f}  delta={ev['candidate_rate']-ev['frozen_rate']:+.3f}")
for name, path in [("update_phase", "protocols/ttrl_v2_seed0.json")]:
    dd = json.load(open(path))
    drift = [u["drift"] for u in dd["update_phase"] if "drift" in u]
    print(f"  behavior drift: {drift[0]:.3f} -> {drift[-1]:.3f} over {len(drift)} updates")
PY

echo
echo "=== 3. 密封留出任务：采样口径（transformers, k=2）——更新后的增益 ==="
"$PY" - <<'PY2'
import json, os
f = "protocols/eval_sampled_tf_band.json"
if os.path.exists(f):
    d = json.load(open(f))
    s = d["summary"]
    print(f"  frozen    = {s['frozen_k']}/{s['frozen_n']} = {s['frozen_k']/s['frozen_n']:.3f}")
    print(f"  candidate = {s['candidate_k']}/{s['candidate_n']} = {s['candidate_k']/s['candidate_n']:.3f}"
          f"   delta={(s['candidate_k']-s['frozen_k'])/s['frozen_n']:+.3f}"
          f"  gained={s['gained']} lost={s['lost']}  McNemar p={s['mcnemar_p']:.3f}")
else:
    print("  (protocols/eval_sampled_tf_band.json missing)")
PY2

echo
echo "=== 4. harness 修复的对照证据（v1 vs v2 prompt，同一批任务） ==="
"$PY" - <<'PY'
import json
d = json.load(open("protocols/diag_prompt_v2.json"))
for tid, t in d["tasks"].items():
    for arm in ("v1_prompt", "v2_prompt"):
        rs = t["arms"][arm]
        ok = sum(1 for r in rs if r["success"])
        mod = sum(r["n_modify"] for r in rs)
        print(f"  task {tid:>3} {arm:9s}: {ok}/{len(rs)} success, modify-calls={mod}")
PY

echo
echo "=== 5. 单元测试 ==="
"$PY" -m pytest tests/ -q 2>&1 | tail -3

echo
echo "（端到端链路复现：bash reproduce.sh；v1 对照产物：protocols/ttrl_seed*.json）"
