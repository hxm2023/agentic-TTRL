"""D7-8 behavior-drift diagnostics on a ttrl run manifest.

Outputs:
- per-update drift trajectory (does behavior change accumulate? direction?)
- credit-outcome correlation (do episodes with more confident/positive credit
  end better?)
- modify-call rate over the update phase (did the model start executing
  state changes?)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", help="ttrl run JSON")
    args = ap.parse_args()

    d = json.load(open(args.manifest, encoding="utf-8"))
    up = d.get("update_phase", [])
    print(f"=== {args.manifest} ===")
    print(f"update episodes: {len(up)}")

    # 1. drift trajectory (chunked)
    drifts = [u["drift"] for u in up]
    print(f"drift: start={drifts[0]:.3f} mid={drifts[len(drifts)//2]:.3f} "
          f"end={drifts[-1]:.3f} mean={sum(drifts)/len(drifts):.3f}")

    # 2. credit-outcome correlation: rows trained vs subsequent success
    rows = [u["rows"] for u in up]
    succ = [1 if u["success"] else 0 for u in up]
    print(f"update success rate (phase): {sum(succ)}/{len(succ)} "
          f"({sum(succ)/len(succ):.3f})")
    print(f"rows trained: total={sum(rows)} mean={sum(rows)/len(rows):.1f}/episode")

    # 3. modify-call rate over the update phase
    mods = [sum(1 for t in (u.get("tool_names") or []) if "modify" in t or t in (
        "cancel_pending_order", "exchange_delivered_order_items",
        "return_delivered_order_items"))
            for u in up]
    # fallback: derive from calls count is not available; use conflicts/tool names
    n_mod = sum(mods)
    print(f"modify-group calls across update phase: {n_mod}")

    # 4. eval contrast
    ev = d.get("eval", {})
    fr, cr = ev.get("frozen_rate"), ev.get("candidate_rate")
    print(f"eval: frozen={fr:.3f} candidate={cr:.3f} "
          f"diff={cr - fr:+.3f}")

    # 5. gate
    g = d.get("gate", {})
    print(f"gate: {g.get('decision')} n={g.get('n_shadow')} "
          f"mean_gain={g.get('mean_gain'):+.3f} mean_harm={g.get('mean_harm'):+.3f} "
          f"lcb_gain={g.get('lcb_gain'):+.3f} ucb_harm={g.get('ucb_harm'):+.3f}")

    # 6. per-task eval detail
    if ev.get("frozen") and ev.get("candidate"):
        flips = [(t["task_id"], t["success"], c["success"])
                 for t, c in zip(ev["frozen"], ev["candidate"])
                 if t["success"] != c["success"]]
        print(f"eval flips (frozen->candidate): {flips}")


if __name__ == "__main__":
    main()
