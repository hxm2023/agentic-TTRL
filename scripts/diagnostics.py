"""Formal diagnostics for a ttrl run manifest (D7-8 deliverables).

1. Learning curve: update-phase success by stream third (early/mid/late).
2. Credit-outcome correlation: does positive-credit (successful) training
   associate with lower subsequent drift (behavioral stability)?
3. Gate ablation: candidate deployed unconditionally (gate OFF) vs
   gate ON (ROLLBACK -> frozen deployed); the contrast + the decision.
4. Behavior-change vs outcome-change table (eval).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    args = ap.parse_args()

    d = json.load(open(args.manifest, encoding="utf-8"))
    up = d["update_phase"]
    n = len(up)
    thirds = [up[i * n // 3:(i + 1) * n // 3] for i in range(3)]
    print(f"== {args.manifest} ==")
    print("update-phase success by third:",
          [f"{sum(1 for u in t if u['success'])}/{len(t)}" for t in thirds])

    # credit-outcome: drift after episodes with vs without positive credit
    succ_ep = [u for u in up if u["success"]]
    fail_ep = [u for u in up if not u["success"]]
    print(f"successes: {len(succ_ep)}/68 (positive-credit episodes)")
    if len(succ_ep) >= 2:
        # drift trajectory around successful episodes
        idxs = [i for i, u in enumerate(up) if u["success"]]
        post = [up[min(i + 1, n - 1)]["drift"] for i in idxs]
        print(f"drift right after successes: {[round(x, 2) for x in post]}")
    print(f"drift: first={up[0]['drift']:.2f} last={up[-1]['drift']:.2f} "
          f"max={max(u['drift'] for u in up):.2f}")

    # gate ablation
    ev = d["eval"]
    g = d["gate"]
    fr, cr = ev["frozen_rate"], ev["candidate_rate"]
    print(f"\nGATE ABLATION: gate OFF (candidate deployed) = {cr:.3f}; "
          f"gate ON (ROLLBACK -> frozen deployed) = {fr:.3f}; "
          f"decision={g['decision']} at n={g['n_shadow']} "
          f"(mean_gain={g['mean_gain']:+.3f})")

    # behavior vs outcome flips
    ef, ec = ev["frozen"], ev["candidate"]
    beh = sum(1 for t, c in zip(ef, ec) if t["turns"] != c["turns"] or t["calls"] != c["calls"])
    outc = sum(1 for t, c in zip(ef, ec) if t["success"] != c["success"])
    print(f"eval: behavior-changed {beh}/46, outcome-changed {outc}/46")
    fr_mean = np.mean([t["turns"] for t in ef])
    cr_mean = np.mean([t["turns"] for t in ec])
    print(f"eval mean turns: frozen={fr_mean:.2f} candidate={cr_mean:.2f}")


if __name__ == "__main__":
    main()
