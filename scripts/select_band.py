"""Aggregate the frozen sweep into a learnable band + pre-registered stream.

Protocol v2 (2026-10-06). Task strata by frozen success frequency (K samples):
- band      : 0 < p_hat < 1  — mixed outcomes; the only stratum with contrast
              for the update to learn from and measured headroom.
- zero      : p_hat = 0      — hard tasks (negative-signal training; regression check)
- saturated : p_hat = 1      — solved; regression check only
Read-only tasks (no write action) are excluded from all strata: with a DB-only
evaluator they are free wins and carry no signal.

Pre-registration: the update/eval split is fixed here by --seed and written to
a JSON that run_stream.py --stream-file consumes.

Usage:
  python scripts/select_band.py --sweep 'protocols/sweeps/sweep_s4_t0.7_shard*.json' \
      --seed 0 --n-update 24 --n-eval 12 --n-zero-update 16 --n-zero-eval 6 \
      --n-sat-eval 6 --out protocols/stream_v2_seed0.json
"""
from __future__ import annotations

import argparse
import glob
import json
import random
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", nargs="+", required=True, help="sweep json globs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-update", type=int, default=24, help="band -> update")
    ap.add_argument("--n-eval", type=int, default=12, help="band -> held-out eval")
    ap.add_argument("--n-zero-update", type=int, default=16,
                    help="zero-success tasks added to the update stream")
    ap.add_argument("--n-zero-eval", type=int, default=6,
                    help="zero-success tasks held out (regression check)")
    ap.add_argument("--n-sat-eval", type=int, default=6,
                    help="saturated tasks held out (no-regression check)")
    ap.add_argument("--out", default="protocols/stream_v2_seed0.json")
    args = ap.parse_args()

    files = []
    for pat in args.sweep:
        files.extend(sorted(glob.glob(pat)))
    assert files, "no sweep files matched"

    agg: dict[str, dict] = {}
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        for tid, t in d["tasks"].items():
            a = agg.setdefault(tid, {"n_write_actions": t["n_write_actions"],
                                     "runs": []})
            assert a["n_write_actions"] == t["n_write_actions"]
            a["runs"].extend(t["runs"])

    band, zero, saturated = [], [], []
    for tid, t in sorted(agg.items(), key=lambda kv: int(kv[0])):
        k = len(t["runs"])
        s = sum(1 for r in t["runs"] if r["success"])
        t["p_hat"] = s / k
        t["n_success"] = s
        t["n_samples"] = k
        if t["n_write_actions"] == 0:
            continue
        if s == 0:
            zero.append(tid)
        elif s == k:
            saturated.append(tid)
        else:
            band.append(tid)

    rng = random.Random(args.seed)
    rng.shuffle(band)
    rng.shuffle(zero)
    rng.shuffle(saturated)

    n_bu = min(args.n_update, max(0, len(band) - 1))
    update_band = band[:n_bu]
    eval_band = band[n_bu:n_bu + args.n_eval]
    update_zero = zero[:args.n_zero_update]
    eval_zero = zero[args.n_zero_update:args.n_zero_update + args.n_zero_eval]
    eval_sat = saturated[:args.n_sat_eval]

    update_ids = update_band + update_zero
    eval_ids = eval_band + eval_zero + eval_sat
    out = {
        "protocol": "v2_fixed_prompt",
        "seed": args.seed,
        "sources": files,
        "band": band,
        "band_n": len(band),
        "zero_success": zero,
        "saturated": saturated,
        "update_ids": update_ids,
        "eval_ids": eval_ids,
        "update_band": update_band,
        "eval_band": eval_band,
        "update_zero": update_zero,
        "eval_zero": eval_zero,
        "eval_saturated": eval_sat,
        "per_task": {tid: {"p_hat": agg[tid]["p_hat"],
                           "n_success": agg[tid]["n_success"],
                           "n_samples": agg[tid]["n_samples"],
                           "n_write_actions": agg[tid]["n_write_actions"]}
                     for tid in sorted(agg, key=int)},
    }
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"write tasks: band={len(band)} zero={len(zero)} sat={len(saturated)}")
    print(f"update: {len(update_ids)} (band {len(update_band)} + zero "
          f"{len(update_zero)})")
    print(f"eval:   {len(eval_ids)} (band {len(eval_band)} + zero {len(eval_zero)}"
          f" + sat {len(eval_sat)})")


if __name__ == "__main__":
    main()
