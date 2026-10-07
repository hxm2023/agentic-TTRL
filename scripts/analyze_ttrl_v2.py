"""Summarize TTRL protocol-v2 runs: frozen band stats + eval deltas.

Reads the pre-registered stream file, the frozen sweep JSONL/JSONs, and one
or more TTRL run JSONs; prints (and optionally writes) the headline numbers
with a paired McNemar test on the eval phase.

Usage:
  python scripts/analyze_ttrl_v2.py \
      --stream protocols/stream_v2_seed0.json \
      --sweep-glob 'protocols/sweeps/sweep_s*_shard*.json' \
      --ttrl protocols/ttrl_v2_seed0.json
"""
from __future__ import annotations

import argparse
import glob
import json
from math import comb


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", required=True)
    ap.add_argument("--sweep-glob", default=None)
    ap.add_argument("--ttrl", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    stream = json.load(open(args.stream, encoding="utf-8"))
    summary: dict = {"stream": args.stream, "runs": []}

    print(f"== pre-registered stream ({args.stream})")
    print(f"   band n={stream['band_n']}  update={len(stream['update_ids'])}  "
          f"eval={len(stream['eval_ids'])}")
    per_task = stream.get("per_task", {})
    for label, ids in (("update", stream["update_ids"]),
                       ("eval", stream["eval_ids"])):
        ps = [per_task[i]["p_hat"] for i in ids if i in per_task]
        if ps:
            print(f"   {label} p_hat: mean={sum(ps)/len(ps):.3f} "
                  f"min={min(ps):.3f} max={max(ps):.3f}")

    if args.sweep_glob:
        rows = []
        for pat in args.sweep_glob.split(","):
            for f in sorted(glob.glob(pat)):
                d = json.load(open(f, encoding="utf-8"))
                rows.extend(d["tasks"].values())
        if rows:
            n = sum(len(r["runs"]) for r in rows)
            k = sum(1 for r in rows for x in r["runs"] if x["success"])
            print(f"== frozen sweep: {k}/{n} = {k/n:.3f} over {len(rows)} tasks")
            summary["sweep"] = {"n": n, "k": k, "rate": k / n}

    for f in args.ttrl:
        d = json.load(open(f, encoding="utf-8"))
        ev = d["eval"]
        fr, cr = ev["frozen_rate"], ev["candidate_rate"]
        frozen = {t["task_id"]: t["success"] for t in ev["frozen"]}
        cand = {t["task_id"]: t["success"] for t in ev["candidate"]}
        b = sum(1 for t in frozen if cand[t] and not frozen[t])
        c = sum(1 for t in frozen if frozen[t] and not cand[t])
        p = mcnemar_exact(b, c)
        up = d.get("update_phase", [])
        n_up_succ = sum(1 for u in up if u.get("success"))
        drift = [u["drift"] for u in up if u.get("drift")]
        print(f"\n== TTRL run {f}")
        print(f"   update phase: {len(up)} tasks, {n_up_succ} solved by the "
              f"current policy ({n_up_succ/max(len(up),1):.3f})")
        if drift:
            print(f"   drift: first={drift[0]:.3f} last={drift[-1]:.3f} "
                  f"peak={max(drift):.2f}")
        print(f"   eval (n={len(ev['frozen'])} held-out tasks, greedy):")
        print(f"     frozen    = {fr:.4f} ({sum(frozen.values())}/{len(frozen)})")
        print(f"     candidate = {cr:.4f} ({sum(cand.values())}/{len(cand)})")
        print(f"     delta = {cr - fr:+.4f}   gained={b} lost={c}   "
              f"McNemar p={p:.4f}")

        def stratum(ids, label):
            if not ids:
                return None
            fb = sum(1 for t in ids if frozen.get(t))
            cb = sum(1 for t in ids if cand.get(t))
            print(f"     [{label:<12}] n={len(ids):>2}  frozen={fb/len(ids):.3f}"
                  f"  candidate={cb/len(ids):.3f}  delta={((cb-fb)/len(ids)):+.3f}")
            return {"n": len(ids), "frozen": fb / len(ids),
                    "candidate": cb / len(ids)}

        strata = {
            "band_heldout": stratum(stream.get("eval_band", []), "band"),
            "zero_heldout": stratum(stream.get("eval_zero", []), "zero"),
            "saturated": stratum(stream.get("eval_saturated", []), "saturated"),
        }
        print(f"   gate: {d.get('gate', {}).get('decision')} "
              f"(lcb_gain={d.get('gate', {}).get('lcb_gain'):.3f})")
        summary["runs"].append({
            "file": f, "frozen_rate": fr, "candidate_rate": cr,
            "delta": cr - fr, "gained": b, "lost": c, "mcnemar_p": p,
            "n_eval": len(frozen), "update_solved": n_up_succ,
            "n_update": len(up), "strata": strata,
            "gate": d.get("gate", {}).get("decision"),
        })

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=1)
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
