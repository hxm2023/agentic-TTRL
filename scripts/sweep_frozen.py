"""Frozen-policy success sweep over all retail tasks (learnable-band analysis).

Protocol v2 (2026-10-06): the agent prompt carries the user's actual request
(`reason_for_call`), a scripted user answers confirmation questions, and
Qwen3.5 thinking is disabled. Write-task success requires the correct DB state.

Each task gets K sampled rollouts at temperature T; per-task results are
appended to a JSONL as they finish (crash-safe, resumable: tasks already in
the JSONL are skipped). Shard the task list across parallel processes.

Usage (server):
  python scripts/sweep_frozen.py --shard 0/4 --samples 4 \
      --endpoint http://localhost:8002/v1 --out-dir protocols/sweeps
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ttrl2.env.tau2_env import Tau2Episode  # noqa: E402
from ttrl2.serving.vllm_client import ServedPolicy  # noqa: E402
from tau2.domains.retail.environment import get_environment, get_tasks  # noqa: E402

MODIFY_TOOLS = {
    "cancel_pending_order", "exchange_delivered_order_items",
    "modify_pending_order_address", "modify_pending_order_items",
    "modify_pending_order_payment", "modify_user_address",
    "return_delivered_order_items",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--endpoint", default="http://localhost:8002/v1")
    ap.add_argument("--base-model", default="qwen3.5-4b")
    ap.add_argument("--out-dir", default="protocols/sweeps")
    ap.add_argument("--task-ids", default=None,
                    help="comma list to restrict the sweep (default: all)")
    ap.add_argument("--tag", default="", help="filename tag (config marker)")
    args = ap.parse_args()

    si, sn = (int(x) for x in args.shard.split("/"))
    env = get_environment()
    policy_doc = env.get_policy()
    tools = env.get_tools()
    tasks = sorted(get_tasks("base"), key=lambda t: int(t.id))
    if args.task_ids:
        wanted = {x.strip() for x in args.task_ids.split(",") if x.strip()}
        tasks = [t for t in tasks if t.id in wanted]
        # keep shard indices stable w.r.t. the full id order
        tasks = [t for i, t in enumerate(tasks) if i % sn == si]
    else:
        tasks = [t for i, t in enumerate(tasks) if i % sn == si]
    shard = tasks

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / f"sweep_s{args.samples}_t{args.temperature}{args.tag}_shard{si}of{sn}.jsonl"
    done: dict[str, dict] = {}
    if jsonl.exists():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
                done[rec["task_id"]] = rec
            except json.JSONDecodeError:
                pass
    print(f"shard {args.shard}: {len(done)}/{len(shard)} tasks already done",
          flush=True)

    sp = ServedPolicy(args.endpoint, args.base_model)
    t0 = time.time()
    for task in shard:
        if task.id in done:
            continue
        n_write = sum(1 for a in (task.evaluation_criteria.actions or [])
                      if a.name in MODIFY_TOOLS)
        runs = []
        for k in range(args.samples):
            try:
                ep = Tau2Episode(task)
                r = sp.rollout_episode(ep, policy_doc, tools,
                                       temperature=args.temperature,
                                       max_tokens=args.max_tokens, seed=k)
                names = [tc["name"] for e in r.transcript
                         for tc in (e.tool_calls or [])]
                runs.append({"success": bool(r.success), "turns": r.turns,
                             "n_calls": r.n_tool_calls,
                             "n_modify": sum(1 for n in names
                                             if n in MODIFY_TOOLS),
                             "calls": names})
                print(f"[{task.id}] sample {k}: success={r.success} "
                      f"modify={runs[-1]['n_modify']} "
                      f"n_calls={r.n_tool_calls}", flush=True)
            except Exception as e:  # noqa: BLE001 — one bad sample != dead shard
                runs.append({"success": False, "error": str(e)[:200],
                             "n_calls": 0, "n_modify": 0, "calls": []})
                print(f"[{task.id}] sample {k}: ERROR {str(e)[:120]}", flush=True)
        rec = {"task_id": task.id, "n_write_actions": n_write, "runs": runs}
        done[task.id] = rec
        with open(jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    results = {tid: {"n_write_actions": r["n_write_actions"], "runs": r["runs"]}
               for tid, r in done.items()}
    out = {"mode": "sweep_frozen", "shard": args.shard, "samples": args.samples,
           "temperature": args.temperature, "tasks": results,
           "elapsed_s": round(time.time() - t0, 1)}
    out_path = out_dir / f"sweep_s{args.samples}_t{args.temperature}{args.tag}_shard{si}of{sn}.json"
    out_path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    n_ok = sum(1 for t in results.values() for r in t["runs"] if r["success"])
    n_all = sum(len(t["runs"]) for t in results.values())
    print(f"shard {args.shard}: {n_ok}/{n_all} success -> {out_path}")


if __name__ == "__main__":
    main()
