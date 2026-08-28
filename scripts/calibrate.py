"""D1 calibration: frozen-policy initial success on tau2-bench retail subsets.

Measures per-task success of the frozen model so we can choose the
difficulty-calibrated subset with initial success in [0.10, 0.30].

Usage:
  python scripts/calibrate.py --endpoint http://localhost:8001/v1 \
      --model qwen3.5-4b --n 24 --seed 0 --out protocols/calib_run1.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ttrl2.agent.loop import rollout  # noqa: E402
from ttrl2.env.tau2_env import Tau2Episode  # noqa: E402
from tau2.domains.retail.environment import get_environment, get_tasks  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://localhost:8001/v1")
    ap.add_argument("--model", default="qwen3.5-4b")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="base", help="train|test|base|all")
    ap.add_argument("--task-ids", default=None, help="explicit comma-separated ids")
    ap.add_argument("--out", default="protocols/calib_run.json")
    args = ap.parse_args()

    tasks = get_tasks(args.split)
    if args.task_ids:
        ids = {x.strip() for x in args.task_ids.split(",")}
        tasks = [t for t in tasks if t.id in ids]
    rng = random.Random(args.seed)
    tasks = rng.sample(tasks, min(args.n, len(tasks)))

    env = get_environment()
    policy = env.get_policy()
    tools = env.get_tools()
    client = OpenAI(base_url=args.endpoint, api_key="EMPTY")

    results = []
    t0 = time.time()
    for i, task in enumerate(tasks):
        ep = Tau2Episode(task)
        r = rollout(client, args.model, ep, policy, tools,
                    max_turns=20, max_tokens=256, temperature=0.7, seed=args.seed)
        results.append({
            "task_id": task.id,
            "success": r.success,
            "turns": r.turns,
            "n_tool_calls": r.n_tool_calls,
            "n_receipt_errors": sum(1 for x in ep.record.receipts if not x.ok),
            "state": ep.snapshot_state(),
        })
        print(f"[{i+1}/{len(tasks)}] {task.id}: "
              f"success={r.success} turns={r.turns} calls={r.n_tool_calls}",
              flush=True)

    n_ok = sum(1 for r in results if r["success"])
    out = {
        "profile": "frozen Qwen3.5-4B, tau2 retail, tool-only protocol, temp 0.7",
        "endpoint": args.endpoint, "model": args.model,
        "n_tasks": len(results), "n_success": n_ok,
        "success_rate": n_ok / len(results) if results else 0.0,
        "elapsed_s": round(time.time() - t0, 1),
        "per_task": results,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1))
    print(f"success_rate={out['success_rate']:.3f} ({n_ok}/{len(results)}) "
          f"-> {out_path}")


if __name__ == "__main__":
    main()
