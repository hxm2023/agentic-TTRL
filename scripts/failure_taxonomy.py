"""Failure-mode taxonomy on the sealed eval set (frozen policy, vLLM).

Classifies each failing episode into:
- no_call_answer: the model answered without any tool call
- early_stop: 1+ successful calls, then a final answer before completion
  (reference trajectory is longer than the executed calls)
- wrong_args: at least one receipt error (bad arguments / not found)
- wrong_tool: calls succeed but the tool sequence diverges (e.g., modify
  attempted with wrong target, or calls unrelated to the reference)

Output: counts + per-task classification. Pure diagnostics (no training use).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ttrl2.env.tau2_env import Tau2Episode  # noqa: E402
from ttrl2.serving.vllm_client import ServedPolicy  # noqa: E402
from tau2.domains.retail.environment import get_environment, get_tasks  # noqa: E402


def classify(ep, result) -> str:
    receipts = ep.record.receipts
    if not receipts:
        return "no_call_answer"
    n_errors = sum(1 for r in receipts if not r.ok)
    if n_errors > 0:
        return "wrong_args"
    ref_len = len(ep.task.evaluation_criteria.actions or [])
    if result.n_tool_calls < ref_len:
        return "early_stop"
    return "wrong_tool"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--endpoint", default="http://localhost:8001/v1")
    ap.add_argument("--out", default="protocols/failure_taxonomy.json")
    args = ap.parse_args()

    env = get_environment()
    policy_doc = env.get_policy()
    tools = env.get_tools()
    tasks = get_tasks("base")
    rng = random.Random(args.seed)
    stream = sorted(tasks, key=lambda t: t.id)
    rng.shuffle(stream)
    eval_tasks = stream[68:]
    sp = ServedPolicy(args.endpoint, "qwen3.5-4b")

    results = []
    t0 = time.time()
    for i, task in enumerate(eval_tasks):
        ep = Tau2Episode(task)
        r = sp.rollout_episode(ep, policy_doc, tools, temperature=0.7, seed=0)
        mode = classify(ep, r) if not r.success else "success"
        results.append({"task_id": task.id, "mode": mode,
                        "success": r.success, "turns": r.turns,
                        "calls": r.n_tool_calls,
                        "ref_len": len(task.evaluation_criteria.actions or [])})
        print(f"[{i+1}/{len(eval_tasks)}] {task.id}: {mode}", flush=True)

    from collections import Counter
    counts = Counter(x["mode"] for x in results)
    out = {"mode": "failure_taxonomy", "seed": args.seed,
           "counts": dict(counts),
           "n": len(results),
           "successes": [x["task_id"] for x in results if x["mode"] == "success"],
           "per_task": results,
           "elapsed_s": round(time.time() - t0, 1)}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1))
    print(f"counts: {dict(counts)} -> {out_path}")


if __name__ == "__main__":
    main()
