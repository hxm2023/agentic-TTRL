"""Few-shot capability probe (2026-08-30): can a worked example elicit the
state-changing (modify) behavior the 4B model never emits?

The structural finding says the model's successes never contain modify calls.
This probe adds ONE worked exchange workflow (from task 0's reference — a
different task; legitimate few-shot, does not leak the eval tasks' targets)
to the system prompt and measures: modify-call rate + success on the sealed
eval set. If the capability is prompt-addressable, the TTRL positive-signal
story changes (the rollout policy can emit modify calls -> learnable).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ttrl2.agent.loop import rollout  # noqa: E402
from ttrl2.env.tau2_env import Tau2Episode  # noqa: E402
from ttrl2.serving.vllm_client import ServedPolicy  # noqa: E402
from tau2.domains.retail.environment import get_environment, get_tasks  # noqa: E402

MODIFY_TOOLS = {"cancel_pending_order", "exchange_delivered_order_items",
                "modify_pending_order_address", "modify_pending_order_items",
                "modify_pending_order_payment", "modify_user_address",
                "return_delivered_order_items"}


def example_trajectory() -> str:
    """One worked exchange workflow from task 0's reference (few-shot only)."""
    tasks = get_tasks("base")
    t0 = next(t for t in tasks if t.id == "0")
    lines = ["Here is an example of completing a similar request:"]
    for a in (t0.evaluation_criteria.actions or []):
        args = ", ".join(f"{k}={v}" for k, v in a.arguments.items())
        lines.append(f"  apis.{a.name}({args})")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--endpoint", default="http://localhost:8001/v1")
    ap.add_argument("--model", default="qwen3.5-4b")
    ap.add_argument("--n", type=int, default=46)
    ap.add_argument("--pool", choices=["eval", "update"], default="eval",
                    help="which stream segment to probe")
    ap.add_argument("--out", default="protocols/fewshot_probe.json")
    args = ap.parse_args()

    env = get_environment()
    policy_doc = env.get_policy()
    tools = env.get_tools()
    tasks = get_tasks("base")
    rng = random.Random(args.seed)
    stream = sorted(tasks, key=lambda t: t.id)
    rng.shuffle(stream)
    if args.pool == "eval":
        pool_tasks = stream[68:68 + args.n]
    else:
        pool_tasks = stream[:args.n]
    sp = ServedPolicy(args.endpoint, args.model)
    example = example_trajectory()

    sys_prompt = (f"You are a retail customer service agent.\n\n"
                  f"Policy:\n{policy_doc}\n\n{example}\n\n"
                  "IMPORTANT: follow the example's pattern — use the tools to "
                  "complete the request fully, including any exchange, return, "
                  "cancel or modification.")

    results = []
    t0 = time.time()
    for i, task in enumerate(pool_tasks):
        ep = Tau2Episode(task)
        instr = task.user_scenario.instructions
        up = instr.task_instructions
        if instr.known_info:
            up += f"\n\nKnown information: {instr.known_info}"
        if instr.unknown_info:
            up += f"\n\nUnknown information: {instr.unknown_info}"
        r = rollout(sp.client, sp.base_model, ep, policy_doc, tools,
                    max_turns=20, max_tokens=256, temperature=0.7, seed=0,
                    system_override=sys_prompt, user_prompt_override=up)
        n_mod = sum(1 for rc in ep.record.receipts if rc.tool_name in MODIFY_TOOLS)
        results.append({"task_id": task.id, "success": r.success,
                        "turns": r.turns, "calls": r.n_tool_calls,
                        "modify_calls": n_mod})
        print(f"[{i+1}/{len(pool_tasks)}] {task.id}: succ={r.success} "
              f"calls={r.n_tool_calls} modify={n_mod}", flush=True)

    rate = sum(1 for x in results if x["success"]) / len(results)
    n_mod = sum(x["modify_calls"] for x in results)
    n_tasks_with_modify = sum(1 for x in results if x["modify_calls"] > 0)
    out = {"mode": "fewshot_probe", "seed": args.seed,
           "success_rate": rate, "n_success": sum(1 for x in results if x["success"]),
           "n": len(results), "modify_calls_total": n_mod,
           "tasks_with_modify": n_tasks_with_modify,
           "per_task": results, "elapsed_s": round(time.time() - t0, 1)}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1))
    print(f"few-shot probe: success={rate:.3f} modify_calls={n_mod} "
          f"tasks_with_modify={n_tasks_with_modify} -> {out_path}")


if __name__ == "__main__":
    main()
