"""Protocol-v2 diagnostic: does the agent actually see the user's request?

v1 of this harness built the agent prompt from `task_instructions`
(simulator-side behaviour guidance, e.g. "You are detail-oriented...") and
never included `reason_for_call` (the actual request). Result: the model
never attempted state-changing calls and only "succeeded" on read-only tasks.

This script runs the same tasks under both prompt constructions and reports,
per arm: success, modify-call attempts, and the tool-call sequence.

Usage (server):
  python scripts/diag_prompt_v2.py --tasks 1,108,25,30,44 --samples 2 \
      --endpoint http://localhost:8002/v1 --out protocols/diag_prompt_v2.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ttrl2.agent.loop import build_user_prompt, rollout  # noqa: E402
from ttrl2.env.tau2_env import Tau2Episode  # noqa: E402
from ttrl2.serving.vllm_client import ServedPolicy  # noqa: E402
from tau2.domains.retail.environment import get_environment, get_tasks  # noqa: E402

MODIFY_TOOLS = {
    "cancel_pending_order", "exchange_delivered_order_items",
    "modify_pending_order_address", "modify_pending_order_items",
    "modify_pending_order_payment", "modify_user_address",
    "return_delivered_order_items",
}


def old_prompt(task) -> str:
    instr = task.user_scenario.instructions
    up = instr.task_instructions
    if instr.known_info:
        up += f"\n\nKnown information: {instr.known_info}"
    if instr.unknown_info:
        up += f"\n\nUnknown information: {instr.unknown_info}"
    return up


def call_sequence(result) -> list[str]:
    names = []
    for e in result.transcript:
        for tc in (e.tool_calls or []):
            names.append(tc["name"])
    return names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="1,108,25,30,44")
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--endpoint", default="http://localhost:8002/v1")
    ap.add_argument("--base-model", default="qwen3.5-4b")
    ap.add_argument("--out", default="protocols/diag_prompt_v2.json")
    args = ap.parse_args()

    env = get_environment()
    policy_doc = env.get_policy()
    tools = env.get_tools()
    all_tasks = {t.id: t for t in get_tasks("base")}
    task_ids = [t.strip() for t in args.tasks.split(",") if t.strip()]

    sp = ServedPolicy(args.endpoint, args.base_model)
    out = {"mode": "diag_prompt_v2", "tasks": {}, "endpoint": args.endpoint}
    t0 = time.time()
    for tid in task_ids:
        task = all_tasks[tid]
        n_write = sum(1 for a in (task.evaluation_criteria.actions or [])
                      if a.name in MODIFY_TOOLS)
        entry = {"n_write_actions": n_write, "arms": {}}
        for arm, up in (("v1_prompt", old_prompt(task)),
                        ("v2_prompt", build_user_prompt(task))):
            runs = []
            for k in range(args.samples):
                ep = Tau2Episode(task)
                r = rollout(sp.client, sp.base_model, ep, policy_doc, tools,
                            max_turns=20, max_tokens=args.max_tokens,
                            temperature=args.temperature, seed=k,
                            user_prompt_override=up)
                seq = call_sequence(r)
                runs.append({
                    "success": r.success, "turns": r.turns,
                    "n_calls": r.n_tool_calls,
                    "n_modify": sum(1 for n in seq if n in MODIFY_TOOLS),
                    "calls": seq,
                })
                print(f"[{tid}] {arm} #{k}: success={r.success} "
                      f"modify={runs[-1]['n_modify']} calls={seq[:8]}", flush=True)
            entry["arms"][arm] = runs
        out["tasks"][tid] = entry
    out["elapsed_s"] = round(time.time() - t0, 1)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1))
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
