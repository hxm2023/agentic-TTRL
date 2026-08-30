"""v5: success-replay booster — the exploration-gap fix taken to its limit.

The main runs' null was mechanistically explained by sparse positive signal
(~6/68 update episodes succeed). This experiment trains ONLY on the model's
own successful trajectories (replayed with positive credit), which is the
standard prioritized-experience idea applied to TTRL:

1. Roll out the known-successful UPDATE-SET tasks with the frozen base
   (deterministic greedy; trajectories reproduce) — never eval-set tasks
   (contamination freeze).
2. Train the LoRA on those rows with +0.5 credit, multiple passes.
3. Eval frozen vs candidate on the sealed 46-task eval set.

Usage (GPU0, server):
  python scripts/success_replay.py --out protocols/success_replay.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ttrl2.agent.transformers_loop import rollout_transformers  # noqa: E402
from ttrl2.env.tau2_env import Tau2Episode  # noqa: E402
from ttrl2.gates.global_gate import decide  # noqa: E402
from ttrl2.trainer.lora_update import (  # noqa: E402
    GroupBaselines, build_training_rows, detect_conflicts, grpo_update,
    make_lora_model,
)
from tau2.domains.retail.environment import get_environment, get_tasks  # noqa: E402

MODEL_DIR = "/root/autodl-tmp/models/Qwen3.5-4B"
SUCCESSFUL_IDS = {"10", "105", "12", "24", "57", "62", "65", "67", "68"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--format", default="qwen3_xml",
                    choices=["qwen3_xml", "llama3_json"])
    ap.add_argument("--no-kl", action="store_true")
    ap.add_argument("--fewshot", action="store_true")
    ap.add_argument("--update-temp", type=float, default=0.0)
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--ids", default=None,
                    help="comma-separated task IDs to replay (measured successes)")
    ap.add_argument("--out", default="protocols/success_replay.json")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = args.model_dir
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, device_map={"": 0},
        trust_remote_code=True).eval()
    policy_model = make_lora_model(base)
    if args.no_kl:
        ref_model = None
        probe_ref = policy_model.base_model
    else:
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=torch.bfloat16, device_map={"": 0},
            trust_remote_code=True).eval()
        probe_ref = ref_model
    env = get_environment()
    policy_doc = env.get_policy()
    tools = env.get_tools()
    schemas = [{"type": "function",
                "function": {"name": t.name,
                             "description": (t.long_desc or t.short_desc or "")[:1024],
                             "parameters": t.params.model_json_schema()}}
               for t in tools]

    tasks = get_tasks("base")
    rng = random.Random(args.seed)
    stream = sorted(tasks, key=lambda t: t.id)
    rng.shuffle(stream)
    update_ids = {t.id for t in stream[:68]}
    eval_tasks = stream[68:]
    # success-replay pool: successful tasks in the UPDATE set only
    replay_ids = SUCCESSFUL_IDS & update_ids
    replay_tasks = [t for t in stream if t.id in replay_ids]
    if args.ids:
        ids = {x.strip() for x in args.ids.split(",")}
        replay_tasks = [t for t in stream if t.id in ids and t.id in update_ids]
    print(f"replay pool: {len(replay_tasks)} tasks "
          f"fewshot={args.fewshot} ids={[t.id for t in replay_tasks]}",
          flush=True)

    sys_prompt = None
    if args.fewshot:
        t0ref = next(t for t in tasks if t.id == "0")
        ex_lines = ["Here is an example of completing a similar request:"]
        for a in (t0ref.evaluation_criteria.actions or []):
            argstr = ", ".join(f"{k}={v}" for k, v in a.arguments.items())
            ex_lines.append(f"  apis.{a.name}({argstr})")
        sys_prompt = (f"You are a retail customer service agent.\n\n"
                      f"Policy:\n{policy_doc}\n\n" + "\n".join(ex_lines) + "\n\n"
                      "IMPORTANT: follow the example's pattern — use the tools "
                      "to complete the request fully, including any exchange, "
                      "return, cancel or modification.")

    def roll(model, task, temperature=0.0, seed=0):
        ep = Tau2Episode(task)
        r = rollout_transformers(model, tokenizer, ep, policy_doc, tools,
                                 max_turns=20, max_tokens=384,
                                 temperature=temperature, seed=seed,
                                 system_override=sys_prompt, fmt=args.format)
        return r, ep

    # ---- 1. gather successful trajectories (frozen base, deterministic) ----
    baselines = GroupBaselines()
    all_rows = []
    t0 = time.time()
    for task in replay_tasks:
        r, ep = roll(probe_ref, task, temperature=args.update_temp)
        instr = task.user_scenario.instructions
        tp = instr.task_instructions
        if instr.known_info:
            tp += f"\n\nKnown information: {instr.known_info}"
        if instr.unknown_info:
            tp += f"\n\nUnknown information: {instr.unknown_info}"
        identified = None
        for rc in ep.record.receipts:
            if rc.tool_name in ("find_user_id_by_name_zip", "find_user_id_by_email") and rc.ok:
                identified = str(rc.output).strip()
        conflicts = detect_conflicts(ep.record.receipts, identified)
        rows = build_training_rows(r.transcript, ep.record.receipts,
                                   r.success or False, baselines, conflicts,
                                   policy_doc, tp, schemas)
        calls = [e.name for e in r.transcript if e.role == "tool"]
        print(f"[replay {task.id}] succ={r.success} turns={r.turns} "
              f"calls={r.n_tool_calls} rows={len(rows)} tools={calls}",
              flush=True)
        if r.success:
            all_rows.extend(rows)

    # ---- 2. train on the successful rows, multiple passes ----
    print(f"training on {len(all_rows)} positive rows x {args.passes} passes",
          flush=True)
    for p in range(args.passes):
        if all_rows:
            grpo_update(policy_model, ref_model, tokenizer, all_rows, schemas,
                        lr=args.lr, kl_beta=0.1, steps=args.steps)
            policy_model.eval()

    # ---- 3. eval: frozen vs candidate on the sealed set ----
    def call_names(ep):
        return [rc.tool_name for rc in ep.record.receipts]

    MODIFY_TOOLS = {"cancel_pending_order", "exchange_delivered_order_items",
                    "modify_pending_order_address", "modify_pending_order_items",
                    "modify_pending_order_payment", "modify_user_address",
                    "return_delivered_order_items"}
    ef, ec = [], []
    cf_modify = 0
    for i, task in enumerate(eval_tasks):
        rf, epf = roll(probe_ref, task)
        rc, epc = roll(policy_model, task)
        cf_modify += sum(1 for t in call_names(epc) if t in MODIFY_TOOLS)
        ef.append({"task_id": task.id, "success": rf.success,
                   "turns": rf.turns, "calls": rf.n_tool_calls})
        ec.append({"task_id": task.id, "success": rc.success,
                   "turns": rc.turns, "calls": rc.n_tool_calls})
        print(f"[eval {i+1}/{len(eval_tasks)}] {task.id}: "
              f"frozen={rf.success} candidate={rc.success}", flush=True)
    print(f"candidate modify-calls in eval: {cf_modify}", flush=True)
    fr = sum(1 for t in ef if t["success"]) / len(ef)
    cr = sum(1 for t in ec if t["success"]) / len(ec)
    flips = [(t["task_id"], t["success"], c["success"])
             for t, c in zip(ef, ec) if t["success"] != c["success"]]
    beh = sum(1 for t, c in zip(ef, ec)
              if t["turns"] != c["turns"] or t["calls"] != c["calls"])
    print(f"EVAL: frozen={fr:.3f} candidate={cr:.3f} (delta {cr-fr:+.3f}); "
          f"flips={flips}; behavior-diff={beh}/46", flush=True)

    # ---- 4. gate on the eval contrast (honest, reported) ----
    gain_diffs = [(1.0 if c["success"] else 0.0) - (1.0 if t["success"] else 0.0)
                  for t, c in zip(ef, ec)]
    gate = decide(1, gain_diffs, gain_diffs)
    print(f"gate (n={len(gain_diffs)}): {gate.decision} "
          f"lcb_gain={gate.lcb_gain:.3f}", flush=True)

    out = {
        "mode": "success_replay", "seed": args.seed,
        "replay_tasks": sorted(replay_ids), "n_rows": len(all_rows),
        "passes": args.passes, "steps": args.steps, "lr": args.lr,
        "eval": {"frozen_rate": fr, "candidate_rate": cr, "delta": cr - fr,
                 "flips": flips, "behavior_diff": beh, "n": len(ef)},
        "gate": {"decision": gate.decision.value, "n": len(gain_diffs),
                 "lcb_gain": gate.lcb_gain},
        "elapsed_s": round(time.time() - t0, 1),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1))
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
