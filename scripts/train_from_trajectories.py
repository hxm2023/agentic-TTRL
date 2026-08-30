"""Train the LoRA candidate on SAVED successful trajectories (positive replay).

Deterministic: uses the exact trajectories from the vLLM probe (which
succeeded), no re-roll variance. Builds training rows from each saved
transcript, trains with +0.5 credit over multiple passes, then evaluates
frozen vs candidate on the sealed eval set.

Usage (GPU0, server):
  python scripts/train_from_trajectories.py \
      --trajectories protocols/fewshot_probe_llama_update.json \
      --format llama3_json --no-kl --fewshot \
      --passes 12 --steps 12 --lr 3e-5 \
      --model-dir /root/autodl-tmp/models/Llama-3.1-8B-Instruct \
      --out protocols/success_replay_llama.json
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectories", required=True,
                    help="probe JSON containing saved successful trajectories")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--passes", type=int, default=12)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--format", default="qwen3_xml",
                    choices=["qwen3_xml", "llama3_json"])
    ap.add_argument("--no-kl", action="store_true")
    ap.add_argument("--fewshot", action="store_true")
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--out", default="protocols/success_replay_llama.json")
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
        probe_ref = policy_model.base_model
    else:
        probe_ref = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=torch.bfloat16, device_map={"": 0},
            trust_remote_code=True).eval()
    env = get_environment()
    policy_doc = env.get_policy()
    tools = env.get_tools()
    schemas = [{"type": "function",
                "function": {"name": t.name,
                             "description": (t.long_desc or t.short_desc or "")[:1024],
                             "parameters": t.params.model_json_schema()}}
               for t in tools]

    sys_prompt = None
    if args.fewshot:
        t0ref = next(t for t in get_tasks("base") if t.id == "0")
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

    # ---- 1. build rows from the SAVED successful trajectories ----
    probe = json.load(open(args.trajectories, encoding="utf-8"))
    saved = probe.get("trajectories", [])
    print(f"saved successful trajectories: {len(saved)}", flush=True)
    tasks_by_id = {t.id: t for t in get_tasks("base")}
    baselines = GroupBaselines()
    all_rows = []
    t0 = time.time()
    for traj in saved:
        task = tasks_by_id[traj["task_id"]]
        # rebuild the transcript entries
        class E:
            def __init__(self, d):
                self.role = d["role"]
                self.content = d["content"]
                self.tool_calls = d.get("tool_calls")
                self.tool_call_id = d.get("tool_call_id")
                self.name = d.get("name")
        transcript = [E(d) for d in traj["transcript"]]
        # rebuild the receipts
        from ttrl2.env.tau2_env import ToolReceipt
        receipts = [ToolReceipt(d["tool_name"], d["arguments"], d["ok"],
                                None, d.get("error")) for d in traj["receipts"]]
        identified = None
        for rc in receipts:
            if rc.tool_name in ("find_user_id_by_name_zip", "find_user_id_by_email") and rc.ok:
                identified = str(rc.arguments)  # output not saved; use args marker
        conflicts = detect_conflicts(receipts, identified)
        rows = build_training_rows(transcript, receipts, True, baselines, conflicts,
                                   policy_doc, traj["task_prompt"], schemas)
        n_mod = sum(1 for d in traj["receipts"] if d["tool_name"] in (
            "cancel_pending_order", "exchange_delivered_order_items",
            "modify_pending_order_address", "modify_pending_order_items",
            "modify_pending_order_payment", "modify_user_address",
            "return_delivered_order_items"))
        print(f"[traj {traj['task_id']}] rows={len(rows)} modify={n_mod} "
              f"conflicts={conflicts}", flush=True)
        all_rows.extend(rows)

    # ---- 2. train on the positive rows, multiple passes ----
    print(f"training on {len(all_rows)} positive rows x {args.passes} passes",
          flush=True)
    for p in range(args.passes):
        if all_rows:
            grpo_update(policy_model, probe_ref if not args.no_kl else None,
                        tokenizer, all_rows, schemas,
                        lr=args.lr, kl_beta=0.1, steps=args.steps)
            policy_model.eval()

    # ---- 3. eval: frozen vs candidate on the sealed set ----
    tasks = get_tasks("base")
    rng = random.Random(args.seed)
    stream = sorted(tasks, key=lambda t: t.id)
    rng.shuffle(stream)
    eval_tasks = stream[68:]
    ef, ec = [], []
    for i, task in enumerate(eval_tasks):
        rf, _ = roll(probe_ref, task)
        rc, _ = roll(policy_model, task)
        ef.append({"task_id": task.id, "success": rf.success,
                   "turns": rf.turns, "calls": rf.n_tool_calls})
        ec.append({"task_id": task.id, "success": rc.success,
                   "turns": rc.turns, "calls": rc.n_tool_calls})
        print(f"[eval {i+1}/{len(eval_tasks)}] {task.id}: "
              f"frozen={rf.success} candidate={rc.success}", flush=True)
    fr = sum(1 for t in ef if t["success"]) / len(ef)
    cr = sum(1 for t in ec if t["success"]) / len(ec)
    flips = [(t["task_id"], t["success"], c["success"])
             for t, c in zip(ef, ec) if t["success"] != c["success"]]
    beh = sum(1 for t, c in zip(ef, ec)
              if t["turns"] != c["turns"] or t["calls"] != c["calls"])
    print(f"EVAL: frozen={fr:.3f} candidate={cr:.3f} (delta {cr-fr:+.3f}); "
          f"flips={flips}; behavior-diff={beh}/46", flush=True)

    gain_diffs = [(1.0 if c["success"] else 0.0) - (1.0 if t["success"] else 0.0)
                  for t, c in zip(ef, ec)]
    gate = decide(1, gain_diffs, gain_diffs)
    print(f"gate (n={len(gain_diffs)}): {gate.decision} "
          f"lcb_gain={gate.lcb_gain:.3f}", flush=True)

    out = {
        "mode": "train_from_trajectories", "seed": args.seed,
        "trajectory_source": args.trajectories,
        "n_rows": len(all_rows), "passes": args.passes, "steps": args.steps,
        "lr": args.lr, "fewshot": args.fewshot, "format": args.format,
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
