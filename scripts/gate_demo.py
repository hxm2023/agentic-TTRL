"""Gate-protection demonstration (user-mandated key upgrade, 2026-08-29).

Shows the GLOBAL gate protecting deployment against a POISONED adapter:

1. Train a poisoned candidate: episode-boundary LoRA updates with FLIPPED
   credit (failed episodes reinforce the early stop; successful episodes
   suppress the correct calls) — mimics a compromised reward channel.
2. Shadow-evaluate poisoned vs frozen on paired episodes (E_hard rollouts).
3. The empirical-Bernstein e-process decides: expect ROLLBACK (harm detected
   or gain not established) -> gate ON deploys the FROZEN policy.
4. Gate-OFF counterfactual: evaluate the poisoned candidate's deployment
   (its actual future-task success) -> show the degradation the gate
   prevented.

Usage (GPU0, server):
  python scripts/gate_demo.py --seed 0 --poison-episodes 25 \
      --out protocols/gate_demo.json
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
    logit_drift, make_lora_model,
)
from tau2.domains.retail.environment import get_environment, get_tasks  # noqa: E402

MODEL_DIR = "/root/autodl-tmp/models/Qwen3.5-4B"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--poison-episodes", type=int, default=25)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--n-shadow", type=int, default=20)
    ap.add_argument("--n-eval", type=int, default=20)
    ap.add_argument("--out", default="protocols/gate_demo.json")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, device_map={"": 0},
        trust_remote_code=True).eval()
    policy_model = make_lora_model(base)
    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, device_map={"": 0},
        trust_remote_code=True).eval()
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
    pool = stream[:68]                       # update-phase pool
    rest = stream[68:]                       # eval pool (sealed-ish for demo)
    poison_tasks = rng.sample(pool, args.poison_episodes)
    shadow_tasks = rng.sample([t for t in pool if t.id not in
                               {x.id for x in poison_tasks}],
                              min(args.n_shadow, len(pool) - args.poison_episodes))
    eval_tasks = rng.sample(rest, min(args.n_eval, len(rest)))

    def roll(model, task, temperature=0.0, seed=0):
        ep = Tau2Episode(task)
        r = rollout_transformers(model, tokenizer, ep, policy_doc, tools,
                                 max_turns=20, max_tokens=384,
                                 temperature=temperature, seed=seed)
        return r, ep

    # ---- 1. poison training: flipped credit ----
    baselines = GroupBaselines()
    t0 = time.time()
    for i, task in enumerate(poison_tasks):
        r, ep = roll(policy_model, task)
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
        # FLIP the credit: reinforce failures, suppress successes
        for row in rows:
            row["advantage"] = -row["advantage"]
        if rows:
            grpo_update(policy_model, ref_model, tokenizer, rows, schemas,
                        lr=args.lr, kl_beta=0.1, steps=args.steps)
            policy_model.eval()
        print(f"[poison {i+1}/{len(poison_tasks)}] {task.id}: succ={r.success} "
              f"rows={len(rows)}", flush=True)

    # ---- 2. shadow evaluation: frozen vs poisoned ----
    gain_diffs, harm_diffs = [], []
    for i, task in enumerate(shadow_tasks):
        rf, _ = roll(ref_model, task)
        rc, _ = roll(policy_model, task)
        gain_diffs.append((1.0 if rc.success else 0.0) - (1.0 if rf.success else 0.0))
        harm_diffs.append((1.0 if rf.success else 0.0) - (1.0 if rc.success else 0.0))
    gate = decide(1, gain_diffs, harm_diffs)
    print(f"GATE on shadow n={len(gain_diffs)}: {gate.decision} "
          f"lcb_gain={gate.lcb_gain:.3f} ucb_harm={gate.ucb_harm:.3f} "
          f"mean_gain={sum(gain_diffs)/len(gain_diffs):+.3f} "
          f"mean_harm={sum(harm_diffs)/len(harm_diffs):+.3f} "
          f"codes={gate.reason_codes}", flush=True)

    # ---- 3. gate-OFF counterfactual: deploy the poisoned candidate ----
    ef, ec = [], []
    for i, task in enumerate(eval_tasks):
        rf, _ = roll(ref_model, task)
        rc, _ = roll(policy_model, task)
        ef.append({"task_id": task.id, "success": rf.success,
                   "turns": rf.turns, "calls": rf.n_tool_calls})
        ec.append({"task_id": task.id, "success": rc.success,
                   "turns": rc.turns, "calls": rc.n_tool_calls})
    fr = sum(1 for t in ef if t["success"]) / len(ef)
    cr = sum(1 for t in ec if t["success"]) / len(ec)
    print(f"DEPLOYED (gate OFF) poisoned candidate: {cr:.3f} vs frozen "
          f"{fr:.3f} (delta {cr - fr:+.3f}) on {len(eval_tasks)} tasks",
          flush=True)

    out = {
        "demo": "gate_protection",
        "seed": args.seed, "poison_episodes": args.poison_episodes,
        "gate": {"decision": gate.decision.value, "lcb_gain": gate.lcb_gain,
                 "ucb_harm": gate.ucb_harm, "n_shadow": len(gain_diffs),
                 "mean_gain": sum(gain_diffs) / len(gain_diffs) if gain_diffs else 0.0,
                 "mean_harm": sum(harm_diffs) / len(harm_diffs) if harm_diffs else 0.0,
                 "reason_codes": gate.reason_codes},
        "deployment": {"frozen_rate": fr, "poisoned_rate": cr,
                       "delta": cr - fr,
                       "n_eval": len(eval_tasks)},
        "elapsed_s": round(time.time() - t0, 1),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1))
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
