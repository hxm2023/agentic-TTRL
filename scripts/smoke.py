"""D2 gate: full single-task pipeline smoke (format -> rollout -> update -> eval).

Runs on autodl3: GPU0 = trainer (this process), GPU1 = vLLM (already serving).
Verifies: rollout works, credit assigns, the LoRA update changes behavior
(logit drift + output change), the adapter uploads to vLLM and rolls out.

Usage (server):
  CUDA_VISIBLE_DEVICES=0 python scripts/smoke.py --task-id 0 --out protocols/smoke_run.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ttrl2.serving.vllm_client import ServedPolicy  # noqa: E402
from ttrl2.trainer.lora_update import (  # noqa: E402
    GroupBaselines, build_training_rows, detect_conflicts, grpo_update,
    logit_drift, make_lora_model,
)
from tau2.domains.retail.environment import get_environment, get_tasks  # noqa: E402
from ttrl2.env.tau2_env import Tau2Episode  # noqa: E402

MODEL_DIR = "/root/autodl-tmp/models/Qwen3.5-4B"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", default="0")
    ap.add_argument("--endpoint", default="http://localhost:8001/v1")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default="protocols/smoke_run.json")
    ap.add_argument("--base-model", default="qwen3.5-4b")
    args = ap.parse_args()

    import torch
    from peft import PeftModel  # noqa: F401
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("[smoke] loading models on GPU0...", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16,
        device_map={"": 0}, trust_remote_code=True)
    policy_model = make_lora_model(base)
    # ref MUST be a separate instance: a no_grad forward on the shared base
    # object poisons the subsequent grad forward (Qwen3.5 hybrid + fla kernels)
    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16,
        device_map={"": 0}, trust_remote_code=True).eval()
    print(f"[smoke] models loaded in {time.time()-t0:.0f}s", flush=True)

    env = get_environment()
    policy_doc = env.get_policy()
    tools = env.get_tools()
    task = next(t for t in get_tasks("base") if t.id == args.task_id)

    sp = ServedPolicy(args.endpoint, args.base_model)
    ep = Tau2Episode(task)
    print(f"[smoke] task {task.id}: task_instructions="
          f"{task.user_scenario.instructions.task_instructions[:100]}...", flush=True)

    # 1. rollout with frozen
    r_frozen = sp.rollout_episode(ep, policy_doc, tools, adapter=None,
                                  temperature=0.7, seed=0)
    print(f"[smoke] frozen rollout: success={r_frozen.success} "
          f"turns={r_frozen.turns} calls={r_frozen.n_tool_calls}", flush=True)

    # 2. credit + update
    baselines = GroupBaselines()
    identified = None
    for i, rc in enumerate(ep.record.receipts):
        if rc.tool_name in ("find_user_id_by_name_zip", "find_user_id_by_email") and rc.ok:
            identified = str(rc.output).strip()
    conflicts = detect_conflicts(ep.record.receipts, identified)
    print(f"[smoke] conflicts: {conflicts}", flush=True)
    instr = task.user_scenario.instructions
    task_prompt = instr.task_instructions
    if instr.known_info:
        task_prompt += f"\n\nKnown information: {instr.known_info}"
    if instr.unknown_info:
        task_prompt += f"\n\nUnknown information: {instr.unknown_info}"
    tool_schemas = [{"type": "function",
                     "function": {"name": t.name,
                                  "description": (t.long_desc or t.short_desc or "")[:1024],
                                  "parameters": t.params.model_json_schema()}}
                    for t in tools]
    rows = build_training_rows(r_frozen.transcript, ep.record.receipts,
                               r_frozen.success or False, baselines, conflicts,
                               policy_doc, task_prompt, tool_schemas)
    print(f"[smoke] training rows: {len(rows)} "
          f"advs={[round(r['advantage'],3) for r in rows]}", flush=True)
    stats = grpo_update(policy_model, ref_model, tokenizer, rows, tool_schemas,
                        lr=args.lr, steps=args.steps)
    print(f"[smoke] update stats: {stats}", flush=True)

    # 3. drift calibration (update must change behavior)
    probe = tokenizer.apply_chat_template(
        [{"role": "system",
          "content": f"You are a retail customer service agent.\n\nPolicy:\n{policy_doc}"},
         {"role": "user", "content": task_prompt}],
        tools=tool_schemas, tokenize=False, add_generation_prompt=True)
    drift = logit_drift(policy_model, ref_model, tokenizer, probe)
    print(f"[smoke] logit drift on probe: {drift:.4f}", flush=True)

    # 4. adapter -> vLLM -> re-rollout with candidate
    adapter_dir = "/root/autodl-tmp/adapters/smoke_candidate"
    policy_model.save_pretrained(adapter_dir)
    sp.load_adapter("smoke_candidate", adapter_dir)
    print("[smoke] adapter uploaded", flush=True)
    ep2 = Tau2Episode(task)
    r_cand = sp.rollout_episode(ep2, policy_doc, tools, adapter="smoke_candidate",
                                temperature=0.7, seed=0)
    print(f"[smoke] candidate rollout: success={r_cand.success} "
          f"turns={r_cand.turns} calls={r_cand.n_tool_calls}", flush=True)
    sp.unload_adapter("smoke_candidate")

    out = {
        "task_id": task.id,
        "frozen": {"success": r_frozen.success, "turns": r_frozen.turns,
                   "calls": r_frozen.n_tool_calls},
        "candidate": {"success": r_cand.success, "turns": r_cand.turns,
                      "calls": r_cand.n_tool_calls},
        "update_stats": stats,
        "conflicts": conflicts,
        "logit_drift": drift,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1))
    print(f"[smoke] -> {out_path}")
    print(f"[smoke] drift >= 0.05 required for 'update changes behavior' "
          f"(got {drift:.4f})")


if __name__ == "__main__":
    main()
