"""Transformers-path rollout probe: transcript + tool-call detail.

The vLLM-path probe (probe_transcript.py) cannot explain why the TTRL
update-phase rollouts (transformers) stop earlier than the vLLM sweep.
This runs the same task through the actual training-path rollout and prints
per-turn content/calls, with switches for thinking and decoding.

Usage (server, one GPU free):
  CUDA_VISIBLE_DEVICES=1 python scripts/probe_transformers.py --task-id 1 \
      --no-think --temperature 0.7 --max-tokens 1024 --samples 2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ttrl2.agent.transformers_loop import rollout_transformers  # noqa: E402
from ttrl2.env.tau2_env import Tau2Episode  # noqa: E402
from ttrl2.trainer.lora_update import make_lora_model  # noqa: E402
from tau2.domains.retail.environment import get_environment, get_tasks  # noqa: E402

MODEL_DIR = "/datac/posttrain-paper/models/Qwen/Qwen3.5-4B"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", default="1")
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--no-think", action="store_true", default=True)
    ap.add_argument("--think", dest="no_think", action="store_false")
    ap.add_argument("--adapter", action="store_true",
                    help="wrap the model with a fresh LoRA in eval mode")
    ap.add_argument("--model-dir", default=MODEL_DIR)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.bfloat16, device_map={"": 0})
    if args.adapter:
        model = make_lora_model(model)
    model.eval()

    env = get_environment()
    policy = env.get_policy()
    tools = env.get_tools()
    task = next(t for t in get_tasks("base") if t.id == args.task_id)

    from ttrl2.agent.loop import build_user_prompt
    print(f'== user prompt: {build_user_prompt(task)[:300]!r}')
    for k in range(args.samples):
        ep = Tau2Episode(task)
        r = rollout_transformers(
            model, tokenizer, ep, policy, tools, max_turns=24,
            max_tokens=args.max_tokens, temperature=args.temperature,
            seed=k, no_think=args.no_think)
        print(f"\n===== sample {k}: success={r.success} turns={r.turns} "
              f"calls={r.n_tool_calls} no_think={args.no_think}")
        for i, e in enumerate(r.transcript):
            if e.role == "assistant":
                tcs = e.tool_calls or []
                if tcs:
                    for tc in tcs:
                        print(f"  [{i}] assistant call: {tc['name']} "
                              f"{tc['arguments'][:120]}")
                else:
                    print(f"  [{i}] assistant (no call) content_tail="
                          f"{(e.content or '')[-220:]!r}")
            elif e.role == "user":
                print(f"  [{i}] user(stub): {e.content!r}")
        ref = [a.name for a in (task.evaluation_criteria.actions or [])]
        print(f"  reference: {ref}")
        print(f"  json: {json.dumps([e.tool_calls and [c['name'] for c in e.tool_calls] for e in r.transcript if e.role == 'assistant'])}")


if __name__ == "__main__":
    main()
