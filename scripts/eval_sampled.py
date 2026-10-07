"""Sampled pass@1 comparison: frozen base vs candidate adapter.

The greedy eval on the held-out band saturates (the frozen policy solves most
band tasks deterministically), so it cannot show the update's effect. This
evaluates the same tasks under sampling (T=0.7, k samples/arm) — the
distribution the test-time updates actually train on — and reports paired
per-sample rates with an exact McNemar test.

Engines:
  transformers — loads base + (adapter) directly; slow but always available.
  vllm         — serves the base and hot-loads the adapter; ~10x faster.

Usage:
  python scripts/eval_sampled.py --stream protocols/stream_v2_seed0_short.json \
      --adapter protocols/ttrl_v2_seed0.adapter --k 4 --engine vllm \
      --endpoint http://localhost:8002/v1 --out protocols/eval_sampled_seed0.json
"""
from __future__ import annotations

import argparse
import json
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

MODEL_DIR = "/datac/posttrain-paper/models/Qwen/Qwen3.5-4B"


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", default="protocols/stream_v2_seed0_short.json")
    ap.add_argument("--adapter", default="protocols/ttrl_v2_seed0.adapter")
    ap.add_argument("--tasks", default="band", choices=["band", "all"])
    ap.add_argument("--arms", default="both", choices=["both", "frozen", "candidate"])
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--engine", default="vllm", choices=["vllm", "transformers"])
    ap.add_argument("--endpoint", default="http://localhost:8002/v1")
    ap.add_argument("--base-model", default="qwen3.5-4b")
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from ttrl2.agent.loop import build_user_prompt  # noqa: F401 (kept for parity)
    from ttrl2.env.tau2_env import Tau2Episode
    from tau2.domains.retail.environment import get_environment, get_tasks

    spec = json.loads(Path(args.stream).read_text(encoding="utf-8"))
    task_ids = spec["eval_band"] if args.tasks == "band" else spec["eval_ids"]
    all_tasks = {t.id: t for t in get_tasks("base")}
    tasks = [all_tasks[i] for i in task_ids]

    env = get_environment()
    policy_doc = env.get_policy()
    tools = env.get_tools()

    if args.engine == "vllm":
        from ttrl2.serving.vllm_client import ServedPolicy
        sp = ServedPolicy(args.endpoint, args.base_model)
        if args.adapter:
            try:
                sp.load_adapter("candidate", args.adapter)
            except Exception as e:  # already loaded on a shared server is fine
                print(f"[eval] load_adapter note: {str(e)[:120]}", flush=True)
        models = {"frozen": None, "candidate": "candidate"}
        models = {k: v for k, v in models.items() if args.arms in ("both", k)}
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from ttrl2.agent.transformers_loop import rollout_transformers
        from ttrl2.trainer.lora_update import make_lora_model
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        base = AutoModelForCausalLM.from_pretrained(
            args.model_dir, dtype=torch.bfloat16, device_map={"": 0})
        base.eval()
        policy_model = make_lora_model(base)
        policy_model.eval()
        from peft import PeftModel
        cand = PeftModel.from_pretrained(policy_model, args.adapter)
        cand.eval()
        models = {"frozen": policy_model, "candidate": cand}
        models = {k: v for k, v in models.items() if args.arms in ("both", k)}

    results = {"stream": args.stream, "adapter": args.adapter, "k": args.k,
               "temperature": args.temperature, "engine": args.engine,
               "tasks": {}}
    for task in tasks:
        entry = {}
        for arm, model in models.items():
            runs = []
            for k in range(args.k):
                ep = Tau2Episode(task)
                if args.engine == "vllm":
                    r = sp.rollout_episode(ep, policy_doc, tools, adapter=model,
                                           temperature=args.temperature, seed=k)
                else:
                    r = rollout_transformers(model, tokenizer, ep, policy_doc,
                                             tools, max_turns=24, max_tokens=2048,
                                             temperature=args.temperature, seed=k)
                runs.append({"success": bool(r.success), "turns": r.turns,
                             "calls": r.n_tool_calls})
            entry[arm] = runs
            print(f"[{task.id}] {arm}: {sum(x['success'] for x in runs)}/{args.k}",
                  flush=True)
        results["tasks"][task.id] = entry

    def rate(arm, ids):
        vals = [x["success"] for i in ids for x in results["tasks"][i][arm]]
        return sum(vals), len(vals)

    fb, fn = rate("frozen", task_ids) if "frozen" in models else (0, 0)
    cb, cn = rate("candidate", task_ids) if "candidate" in models else (0, 0)
    b = sum(1 for i in task_ids
            for f, c in zip(results["tasks"][i]["frozen"],
                            results["tasks"][i]["candidate"]) if c["success"] and not f["success"])
    c_ = sum(1 for i in task_ids
             for f, c in zip(results["tasks"][i]["frozen"],
                             results["tasks"][i]["candidate"]) if f["success"] and not c["success"])
    p = mcnemar_exact(b, c_)
    results["summary"] = {"frozen_n": fn, "frozen_k": fb, "candidate_n": cn,
                          "candidate_k": cb, "gained": b, "lost": c_,
                          "mcnemar_p": p}
    print(f"\n== sampled pass@1 ({args.engine}, k={args.k}, n={len(task_ids)} tasks)")
    if fn:
        print(f"   frozen    = {fb}/{fn} = {fb/fn:.3f}")
    if cn:
        print(f"   candidate = {cb}/{cn} = {cb/cn:.3f}"
              + (f"   delta={(cb-fb)/fn:+.3f}  gained={b} lost={c_}  McNemar p={p:.4f}"
                 if fn else ""))
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1), encoding="utf-8")
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
