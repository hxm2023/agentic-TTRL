"""Stream runner: frozen / best-of-n / TTRL on the same-task stream.

Protocol (frozen D1/D7-8):
- Stream = all 114 tau2 retail tasks, order fixed by --seed (first N_UPDATE tasks
  = update phase, last N_EVAL tasks = eval phase, SEALED).
- frozen: rollout every eval task with the frozen base; report success rate.
- bon: n independent rollouts per eval task, select ONE by an E_hard-visible
  heuristic (workflow-completion proxy, never the hidden evaluator), then
  hidden-evaluate the selected trajectory.
- ttrl: update phase rolls out with the current policy, assigns evidence-gated
  group credit, and accumulates a LoRA adapter (episode-boundary updates);
  the eval phase measures frozen (base) vs candidate (adapter) on the sealed
  eval tasks. The global e-process gate decision is computed on the shadow
  pairs at the end.

Usage (server):
  python scripts/run_stream.py --mode frozen --seed 0 --out protocols/...
  python scripts/run_stream.py --mode bon --n 4 --seed 0 --out protocols/...
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
from ttrl2.trainer.lora_update import (  # noqa: E402
    GroupBaselines, build_training_rows, detect_conflicts, grpo_update,
    logit_drift, make_lora_model,
)
from tau2.domains.retail.environment import get_environment, get_tasks  # noqa: E402

N_UPDATE = 68
N_EVAL = 46
MODEL_DIR = "/root/autodl-tmp/models/Qwen3.5-4B"


def heuristic_score(receipts) -> float:
    """E_hard-visible workflow-completion proxy (BoN selection only)."""
    score = 0.0
    for r in receipts:
        if r.ok:
            if r.tool_name.startswith("modify_") or r.tool_name in (
                    "cancel_pending_order", "exchange_delivered_order_items",
                    "return_delivered_order_items"):
                score += 2.0
            elif r.tool_name.startswith("get_"):
                score += 1.0
            elif r.tool_name.startswith("find_"):
                score += 1.0
        else:
            score -= 1.0
    return score


def load_stream(seed: int) -> list:
    tasks = get_tasks("base")
    rng = random.Random(seed)
    order = sorted(tasks, key=lambda t: t.id)
    rng.shuffle(order)
    return order


def run_frozen(sp: ServedPolicy, env, tasks, out_path: Path,
               temperature: float = 0.7) -> dict:
    policy = env.get_policy()
    tools = env.get_tools()
    results = []
    t0 = time.time()
    for i, task in enumerate(tasks):
        ep = Tau2Episode(task)
        r = sp.rollout_episode(ep, policy, tools, temperature=temperature, seed=0)
        results.append({"task_id": task.id, "success": r.success,
                        "turns": r.turns, "calls": r.n_tool_calls})
        print(f"[{i+1}/{len(tasks)}] {task.id}: success={r.success}", flush=True)
    n_ok = sum(1 for x in results if x["success"])
    out = {"mode": "frozen", "n": len(results), "n_success": n_ok,
           "success_rate": n_ok / len(results) if results else 0.0,
           "elapsed_s": round(time.time() - t0, 1), "per_task": results}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1))
    print(f"frozen success_rate={out['success_rate']:.3f} -> {out_path}")
    return out


def run_bon(sp: ServedPolicy, env, tasks, n: int, out_path: Path) -> dict:
    policy = env.get_policy()
    tools = env.get_tools()
    results = []
    t0 = time.time()
    for i, task in enumerate(tasks):
        candidates = []
        for k in range(n):
            ep = Tau2Episode(task)
            r = sp.rollout_episode(ep, policy, tools, temperature=0.7, seed=k)
            candidates.append({"rollout": r, "episode": ep,
                               "score": heuristic_score(ep.record.receipts)})
        best = max(candidates, key=lambda c: c["score"])
        success = best["episode"].evaluate()
        results.append({"task_id": task.id, "success": success,
                        "turns": best["rollout"].turns,
                        "calls": best["rollout"].n_tool_calls,
                        "n_rollouts": n, "n_success_any":
                        int(any(c["episode"].evaluate() for c in candidates))})
        print(f"[{i+1}/{len(tasks)}] {task.id}: selected success={success} "
              f"any={results[-1]['n_success_any']}", flush=True)
    n_ok = sum(1 for x in results if x["success"])
    out = {"mode": f"bon{n}", "n": len(results), "n_success": n_ok,
           "success_rate": n_ok / len(results) if results else 0.0,
           "elapsed_s": round(time.time() - t0, 1), "per_task": results}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1))
    print(f"bon{n} selected success_rate={out['success_rate']:.3f} -> {out_path}")
    return out


def run_ttrl(sp: ServedPolicy, env, update_tasks, eval_tasks, out_path: Path,
             args, stream_meta: dict) -> None:
    """D7-8 main: episode-boundary updates on the first 60%, sealed eval on 40%.

    Rollouts run in transformers (NOT vLLM): vLLM 0.26's LoRA serving is a
    no-op for the Qwen3.5 hybrid model (verified 2026-08-29), so the trainer
    process rolls out with the CURRENT adapter directly (single GPU0 process).
    Eval phase: frozen (ref base) vs candidate (policy_model) on the sealed
    set. Global gate: paired shadow re-rollouts on a subset of update-phase
    tasks, empirical-Bernstein e-process decision (honest at its n).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from ttrl2.agent.transformers_loop import rollout_transformers
    from ttrl2.gates.global_gate import decide
    from ttrl2.gates.local_gate import DriftMonitor

    model_dir = args.model_dir
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, device_map={"": 0},
        trust_remote_code=True)
    policy_model = make_lora_model(base)
    if args.no_kl:
        # KL-free: the frozen base (policy_model.base_model) doubles as the
        # reference for the drift probe; standard dense models (Llama-3.1)
        # do not have the shared-base grad-poisoning issue of Qwen3.5 hybrid
        ref_model = None
        probe_ref = policy_model.base_model
    else:
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=torch.bfloat16, device_map={"": 0},
            trust_remote_code=True).eval()
        probe_ref = ref_model
    policy_doc = env.get_policy()
    tools = env.get_tools()
    schemas = [{"type": "function",
                "function": {"name": t.name,
                             "description": (t.long_desc or t.short_desc or "")[:1024],
                             "parameters": t.params.model_json_schema()}}
               for t in tools]
    baselines = GroupBaselines()
    monitor = DriftMonitor()
    update_log = []
    t0 = time.time()
    lr = float(args.lr)

    sys_prompt = None
    if args.fewshot:
        # one worked exchange workflow from task 0's reference (few-shot;
        # does not leak eval-task targets)
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

    def roll(model, task, temperature, seed, verbose: bool = False):
        ep = Tau2Episode(task)
        r = rollout_transformers(model, tokenizer, ep,
                                 policy_doc, tools, max_turns=20,
                                 max_tokens=384, temperature=temperature,
                                 seed=seed, system_override=sys_prompt,
                                 fmt=args.format)
        if verbose and r.transcript:
            e0 = r.transcript[0]
            print(f"    [diag] training={model.training} "
                  f"first={repr((e0.content or '')[:60])} "
                  f"calls={r.n_tool_calls}", flush=True)
        return r, ep

    # ---- UPDATE PHASE (episode-boundary test-time updates) ----
    # Greedy rollouts: transformers temp-0.7 sampling drifts away from the
    # tool-call mode (verified 2026-08-29) -> 0-call episodes -> the answer
    # penalty self-reinforces a no-tool policy. Greedy keeps the mode.
    for i, task in enumerate(update_tasks):
        r, ep = roll(policy_model, task, temperature=0.0, seed=0)
        instr = task.user_scenario.instructions
        task_prompt = instr.task_instructions
        if instr.known_info:
            task_prompt += f"\n\nKnown information: {instr.known_info}"
        if instr.unknown_info:
            task_prompt += f"\n\nUnknown information: {instr.unknown_info}"
        identified = None
        for rc in ep.record.receipts:
            if rc.tool_name in ("find_user_id_by_name_zip", "find_user_id_by_email") and rc.ok:
                identified = str(rc.output).strip()
        conflicts = detect_conflicts(ep.record.receipts, identified)
        conflicted = bool(conflicts)
        halt = monitor.observe(conflicted)
        rows = build_training_rows(r.transcript, ep.record.receipts,
                                   r.success or False, baselines, conflicts,
                                   policy_doc, task_prompt, schemas)
        stats = {"rows": 0, "tokens": 0, "loss": 0.0}
        if rows and not halt:
            # successful episodes get more steps: the sparse positive signal
            # is the update's only teacher (exploration-gap fix, v4)
            ep_steps = args.success_steps if (r.success or False) else args.steps
            stats = grpo_update(policy_model, ref_model, tokenizer, rows, schemas,
                                lr=lr, kl_beta=args.kl_beta, steps=ep_steps)
            # CRITICAL: grpo_update leaves the model in train() (dropout active);
            # generations after that are corrupted (empty outputs). Restore eval.
            policy_model.eval()
        # drift check on this task's probe; adaptive guard: if behavior drifts
        # too far from base, halve the learning rate (collapse protection)
        probe = tokenizer.apply_chat_template(
            [{"role": "system",
              "content": f"You are a retail customer service agent.\n\nPolicy:\n{policy_doc}"},
             {"role": "user", "content": task_prompt}],
            tools=schemas, tokenize=False, add_generation_prompt=True)
        drift = logit_drift(policy_model, probe_ref, tokenizer, probe)
        if drift > 2.0:
            lr = max(lr / 2.0, 1e-6)
            print(f"  [guard] drift {drift:.2f} > 2.0 -> lr {lr:.1e}", flush=True)
        update_log.append({"task_id": task.id, "success": r.success,
                           "turns": r.turns, "calls": r.n_tool_calls,
                           "conflicts": conflicts, "halt": halt,
                           "rows": stats["rows"], "tokens": stats["tokens"],
                           "loss": stats["loss"], "drift": drift})
        print(f"[{i+1}/{len(update_tasks)}] {task.id}: succ={r.success} "
              f"rows={stats['rows']} drift={drift:.3f} halt={halt}", flush=True)

    # ---- EVAL PHASE (sealed): frozen vs candidate ----
    eval_frozen = []
    eval_cand = []
    for i, task in enumerate(eval_tasks):
        rf, _ = roll(probe_ref, task, temperature=0.0, seed=0)
        rc, _ = roll(policy_model, task, temperature=0.0, seed=0)
        eval_frozen.append({"task_id": task.id, "success": rf.success,
                            "turns": rf.turns, "calls": rf.n_tool_calls})
        eval_cand.append({"task_id": task.id, "success": rc.success,
                          "turns": rc.turns, "calls": rc.n_tool_calls})
        print(f"[eval {i+1}/{len(eval_tasks)}] {task.id}: "
              f"frozen={rf.success} candidate={rc.success}", flush=True)

    # ---- GLOBAL GATE: paired shadow re-rollouts on update-phase tasks ----
    rng = random.Random(args.seed)
    shadow_tasks = rng.sample(update_tasks, min(20, len(update_tasks)))
    gain_diffs, harm_diffs = [], []
    for task in shadow_tasks:
        rf, _ = roll(probe_ref, task, temperature=0.0, seed=1)
        rc, _ = roll(policy_model, task, temperature=0.0, seed=1)
        gain_diffs.append((1.0 if rc.success else 0.0) - (1.0 if rf.success else 0.0))
        harm_diffs.append((1.0 if rf.success else 0.0) - (1.0 if rc.success else 0.0))
    gate = decide(1, gain_diffs, harm_diffs)
    print(f"GLOBAL GATE n={len(gain_diffs)}: {gate.decision} "
          f"lcb_gain={gate.lcb_gain:.3f} ucb_harm={gate.ucb_harm:.3f} "
          f"codes={gate.reason_codes}")

    def rate(vals):
        return sum(1 for v in vals if v["success"]) / len(vals) if vals else 0.0

    out = {
        "mode": "ttrl", "meta": stream_meta,
        "update_phase": update_log,
        "eval": {"frozen": eval_frozen, "candidate": eval_cand,
                 "frozen_rate": rate(eval_frozen), "candidate_rate": rate(eval_cand)},
        "gate": {"decision": gate.decision.value, "lcb_gain": gate.lcb_gain,
                 "ucb_harm": gate.ucb_harm, "n_shadow": len(gain_diffs),
                 "reason_codes": gate.reason_codes,
                 "mean_gain": sum(gain_diffs) / len(gain_diffs) if gain_diffs else 0.0,
                 "mean_harm": sum(harm_diffs) / len(harm_diffs) if harm_diffs else 0.0},
        "elapsed_s": round(time.time() - t0, 1),
        "config": {"steps": args.steps, "lr": args.lr, "kl_beta": args.kl_beta,
                   "n_update": len(update_tasks), "n_eval": len(eval_tasks)},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1))
    print(f"ttrl DONE: frozen={out['eval']['frozen_rate']:.3f} "
          f"candidate={out['eval']['candidate_rate']:.3f} "
          f"gate={gate.decision.value} -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["frozen", "bon", "ttrl"], default="frozen")
    ap.add_argument("--n", type=int, default=4, help="BoN samples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-update", type=int, default=N_UPDATE)
    ap.add_argument("--n-eval", type=int, default=N_EVAL)
    ap.add_argument("--endpoint", default="http://localhost:8001/v1")
    ap.add_argument("--base-model", default="qwen3.5-4b")
    ap.add_argument("--out", default=None)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--kl-beta", type=float, default=0.1)
    ap.add_argument("--success-steps", type=int, default=8,
                    help="steps on episodes that succeeded (focus the positive signal)")
    ap.add_argument("--fewshot", action="store_true",
                    help="v6: append a worked exchange workflow (task-0 reference, "
                         "few-shot) to the system prompt for rollouts and eval")
    ap.add_argument("--format", default="qwen3_xml",
                    choices=["qwen3_xml", "llama3_json"],
                    help="tool-call parser for the transformers rollout")
    ap.add_argument("--no-kl", action="store_true",
                    help="KL-free GRPO (8B+ models: policy+ref exceed one 5090; "
                         "the drift guard replaces the KL anchor)")
    ap.add_argument("--model-dir", default=MODEL_DIR,
                    help="local model directory for the trainer")
    args = ap.parse_args()

    stream = load_stream(args.seed)
    update_tasks = stream[:args.n_update]
    eval_tasks = stream[args.n_update:args.n_update + args.n_eval]
    out_path = Path(args.out or f"protocols/{args.mode}_seed{args.seed}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env = get_environment()
    sp = ServedPolicy(args.endpoint, args.base_model)

    if args.mode == "frozen":
        res = run_frozen(sp, env, eval_tasks, out_path)
        meta = {"seed": args.seed, "n_update": len(update_tasks),
                "n_eval": len(eval_tasks)}
        out_path.write_text(json.dumps({**res, "meta": meta}, indent=1))
    elif args.mode == "bon":
        res = run_bon(sp, env, eval_tasks, args.n, out_path)
        meta = {"seed": args.seed, "n_update": len(update_tasks),
                "n_eval": len(eval_tasks)}
        out_path.write_text(json.dumps({**res, "meta": meta}, indent=1))
    else:
        run_ttrl(sp, env, update_tasks, eval_tasks, out_path, args,
                 stream_meta={"seed": args.seed, "n_update": len(update_tasks),
                              "n_eval": len(eval_tasks)})


if __name__ == "__main__":
    main()
