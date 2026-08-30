import sys, torch
sys.path.insert(0, "/root/autodl-tmp/agent-ttrl2/src")
sys.path.insert(0, "/root/autodl-tmp/tau2-bench/src")
from transformers import AutoModelForCausalLM, AutoTokenizer
from ttrl2.trainer.lora_update import (make_lora_model, GroupBaselines,
    build_training_rows, detect_conflicts)
from ttrl2.agent.transformers_loop import rollout_transformers
from ttrl2.env.tau2_env import Tau2Episode
from tau2.domains.retail.environment import get_environment, get_tasks
import random
M = "/root/autodl-tmp/models/Llama-3.1-8B-Instruct"
tk = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
base = AutoModelForCausalLM.from_pretrained(M, torch_dtype=torch.bfloat16, device_map={"": 0}).eval()
pm = make_lora_model(base); pm.eval()
env = get_environment()
tasks = get_tasks("base")
rng = random.Random(0)
stream = sorted(tasks, key=lambda t: t.id); rng.shuffle(stream)

# few-shot system prompt (same as run_stream --fewshot)
t0ref = next(t for t in tasks if t.id == "0")
ex_lines = ["Here is an example of completing a similar request:"]
for a in (t0ref.evaluation_criteria.actions or []):
    argstr = ", ".join(f"{k}={v}" for k, v in a.arguments.items())
    ex_lines.append(f"  apis.{a.name}({argstr})")
sys_prompt = (f"You are a retail customer service agent.\n\n"
              f"Policy:\n{env.get_policy()}\n\n" + "\n".join(ex_lines) + "\n\n"
              "IMPORTANT: follow the example's pattern — use the tools to "
              "complete the request fully, including any exchange, return, "
              "cancel or modification.")

task = stream[1]  # task 86
ep = Tau2Episode(task)
r = rollout_transformers(pm, tk, ep, env.get_policy(), env.get_tools(),
                         max_turns=12, max_tokens=384, temperature=0.5, seed=0,
                         fmt="llama3_json", system_override=sys_prompt)
print(f"task {task.id}: calls={r.n_tool_calls} turns={r.turns}")
identified = None
for rc in ep.record.receipts:
    if rc.tool_name in ("find_user_id_by_name_zip", "find_user_id_by_email") and rc.ok:
        identified = str(rc.output).strip()
print("identified:", identified)
conflicts = detect_conflicts(ep.record.receipts, identified)
print("conflicts:", {k: v for k, v in conflicts.items()})
instr = task.user_scenario.instructions
tp = instr.task_instructions + (f"\n\nKnown information: {instr.known_info}" if instr.known_info else "")
rows = build_training_rows(r.transcript, ep.record.receipts, r.success or False,
                           GroupBaselines(), conflicts, env.get_policy(), tp, [])
print("rows:", len(rows), "advs:", [round(x["advantage"], 2) for x in rows])
# what are the modify receipts?
modify_tools = {"cancel_pending_order", "exchange_delivered_order_items",
                "modify_pending_order_address", "modify_pending_order_items",
                "modify_pending_order_payment", "modify_user_address",
                "return_delivered_order_items"}
for rc in ep.record.receipts:
    if rc.tool_name in modify_tools:
        uid = getattr(rc.output, "user_id", None) if rc.ok else None
        print(f"  modify: {rc.tool_name} ok={rc.ok} user={uid}")
