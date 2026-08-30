import sys, torch
sys.path.insert(0, "/root/autodl-tmp/agent-ttrl2/src")
sys.path.insert(0, "/root/autodl-tmp/tau2-bench/src")
from transformers import AutoModelForCausalLM, AutoTokenizer
from ttrl2.trainer.lora_update import make_lora_model
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
task = stream[1]
ep = Tau2Episode(task)
r = rollout_transformers(pm, tk, ep, env.get_policy(), env.get_tools(),
                         max_turns=20, max_tokens=384, temperature=0.5, seed=0,
                         fmt="llama3_json")
print(f"task {task.id}: calls={r.n_tool_calls} turns={r.turns} succ={r.success}")
for e in r.transcript[:10]:
    print("  ", e.role, "calls=", len(e.tool_calls or []), repr((e.content or "")[:40]))
