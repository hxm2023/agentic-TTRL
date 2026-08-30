import sys, torch
sys.path.insert(0, "/root/autodl-tmp/agent-ttrl2/src")
sys.path.insert(0, "/root/autodl-tmp/tau2-bench/src")
from transformers import AutoModelForCausalLM, AutoTokenizer
from ttrl2.trainer.lora_update import make_lora_model
from ttrl2.agent.loop import build_tool_schemas
from tau2.domains.retail.environment import get_environment, get_tasks
M = "/root/autodl-tmp/models/Llama-3.1-8B-Instruct"
tk = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
base = AutoModelForCausalLM.from_pretrained(M, torch_dtype=torch.bfloat16, device_map={"": 0}).eval()
pm = make_lora_model(base); pm.eval()
env = get_environment()
schemas = build_tool_schemas(env.get_tools())
task = next(t for t in get_tasks("base") if t.id == "108")
instr = task.user_scenario.instructions
up = instr.task_instructions + (f"\n\nKnown information: {instr.known_info}" if instr.known_info else "")
msgs = [{"role": "system", "content": f"You are a retail customer service agent.\n\nPolicy:\n{env.get_policy()}"},
        {"role": "user", "content": up}]
rendered = tk.apply_chat_template(msgs, tools=schemas, tokenize=False, add_generation_prompt=True)
ids = tk(rendered, return_tensors="pt").to("cuda")
with torch.no_grad():
    out = pm.generate(input_ids=ids["input_ids"], max_new_tokens=384, temperature=0.0,
                      do_sample=False, pad_token_id=tk.pad_token_id or tk.eos_token_id,
                      eos_token_id=tk.eos_token_id)
t = tk.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=False)
print("RAW_START")
print(t[:500])
print("RAW_END")
