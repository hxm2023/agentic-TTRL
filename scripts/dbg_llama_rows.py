import sys
sys.path.insert(0, "/root/autodl-tmp/agent-ttrl2/src")
sys.path.insert(0, "/root/autodl-tmp/tau2-bench/src")
from transformers import AutoTokenizer
from ttrl2.trainer.lora_update import GroupBaselines, build_training_rows
from ttrl2.agent.loop import TranscriptEntry
from ttrl2.env.tau2_env import ToolReceipt
from tau2.domains.retail.environment import get_environment
from ttrl2.agent.loop import build_tool_schemas

M = "/root/autodl-tmp/models/Llama-3.1-8B-Instruct"
tk = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
env = get_environment()
schemas = build_tool_schemas(env.get_tools())

# per-call transcript (llama3_json style)
transcript = [
    TranscriptEntry(role="assistant", content="",
                    tool_calls=[{"id": "t0", "name": "find_user_id_by_name_zip",
                                 "arguments": '{"first_name": "Y"}', }]),
    TranscriptEntry(role="tool", content="yusuf", tool_call_id="t0", name="find_user_id_by_name_zip"),
    TranscriptEntry(role="assistant", content="", tool_calls=None),
    TranscriptEntry(role="assistant", content="I could not complete it."),
]
receipts = [ToolReceipt("find_user_id_by_name_zip", {"first_name": "Y"}, True, "yusuf")]
rows = build_training_rows(transcript, receipts, outcome=False, baselines=GroupBaselines(),
                           conflicts={}, policy="POLICY", task_instr="TASK",
                           tool_schemas=schemas)
print("rows:", len(rows))
for r in rows:
    n_calls = [len(m.get("tool_calls", [])) for m in r["messages"] if "tool_calls" in m]
    print("  adv:", round(r["advantage"], 2), "assistant tool_calls counts:", n_calls)
    try:
        tk.apply_chat_template(r["messages"], tools=schemas, tokenize=False)
        print("  render OK")
    except Exception as e:
        print("  RENDER FAIL:", str(e)[:100])
