"""Full-transcript probe: why does the model stop before the write call?

Runs ONE task with the v2 prompt through vLLM and prints, per turn:
finish_reason, tool calls, and the tail of the content. Then evaluates the DB.

Usage (server):
  python scripts/probe_transcript.py --task-id 1 --max-tokens 1024 \
      --endpoint http://localhost:8002/v1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openai import OpenAI  # noqa: E402

from ttrl2.agent.loop import build_tool_schemas, build_user_prompt  # noqa: E402
from ttrl2.env.tau2_env import Tau2Episode  # noqa: E402
from tau2.domains.retail.environment import get_environment, get_tasks  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", default="1")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-turns", type=int, default=20)
    ap.add_argument("--user-stub", type=int, default=0,
                    help="max scripted user replies after a no-tool-call turn")
    ap.add_argument("--stub-reply", default="Yes, please go ahead and proceed.")
    ap.add_argument("--no-think", action="store_true",
                    help="pass chat_template_kwargs.enable_thinking=false")
    ap.add_argument("--endpoint", default="http://localhost:8002/v1")
    ap.add_argument("--base-model", default="qwen3.5-4b")
    args = ap.parse_args()

    env = get_environment()
    policy = env.get_policy()
    tools = env.get_tools()
    schemas = build_tool_schemas(tools)
    task = next(t for t in get_tasks("base") if t.id == args.task_id)
    ep = Tau2Episode(task)
    client = OpenAI(base_url=args.endpoint, api_key="EMPTY")

    messages = [
        {"role": "system",
         "content": f"You are a retail customer service agent.\n\nPolicy:\n{policy}"},
        {"role": "user", "content": build_user_prompt(task)},
    ]
    print(f"== task {args.task_id} | user prompt: {messages[1]['content'][:200]!r}")
    stub_left = args.user_stub

    for turn in range(args.max_turns):
        extra = ({"chat_template_kwargs": {"enable_thinking": False}}
                 if args.no_think else None)
        resp = client.chat.completions.create(
            model=args.base_model, messages=messages, tools=schemas,
            temperature=args.temperature, max_tokens=args.max_tokens,
            extra_body=extra)
        ch = resp.choices[0]
        msg = ch.message
        tcs = msg.tool_calls or []
        print(f"\n-- turn {turn}: finish={ch.finish_reason} "
              f"n_calls={len(tcs)} content_len={len(msg.content or '')}")
        if tcs:
            for tc in tcs:
                print(f"   call: {tc.function.name} {tc.function.arguments[:200]}")
        else:
            print(f"   final content: {(msg.content or '')[-600:]!r}")
            if stub_left > 0:
                stub_left -= 1
                print(f"   [user-stub] {args.stub_reply!r} ({stub_left} left)")
                messages.append({"role": "assistant", "content": msg.content or ""})
                messages.append({"role": "user", "content": args.stub_reply})
                continue
            break
        messages.append({"role": "assistant", "content": msg.content,
                         "tool_calls": [{"id": x.id, "type": "function",
                                         "function": {"name": x.function.name,
                                                      "arguments": x.function.arguments}}
                                        for x in tcs]})
        for tc in tcs:
            try:
                a = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                a = {}
            r = ep.step(tc.function.name, a)
            out = json.dumps(r.output, default=str)[:1500] if r.ok else f"Error: {r.error}"
            print(f"   receipt ok={r.ok}: {out[:200]}")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": out})
    print(f"\n== success: {ep.evaluate()}")
    ref = [a.name for a in (task.evaluation_criteria.actions or [])]
    print(f"== reference actions: {ref}")


if __name__ == "__main__":
    main()
