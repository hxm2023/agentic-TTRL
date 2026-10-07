"""Tool-calling agent loop against a vLLM OpenAI-compatible endpoint.

Rollout protocol (frozen D1): system prompt = domain policy + tool schemas;
user prompt = task instructions; the model may emit multiple tool calls per
turn; every receipt is appended and the loop continues until the model stops
calling tools or the turn cap is hit. The full transcript is recorded for
E_hard evidence and drift diagnostics.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from openai import OpenAI


@dataclass
class TranscriptEntry:
    role: str
    content: str | None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    name: str | None = None


@dataclass
class RolloutResult:
    success: bool | None
    turns: int
    n_tool_calls: int
    transcript: list[TranscriptEntry] = field(default_factory=list)
    conflict_events: list[str] = field(default_factory=list)


def build_user_prompt(task) -> str:
    """The user's request, as the agent would hear it (single-shot protocol).

    tau2's StructuredUserInstructions separates the user's actual request
    (`reason_for_call`) from simulator-side behaviour guidance
    (`task_instructions`). Only the former belongs in the agent's prompt;
    `known_info` is information the user can supply when asked.
    """
    instr = task.user_scenario.instructions
    if isinstance(instr, str):
        return instr
    parts = [instr.reason_for_call]
    if instr.known_info:
        parts.append(f"Known information: {instr.known_info}")
    return "\n\n".join(parts)


def stub_user_reply(assistant_text: str) -> str:
    """Scripted user reply for the no-user-simulator protocol (v2).

    The retail policy requires the agent to confirm before writing. Without a
    user simulator the episode would stall at the confirmation question, so
    the harness answers as the customer would: confirm when asked, otherwise
    push the agent to proceed.
    """
    t = (assistant_text or "").lower()
    if "confirm" in t or "proceed" in t or "go ahead" in t:
        return "Yes, please proceed."
    if "?" in t:
        return ("I'm not sure about that — please choose whatever matches my "
                "request and go ahead with it.")
    return "Please go ahead and complete my request."


def build_tool_schemas(tools: list) -> list[dict]:
    """Convert tau2 Tool objects to OpenAI function-calling schemas."""
    schemas = []
    for t in tools:
        try:
            params = t.params.model_json_schema()
        except AttributeError:
            params = {"type": "object", "properties": {}}
        schemas.append({
            "type": "function",
            "function": {
                "name": t.name,
                "description": (t.long_desc or t.short_desc or "")[:1024],
                "parameters": params,
            },
        })
    return schemas


def rollout(
    client: OpenAI,
    model: str,
    episode,
    policy: str,
    tools: list,
    max_turns: int = 20,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    seed: int | None = None,
    system_override: str | None = None,
    user_prompt_override: str | None = None,
    n_stub: int = 3,
    no_think: bool = False,
) -> RolloutResult:
    """Run one episode: prompt -> tool calls -> receipts -> repeat.

    Protocol v2: the user prompt is the actual request (reason_for_call), a
    scripted user answers when the agent talks instead of acting (`n_stub`
    replies max), and Qwen3.5 thinking is disabled for compact tool-call turns.
    `episode` must expose `step(tool_name, arguments) -> receipt` and
    `evaluate() -> bool`; the overrides are for diagnostic probes only.
    """
    schemas = build_tool_schemas(tools)
    task = episode.task
    if user_prompt_override is not None:
        user_prompt = user_prompt_override
    else:
        user_prompt = build_user_prompt(task)

    messages: list[dict] = [
        {"role": "system",
         "content": system_override or f"You are a retail customer service agent.\n\nPolicy:\n{policy}"},
        {"role": "user", "content": user_prompt},
    ]
    result = RolloutResult(success=None, turns=0, n_tool_calls=0)
    n_tool_calls = 0
    stubs_left = n_stub

    for turn in range(max_turns):
        kwargs = dict(temperature=temperature, max_tokens=max_tokens)
        if seed is not None:
            kwargs["seed"] = seed
        if no_think:
            kwargs["extra_body"] = {"chat_template_kwargs":
                                    {"enable_thinking": False}}
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages,
                tools=schemas or None, **kwargs)
        except Exception as e:  # vLLM 400 on malformed tool-call args
            try:
                resp = client.chat.completions.create(
                    model=model, messages=messages,
                    tools=schemas or None, **kwargs)
            except Exception as e2:
                result.transcript.append(TranscriptEntry(
                    role="assistant", content=f"[api-error] {e2}"))
                break
        msg = resp.choices[0].message
        api_calls = [{"id": tc.id, "name": tc.function.name,
                      "arguments": tc.function.arguments}
                     for tc in (msg.tool_calls or [])]
        content = msg.content
        if not api_calls and msg.content:
            # server-side parsing may drop calls it cannot JSON-ify; parse
            # the qwen3 XML client-side as a fallback (same as transformers)
            from ttrl2.agent.transformers_loop import _TOOL_CALL_RE, parse_tool_calls
            local = parse_tool_calls(msg.content, "qwen3_xml")
            if local:
                api_calls = [{"id": f"t{i}", "name": c["name"],
                              "arguments": json.dumps(c["arguments"])}
                             for i, c in enumerate(local)]
                content = _TOOL_CALL_RE.sub("", msg.content).strip()
        result.transcript.append(TranscriptEntry(
            role="assistant", content=content, tool_calls=api_calls))
        if not api_calls:
            if stubs_left > 0:
                stubs_left -= 1
                reply = stub_user_reply(content)
                messages.append({"role": "assistant", "content": content or ""})
                messages.append({"role": "user", "content": reply})
                result.transcript.append(TranscriptEntry(role="user", content=reply))
                result.turns = turn + 1
                continue
            break
        messages.append({
            "role": "assistant", "content": content,
            "tool_calls": [{"id": c["id"], "type": "function",
                            "function": {"name": c["name"],
                                         "arguments": c["arguments"]}}
                           for c in api_calls]})
        for tc in api_calls:
            try:
                args = json.loads(tc["arguments"])
            except json.JSONDecodeError:
                args = {}
            receipt = episode.step(tc["name"], args)
            n_tool_calls += 1
            messages.append({
                "role": "tool", "tool_call_id": tc["id"],
                "content": _receipt_to_text(receipt)})
            result.transcript.append(TranscriptEntry(
                role="tool", content=_receipt_to_text(receipt),
                tool_call_id=tc["id"], name=tc["name"]))
        result.turns = turn + 1
    result.n_tool_calls = n_tool_calls
    result.success = episode.evaluate()
    return result


def _receipt_to_text(receipt) -> str:
    if not receipt.ok:
        return f"Error: {receipt.error}"
    try:
        return json.dumps(receipt.output, default=str)[:2000]
    except TypeError:
        return str(receipt.output)[:2000]
