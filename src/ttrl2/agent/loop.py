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
    max_tokens: int = 256,
    temperature: float = 0.7,
    seed: int | None = None,
    system_override: str | None = None,
    user_prompt_override: str | None = None,
) -> RolloutResult:
    """Run one episode: prompt -> tool calls -> receipts -> repeat.

    `episode` must expose `step(tool_name, arguments) -> receipt` and
    `evaluate() -> bool`. `system_override` / `user_prompt_override` are for
    diagnostic probes only (never used in the main protocol).
    """
    schemas = build_tool_schemas(tools)
    task = episode.task
    instr = task.user_scenario.instructions
    if user_prompt_override is not None:
        user_prompt = user_prompt_override
    else:
        user_prompt = instr.task_instructions
        if instr.known_info:
            user_prompt += f"\n\nKnown information: {instr.known_info}"
        if instr.unknown_info:
            user_prompt += f"\n\nUnknown information: {instr.unknown_info}"

    messages: list[dict] = [
        {"role": "system",
         "content": system_override or f"You are a retail customer service agent.\n\nPolicy:\n{policy}"},
        {"role": "user", "content": user_prompt},
    ]
    result = RolloutResult(success=None, turns=0, n_tool_calls=0)
    n_tool_calls = 0

    for turn in range(max_turns):
        kwargs = dict(temperature=temperature, max_tokens=max_tokens)
        if seed is not None:
            kwargs["seed"] = seed
        resp = client.chat.completions.create(
            model=model, messages=messages,
            tools=schemas or None, **kwargs)
        msg = resp.choices[0].message
        result.transcript.append(TranscriptEntry(
            role="assistant", content=msg.content,
            tool_calls=[{"id": tc.id, "name": tc.function.name,
                         "arguments": tc.function.arguments}
                        for tc in (msg.tool_calls or [])]))
        if not msg.tool_calls:
            break
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            receipt = episode.step(tc.function.name, args)
            n_tool_calls += 1
            messages.append({
                "role": "assistant", "content": msg.content,
                "tool_calls": [{"id": x.id, "type": "function",
                                "function": {"name": x.function.name,
                                             "arguments": x.function.arguments}}
                               for x in msg.tool_calls]})
            messages.append({
                "role": "tool", "tool_call_id": tc.id,
                "content": _receipt_to_text(receipt)})
            result.transcript.append(TranscriptEntry(
                role="tool", content=_receipt_to_text(receipt),
                tool_call_id=tc.id, name=tc.function.name))
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
