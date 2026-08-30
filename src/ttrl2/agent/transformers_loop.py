"""Tool-calling rollout via transformers generate (no vLLM hot-swap).

Used for the TTRL loop: the trainer process rolls out with the CURRENT adapter
directly, so no LoRA serving (broken for Qwen3.5 hybrid in vLLM 0.26) is
needed. Parses the qwen3_xml <tool_call> output format.
"""
from __future__ import annotations

import json
import re

import torch

from ttrl2.agent.loop import RolloutResult, TranscriptEntry, build_tool_schemas

_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=([^>]+)>(.*?)</tool_call>", re.DOTALL)
_PARAM_RE = re.compile(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", re.DOTALL)
_JSON_OBJ_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def parse_qwen3_xml(text: str) -> list[dict]:
    """Parse assistant output into [{name, arguments}...] tool calls."""
    calls = []
    for m in _TOOL_CALL_RE.finditer(text):
        name = m.group(1).strip()
        args = {}
        for pm in _PARAM_RE.finditer(m.group(2)):
            key = pm.group(1).strip()
            val = pm.group(2).strip()
            try:
                args[key] = json.loads(val)
            except json.JSONDecodeError:
                args[key] = val
        calls.append({"name": name, "arguments": args})
    return calls


def _iter_json_objects(text: str):
    """Yield balanced-brace JSON object substrings (nested-safe)."""
    i = 0
    while True:
        start = text.find("{", i)
        if start < 0:
            return
        depth = 0
        j = start
        while j < len(text):
            c = text[j]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    yield text[start:j + 1]
                    i = j + 1
                    break
            j += 1
        else:
            i = start + 1


def parse_llama3_json(text: str) -> list[dict]:
    """Parse Llama-3.1-style JSON tool calls.

    The model emits JSON objects like {"name": ..., "parameters": {...}}
    (optionally wrapped in {"tool_calls": [...]}), possibly several per
    message. Take every object carrying a tool name.
    """
    def args_of(fn: dict) -> dict:
        # Llama-3.1 emits {"name":..., "parameters": {...}} (verified 2026-08-30)
        a = fn.get("arguments", fn.get("parameters", {}))
        return a if isinstance(a, dict) else {}

    calls = []
    try:
        full = json.loads(text)
        if isinstance(full, dict) and isinstance(full.get("tool_calls"), list):
            for tc in full["tool_calls"]:
                fn = tc.get("function", tc)
                if isinstance(fn, dict) and fn.get("name"):
                    calls.append({"name": fn["name"], "arguments": args_of(fn)})
            return calls
    except json.JSONDecodeError:
        pass
    for chunk in _iter_json_objects(text):
        try:
            obj = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            fn = obj.get("function", obj)
            if isinstance(fn, dict) and fn.get("name"):
                calls.append({"name": fn["name"], "arguments": args_of(fn)})
    return calls


def parse_tool_calls(text: str, fmt: str = "qwen3_xml") -> list[dict]:
    if fmt == "llama3_json":
        return parse_llama3_json(text)
    return parse_qwen3_xml(text)


def rollout_transformers(
    model,
    tokenizer,
    episode,
    policy: str,
    tools: list,
    max_turns: int = 20,
    max_tokens: int = 256,
    temperature: float = 0.7,
    seed: int | None = None,
    system_override: str | None = None,
    fmt: str = "qwen3_xml",
) -> RolloutResult:
    """Rollout with a transformers/peft model (adapter already active).

    `system_override` replaces the default system prompt (few-shot probe /
    diagnostic use only). `fmt` selects the tool-call parser
    (qwen3_xml | llama3_json).
    """
    schemas = build_tool_schemas(tools)
    task = episode.task
    instr = task.user_scenario.instructions
    user_prompt = instr.task_instructions
    if instr.known_info:
        user_prompt += f"\n\nKnown information: {instr.known_info}"
    if instr.unknown_info:
        user_prompt += f"\n\nUnknown information: {instr.unknown_info}"

    messages: list[dict] = [
        {"role": "system",
         "content": system_override or
                    f"You are a retail customer service agent.\n\nPolicy:\n{policy}"},
        {"role": "user", "content": user_prompt},
    ]
    result = RolloutResult(success=None, turns=0, n_tool_calls=0)
    if seed is not None:
        torch.manual_seed(seed)

    for turn in range(max_turns):
        rendered = tokenizer.apply_chat_template(
            messages, tools=schemas, tokenize=False,
            add_generation_prompt=True)
        prompt = tokenizer(rendered, return_tensors="pt")["input_ids"]
        prompt = prompt.to(model.device)
        with torch.no_grad():
            out = model.generate(
                input_ids=prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=(temperature > 0),
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_tokens = out[0][prompt.shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=False)
        # trim at the end-of-message marker (template-specific)
        for marker in ("<|im_end|>", "<|eot_id|>"):
            text = text.split(marker)[0]
        calls = parse_tool_calls(text, fmt)
        if not calls:
            result.transcript.append(TranscriptEntry(
                role="assistant", content=text))
            break
        if fmt == "llama3_json":
            # Llama-3.1 template: one tool call per assistant message; the
            # transcript mirrors that (per-call assistant entries) so the
            # training rows render identically
            for i, c in enumerate(calls):
                result.transcript.append(TranscriptEntry(
                    role="assistant", content="",
                    tool_calls=[{"id": f"t{i}", "name": c["name"],
                                 "arguments": json.dumps(c["arguments"])}]))
                messages.append({"role": "assistant", "content": "",
                                 "tool_calls": [{"id": f"t{i}", "type": "function",
                                                 "function": {"name": c["name"],
                                                              "arguments": c["arguments"]}}]})
                receipt = episode.step(c["name"], c["arguments"])
                result.n_tool_calls += 1
                content = ("" if receipt.ok else f"Error: {receipt.error}")
                if receipt.ok:
                    try:
                        content = json.dumps(receipt.output, default=str)[:2000]
                    except TypeError:
                        content = str(receipt.output)[:2000]
                messages.append({"role": "tool", "tool_call_id": f"t{i}",
                                 "content": content})
                result.transcript.append(TranscriptEntry(
                    role="tool", content=content, tool_call_id=f"t{i}",
                    name=c["name"]))
        else:
            result.transcript.append(TranscriptEntry(
                role="assistant", content=text,
                tool_calls=[{"id": f"t{i}", "name": c["name"],
                             "arguments": json.dumps(c["arguments"])}
                            for i, c in enumerate(calls)]))
            messages.append({"role": "assistant", "content": text,
                             "tool_calls": [{"id": f"t{i}", "type": "function",
                                             "function": {"name": c["name"],
                                                          "arguments": c["arguments"]}}
                                            for i, c in enumerate(calls)]})
            for i, c in enumerate(calls):
                receipt = episode.step(c["name"], c["arguments"])
                result.n_tool_calls += 1
                content = ("" if receipt.ok else f"Error: {receipt.error}")
                if receipt.ok:
                    try:
                        content = json.dumps(receipt.output, default=str)[:2000]
                    except TypeError:
                        content = str(receipt.output)[:2000]
                messages.append({"role": "tool", "tool_call_id": f"t{i}",
                                 "content": content})
                result.transcript.append(TranscriptEntry(
                    role="tool", content=content, tool_call_id=f"t{i}",
                    name=c["name"]))
        result.turns = turn + 1
    result.success = episode.evaluate()
    return result
