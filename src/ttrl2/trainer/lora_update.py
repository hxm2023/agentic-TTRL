"""Episode-boundary LoRA update with evidence-gated group credit.

Design (frozen D1): after each update-phase episode, credit is assigned at
structured action-group granularity:

- Groups: IDENTIFY (user lookup), READ (order/product/user info), MODIFY
  (state-changing calls), OTHER.
- Group credit: A_g = r_episode - b_g, where r_episode in {0,1} is the hidden
  evaluator outcome and b_g is the running mean outcome of prior episodes in
  which group g acted (group-level baseline).
- LOCAL gate: a group whose E_hard evidence conflicts (repeated failure with
  identical args, modify-before-identity, user mismatch on modify) gets
  zeroed credit (abstain) — never reinforce wrong consensus.
- The update is an advantage-weighted policy gradient (GRPO-style) on the
  tool-call token spans of each assistant turn, with a KL penalty against the
  frozen base policy (conservatism at optimizer level):
    L = -mean(A_turn * logp(action | ctx)) + kl_beta * mean(KL)

The adapter accumulates over the update phase; the global gate
(ttrl2.gates.global_gate) decides commit/rollback of the chain.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from peft import LoraConfig, get_peft_model

GROUP_IDENTIFY = "identify"
GROUP_READ = "read"
GROUP_MODIFY = "modify"
GROUP_OTHER = "other"
GROUP_STOP = "stop"  # final answer turn (terminate action); gets episode credit

# Tool -> group mapping for tau2 retail (frozen D1).
TOOL_GROUPS = {
    "find_user_id_by_name_zip": GROUP_IDENTIFY,
    "find_user_id_by_email": GROUP_IDENTIFY,
    "get_user_details": GROUP_IDENTIFY,
    "get_order_details": GROUP_READ,
    "get_product_details": GROUP_READ,
    "get_item_details": GROUP_READ,
    "list_all_product_types": GROUP_READ,
    "cancel_pending_order": GROUP_MODIFY,
    "exchange_delivered_order_items": GROUP_MODIFY,
    "modify_pending_order_address": GROUP_MODIFY,
    "modify_pending_order_items": GROUP_MODIFY,
    "modify_pending_order_payment": GROUP_MODIFY,
    "modify_user_address": GROUP_MODIFY,
    "return_delivered_order_items": GROUP_MODIFY,
    "calculate": GROUP_OTHER,
    "transfer_to_human_agents": GROUP_OTHER,
}


@dataclass
class CreditRow:
    action_idx: int
    group: str
    credit: float
    raw_credit: float
    gate_passed: bool
    reason: str


@dataclass
class GroupBaselines:
    """Running outcome means per group (credit baseline)."""
    means: dict[str, float] = field(default_factory=lambda: {g: 0.5 for g in
                                     (GROUP_IDENTIFY, GROUP_READ, GROUP_MODIFY, GROUP_OTHER)})

    def update(self, group: str, outcome: float) -> None:
        m = self.means
        m[group] = 0.9 * m.get(group, 0.5) + 0.1 * outcome

    def get(self, group: str) -> float:
        return self.means.get(group, 0.5)


def assign_group_credit(receipts, outcome: bool, baselines: GroupBaselines,
                        conflicts: dict[str, list[str]]) -> list[CreditRow]:
    """Failure-aware per-group credit; local gate zeroes conflicted groups.

    Success: +0.5 for every group that acted (reinforce the full pattern).
    Failure: identify/read are CORRECT prerequisites — neutral (0); modify
    attempts that did not lead to success are penalized (-0.3); the early
    stop is penalized via the terminate turn (-0.5). This avoids the
    perverse uniform-negative signal that killed tool calling (verified
    2026-08-29: uniform -0.5 on failed episodes -> drift-13 collapse).
    """
    rows: list[CreditRow] = []
    for i, r in enumerate(receipts):
        g = TOOL_GROUPS.get(r.tool_name, GROUP_OTHER)
        gate_conflicts = conflicts.get(g, [])
        if gate_conflicts:
            rows.append(CreditRow(i, g, 0.0, 0.0, False,
                                  f"EVIDENCE_CONFLICT:{';'.join(gate_conflicts)}"))
            continue
        if outcome:
            raw = 0.5
        elif g == GROUP_MODIFY:
            raw = -0.3 if r.ok else -0.1
        else:
            raw = 0.0
        rows.append(CreditRow(i, g, raw, raw, True,
                              "OK" if r.ok else "CALL_FAILED"))
    return rows


def detect_conflicts(receipts, identified_user: str | None) -> dict[str, list[str]]:
    """E_hard vs E_soft conflict detection per action group (deterministic).

    Conflicts mean the credit signal for that group is unreliable -> the local
    gate zeroes the group's credit (fail-closed):
    - REPEATED_FAIL_SAME_ARGS: a failed call is retried verbatim -> the model
      did not absorb the receipt; its behavior is evidence-conflicted.
    - MISSING_IDENTITY_BEFORE_MODIFY: modify without a prior successful
      identity resolution (policy violation).
    - USER_MISMATCH_ON_MODIFY: modify targets an order whose owner differs
      from the identified user (cross-check of receipts vs. state).
    """
    conflicts: dict[str, list[str]] = {}
    modify_tools = {t for t, g in TOOL_GROUPS.items() if g == GROUP_MODIFY}
    for i, r in enumerate(receipts):
        g = TOOL_GROUPS.get(r.tool_name, GROUP_OTHER)
        cs = conflicts.setdefault(g, [])
        if not r.ok and i + 1 < len(receipts):
            nxt = receipts[i + 1]
            if (nxt.tool_name == r.tool_name
                    and nxt.arguments == r.arguments
                    and not nxt.ok):
                cs.append("REPEATED_FAIL_SAME_ARGS")
        if r.tool_name in modify_tools:
            if identified_user is None:
                cs.append("MISSING_IDENTITY_BEFORE_MODIFY")
            else:
                out = r.output
                uid = getattr(out, "user_id", None)
                if uid is not None and uid != identified_user:
                    cs.append("USER_MISMATCH_ON_MODIFY")
    return {g: v for g, v in conflicts.items() if v}


def make_lora_model(base_model, r: int = 16, alpha: int = 32,
                    target_modules: list[str] | None = None,
                    lora_dropout: float = 0.05):
    """Wrap a frozen base with a trainable LoRA adapter (peft)."""
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]
    cfg = LoraConfig(task_type="CAUSAL_LM", r=r, lora_alpha=alpha,
                     lora_dropout=lora_dropout, target_modules=target_modules)
    model = get_peft_model(base_model, cfg)
    model.print_trainable_parameters()
    return model


def build_training_rows(transcript, receipts, outcome: bool, baselines: GroupBaselines,
                        conflicts: dict[str, list[str]],
                        policy: str, task_instr: str,
                        tool_schemas: list[dict]) -> list[dict]:
    """Turn a rollout transcript into per-assistant-turn training rows.

    Each row: the full chat sequence up to and including one assistant
    tool-call turn, the byte span of its <tool_call> blocks in the rendered
    prompt, and the turn advantage (mean group credit over its calls).
    """
    credits = {r.action_idx: r for r in assign_group_credit(
        receipts, outcome, baselines, conflicts)}
    rows: list[dict] = []
    messages: list[dict] = [
        {"role": "system",
         "content": f"You are a retail customer service agent.\n\nPolicy:\n{policy}"},
        {"role": "user", "content": task_instr},
    ]
    pending: dict | None = None  # assistant turn being accumulated
    last_entry: dict | None = None

    def flush() -> None:
        nonlocal pending
        if pending is None:
            return
        calls = pending["calls"]
        msg = {"role": "assistant", "content": pending["content"],
               "tool_calls": [c for c in calls]}
        if calls:
            row_advs = []
            idx = pending["start_idx"]
            for c in calls:
                g = TOOL_GROUPS.get(c["function"]["name"], GROUP_OTHER)
                cred = credits.get(idx)
                row_advs.append(cred.credit if cred else 0.0)
                idx += 1
            adv = sum(row_advs) / len(row_advs) if row_advs else 0.0
            rows.append({
                "messages": messages + [msg],
                "advantage": adv,
                "tool_names": [c["function"]["name"] for c in calls],
            })
        messages.append(msg)
        pending = None

    start_idx = 0
    for entry in transcript:
        if entry.role == "assistant":
            flush()
            pending = {"content": entry.content, "calls": [], "start_idx": start_idx}
            last_entry = entry
            for tc in (entry.tool_calls or []):
                try:
                    args = json.loads(tc["arguments"])
                except (json.JSONDecodeError, KeyError):
                    args = {}
                pending["calls"].append({"id": tc.get("id", ""), "type": "function",
                                         "function": {"name": tc["name"], "arguments": args}})
                start_idx += 1
        elif entry.role == "tool":
            flush()  # closes the pending assistant turn (appends its row)
            # receipt context truncated: it is context, not action; keeps the
            # training sequence within the 5090 memory envelope
            content = (entry.content or "")[:400]
            messages.append({"role": "tool", "tool_call_id": entry.tool_call_id,
                             "content": content})
    flush()
    # the final answer turn (no tool calls) is the TERMINATE action: give it
    # the episode credit so "stop early on failure" is penalized and
    # "stop after completion" is reinforced (agent-ttrl D12: early stopping
    # was the dominant failure mode).
    if last_entry is not None and not (last_entry.tool_calls or []):
        msg = {"role": "assistant", "content": last_entry.content}
        adv = 0.5 if outcome else -0.3  # soft stop penalty (v4): too hard a
        # penalty (v3's -0.5) pushed the model out of answering AND out of
        # tool calling on unseen tasks; -0.3 keeps the direction, less damage
        rows.append({"messages": messages + [msg], "advantage": adv,
                     "tool_names": []})
        baselines.update(GROUP_STOP, 1.0 if outcome else 0.0)
    return [r for r in rows if r["advantage"] != 0.0]


def _tool_call_span(rendered: str) -> tuple[int, int] | None:
    """Byte span of the current turn's action in the rendered prompt.

    Prefers the LAST <tool_call> block; falls back to the last assistant turn
    (used for the terminate/answer turn, which carries episode credit).
    """
    start = rendered.rfind("<tool_call>")
    if start >= 0:
        end = rendered.rfind("</tool_call>")
        if end > start:
            return start, end + len("</tool_call>")
    a_start = rendered.rfind("<|im_start|>assistant")
    if a_start >= 0:
        return a_start, len(rendered)
    return None


def _chunked_logp(logits: "torch.Tensor", target: "torch.Tensor",
                  chunk_vocab: int = 16384) -> "torch.Tensor":
    """Per-target-token log-probs via chunked logsumexp (avoids full softmax).

    `logits` [T, V]; `target` [T] token ids. Returns [T] logp with gradient
    flowing only through the gathered columns + chunked denominators. The
    float32 conversion happens per chunk so no full [T, V] float copy exists.
    """
    import torch

    T, V = logits.shape
    target_logits = logits.float().gather(-1, target.unsqueeze(-1)).squeeze(-1)
    lse = torch.full((T,), -float("inf"), dtype=torch.float32, device=logits.device)
    for v0 in range(0, V, chunk_vocab):
        chunk = logits[:, v0:v0 + chunk_vocab].float()
        chunk_max = chunk.max(dim=-1).values
        chunk_lse = chunk_max + torch.log(
            torch.exp(chunk - chunk_max.unsqueeze(-1)).sum(dim=-1))
        lse = torch.logaddexp(lse, chunk_lse)
    return target_logits - lse


def grpo_update(model, ref_model, tokenizer, rows, tool_schemas,
                lr: float = 5e-5, kl_beta: float = 0.1, steps: int = 4,
                max_seq_len: int = 8192, grad_clip: float = 1.0) -> dict:
    """Advantage-weighted policy gradient on tool-call spans + KL vs frozen ref.

    Ref = frozen base policy (parent) -> conservatism at optimizer level.
    """
    import torch

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=lr)
    model.train()
    ref_model.eval()
    # Requires flash-linear-attention (fla) kernels: Qwen3.5's torch fallback
    # gated-delta-rule kernel OOMs / breaks autograd on long contexts. With
    # fla installed, a single full-sequence forward + backward fits on one 5090.
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    total_loss = 0.0
    total_tokens = 0
    n_rows = 0

    for _ in range(steps):
        opt.zero_grad()
        for row in rows:
            msgs = row["messages"]
            rendered = tokenizer.apply_chat_template(
                msgs, tools=tool_schemas, tokenize=False,
                add_generation_prompt=False)
            span = _tool_call_span(rendered)
            if span is None:
                continue
            enc = tokenizer(rendered, return_offsets_mapping=True)
            seq = enc["input_ids"]
            if len(seq) > max_seq_len:
                continue
            offsets = enc["offset_mapping"]
            s, e = span
            tok_s = next((i for i, (a, b) in enumerate(offsets) if b > s), 0)
            tok_e = next((i for i, (a, b) in enumerate(offsets) if a >= e),
                         len(seq))
            tok_e = max(tok_e, tok_s + 1)
            if tok_e > len(seq) or tok_s >= tok_e:
                continue
            adv = torch.tensor([row["advantage"]], dtype=torch.float32).to(model.device)
            target = torch.tensor(seq[tok_s:tok_e], dtype=torch.long).to(model.device)
            inp = torch.tensor([seq], dtype=torch.long).to(model.device)
            attn = torch.ones_like(inp)
            # ref first and freed before the grad-forward, so peak memory
            # holds only one logits tensor (bf16) at a time
            with torch.no_grad():
                ref_out = ref_model(input_ids=inp, attention_mask=attn)
                ref_logp = _chunked_logp(ref_out.logits[0, tok_s - 1:tok_e - 1], target)
            del ref_out
            torch.cuda.empty_cache()
            out = model(input_ids=inp, attention_mask=attn)
            tok_logp = _chunked_logp(out.logits[0, tok_s - 1:tok_e - 1], target)
            kl = (tok_logp.detach() - ref_logp).clamp(min=0)
            loss = -(adv * tok_logp).mean() + kl_beta * kl.mean()
            loss.backward()
            total_loss += loss.item()
            total_tokens += len(target)
            n_rows += 1
        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, model.parameters()), grad_clip)
        opt.step()
    return {"loss": total_loss / max(n_rows, 1), "rows": n_rows, "tokens": total_tokens}


def logit_drift(model, ref_model, tokenizer, prompt_text: str,
                top_k: int = 50) -> float:
    """Mean |logp_policy - logp_ref| over the top-k next-token distribution.

    Behavior-drift calibration: must be measurable and directional BEFORE
    downstream gains are interpreted (agent-ttrl D12 lesson).
    """
    import torch

    enc = tokenizer(prompt_text, return_tensors="pt")
    inp = enc["input_ids"].to(model.device)
    attn = enc["attention_mask"].to(model.device)
    with torch.no_grad():
        lr = torch.log_softmax(ref_model(input_ids=inp, attention_mask=attn).logits.float(), dim=-1)
        lp = torch.log_softmax(model(input_ids=inp, attention_mask=attn).logits.float(), dim=-1)
    top_idx = lr[0, -1].topk(top_k).indices
    d = (lp[0, -1, top_idx] - lr[0, -1, top_idx]).abs().mean().item()
    return d
