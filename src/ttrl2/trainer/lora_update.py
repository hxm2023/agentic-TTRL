"""Episode-boundary LoRA update with evidence-gated group credit.

Design (frozen D1): after each update-phase episode, credit is assigned at
structured action-group granularity:

- Groups: IDENTIFY (user lookup), READ (order/product/user info), MODIFY
  (state-changing calls), OTHER.
- Group credit: A_g = r_episode - b_g, where r_episode in {0,1} is the hidden
  evaluator outcome and b_g is the running mean outcome of prior episodes in
  which group g acted (group-level baseline).
- LOCAL gate: a group whose E_hard evidence conflicts (receipt vs state
  projection mismatch, schema violations on mutating calls) gets zeroed
  credit (abstain) — never reinforce wrong consensus.
- The update is a GRPO-style step on the tool-call token spans of each
  non-abstained group: maximize log p(action tokens | context) * A_g.

The adapter lifecycle (train -> shadow-eval -> commit/rollback) is managed by
the global gate (ttrl2.gates.global_gate) at chain level.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from peft import LoraConfig, get_peft_model

GROUP_IDENTIFY = "identify"
GROUP_READ = "read"
GROUP_MODIFY = "modify"
GROUP_OTHER = "other"

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
                        conflicts: dict[int, list[str]]) -> list[CreditRow]:
    """Per-group credit from episode outcome; local gate zeroes conflicted groups."""
    rows: list[CreditRow] = []
    acted = {GROUP_IDENTIFY: False, GROUP_READ: False, GROUP_MODIFY: False, GROUP_OTHER: False}
    for i, r in enumerate(receipts):
        g = TOOL_GROUPS.get(r.tool_name, GROUP_OTHER)
        acted[g] = True
    reward = 1.0 if outcome else 0.0
    for i, r in enumerate(receipts):
        g = TOOL_GROUPS.get(r.tool_name, GROUP_OTHER)
        b = baselines.get(g)
        raw = reward - b
        gate_conflicts = conflicts.get(g, [])
        if gate_conflicts:
            rows.append(CreditRow(i, g, 0.0, raw, False,
                                  f"EVIDENCE_CONFLICT:{';'.join(gate_conflicts)}"))
        elif not r.ok:
            # failed calls carry their own (negative) evidence; keep small negative
            rows.append(CreditRow(i, g, min(raw, -0.1), raw, True, "CALL_FAILED"))
        else:
            rows.append(CreditRow(i, g, raw, raw, True, "OK"))
        if acted[g]:
            baselines.update(g, reward)
    return rows


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
    for p in model.base_model.model.parameters():
        p.requires_grad = False
    return model
