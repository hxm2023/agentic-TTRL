"""Local gate: evidence-gated action credit with E_hard/E_soft conflict detection.

The local gate decides, per structured action group, whether E_hard evidence
(API receipts, schema validation, state projections) supports the credit
signal. When E_hard and E_soft conflict (e.g., a success receipt whose state
projection is impossible), the gate abstains: zero credit for the whole group
and feeds a drift counter that can halt further adaptation (fail-closed).

Structured action groups (agent-ttrl R003 lesson): never rely on free-form
model branches. The environment fixes the group structure (identify -> read ->
modify), and credit is assigned at group granularity.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GroupVerdict:
    """Credit verdict for one structured action group."""
    status: str
    reason_code: str
    rows: list = field(default_factory=list)


@dataclass
class ConflictGateOutcome:
    abstained: bool
    reason_code: str | None
    verdict: GroupVerdict
    conflicts: list[str] = field(default_factory=list)


def apply_conflict_gate(verdict: GroupVerdict, conflicts: list[str],
                        abstain_on_conflict: bool = True) -> ConflictGateOutcome:
    """Zero all credit rows when the decision-state evidence is conflicted."""
    if abstain_on_conflict and conflicts:
        rows = []
        for r in (verdict.rows or []):
            rows.append(type(r)(action_idx=r.action_idx, credit=0.0,
                                raw_credit=r.raw_credit, gate_passed=False,
                                reason="EVIDENCE_CONFLICT_ABSTAIN"))
        zeroed = GroupVerdict(status="OK", reason_code="EVIDENCE_CONFLICT_ABSTAIN", rows=rows)
        return ConflictGateOutcome(abstained=True, reason_code="EVIDENCE_CONFLICT_ABSTAIN",
                                   verdict=zeroed, conflicts=conflicts)
    return ConflictGateOutcome(abstained=False, reason_code=None,
                               verdict=verdict, conflicts=conflicts)


@dataclass
class DriftMonitor:
    """Accumulated evidence conflicts can halt adaptation (fail-closed)."""
    window: int = 5
    threshold: int = 3
    _recent: list[bool] = field(default_factory=list)

    def observe(self, conflicted: bool) -> bool:
        self._recent.append(conflicted)
        if len(self._recent) > self.window:
            self._recent.pop(0)
        return sum(self._recent) >= self.threshold

    def halt_recommended(self) -> bool:
        return sum(self._recent) >= self.threshold
