"""Global gate: empirical-Bernstein e-process commit/rollback for policy updates.

FROZEN 2026-08-22 (agent-ttrl D6) by a 162-config coverage simulator
(agent-ttrl/protocols/sweep_coverage_results.json): fixed-n Hoeffding passed no
operating point; the empirical-Bernstein e-process at these values passes
null-familywise 0.000 <= 0.05 with non-degenerate power:

  alpha_total=0.05, eps_gain=0.01, eps_harm=0.10, n=512, lambda=0.5
  -> null_rate 0.000, sesoi(0.08) 0.111 >= 0.10 floor, strong(0.15) 0.646,
     poisoned 0.000 < sesoi

The gate is paired shadow evaluation of a candidate adapter against the parent:
gain_diffs = candidate - parent on fresh tasks (want > eps_gain);
harm_diffs = parent - candidate on anchor tasks (want <= eps_harm).
"""  # noqa: E501
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

EPS_GAIN = 0.01
EPS_HARM = 0.10
N_FIXED = 512
ALPHA_TOTAL = 0.05
LAMBDA = 0.5


class GateDecision(str, Enum):
    COMMIT = "COMMIT"
    ROLLBACK = "ROLLBACK"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass
class GateOutcome:
    decision: GateDecision
    lcb_gain: float
    ucb_harm: float
    alpha_k: float
    n_gain: int
    n_anchor: int
    reason_codes: list[str]


def alpha_k(k: int, alpha_total: float = ALPHA_TOTAL) -> float:
    """Cross-candidate error budget (summable: sum 6/(pi^2 k^2) = 1)."""
    return 6.0 * alpha_total / (math.pi**2 * k**2)


def _eb_eprocess_radius(alpha_side: float, n: int, variance: float) -> float:
    """Empirical-Bernstein anytime-valid one-sided radius (Waudby-Smith-Ramdas)."""
    if n < 2 or variance <= 0:
        return LAMBDA / 8.0 + math.log(1.0 / alpha_side) / (LAMBDA * n)
    log_inv = math.log(1.0 / alpha_side)
    return math.sqrt(2.0 * variance * log_inv / n) + 7.0 * log_inv / (3.0 * (n - 1))


def decide(
    k: int,
    gain_diffs: list[float],
    harm_diffs: list[float],
    eps_gain: float = EPS_GAIN,
    eps_harm: float = EPS_HARM,
    alpha_total: float = ALPHA_TOTAL,
) -> GateOutcome:
    """Commit iff LCB_gain >= eps_gain and UCB_harm <= eps_harm; else rollback."""
    ak = alpha_k(k, alpha_total)
    a_side = ak / 2.0
    n_gain, n_anchor = len(gain_diffs), len(harm_diffs)
    if n_gain < 2 or n_anchor < 2:
        return GateOutcome(GateDecision.INCONCLUSIVE, 0.0, 0.0, ak, n_gain, n_anchor,
                           ["INSUFFICIENT_SHADOW_SAMPLE"])
    mg = sum(gain_diffs) / n_gain
    mh = sum(harm_diffs) / n_anchor
    vg = sum((x - mg) ** 2 for x in gain_diffs) / (n_gain - 1)
    vh = sum((x - mh) ** 2 for x in harm_diffs) / (n_anchor - 1)
    lcb = mg - _eb_eprocess_radius(a_side, n_gain, vg)
    ucb = mh + _eb_eprocess_radius(a_side, n_anchor, vh)
    codes: list[str] = []
    if lcb < eps_gain:
        codes.append("GAIN_NOT_ESTABLISHED")
    if ucb > eps_harm:
        codes.append("HARM_NOT_BOUNDED")
    decision = GateDecision.COMMIT if (lcb >= eps_gain and ucb <= eps_harm) else GateDecision.ROLLBACK
    return GateOutcome(decision, lcb, ucb, ak, n_gain, n_anchor, codes)
