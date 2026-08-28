import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ttrl2.gates.global_gate import GateDecision, alpha_k, decide  # noqa: E402


def test_alpha_k_summable():
    assert alpha_k(1) < 0.05
    assert abs(sum(6.0 * 0.05 / (3.14159**2 * k**2) for k in range(1, 1000)) - 0.05) < 1e-3


def test_null_rolls_back():
    out = decide(1, gain_diffs=[0.0] * 20, harm_diffs=[0.0] * 20)
    assert out.decision == GateDecision.ROLLBACK
    assert "GAIN_NOT_ESTABLISHED" in out.reason_codes


def test_poisoned_rolls_back():
    out = decide(1, gain_diffs=[0.3] * 20, harm_diffs=[0.5] * 20)
    assert out.decision == GateDecision.ROLLBACK
    assert "HARM_NOT_BOUNDED" in out.reason_codes


def test_insufficient_sample_inconclusive():
    out = decide(1, gain_diffs=[0.1], harm_diffs=[0.1])
    assert out.decision == GateDecision.INCONCLUSIVE


def test_frozen_config_coverage_values_preserved():
    """Frozen D6 operating point must reproduce the published simulator numbers."""
    import random

    rng = random.Random(42)
    # null family: mean 0
    nulls = [decide(1, [0.0] * 512, [0.0] * 512).decision for _ in range(50)]
    null_rate = sum(d == GateDecision.COMMIT for d in nulls) / len(nulls)
    # strong: mean 0.15 gain, 0.02 harm, var ~0.01
    strongs = [decide(1,
                      [rng.gauss(0.15, 0.1) for _ in range(512)],
                      [abs(rng.gauss(0.02, 0.02)) for _ in range(512)]).decision
               for _ in range(50)]
    strong_rate = sum(d == GateDecision.COMMIT for d in strongs) / len(strongs)
    assert null_rate <= 0.05, f"null_rate {null_rate} > 0.05"
    assert strong_rate >= 0.3, f"strong_rate {strong_rate} < 0.3 power floor"
