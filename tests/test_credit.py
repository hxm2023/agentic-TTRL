import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ttrl2.env.receipts import ToolReceipt  # noqa: E402
from ttrl2.trainer.lora_update import (  # noqa: E402
    GROUP_MODIFY, GroupBaselines, assign_group_credit, detect_conflicts,
)


def _r(tool, ok=True, output=None, error=None, args=None):
    return ToolReceipt(tool, args or {}, ok, output, error)


def test_success_credit_positive():
    receipts = [_r("find_user_id_by_name_zip"),
                _r("return_delivered_order_items")]
    rows = assign_group_credit(receipts, outcome=True, baselines=GroupBaselines(),
                               conflicts={})
    assert all(r.credit == 0.5 for r in rows)
    assert all(r.gate_passed for r in rows)


def test_failure_credit_failure_aware():
    receipts = [_r("find_user_id_by_name_zip"),      # identify: neutral on failure
                _r("get_order_details"),              # read: neutral
                _r("return_delivered_order_items")]   # modify: penalized
    rows = assign_group_credit(receipts, outcome=False, baselines=GroupBaselines(),
                               conflicts={})
    by_tool = {r.action_idx: r for r in rows}
    assert by_tool[0].credit == 0.0        # identify neutral
    assert by_tool[1].credit == 0.0        # read neutral
    assert by_tool[2].credit == -0.3       # modify penalized


def test_conflict_zeroes_group():
    receipts = [_r("return_delivered_order_items")]
    rows = assign_group_credit(receipts, outcome=False, baselines=GroupBaselines(),
                               conflicts={GROUP_MODIFY: ["USER_MISMATCH_ON_MODIFY"]})
    assert rows[0].credit == 0.0
    assert not rows[0].gate_passed
    assert "EVIDENCE_CONFLICT" in rows[0].reason


def test_detect_repeated_fail_same_args():
    receipts = [_r("get_order_details", ok=False, error="Order not found",
                   args={"order_id": "#X"}),
                _r("get_order_details", ok=False, error="Order not found",
                   args={"order_id": "#X"})]
    conflicts = detect_conflicts(receipts, identified_user="u1")
    assert "REPEATED_FAIL_SAME_ARGS" in conflicts["read"]


def test_detect_missing_identity_before_modify():
    receipts = [_r("return_delivered_order_items")]
    conflicts = detect_conflicts(receipts, identified_user=None)
    assert "MISSING_IDENTITY_BEFORE_MODIFY" in conflicts["modify"]


def test_detect_user_mismatch():
    class FakeOrder:
        user_id = "other_user"
    receipts = [_r("return_delivered_order_items", output=FakeOrder())]
    conflicts = detect_conflicts(receipts, identified_user="u1")
    assert "USER_MISMATCH_ON_MODIFY" in conflicts["modify"]
