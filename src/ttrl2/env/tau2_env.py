"""tau2-bench retail tool-only episode wrapper.

Protocol (frozen D1): agent -> `apis.*` tool calls -> tool receipts ->
hidden DB-state evaluation. No user simulator in the loop.

- Replayable: each episode loads a FRESH RetailDB from the canonical file and
  applies the task's initialization actions, so the same action sequence on
  the same task ID is deterministic.
- E_hard evidence: every tool receipt (call validity, schema check, output)
  is recorded for the local gate.
- Hidden evaluator: success = final DB state equals the target state derived
  by replaying the task's hidden reference trajectory on a fresh DB. The
  reference trajectory is NEVER shown to the agent.
"""
from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field

from tau2.data_model.tasks import Task
from tau2.domains.retail.data_model import RetailDB
from tau2.domains.retail.utils import RETAIL_DB_PATH
from tau2.domains.retail.tools import RetailTools


@dataclass
class ToolReceipt:
    tool_name: str
    arguments: dict
    ok: bool
    output: object
    error: str | None = None


@dataclass
class EpisodeRecord:
    task_id: str
    receipts: list[ToolReceipt] = field(default_factory=list)
    success: bool | None = None
    turns: int = 0


class Tau2Episode:
    """One replayable task episode over a private RetailDB instance."""

    def __init__(self, task: Task, db: RetailDB | None = None,
                 db_path: str | None = RETAIL_DB_PATH):
        self.task = task
        self._base_db = db if db is not None else RetailDB.load(db_path)
        self.db = deepcopy(self._base_db)
        self.tools = RetailTools(self.db)
        self.record = EpisodeRecord(task_id=task.id)
        self._apply_initial_state()

    def _apply_initial_state(self) -> None:
        init = self.task.initial_state
        if init and init.initialization_actions:
            for fc in init.initialization_actions:
                self.tools.__getattribute__(fc.name)(**fc.arguments)

    def step(self, tool_name: str, arguments: dict) -> ToolReceipt:
        """Execute one tool call, returning the receipt (E_hard evidence)."""
        fn = getattr(self.tools, tool_name, None)
        receipt = ToolReceipt(tool_name=tool_name, arguments=arguments,
                              ok=False, output=None)
        if fn is None:
            receipt.error = f"unknown tool {tool_name}"
            self.record.receipts.append(receipt)
            return receipt
        try:
            out = fn(**arguments)
            receipt.ok = True
            receipt.output = out
        except Exception as e:  # noqa: BLE001 — tool failures are evidence, not crashes
            receipt.error = str(e)
        self.record.receipts.append(receipt)
        return receipt

    def evaluate(self) -> bool:
        """Hidden evaluation: DB-state match vs replayed reference trajectory."""
        target_db = deepcopy(self._base_db)
        target_tools = RetailTools(target_db)
        for a in (self.task.evaluation_criteria.actions or []):
            fn = getattr(target_tools, a.name, None)
            if fn is None:
                continue
            try:
                fn(**a.arguments)
            except Exception:  # reference trajectory must replay; skip bad rows
                continue
        self.record.success = _db_equal(self.db, target_db)
        return self.record.success

    def snapshot_state(self) -> dict:
        """Serializable state fingerprint (for diagnostics, not credit)."""
        return {"orders": len(self.db.orders),
                "users": len(self.db.users),
                "products": len(self.db.products)}


def _db_equal(db_a: RetailDB, db_b: RetailDB) -> bool:
    return db_a.model_dump() == db_b.model_dump()
