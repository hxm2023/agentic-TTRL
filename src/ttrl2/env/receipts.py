"""Core evidence types (decoupled from the tau2 package for testability)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ToolReceipt:
    """E_hard evidence: one executed tool call's outcome."""
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
