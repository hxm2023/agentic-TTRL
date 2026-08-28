"""Served policy: vLLM OpenAI-compatible endpoint with dynamic LoRA adapters.

Lifecycle (frozen D1): GPU1 serves the frozen base + named LoRA adapters.
The trainer (GPU0) produces a candidate adapter directory; the served policy
loads it via POST /v1/load_lora_adapter and rolls out with
model=<adapter_name>. Commit = keep adapter active; rollback = unload and
fall back to the base model name.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from openai import OpenAI

from ttrl2.agent.loop import TranscriptEntry, rollout


@dataclass
class ServedPolicy:
    endpoint: str
    base_model: str
    client: OpenAI | None = None

    def __post_init__(self) -> None:
        self.client = OpenAI(base_url=self.endpoint, api_key="EMPTY")

    def load_adapter(self, name: str, path: str) -> None:
        resp = self.client._client.post(
            f"{self.endpoint}/load_lora_adapter",
            json={"lora_name": name, "lora_path": path})
        resp.raise_for_status()

    def unload_adapter(self, name: str) -> None:
        try:
            resp = self.client._client.post(
                f"{self.endpoint}/unload_lora_adapter",
                json={"lora_name": name})
            resp.raise_for_status()
        except Exception:  # unloading a missing adapter is harmless
            pass

    def rollout_episode(self, episode, policy: str, tools: list,
                        adapter: str | None = None,
                        max_turns: int = 20, max_tokens: int = 256,
                        temperature: float = 0.7, seed: int | None = None):
        model = adapter if adapter else self.base_model
        return rollout(self.client, model, episode, policy, tools,
                       max_turns=max_turns, max_tokens=max_tokens,
                       temperature=temperature, seed=seed)
