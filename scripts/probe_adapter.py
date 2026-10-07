"""Sanity check: does the loaded LoRA adapter actually change generations?

Two checks: greedy text over 200 tokens, and a logprob-level comparison of the
next-token distribution. If both are identical the adapter is a no-op (a
failure mode seen with some vLLM versions) and downstream comparisons would
be invalid.

Usage: python scripts/probe_adapter.py --endpoint http://localhost:8003/v1 \
           --base qwen3.5-4b --adapter candidate
"""
from __future__ import annotations

import argparse
import json
import urllib.request


def post(endpoint: str, body: dict) -> dict:
    req = urllib.request.Request(f"{endpoint}/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=300).read())


def ask(endpoint: str, model: str, prompt: str, max_tokens: int = 60) -> str:
    d = post(endpoint, {"model": model, "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens, "temperature": 0.0})
    return d["choices"][0]["message"].get("content") or ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://localhost:8003/v1")
    ap.add_argument("--base", default="qwen3.5-4b")
    ap.add_argument("--adapter", default="candidate")
    args = ap.parse_args()

    prompts = [
        "Say hello in exactly three words.",
        "What is the capital of France? Answer in one word.",
        "Complete: The retail agent should first",
    ]
    n_diff = 0
    for p in prompts:
        a = ask(args.endpoint, args.base, p, max_tokens=200)
        b = ask(args.endpoint, args.adapter, p, max_tokens=200)
        same = a == b
        n_diff += 0 if same else 1
        print(f"{'SAME' if same else 'DIFF'} | base={a[:60]!r} | adapter={b[:60]!r}")
    print(f"\n{n_diff}/{len(prompts)} prompts differ (200 tokens each)")

    def lp(model: str):
        d = post(args.endpoint, {"model": model, "max_tokens": 1, "temperature": 0.0,
                                 "logprobs": True, "top_logprobs": 5,
                                 "messages": [{"role": "user",
                                               "content": "The retail agent should"}]})
        tl = d["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        return [(t["token"], round(t["logprob"], 4)) for t in tl]

    lb, la = lp(args.base), lp(args.adapter)
    print("\ntop-5 logprobs base   :", lb)
    print("top-5 logprobs adapter:", la)
    print("IDENTICAL" if lb == la else "DIFFERENT -> adapter is active")


if __name__ == "__main__":
    main()
