"""Generate the project figures from the result manifests (CPU-only).

Outputs (figures/):
- eval_contrast.png    all arms' sealed-eval success rates
- drift_trajectories.png  per-update logit drift for the main runs
- failure_taxonomy.png failure-mode counts on the sealed eval set
- behavior_outcome.png behavior-change vs outcome-change dissociation
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "protocols"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)


def load(name: str) -> dict:
    return json.load(open(P / name, encoding="utf-8"))


def fig_eval_contrast() -> None:
    arms = [
        ("Frozen (vLLM)", load("frozen_seed0.json")["success_rate"]),
        ("BoN-4 @0.7", load("bon4_seed0.json")["success_rate"]),
        ("BoN-4 @1.2", load("bon4_t1.2_seed0.json")["rate"]),
        ("Prompt probe", load("probe_prompt_seed0.json")["rate"]),
        ("Few-shot 4B", load("fewshot_probe.json")["success_rate"]),
        ("Few-shot 9B", load("fewshot_probe_9b.json")["success_rate"]),
        ("Few-shot Llama", load("fewshot_probe_llama.json")["success_rate"]),
        ("TTRL v3 s0", load("ttrl_seed0_v3.json")["eval"]["candidate_rate"]),
        ("TTRL v3 s1", load("ttrl_seed1_v3.json")["eval"]["candidate_rate"]),
        ("TTRL v4", load("ttrl_seed0_v4.json")["eval"]["candidate_rate"]),
        ("Strong replay", load("success_replay_strong.json")["eval"]["candidate_rate"]),
        ("Llama replay", load("candidate_llama_eval.json")["success_rate"]),
    ]
    names = [a[0] for a in arms]
    vals = [a[1] for a in arms]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    bars = ax.bar(names, vals, color="#4C72B0", edgecolor="black", linewidth=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}",
                ha="center", fontsize=8)
    ax.axhline(0.109, color="red", linestyle="--", linewidth=1)
    ax.text(len(names) - 0.5, 0.113, "frozen baseline 0.109", color="red",
            fontsize=9, ha="right")
    ax.set_ylim(0, 0.14)
    ax.set_ylabel("sealed eval success rate (46 tasks)")
    ax.set_title("All arms: no intervention moves the sealed rate (exploratory, n=1-2)")
    plt.xticks(rotation=30, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "eval_contrast.png", dpi=150)
    plt.close()


def fig_drift() -> None:
    runs = [("v3 seed0", "ttrl_seed0_v3.json"), ("v3 seed1", "ttrl_seed1_v3.json"),
            ("v4 seed0", "ttrl_seed0_v4.json")]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for label, name in runs:
        d = load(name)
        drifts = [u["drift"] for u in d["update_phase"]]
        ax.plot(drifts, label=f"{label} (max {max(drifts):.1f})", linewidth=1.5)
    ax.axhline(2.0, color="gray", linestyle=":", linewidth=1)
    ax.text(1, 2.05, "drift guard threshold", color="gray", fontsize=8)
    ax.set_xlabel("update episode")
    ax.set_ylabel("logit drift vs frozen base (top-50 next-token)")
    ax.set_title("Behavior drift: updates change the policy (verified)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG / "drift_trajectories.png", dpi=150)
    plt.close()


def fig_taxonomy() -> None:
    d = load("failure_taxonomy.json")
    counts = d["counts"]
    labels = list(counts.keys())
    vals = [counts[k] for k in labels]
    colors = ["#C44E52", "#DD8452", "#55A868", "#8172B3", "#4C72B0"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, vals, color=colors[:len(labels)], edgecolor="black",
                  linewidth=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.3, str(v), ha="center")
    ax.set_ylabel("tasks (46 sealed)")
    ax.set_title("Failure-mode taxonomy (frozen policy): 70% behavioral")
    plt.tight_layout()
    plt.savefig(FIG / "failure_taxonomy.png", dpi=150)
    plt.close()


def fig_behavior_outcome() -> None:
    runs = [("v3 s0", "ttrl_seed0_v3.json"), ("v3 s1", "ttrl_seed1_v3.json"),
            ("v4 s0", "ttrl_seed0_v4.json"), ("strong replay", "success_replay_strong.json")]
    labels, beh, outc = [], [], []
    for label, name in runs:
        d = load(name)
        ev = d["eval"]
        beh.append(ev.get("behavior_diff", ev.get("behavior-diff", 0)))
        outc.append(len(ev.get("flips", [])))
        labels.append(label)
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([i - 0.2 for i in x], beh, width=0.4, label="behavior-changed tasks",
           color="#4C72B0", edgecolor="black", linewidth=0.6)
    ax.bar([i + 0.2 for i in x], outc, width=0.4, label="outcome flips",
           color="#C44E52", edgecolor="black", linewidth=0.6)
    for i, v in zip([i - 0.2 for i in x], beh):
        ax.text(i, v + 0.5, str(v), ha="center", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("tasks (of 46 eval)")
    ax.set_title("Behavior changes but outcomes do not (the dissociation)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG / "behavior_outcome.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    fig_eval_contrast()
    fig_drift()
    fig_taxonomy()
    fig_behavior_outcome()
    print("figures written to", FIG)
