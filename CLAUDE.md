# agent-ttrl2 — Agent Test-Time RL, Fast-Value Project (Resume Core)

**Goal**: A HIGH-STANDARD, FAST-VALUE project on **test-time RL for agents** —
deployment-period policy updates that measurably improve subsequent task performance
in stateful tool-using agents. NOT paper-first (no top-venue requirement): the
deliverable is **publishable-grade engineering + reproducible, honest experimental
evidence** within weeks, strong enough to be a resume CORE project for
大模型后训练算法工程师 positions (a bar that demands: real training, real
improvement, real engineering, verifiable numbers).

**Why agent-ttrl2 (fresh start)**: agent-ttrl (predecessor) ran M0-M6 over ~2 weeks
and ended with a null result (no prequential gain on an easy environment) and a
heavy protocol that slowed everything down. Its lessons are the requirements of
this project.

## Lessons from agent-ttrl (MANDATORY — read `C:\Users\w1828\repos\agent-ttrl\phase01\EXPERIMENT_DECISION_LOG.md` D10-D14 first)

| Lesson | What went wrong | agent-ttrl2 rule |
|--------|----------------|------------------|
| **Environment choice decides everything** | CTS stream initial success 0.625 — no headroom; 8-task stream too short for cumulative learning | **Pick an env with LOW initial success (≤0.3) and real learning headroom FIRST**; verify headroom with the frozen baseline BEFORE building any method |
| **Model-generated branches are unreliable** | branch protocol depended on model action quality → ~1/8 branch updates, weak credit signal | Use **fixed/structured action groups** (R003-style) or strong tool-format enforcement; do not rely on free-form model branch generation |
| **LoRA updates too small to matter** | 4-step updates at lr 5e-6 → negligible behavior drift | **More updates, higher lr, verified drift** (logit drift must be measurable and directional); calibrate "an update changes behavior" BEFORE measuring downstream gain |
| **Pipeline bugs ate the timeline** | max_tokens truncation (128) broke tool JSON; stop_server self-kill; judge2 arg mismatch | **Smoke the FULL pipeline on 1 task end-to-end before any batch** (format → rollout → update → eval); freeze configs (compatibility profile) |
| **Protocol overweight vs speed** | prequential/budget ledger/pre-registered statistics × multi-stage gates → 2+ weeks with no deliverable | **Fast-value protocol**: honest matched-compute comparisons + ≥3 seeds on the MAIN claim, single pre-registered endpoint, everything else as diagnostic. Gates are about correctness of the pipeline, not ceremony |
| **Null = no deliverable** | null result documented but project stalled | **Define the deliverable ladder**: (a) framework release (b) env + baselines (c) main result (improvement OR honest null WITH a usable artifact). A null with a released framework is still resume-worthy; a null with nothing is not |

## Deliverable ladder (4-6 weeks target — check off in order)

1. **M1 (week 1): Test-time RL framework released** — src/ + reproduce.sh + CI + README
   (trl/peft-based LoRA adapter lifecycle: update-at-episode-boundary, shadow eval,
   commit/rollback) + compatibility profile (Qwen3.5-4B-class model, pinned env).
2. **M2 (week 2): Environment + baselines** — chosen env (low initial success,
   replayable), frozen baseline (GRPO-free: frozen policy), BoN/reflexion-style
   test-time baselines, matched budget.
3. **M3 (week 3-4): Main result** — test-time RL vs frozen vs best baseline on
   prequential/future-task success, ≥3 seeds, honest matched compute; PLUS
   mechanism diagnostics (does the credit signal correlate with outcome?).
4. **M4 (week 5-6): Hardening + resume story** — second model family (if budget),
   ablations (commit gate on/off, credit source), technical report
   (TECH_REPORT.md — the resume artifact), GitHub release tag.

**Definition of done for resume**: a GitHub repo with CI green, reproduce.sh working,
a main-result table with honest numbers, and a 2-page technical report — regardless
of whether the main result is positive or null (null is acceptable IF the framework
+ baselines + diagnostics are real and the null is explained mechanistically).

## Model & Environment

- **Model**: latest open-source agent-capable model, **Qwen3.5-4B-class** (verify
  availability + agent/tool-calling format on the server before freezing the
  profile; fall back to Qwen3-4B if 3.5 is not accessible). Frozen base + LoRA.
- **Environment**: replayable tool-calling env with LOW initial success (e.g.,
  AppWorld-style or τ²-bench mock, or a controlled custom env with injectable
  difficulty). Env Gate: frozen policy initial success must be ≤0.3 and show
  headroom (a strong BoN baseline must beat it by a healthy margin).
- Evidence tiers: E_hard (API/schema/state) usable for credit; hidden evaluator
  NEVER in training/selection.

## High standards (resume-level — non-negotiable)

- **Honest evaluation**: unified eval entry, same test set/metrics/units, matched
  compute (budget ledger simple but real), ≥3 seeds on main claim, report effect
  size + CI, contamination freeze (hidden evaluator never in training/selection).
- **Reproducible**: everything in src/, `bash reproduce.sh`, pinned deps
  (uv.lock), artifacts + SHA256SUMS.
- **Publishable-grade engineering**: CI, README (architecture + quickstart +
  results + limitations), decision log (what changed and why), release tags.
- **No fake claims**: any claim traceable to artifact; null results reported with
  mechanism explanation; forbidden: hidden-evaluator-as-reward, same-task-only
  reporting, gain-from-more-tokens explanations.

## Compute

- **Server: autodl3 (TBD — ssh details pending; record here when given)** —
  assume 1-2× 48-84GB GPUs. 4B LoRA + vLLM fits one 48GB card; two cards = trainer
  + rollout separation. Checkpoint to persistent disk; resume-from-checkpoint
  default. Results rsync + git push continuously.
- Budget discipline: this is a resume project — a few hundred GPU·h total is the
  envelope; fast iterations beat big batches.

## GitHub & Deliverables

- **Repo: https://github.com/hxm2023/agentic-TTRL** (code-first: push as we go).
- Release tags: v0.1 (framework), v0.2 (baselines), v0.3 (main result +
  tech report) — each with CI green + SHA256SUMS.
- The resume artifact: TECH_REPORT.md (2 pages: problem, method, results table,
  honest limitations, engineering highlights).

## Pipeline (adapted — fast-value, no ceremony)

- Phase 0 (≤1 day): lesson re-read + env selection + model verification +
  compatibility profile freeze.
- Phase 1 (≤2 days): claim definition (single pre-registered endpoint:
  prequential future-task success; main comparison: TTRL vs frozen vs best
  baseline, ≥3 seeds) + minimal protocol + decision log.
- Phase 2 (weeks 1-5): deliverable ladder M1→M4 with per-milestone review.
- Phase 3 (week 6): TECH_REPORT.md + release + resume bullets.

## Compliance

- Human owns the project; AI participation disclosed where required (open-source
  project: README acknowledgment).
<!-- ARIS:BEGIN -->
## ARIS Skill Scope
ARIS skills installed in this project: 108 entries.
Manifest: `.aris/installed-skills.txt` (lists every skill ARIS installed and its upstream target).
For ARIS workflows, prefer the project-local skills under `.claude/skills/` over global skills.
Do not modify or delete files inside any skill that is a symlink (symlinks point into `/c/Users/w1828/repos/aris_repo`).
Update with: `bash /c/Users/w1828/repos/aris_repo/tools/install_aris.sh`  (re-runnable; reconciles new/removed skills).
<!-- ARIS:END -->
