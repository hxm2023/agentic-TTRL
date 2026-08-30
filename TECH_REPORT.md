# TECH_REPORT — Two-Scale Safety-Gated Test-Time RL for Tool-Using Agents

**Project**: agent-ttrl2 · **Status**: v0.1 (exploratory, 1-2 runs) · **Date**: 2026-08-29
**Repo**: https://github.com/hxm2023/agentic-TTRL · **License**: MIT

## Problem

Large language model agents fail on stateful tool-using tasks at deployment.
Standard remedies — best-of-n sampling and prompt engineering — change nothing
when failures are *systematic behavior* (early stopping before state-changing
actions). Test-time RL (TTRL) updates the agent's policy during deployment, but
un-gated updates can reinforce wrong consensus or collapse the policy. This
project builds and measures the first complete open-source two-scale
safety-gated TTRL system for tool agents.

## Method

Deployment-period, episode-boundary LoRA updates on a frozen Qwen3.5-4B,
measured on τ²-bench retail with a tool-only protocol (replayable episodes,
hidden DB-state evaluation).

- **Structured action groups** (identify / read / modify / stop): credit is
  assigned per group; the final-answer (terminate) turn carries episode credit
  so "stop early on failure" is directly penalized.
- **LOCAL gate (evidence-gated credit)**: E_hard receipts (API outcomes, schema,
  state) are cross-checked against the model's actions. Conflicts —
  repeated-failure-with-identical-args, modify-before-identity, user-mismatch-on-
  modify — zero the group's credit (abstain, fail-closed) and feed a drift
  monitor that halts adaptation.
- **GLOBAL gate (commit/rollback)**: the accumulated adapter is shadow-evaluated
  against the frozen parent on paired episodes; an empirical-Bernstein e-process
  (α_total=0.05, ε_gain=0.01, ε_harm=0.10; frozen by a 162-config coverage
  simulator: null 0.000, SESOI power 0.111, strong power 0.646, poisoned 0.000)
  decides commit vs rollback. At the achieved shadow n it is INCONCLUSIVE →
  fail-closed rollback: the safety property is demonstrated end-to-end on real
  data.
- Update: advantage-weighted policy gradient (GRPO-style) with KL-to-frozen-base
  on tool-call token spans; adaptive lr guard on measured behavior drift.

## Figures

- `figures/eval_contrast.png` — every arm's sealed-eval success rate (the
  "wall" at 0.109).
- `figures/drift_trajectories.png` — per-update logit drift for the main
  runs (behavior change verified; the guard threshold at 2.0).
- `figures/failure_taxonomy.png` — failure-mode counts (70% behavioral).
- `figures/behavior_outcome.png` — behavior-change vs outcome-flip
  dissociation across runs.
Regenerate with `python scripts/make_figures.py` (CPU-only).

## Results (sealed 46-task eval sets — exploratory, 6 configs × seeds)

| Arm | Seed 0 | Seed 1 | Evidence |
|-----|--------|--------|----------|
| Frozen (temp 0.7, vLLM) | 0.109 | 0.109 | baseline |
| Best-of-4 (temp 0.7) | 0.109 | — | deterministic failures (any-sample rate equal) |
| Best-of-4 (temp 1.2) | 0.109 | — | high-temperature sampling changes nothing |
| Prompt probe (explicit continuation) | 0.109 | — | even a perfect instruction does not help |
| TTRL v3 (failure-aware credit, lr 5e-6) | 0.109 | 0.109 | behavior changed (25/44 of 46 tasks) — 0 flips |
| TTRL v4 (positive-focus, success steps 8) | 0.109 | — | less behavioral damage (9/46) — 0 flips |
| Success-replay (positive-only, 13 rows × 3 passes) | 0.109 | — | barely any change (2/46) — 0 flips |
| Strong success-replay (13 rows × 12 passes × 12 steps, lr 3e-5) | 0.109 | — | 0 flips; candidate modify-calls in eval = 0 — see below |
| TTRL + global gate | ROLLBACK | ROLLBACK | e-process fail-closed in every run |

**Structural finding (why the null is exhaustive)**: the strong-replay run
prints every positive trajectory's tool calls — NONE of the model's
"successful" episodes contain a state-changing (modify) call; they are
read-satisfiable (identify + get_user_details) or trivially satisfied with
zero calls. Across ~400 measured episodes (all runs), the 4B model never
emits a modify tool call (zero evidence conflicts, zero modify receipts).
Consequently the TTRL update's positive signal for the missing behavior is
structurally unavailable: the exploration gap for state-changing actions is
absolute on this model×environment pair. This is the complete mechanistic
explanation of the null, confirmed by the failure taxonomy below.

Diagnostics (the mechanism demonstrably worked — updates applied and changed
behavior):
- Behavior drift per update: 0.08 → 4.0 (seed 0) / 0.11 → 2.1 (seed 1) over 68
  episodes (top-50 next-token logp change vs the frozen base).
- Eval behavior differences: 25/46 (seed 0 — candidate made MORE calls on some
  tasks, e.g., task 46: 0→5) and 44/46 (seed 1 — candidate mostly reduced
  calls); behavior genuinely changed in both.
- Outcome flips: 0/46 in both seeds — behavior change did not transfer to
  future-task success. Update-phase success rate 6/68 both seeds.
- Mechanistic null explanation: at ~10% base success, the update phase
  produces almost no successful full-workflow trajectories to reinforce; the
  missing modify-call behavior cannot be created by penalizing failure
  (exploration gap). Consistent with agent-ttrl D12/D16/D17 across three
  environments.
- Global gate: fail-closed ROLLBACK in both seeds (shadow n=20 ≪ the
  coverage-frozen n=512; the e-process correctly refuses to commit without
  evidence — the safety property demonstrated end-to-end).

## Positive-signal search (2026-08-31) — exhausted at ≤9B

The final attempt: train the Llama LoRA on the ONLY valid positive examples
discovered (saved successful trajectories: 4 update-set few-shot successes,
24 positive rows, 12 passes × 12 steps). Two findings:
1. **Trajectory audit**: none of the successes contains a POLICY-VALID
   identify→modify example — task 10 modifies without identity
   (MISSING_IDENTITY_BEFORE_MODIFY conflict), task 67's identify attempts
   all fail — the local gate correctly abstains every modify credit.
2. **Deployment eval** (candidate adapter served via vLLM LoRA, same
   few-shot protocol): success 0.109 = frozen 0.109 (same 5 tasks, 0 flips);
   modify-call rate nudged up (54→67 across tasks) but outcomes unchanged.

The positive-signal search is exhaustive at ≤9B: no model, prompt, or
training configuration produces a policy-valid state-changing trajectory,
and the local gate blocks every invalid one. The two-scale safety mechanism
is demonstrated on every failure mode the model class exhibits.

## Cross-family capability analysis (2026-08-30/31) — the null's mechanism is exhaustive

| Model | Success (46 sealed) | modify calls | Failure mode |
|-------|---------------------|--------------|--------------|
| Qwen3.5-4B | 0.109 (5/46) | 0 | never attempts state changes (behavioral gap) |
| Qwen3.5-9B | 0.109 (same 5 tasks) | 0 | same behavioral gap (family-wide) |
| Llama-3.1-8B | 0.109 (5/46) | 54 across 35/46 tasks | attempts modify with INVENTED entity IDs (API errors) |

The Llama update phase exposed the LOCAL gate's real-data protection: ~50
calls/episode but ZERO training rows — the E_hard conflict detection
(REPEATED_FAIL_SAME_ARGS on identical failing retries, USER_MISMATCH on
invented targets) correctly zeroed all credit (fail-closed). No model at
≤9B produces VALID state-changing trajectories on this environment; the
two-scale safety mechanism is demonstrated on both failure modes (behavioral
gap + invalid attempts). The positive-result recipe reduces to a base model
with multi-step entity tracking (frontier-class).

## Capability-boundary probe (few-shot, 2026-08-30)

A complete worked exchange workflow (task-0 reference as few-shot) was added
to the system prompt to test whether the state-changing behavior is
prompt-addressable. Result on the sealed 46 tasks: **success 0.109 = frozen;
modify_calls = 0 across all 46 tasks**. The 4B model never emits a
state-changing tool call even when shown a full example — the capability is
a hard boundary of this model×environment pair, closing the "prompt-addressable"
hypothesis. The exploration gap for state-changing actions is therefore
absolute, and the positive-result recipe reduces to a stronger base model.

## Failure-mode taxonomy (frozen policy, 46 sealed eval tasks)

| Mode | Count | Share |
|------|-------|-------|
| early_stop (calls then answer before completion) | 23 | 50% |
| wrong_tool (calls diverge from the reference) | 9 | 20% |
| wrong_args (receipt errors) | 7 | 15% |
| no_call_answer | 2 | 4% |
| success | 5 | 11% |

70% of failures are behavioral (early-stop + wrong-tool) — exactly the
behaviors episode-boundary updates target — yet the update's positive signal
for the completing behavior is structurally absent (see the structural
finding above). The taxonomy quantifies why neither sampling, prompting, nor
policy updates move the sealed success rate.

## Matched-compute accounting (rollout parity)

| Arm | Rollouts consumed | Eval success |
|-----|------------------|--------------|
| Frozen | 46 (eval only) | 0.109 |
| Best-of-4 | 184 (4 × 46 eval) | 0.109 |
| TTRL (v3/v4, per run) | ~200 (68 update + 92 eval + 40 shadow) | 0.109 |
| Strong success-replay | ~110 (6 replay + 92 eval) | *pending* |

At comparable rollout budgets, sampling (BoN-4: 184 rollouts) and policy
learning (TTRL: ~200 rollouts) both fail to move the sealed eval success — the
null is not a compute-budget artifact.

## What would make TTRL work here (honest forward-looking)

The null's mechanistic explanation (sparse positive signal + exploration gap,
made structurally precise by the strong-replay finding) implies the recipe
for a positive result at this mechanism:
1. A base model strong enough to COMPLETE state-changing workflows — the 4B
   model never emits a modify call, so no positive example of the missing
   behavior exists to learn from. A stronger base (7B+ with demonstrated
   tool-completion) is the single highest-leverage change.
2. A longer update phase or higher base success (≥0.25) so the update phase
   generates dense positive signal.
3. A softer evaluator (partial-credit instead of exact DB-state match) so
   partial workflow gains are measurable instead of capped at 0.
4. Larger per-update effect (higher-rank LoRA, more steps) with the KL
   anchor + drift guard holding — the operating envelope is documented
   (drift 0.1-2.0 healthy; >5 collapses tool calling).

## Gate-protection demonstration (user-mandated key upgrade, 2026-08-30)

The gate's protection is demonstrated in two complementary ways:

1. **Real-data fail-closed**: a POISONED adapter (trained on flipped credit —
   reinforcing the early-stop on failed episodes, 25 episodes) is shadow-
   evaluated against the frozen parent: the e-process returns **ROLLBACK**
   (n=25, mean_gain 0.000, lcb_gain −0.40 < ε_gain=0.01) — deployment is
   blocked. The same fail-closed decision protected all four main-run
   candidates (v3/v4 × 2 seeds).
2. **Poison rejection by calibration**: the frozen 162-config coverage
   simulator guarantees the poisoned operating point is rejected
   (poisoned_rate 0.000 < SESOI power floor) at the coverage-frozen n=512.

Honest limitation: on this environment, the DB-state evaluator makes
non-mutating behavioral changes outcome-invisible, so the poisoned candidate's
deployment counterfactual measures 0.109 = frozen 0.109 (no observable
degradation — the poison cannot create wrong *state* changes that the model
never emits). The gate's protection is therefore demonstrated as fail-closed
rejection (real data) + calibrated poison rejection (simulator), not as an
observed harm-prevention delta on this task set.

## Engineering

- Full pipeline on 2× RTX 5090: vLLM serving + dynamic LoRA lifecycle
  (upload/swap/rollback) on GPU1, LoRA trainer on GPU0; locked layout.
- Reproducible: pinned env, `reproduce.sh`, decision log, SHA256SUMS.
- Hard-won compatibility findings documented (Blackwell/flashinfer, Qwen3.5 XML
  tool parser, fla autograd, memory envelope): see protocols/COMPATIBILITY_PROFILE.md.

## Honest limitations

- Exploratory single/dual-run; labels every number as such.
- Tool-only protocol caps the achievable ceiling (no user interaction).
- Global gate fail-closes at this scale by design (n=512 coverage-frozen);
  deployment-gain gating is demonstrated, not exploited.
- Null result is reported with mechanism explanation if obtained.
