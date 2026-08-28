# Marathon Prompt (mission order — CLAUDE.md is the constitution)

```
/research-pipeline "Run the agent-ttrl2 project (10-day fast plan). READ CLAUDE.md in
the project root FIRST — it is the constitution: fast-value, high-standard test-time
RL for agents as a resume CORE project (no top-venue paper; deliverable = released
framework + honest reproducible results + 1-page TECH_REPORT, URGENT 10-day window).
INNOVATION ANGLE (user-mandated): two-scale safety-gated test-time RL — LOCAL gate
(evidence-gated action credit: E_hard/E_soft conflict detection + structured action
groups) + GLOBAL gate (empirical-Bernstein e-process commit/rollback, α=0.05, frozen
by agent-ttrl D6's 162-config coverage simulator). Reuse agent-ttrl's validated
design assets (read C:\Users\w1828\repos\agent-ttrl\phase01\EXPERIMENT_DECISION_LOG.md
D6/D10-D14 + CLAUDE.md lessons table): env FIRST with initial success ≤0.3 + verified
headroom; structured action groups not free-form branches; calibrate 'update changes
behavior' (logit drift) before measuring downstream gain; full-pipeline single-task
smoke before any batch; slimmed stats (1-2 runs labeled exploratory, simple rollout
count, one commit-gate on/off ablation, no second model); null is acceptable IF
framework+baselines+diagnostics are real and null is mechanistically explained.
DELIVERABLE LADDER: D1 env+model (Qwen3.5-4B, fallback Qwen3-4B) + compatibility
profile → D2-4 minimal framework (episode-boundary LoRA updates trl/peft + frozen
policy + vLLM + structured action groups + single-task smoke) → D5-6 frozen baseline
+ headroom verification → D7-8 MAIN run (same-task-stream contrast: update first 60%
tasks, evaluate last 40%, 1-2 runs, behavior-drift diagnostics) → D9 GitHub v0.1
release (README + reproduce.sh + results + limitations) → D10 TECH_REPORT + resume
bullets. HIGH STANDARDS (keep): industry-recognized replayable env, honest
measurement incl. null, matched-ish rollout accounting, contamination freeze
(hidden evaluator never in training), reproduce.sh works, CI, decision log, no fake
claims. SERVER: ssh autodl3 (2×RTX 5090 32GB, GPU0=train ~20GB, GPU1=vLLM rollout
~16GB, NEVER co-locate on one 5090; 754GB RAM; ~60-80 GPU·h budget; expand data disk
to 200GB; resume-from-checkpoint; results rsync + git push continuously; most
engineering on local RTX 5060). GITHUB: https://github.com/hxm2023/agentic-TTRL
code-first, release tag v0.1 at D9. Human owns the project." --deep_mode: false,
auto_write: true, auto_proceed: true, venue: "engineering deliverable + tech report (not a paper)"
```
