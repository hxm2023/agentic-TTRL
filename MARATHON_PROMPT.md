# Marathon Prompt (mission order — CLAUDE.md is the constitution)

```
/research-pipeline "Run the agent-ttrl2 project. READ CLAUDE.md in the project root
FIRST — it is the constitution: fast-value, high-standard test-time RL for agents
(resume CORE project for 大模型后训练算法工程师 — no top-venue paper required; the
deliverable is a released framework + honest reproducible results + tech report).
MANDATORY: read the agent-ttrl lessons table (CLAUDE.md) + the predecessor decision
log at C:\Users\w1828\repos\agent-ttrl\phase01\EXPERIMENT_DECISION_LOG.md D10-D14 —
rules: env FIRST with low initial success (≤0.3) and verified headroom; structured
action groups not free-form branches; calibrate 'update changes behavior' before
measuring downstream gain; full-pipeline smoke on 1 task before any batch; fast-value
protocol (single pre-registered endpoint, ≥3 seeds, honest matched compute, no
ceremony); deliverable ladder M1 framework release → M2 env+baselines → M3 main
result → M4 hardening+tech report (4-6 weeks). Model: Qwen3.5-4B-class agent-capable
open model (verify + freeze compatibility profile; fallback Qwen3-4B). Env:
replayable tool-calling with headroom (AppWorld-style / τ² mock / controlled custom).
High standards: unified honest eval, matched compute, ≥3 seeds, contamination freeze,
src/ + reproduce.sh + CI + artifacts + SHA256SUMS, decision log, GitHub
https://github.com/hxm2023/agentic-TTRL code-first with release tags v0.1/v0.2/v0.3.
Server: autodl3 (ssh details pending — record when given). Pipeline: Phase 0 (≤1 day
env+model+profile) → Phase 1 (≤2 days claim+protocol) → Phase 2 (M1-M4 ladder with
per-milestone review) → Phase 3 (TECH_REPORT.md + release + resume bullets). A null
main result is acceptable IF framework+baselines+diagnostics are real and the null
is mechanistically explained. Human owns the project." --deep_mode: false,
auto_write: true, auto_proceed: true, venue: "engineering deliverable + tech report (not a paper)"
```
