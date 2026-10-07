# Protocol v2 — what was wrong with v1 and why the null result was a harness artifact

Status: v2 frozen 2026-10-06. All experiments after this date use v2
(`protocols/stream_v2_seed*.json`, `protocols/sweeps/*`, `protocols/ttrl_v2_seed*.json`).

## The v1 result being replaced

v1 reported: frozen baseline 0.109 (5/46) on τ²-bench retail, TTRL updates
changed behavior (logit drift 0.08 → 4.0) but moved success by zero flips, and
the mechanism story was "successful episodes never contain state-changing
calls, so positive signal is structurally unavailable".

## Three defects found in the v1 harness (2026-10-06)

1. **The user's request was never sent to the agent.**
   τ²-bench's `StructuredUserInstructions` separates `reason_for_call` (what the
   user wants, e.g. "You received your order #W2378156 and wish to exchange the
   mechanical keyboard...") from `task_instructions` (simulator-side behaviour
   guidance, e.g. "You are detail-oriented..."). The v1 prompt builder used
   `task_instructions` + known/unknown info and **omitted `reason_for_call`**.
   The model was asked to serve a customer without ever being told what the
   customer wanted → it poked read-only tools and stopped (0 modify calls in
   46/46 episodes, including with few-shot prompting) → the only "successes"
   were the 10 read-only tasks where doing nothing leaves the DB matching.
   Evidence: `protocols/diag_prompt_v2.json` runs both prompts on the same
   tasks: v1 fails every write task, v2 solves them (tasks 1/44/108: v1
   0/6 → v2 6/6).

2. **Wrong vLLM tool-call parser.** The Qwen3.5 template emits
   `<tool_call><function=NAME><parameter=K>V</parameter></function></tool_call>`;
   vLLM was started with `--tool-call-parser hermes` (JSON-style), which never
   matches → `msg.tool_calls` was always empty and every episode ended after
   one turn. Fixed with `--tool-call-parser qwen3_xml`. The harness also parses
   the XML client-side as a fallback and survives vLLM 400s on malformed args.

3. **No user for the confirmation turn.** The retail policy requires explicit
   customer confirmation before exchanges/returns. With no user simulator the
   episode stalled at "Do you confirm...?" — v2 adds a deterministic scripted
   user (`stub_user_reply`) that answers confirmation questions and pushes the
   agent to proceed, up to 3 replies per episode.

## Two engine-parity defects (transformers rollout path)

4. **Duplicated tool calls in the re-rendered history.** The transformers path
   stored the raw completion (including `<tool_call>` XML) as message *content*
   AND as structured `tool_calls`; the template then rendered every call twice,
   driving read-call loops. Fixed by stripping the XML from content.

5. **Type-loose XML arguments.** The transformers XML parser JSON-decodes bare
   parameter values (`zip 19122` → int), while vLLM's parser keeps strings. The
   retail DB lookups compare strings, so int arguments silently returned
   "User not found" and the agent looped. Fixed by coercing arguments to the
   tool signature at the `Tau2Episode.step` boundary (`_coerce_arguments`).

6. **Thinking mode matters.** With Qwen3.5-4B, `enable_thinking=False` makes
   the agent chatter instead of acting (probe: task 6 fails with thinking off,
   succeeds with thinking on). v2 runs thinking ON in both engines.

## v2 protocol summary

- Prompt: `reason_for_call` + known information; scripted user replies; assistant
  thinking enabled; `max_tokens=1024–2048`/turn, ≤24 turns.
- Evaluation unchanged: hidden DB-state equality against the replayed reference
  trajectory. Read-only tasks (no write action) are excluded from analysis —
  a DB-only evaluator gives them free wins.
- Frozen sweep (vLLM, T=0.7, 4 samples/task, `protocols/sweeps/`): overall
  success ≈ 0.58; write tasks split into band 41 / zero 25 / saturated 38.
  The band (0 < p̂ < 1) is the learnable stratum; the split of band tasks into
  update/eval is pre-registered per seed (`select_band.py`).
- TTRL run: episode-boundary LoRA updates on the stream (band + zero tasks),
  sealed eval on held-out band/zero/saturated tasks, plus the two-scale gate
  and drift diagnostics.

## Honest caveats

- No user simulator: tasks whose difficulty comes from user behaviour cannot
  play out; known_info is given up front instead of being elicited.
- DB-only evaluation: `nl_assertions`/communicate checks are not scored.
- Success numbers are therefore comparable **within this protocol only** — they
  are not τ²-bench leaderboard numbers.
