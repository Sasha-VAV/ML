# Trip Planner — Управляемый ИИ-агент (Assignment 3)

Deadline: **6 September**. Grading turnaround: 3 days after.
Model: open question — see "GigaChat vs OpenAI-first" below; seminar material
suggests dev-against-GigaChat-directly is now the lower-risk option.

## Course reference material (from Занятия 1–4 notebooks) — READ THIS FIRST
The four seminars build a reference implementation across exactly the 6 criteria,
one criterion-cluster per seminar. Use their patterns instead of reinventing:

- **Занятие 1** → criteria 1–2 basics: `gate_tool_call`, `execute_tool_call`,
  `run_agent` — a plain bounded loop, `ToolMessage` with `tool_call_id` preserved,
  `PROCESS_STATES`, idempotent-effect dedup via a simple effects-seen set.
- **Занятие 2** → criterion 3: `transition_state` / `event_from_observation`
  (hand-written FSM, not LangGraph), `read_with_retry_cache`, `read_or_degrade`,
  `run_refund_saga` (compensation), `classify_request`/`route_request` (cheap-path
  routing that skips the model), `run_react_agent`, `compare_react_runs` (cost
  comparison across routes — this is your criterion-5 "savings" evidence pattern).
- **Занятие 3** → criterion 4 + 5: a hand-rolled `AgentHarness`
  (`EffectProposal`, `ToolAction`, `DelegateAction`, `Finish`, `Handoff`,
  `SubagentResult`, `SubagentRegistry`, `StepBudget`) **and**, separately, a
  `deepagents`-based child agent (`build_billing_subagent`,
  `build_deep_support_agent`). Criterion 4's "Deep Agents or explicit
  AgentHarness" is literally naming these two paths — pick one, don't invent a
  third. Observability primitives (`TraceEvent`, `TraceRecorder`, `MetricsHook`)
  are also introduced here — **plain Python classes, this is what satisfies
  criterion 5**, not an external tool.
- **Занятие 4** → criterion 6: `PriorityTaskQueue`, `Task`, `Delivery`,
  `ProviderGate` (backpressure), `consume_once`, `handle_delivery`,
  `run_worker_once` — an **in-process** queue/worker, no real broker. This is
  the expected shape for publish/reserve/save/ack + DLQ + idempotency.

**Correction to earlier advice:** don't reach for LangGraph `StateGraph` /
`add_conditional_edges` for the core harness — the reference solution uses a
manual FSM + loop instead, and only brings in LangGraph for the `deepagents`
child-agent path. Given limited LangGraph experience, following the taught
manual-FSM+harness pattern for the core (criteria 1–3, 5, 6) and reserving
LangGraph specifically for wherever `deepagents` is used (criterion 4) is both
lower-risk and closer to what's actually graded.

**Correction on observability:** Langfuse does not appear in any of the four
seminars — it's not what satisfies criterion 5. Build the `TraceRecorder`
pattern from Занятие 3 first (it's literally handed to you and is sufficient
for the criterion); treat piping it into Langfuse afterward as a personal
stretch goal, not a dependency of the graded core.

## Why this assignment / this domain
- Assignment 3 chosen over 1/2: closest to production-agent work, and criterion 6
  (idempotent reliable execution: publish/reserve/ack/DLQ) is the genuinely new
  skill relative to existing experience — everything else has some prior exposure.
- Trip planner chosen as domain: booking a flight/hotel is a textbook confirmable
  effect with an obvious double-booking risk (natural idempotency story), and
  "checklist → search → propose → confirm → book" gives a clean state machine
  without forcing it.
- Explicitly out of scope for grading, do NOT let these eat the schedule:
  - Long-term memory (that's Assignment 1's spec, not 3's) — nice-to-have, build
    last, piggyback on LangGraph's Store if built at all.
  - Assignment 4 (evaluation / LLM-as-a-judge) — not attempted; Langfuse is used
    for tracing/datasets/cost only, not judge scoring pipelines.
  - Async/streaming, uv/pyproject hygiene, LangGraph CLI — not tracked
    deliverables, just write async from line one and use uv normally.

## Official grading criteria (map everything back to these 6 + 5 demos)
1. **GigaChat + function calling** — ≥2 tools, ≥1 reads live data, `tool_call_id`
   round-trips into `ToolMessage`.
2. **Trust boundary** — Python validates tool name / closed schema / arg types /
   permission / current state *before* execution; effects need explicit confirm;
   step & model-call budget enforced in code.
3. **Branching, robust ReAct loop** — explicit state, ≥2 routes, one chosen by a
   tool observation, invalid transitions blocked, bounded retries + cache for
   safe reads, graceful degrade on exhaustion.
4. **Skill + child agent** — ≥1 versioned `SKILL.md`, parent delegates a narrow
   task (subset of facts/tools/budget) to a child, parent validates the child's
   result (author/status/facts/source) before accepting.
5. **Observability** — safe trace of route/state transitions/tool+child
   calls/retries/termination reason, ≥2 metrics, no secrets/PII.
6. **Reliable execution** — publish → reserve → save result → ack; priority +
   bounded retries + DLQ; idempotent op keyed by `message_id` (dedupe delivery)
   and `idempotency_key` (dedupe business command).

5 demo scenarios (build as pytest cases from day one — spec, test suite, and
screencast script all at once, deterministic failure injection, not randomness):
1. Happy path — full chain, multi-step, branch, skill + child agent, verified result.
2. Cheap/short path — rule or early observation short-circuits, visible savings in trace.
3. Rejected proposal — bad tool/args/transition/unconfirmed effect → blocked, clear reason.
4. Transient failure — bounded retry, then success (from cache) or graceful degrade.
5. Crash-before-ack redelivery — result saved but not acked, simulate crash, redelivery
   completes without repeating the business op; separately show retry exhaustion → DLQ.

## Domain design

**Phases:** `intake → checklist → search → propose → confirm → book → done/degraded`
- Intake is **single-shot** (one structured form-like input), not multi-turn slot
  filling — multi-turn intake is Assignment 2's problem, adds complexity with zero
  criterion credit here.
- `intake → checklist`: turn freeform preferences into a structured constraint
  object (dates, budget, party size, must-haves) — the one legitimate
  structured-output moment outside the trust boundary itself.

**Tools:** flight-search (live-read), hotel-search (live-read), booking/reservation
(the write). Mock all three as fake providers with injectable latency/failure —
need deterministic failure for demos, real APIs won't cooperate.

**Branch (criterion 3):** hotel/flight search returns "over budget"/"unavailable"
→ route to alternate-search (relax constraint, try nearby dates/area) vs "within
constraints" → straight to propose. Doubles as cheap-path (no re-search) vs
long-path (re-search + re-propose) demo.

**Idempotency key (criterion 6):** `(trip_id, item_type, item_ref)` — booking the
same flight for the same trip twice collapses to one reservation even under
retried/duplicated requests.

## State & orchestration (manual FSM + loop, per Занятие 2's pattern)
- Typed state (pydantic/TypedDict/NamedTuple) with explicit phase enum — this
  state *is* the explicit-state requirement; don't let it degenerate into a
  message list.
- Checklist lives in state as a structured constraint object, checked
  programmatically at `propose`, not left to the model to "remember."
- `transition_state(state, event) -> state`, mirroring Занятие 2: invalid
  transitions raise/reject, literal `if next_phase not in ALLOWED[current_phase]`.
- `event_from_observation(state, tool_name, data) -> event`: the router reading
  the **last tool's structured result** (found/not-found/over-budget), not the
  model's narration — keep the branch decision unit-testable without an LLM call.
- `run_react_agent`-style bounded loop drives the whole thing; no LangGraph
  StateGraph needed for this part (see correction above) — reserve LangGraph
  for the `deepagents` child-agent path only, if you use `deepagents`.

## Trust boundary + reliable execution
- Only `book` gets full trust-boundary + confirm-before-execute treatment;
  searches are reads, no confirmation — keep two visibly separate code paths,
  not one "maybe confirm" flag.
- Pydantic models per tool, closed schema (reject unknown fields) for arg
  validation — validation sits in a single chokepoint every tool call passes
  through (ahead of the execute node), demoable in isolation.
- `propose` step assembles a `Proposal` object (line items, total cost, computed
  idempotency keys per item) — shown to user for confirmation, consumed by `book`
  after approval. Unifies criterion 2 confirmation with criterion 6 key generation.
- Step/model-call budget: counter on state, checked by routing logic, not hope.
  Feeds observability metrics directly.
- Publish/reserve/save/ack mapping: `publish` = booking request enters queue,
  `reserve` = hold with (mock) provider, `save result` = record confirmation
  number, `ack` = mark done. Implement as four separately-callable functions
  (in-process queue, e.g. deque/SQLite table is fine) so the crash-before-ack
  demo can literally stop between two of them in a test.
- Two separate dedupe tables: `message_id` (drop redelivered messages) vs
  `idempotency_key` (drop the business effect even if message body differs) —
  don't conflate them, the assignment is explicitly testing the distinction.
- Bounded retry with backoff; exhaustion routes to a DLQ table/list — reuse the
  same retry/degrade logic for safe-read retries and write retries where shapes
  align. Provider timeout is the natural retry trigger.
- Priority: a numeric field the dequeue function sorts on — no priority-queue
  library needed.

## Skill + child agent
- Decide: hand-rolled `AgentHarness`+`Handoff` (Занятие 3 section 7 pattern) vs
  `deepagents` subagent (Занятие 3 section 8 pattern). Given rusty multi-agent
  experience, the hand-rolled `Handoff`/`SubagentResult` path is more legible
  and debuggable than `deepagents`' config-driven graph — lean that way unless
  you specifically want the `deepagents` practice.
- `SKILL.md` (e.g. "trip-search-strategy.md"): purpose, instructions,
  constraints, tools, expected output — how to trade off budget vs preferences,
  when to widen dates, when to accept a compromise. Load its text into the
  child's system prompt, don't hardcode inline (the versioned/loadable artifact
  is part of the criterion). Mirror `build_billing_skill()`'s shape.
- Child agent: narrow "option-finder" — one category (flights *or* hotels), the
  relevant slice of the checklist via `Handoff`, its own small tool set (just
  that category's search tool), its own step budget (`StepBudget`). Reuse the
  same trust-boundary validator, don't build a second one.
- Child's return contract: `SubagentResult`-style object — `author` (which
  provider/tool), `status` (found/none/partial), `facts` (options + prices),
  `source` (provider name + query used). Parent uses `accept_subagent_result`-
  style check: did it actually call the tool it claims, do facts match checklist
  constraints, is source real, does `expected_owner` match.
- Keep the child's internal loop small (2–3 steps) first, get the parent↔child
  contract working end-to-end, then richen if time allows.

## Observability — build TraceRecorder first (Занятие 3 pattern), Langfuse is a stretch add-on
- Core (graded): `TraceEvent`/`TraceRecorder`/`MetricsHook` per Занятие 3 —
  plain Python, records route/state transitions/tool+child calls/retries/
  termination reason. This alone satisfies criterion 5. Do this before touching
  Langfuse at all.
- Emit ≥2 required metrics (model-call count, tool-call count), plus a domain
  metric like "search iterations before viable itinerary" — use
  `compare_react_runs`-style comparison (Занятие 2) as your cheap-path savings
  evidence for demo scenario 2.
- Scrub before emitting (allowlist fields on the trace-event constructor), not
  a regex filter after the fact — makes "no secrets/PII" structural.
- Stretch, after the graded core works: pipe `TraceRecorder` events into
  Langfuse (sessions to tie one request's full parent+child+retries trace
  together, cost/latency dashboards) purely to explore the ecosystem beyond
  simple tracing — not a criterion-5 dependency, don't let it block anything.
  Stop before judge/scoring features — that's Assignment 4 territory.

## Structured output — where it's actually needed
- OpenAI function-calling format for tool args, pydantic-validated on receipt.
- The `Proposal` object (criterion 2/6 boundary-crossing object).
- The child agent's result contract (criterion 4).
- NOT needed for: intake parsing beyond the single-shot form, or the model's
  intermediate ReAct reasoning text — only boundary-crossing objects need a schema.

## Async
- Tool functions and graph nodes `async def` from the start — LangGraph nodes
  are async-native, this isn't an add-on, it's just how the graph is written.
  Retrofitting later is the expensive path, not doing it now.
- `asyncio.gather` for independent reads (e.g. flight + hotel search in parallel
  when not dependent on each other).
- Queue functions (publish/reserve/ack) async too for consistency, even though
  the queue is in-process — avoids a sync/async seam in the part of the code
  most likely to grow into real async I/O later.

## GigaChat vs OpenAI-first — reconsider
- Занятие 1 hands you a complete, tested GigaChat setup: `langchain_gigachat.
  GigaChat`, exact kwargs (`credentials`, `scope`, `model="GigaChat-2-Max"`,
  `verify_ssl_certs=False`, optional `ca_bundle_file`/`base_url`), `.env` /
  Colab-Secrets pattern, curator-issued credentials. Setup friction — the
  original reason to dev on OpenAI first — is close to zero here.
- If still swapping later: isolate the model client behind a thin interface
  (`call_model(messages, tools) -> response`) so the swap is one line, design
  tool schemas in OpenAI function-calling format (gpt2giga translates it), and
  **re-run all 5 demo scenarios before recording the screencast** as a mandatory
  regression pass — not a formality.
- If building on GigaChat directly from the start: no swap risk at all, and
  you get GigaChat-specific tool-calling behavior (forced `tool_choice`,
  parallel-call support, retry semantics) validated early instead of at the end
  when it's expensive to fix. Given the setup is turnkey, this is now the
  lower-risk default — worth actively choosing rather than defaulting to the
  original plan.

## Repo / submission format
- `uv` + `pyproject.toml`, `.env.example` (var names, no values), one
  `SKILL.md` + reference material.
- Single `make demo` / `run.sh` (or pytest target) that runs all 5 scenarios
  in one command.
- README.md: task, architecture, run commands, how to check each demo scenario.
- Screencast ≤10 min: architecture walkthrough + successful and failure-path demos.