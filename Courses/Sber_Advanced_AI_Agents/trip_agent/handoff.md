# Handoff — read this alongside task.md

task.md has the requirements mapping (course criteria → design decisions).
This file has the architectural conventions and current build state that
emerged after task.md was written — treat this as the source of truth where
the two disagree.

## Architecture decisions (settled, don't relitigate)
- **No clean architecture / ports-adapters / DDD layering.** Single
  implementation of everything except the LLM client — one layering pass
  (`ports/`, `domain/entity/`, etc.) was tried and explicitly rejected twice.
  Rule going forward: a folder name is justified only if it names a real
  swappable boundary that exists today, not architecture vocabulary applied
  because it's familiar.
- **Flat layout, no `src/`, no clean-arch subpackages:**
  ```
  trip_agent/
  ├── main.py
  ├── pyproject.toml
  ├── agent/
  │   ├── state.py       # TripState (pydantic), Phase enum, ALLOWED_TRANSITIONS
  │   ├── llm.py          # get_router_llm(), get_agent_llm()
  │   ├── skill.py         # load_skill() — parses SKILL.md frontmatter + body
  │   ├── graph.py          # build_graph() — StateGraph wiring only
  │   └── nodes/
  │       ├── router.py        # cheap structured-output classifier
  │       ├── destination_advisor.py
  │       └── stubs.py         # dead-end placeholders for unbuilt nodes
  ├── skills/
  │   └── destination-advisor/
  │       ├── SKILL.md
  │       └── places.md
  └── tests/
      └── test_router.py
  ```
  Split a module into a subpackage only when file count in it actually grows
  (~5-6 files), not preemptively.
- **Async everywhere, no exceptions except pure sync routing functions.**
  Every node is `async def`, every LLM/tool call is `.ainvoke(...)`, graph
  invocation is `graph.ainvoke(...)`. Only plain `def route_x(state) -> str`
  functions used in `add_conditional_edges` stay sync (no I/O in them).
- **LangGraph `StateGraph`, not the hand-rolled AgentHarness from the course
  notebooks.** Course seminars (Занятия 1-4, uploaded earlier) build a manual
  FSM + custom classes as teaching scaffolding — deliberately not mirrored
  here; using real LangGraph primitives instead (`add_conditional_edges`,
  will use `interrupt()` for human-in-the-loop confirm, subgraph invoked as a
  node for the child agent).
- **`TripState` is a pydantic `BaseModel`**, not `TypedDict` — mutable defaults
  (`messages: Annotated[list, add_messages] = Field(default_factory=list)`)
  are safe in pydantic (per-instance deep copy), would NOT be safe in a plain
  dataclass or function default.
- **Skills are on-disk artifacts under top-level `skills/`**, each own folder
  with `SKILL.md` (YAML frontmatter: `name`, `version` semver) + optional
  resource files (e.g. `places.md`). Loaded once at module import time via
  `agent/skill.py`'s `load_skill()`, not re-parsed per call — version is a
  stable identity for the process lifetime. "Versioned" = the version string
  is threaded through and referenced wherever the skill is used/logged, not
  that it changes automatically.
- **Router is a cheap structured-output classifier**, not a tool-calling
  ReAct loop — `with_structured_output(RouteDecision)`, single call, no tools.
  Re-entered every turn (no explicit self-loop edge needed): because it reads
  full message history each time and a `MemorySaver` checkpointer persists
  state across turns, "keep chatting until decided" falls out for free from
  re-routing rather than needing loop logic in the node itself.
- **Checkpointer required** (`MemorySaver()` for dev) once any multi-turn
  behavior matters — `graph.compile(checkpointer=MemorySaver())`, invoked
  with a `thread_id` in config. Will need to become a persisted checkpointer
  (SQLite) if the crash-before-ack demo scenario needs to survive a process
  restart later (criterion 6 territory, not built yet).
- **GigaChat vs OpenAI-first: unresolved, currently building against OpenAI**
  (`langchain_openai.ChatOpenAI`) via `agent/llm.py`'s two functions
  (`get_router_llm()` cheap model, `get_agent_llm()` stronger model) as the
  one deliberate swap point. Revisit before the deadline — Занятие 1's
  notebook has a complete, tested GigaChat setup (`langchain_gigachat`,
  documented kwargs, curator-issued credentials) that may make dev-on-GigaChat
  lower-risk than swapping late; no decision made yet either way.

## Built so far
- `agent/state.py`, `agent/llm.py`, `agent/graph.py` — router + destination
  advisor wired; `trip_booking` is still `stubs.trip_booking_stub`.
- `agent/nodes/router.py` — working, tested against both branches.
- `agent/nodes/destination_advisor.py` + `skills/destination-advisor/` —
  working: system-prompt-only chat node, no tools, no extra state, suggests
  only from `places.md`, confirms the pick back so the router can act on it
  next turn.
- `agent/skill.py` — `load_skill(dir_name, resource_files=(...))` loader.
- `tests/test_router.py` — parametrized against both routing branches.
- `main.py` — multi-turn REPL loop with `MemorySaver` + fixed `thread_id`.

## Not built yet (in rough build order)
1. `trip_booking` child agent — real ReAct subgraph (own tools: search_flights,
   search_hotels, record_choice; own SKILL.md; loops on itself, exits once
   required slots filled — see task.md's `TripSlots.is_complete()` design).
2. `propose` node — structured `Proposal` output from filled slots +
   computed `idempotency_key`.
3. `confirm` node — `interrupt()`-based human-in-the-loop gate.
4. `trust.py` — validation chokepoint (schema/permission/state) gating the
   `book` write tool specifically; searches stay ungated (read/write asymmetry
   should be visible in code, two paths not one flag).
5. `book` node + `queue/` (models, store, worker) — publish/reserve/save/ack,
   bounded retry + DLQ, dedupe on `message_id` vs `idempotency_key` (two
   separate tables/checks, don't conflate).
6. `trace.py` — structured logging w/ correlation id (not Langfuse — that's
   an optional stretch layer on top later, not required for criterion 5).
7. `tests/test_scenarios.py` — the 5 required demo scenarios as pytest cases
   (happy path, cheap path, rejected proposal, transient failure+retry,
   crash-before-ack redelivery). Write early where possible, not last.

## Constraints to keep in view
- Deadline **6 September**. Model swap decision above should be made early,
  not near the deadline.
- Long-term memory, Langfuse, evaluation/LLM-as-judge (Assignment 4), async
  streaming-to-client, and generic code-quality tooling (pyproject/lint/etc.)
  are explicitly deprioritized/out of scope for the graded core — see task.md
  for the full reasoning, don't reintroduce them as "must-haves."