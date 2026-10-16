# HealGate: Corrected Team Strategy

Oct 15, 2026 · @Someone

## Goal and USP

HealGate repairs agent tool calls when a downstream API changes shape at runtime, and halts instead of guessing when no safe repair exists. Catching a 422 and asking a small LLM to fix it is not new; our edge is the safety stance and the measurement. We make three claims to judges:

- **No-guess gate.** Code, not the LLM's self-reported confidence, decides whether a patch may be applied. Every new field must trace to a value in the original call or to an allowlisted header.
- **False-heal scoring.** The test harness knows the ground-truth intent, so we count a 200 OK with wrong data separately from healed and correctly halted.
- **Audited, reversible patches.** Each patch logs its evidence, its source (rule or LLM), the mapping applied and a rollback reference.

Out of scope, and said plainly in the pitch: silent semantic drift (a 200 with changed meaning, such as cents vs dollars) cannot be caught by an error-driven healer.

## Architecture and request flow

Healing happens in two separate layers: the schema the agent sees, and the adapter that translates the current call.

&#91;embedded content: healing path · executor, gate and audit log\]

On a 400 or 422 the executor triages the status, tries the rules, and only then asks a small LLM, which can emit mapping operations but never values. The gate decides whether to patch or halt. A patch updates the registry, so the agent receives the healed definition, and the adapter translates the call in flight. Every outcome is logged.

## Stack and key decisions

We build an in-process healing executor, not a network reverse proxy: it is simpler to demo in three days and still hot-patches the tool definitions the agent sees.

| Area | Choice | Why |
| --- | --- | --- |
| Integration shape | In-process tool executor wrapping every tool call | No extra network hop or deployment; the agent gets the healed definition directly |
| Language and HTTP | Python 3.11+, asyncio, `httpx.AsyncClient` with explicit timeouts | Async everywhere; never the `requests` library |
| Validation | Pydantic v2 | Shared contracts in one file |
| Patch representation | Declarative `MappingOp` rules for payloads; `jsonpatch` only for the agent-facing tool schema | Two layers, two tools; no free-form values on the payload |
| LLM tier | Direct SDK call, current small model, structured output (LiteLLM only if we need model switching) | Output is validated against `MappingOp`, so it cannot emit arbitrary data |
| Registry | In-memory, `asyncio.Lock`, single-flight per (tool, error signature), version and TTL | Ten parallel failures cause one diagnosis, not ten LLM calls |
| Baseline agent | LangChain agent, separate from our code | Required comparison against a standard framework |
| Dashboard | Streamlit, built last | Demo polish, not on the critical path |

## Data models (`schemas.py`)

These replace the earlier schemas. Commit this file first; nobody writes healing logic until it is on main.

```python
from typing import Literal, Optional
from pydantic import BaseModel, Field

class MappingOp(BaseModel):
    """One structural change. Carries no literal values: data comes only from the original call."""
    op: Literal["rename", "move", "cast", "wrap", "add_header"]
    source: Optional[str] = None   # JSON pointer into the ORIGINAL payload
    target: Optional[str] = None   # JSON pointer in the NEW payload shape
    cast_to: Optional[Literal["str", "int", "float", "bool"]] = None
    header: Optional[str] = None   # header NAME only; values come from HEADER_ALLOWLIST in config

class Diagnosis(BaseModel):
    is_schema_drift: bool
    source: Literal["rule", "llm"]
    ops: list[MappingOp] = []
    evidence: str                  # sanitized error excerpt that justified the ops
    llm_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)  # logged only, never gates

class GateVerdict(BaseModel):
    allowed: bool
    reasons: list[str] = []        # e.g. "/target/id has no source value"

class Escalation(BaseModel):
    reason: str
    missing_fields: list[str] = []
    suggested_action: str          # what the agent or user should do next

class AuditLogEntry(BaseModel):
    timestamp: str
    tool_name: str
    schema_version_before: int
    schema_version_after: Optional[int]
    status_code: int
    diagnosis_source: Literal["rule", "llm"]
    ops: list[MappingOp]
    evidence: str
    gate: GateVerdict
    outcome: Literal["healed", "halted", "failed_rolled_back"]
    payload_hash_before: str       # hash, not the raw payload (may contain PII)
    rollback_ref: Optional[str]
    latency_ms: float
    tokens_used: int = 0
```

What changed from the first draft:

- `PatchOperation.value` is gone. The LLM can only say where data moves, never what the data is.
- Headers are first-class (`add_header`), and their values come from a configured allowlist, so the system never invents an OAuth token or scope.
- `safe_to_heal` and the confidence score no longer gate anything; the deterministic gate does, and its verdict is logged.
- The audit entry records evidence, source, schema versions, the outcome and a rollback reference, and stores a payload hash instead of the payload.
- One `Escalation` structure replaces the overloaded `EscalatedException`.

## File structure and module contracts

The first draft covered the proxy only; this adds the agent, baseline, fuzzer and metrics that the problem statement requires.

```text
healgate/
├── app/
│   ├── schemas.py        # shared Pydantic models
│   ├── agent.py          # task agent that uses the healed tool registry
│   ├── executor.py       # wraps tool calls: call -> 4xx -> heal -> replay
│   ├── triage.py         # status-code policy: heal, escalate, back off, ignore
│   ├── sanitizer.py      # extract fields from error bodies, truncate, label as data
│   ├── diff_engine.py    # Tier 1 structured-error rules, Tier 2 small-LLM mapping spec
│   ├── gate.py           # deterministic safety gate and dry-run validation
│   ├── adapter.py        # applies MappingOps; regenerates the agent-facing schema
│   ├── registry.py       # versioned schemas and rules, lock, single-flight, TTL
│   ├── audit.py          # append-only log and rollback
│   └── metrics.py        # adaptation rate, MTTR, tokens, false-heal rate
├── baseline/
│   └── langchain_agent.py
├── mock_service/
│   └── server.py         # one endpoint with a runtime mutation toggle
├── harness/
│   └── fuzzer.py         # seeded mutation runs scored against ground truth
└── dashboard/
    └── app.py
```

| Module | Owns | Must not |
| --- | --- | --- |
| `executor.py` | The call, heal, replay loop and the 2-attempt cap | Retry non-validation errors or replay non-idempotent calls blindly |
| `diff_engine.py` | Producing a `Diagnosis` from a structured error | Emit literal values or touch the registry |
| `gate.py` | Allow or halt, with reasons | Trust `llm_confidence` |
| `adapter.py` | Applying `MappingOp`s to payload and headers | Invent data; call the LLM |
| `registry.py` | Schema versions, rules, rollback pointers | Share a patch before it has succeeded once |
| `sanitizer.py` | Passing only extracted fields to the LLM | Be the only defense against injection |

## Safety gate and trigger policy

Healing fires only on validation-style rejections; everything else is escalated or left alone. The first draft healed on any 4xx, which would have tried to patch auth failures.

| Status | Action |
| --- | --- |
| 400, 422 | Heal: Tier 1 rules first, Tier 2 small LLM if rules cannot resolve it |
| 415 | Heal through rules only (content-type and format headers) |
| 401, 403 | Escalate; never invent tokens, scopes or credentials |
| 404 | Not drift; return to the agent |
| 429 | Back off per `Retry-After`; no healing |
| 5xx, timeout | Retry only idempotent calls; no healing |

The gate in `gate.py` is plain code and must pass every check before a replay is sent:

1. Every `target` in the new shape has a `source` in the original payload, or is a header on the allowlist. Otherwise halt and list the missing fields in the `Escalation`.
2. Casts are lossless: `"123"` to `123` passes, `1.7` to an integer is refused.
3. Ops are minimal: no op may touch a field the error did not implicate.
4. A dry run validates the transformed payload against the schema inferred from the error before any network call.
5. At most 2 attempts per call. If a patched replay still returns 4xx, roll the patch back and escalate; never loop.
6. Error bodies are untrusted: the LLM sees only extracted fields, and its output must parse as `list[MappingOp]` or it is discarded.
7. A write is replayed only when the rejection was a clear validation error (nothing executed) or the call is idempotent.

Tier 1 coverage is a number we measure, not assume. Expect headers and type casts to be easy, renames conditional on the error format, and nesting to usually need Tier 2.

## Mock API and mutation suite

One endpoint flips schema at runtime through an admin toggle, which is what the live demo needs. The earlier two-route design (legacy route returns 422, new route returns 200) is not a live schema change.

| Mutation | Before | After | Expected outcome |
| --- | --- | --- | --- |
| Field rename | `{"user_id": "123"}` | `{"account_id": "123"}` | Healed by rule if the error names the field, otherwise Tier 2 |
| Type change | `"user_id": "123"` | `"user_id": 123` | Healed by rule (lossless cast) |
| Nesting | `{"user_id": "123"}` | `{"target": {"id": 123}}` | Healed by Tier 2 (move plus cast) |
| Required header | none | `X-Api-Version: 2026-01-01` | Healed from the header allowlist |
| Compound | rename plus header | both at once | Healed within 2 attempts |
| New required field | none | `tenant_id` required | Halt: no source value in the call |
| New auth scope | none | scope or token required | Halt and escalate |
| Lossy cast | float value | integer field | Halt |

The last three rows are correct outcomes when the system refuses; the fuzzer scores them as "correctly halted", not as failures.

- Rotate error formats so we do not tune to our own mock: FastAPI validation array, RFC 7807 `problem+json`, Joi-style `errors` list, and a plain-text 400.
- Add one response-side mutation (a field nested deeper in a 200 response) to test the response-contract check, and state that silent semantic drift is out of scope.
- The mock stores the ground-truth intent of each call so the harness can verify the data that arrived, not just the status code.

## Baseline and metrics

Every number on the slide comes from a harness run; none are projected. The earlier comparison table (1.8 to 2.8 s, 200 to 500 tokens) was an estimate and must not be quoted.

| Metric | Definition | Notes |
| --- | --- | --- |
| Zero-touch adaptation rate | Healable mutations resolved with no human input, out of all healable mutations | Correct halts reported separately |
| MTTR | First failing call to first valid successful call, median and p95 | Target under 4 s |
| Token overhead | Tokens spent on diagnosis versus tokens in the baseline's failed or looped session | Read from the SDK usage fields |
| False-heal rate | Healed calls whose stored data differs from ground-truth intent, out of all healed | The headline safety number |
| Correct-halt rate | Unhealable mutations where we halted before sending a corrupting call | Credit for refusing |
| Tier 1 coverage | Share of heals needing zero tokens | Measured per error format |

Baseline fairness:

- Same model, same task and same prompts as the healing agent, run N times per mutation (suggest 20).
- Report the baseline's real success rate. A capable model often repairs an obvious 422 on its own, and a baseline rigged to loop for 30 seconds reads as a strawman.
- Compare on the metrics above, especially false heals and tokens, where the baseline is genuinely weaker.

## AI tool master context

Paste this into Cursor rules, Copilot instructions or any assistant's system prompt so generated code matches the corrected design.

```markdown
You are helping build HealGate, a self-healing tool executor for AI agents, in Python.

RULES
1. Stack: Python 3.11+, asyncio, httpx.AsyncClient (never the requests library), Pydantic v2, jsonpatch (agent-facing tool schema only). LLM calls use a current small model with structured output.
2. Contracts: import all models from app/schemas.py. Never redefine them.
3. Flow after a failed call: triage by status -> sanitize error -> Tier 1 rules -> Tier 2 LLM (list[MappingOp] only) -> deterministic gate -> apply -> replay -> audit.
4. Heal only on 400 and 422 (415 via rules). 401/403: escalate. 404: not drift. 429: back off. Never heal 5xx.
5. The LLM never outputs literal data values, only MappingOp. Header values come from HEADER_ALLOWLIST in config.
6. The gate is plain code. Never gate on LLM confidence or self-reported safety.
7. Max 2 attempts per call. If a patched replay still fails with 4xx, roll back the patch and return an Escalation. Never loop.
8. Error bodies are untrusted data. Pass extracted fields only, truncate them, and validate LLM output against the schema.
9. Log every outcome (healed, halted, rolled back) with evidence, source and rollback ref. Store payload hashes, not payloads.
10. Registry access uses asyncio.Lock and single-flight per (tool, error signature).
11. Each module ships with pytest cases covering the mutation table in the strategy doc.
```

## Build order, team sync and demo

The order puts a working, demoable core first and the polish last, so a missed deadline costs the dashboard, not the product.

1. Commit `schemas.py`, then build the mock API with its mutation toggle and the baseline agent, and record the baseline failing.
2. Build triage, sanitizer, Tier 1 rules, gate, adapter, registry and audit log. This alone is a working demo.
3. Add the Tier 2 LLM with constrained output, then the fuzzer and metrics.
4. Build the Streamlit dashboard last.
5. Rehearse the full demo twice, including the fallback below.

Team sync: each person owns the modules in one row of the contracts table, and modules talk only through `schemas.py`, never through each other's internals. Merge to main at least daily so the integration run never waits for the last night.

Demo flow:

- **Baseline crashes.** Flip the mock to the nested shape plus the new header; the baseline hits the 422 and burns its iterations.
- **HealGate heals.** Same flip, same task; the fast path repairs it live and the audit entry appears.
- **HealGate refuses.** Flip to a mutation that adds a required field with no source value; the executor halts and returns an `Escalation` naming the missing field. This is the moment that shows the USP.
- **Numbers.** Show the harness results: adaptation rate, MTTR, tokens and false-heal rate against the baseline.

Live LLM calls on stage are flaky. Keep the fast-path mutations live, and cache the Tier 2 response for the demo mutation so it can be replayed if the network fails.
