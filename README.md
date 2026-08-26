# Aster & Row — Reliable RAG Support Agent

A customer-support agent for the fictional store *Aster & Row*, built for reliability
on a deliberately messy corpus: superseded policies, an unapproved draft with a
prompt-injection payload, two genuinely-conflicting active documents, and mock
order data containing PII, stale fields, and injected "instructions" inside internal
notes.

The agent answers policy/product questions with **RAG over `knowledge-base/`**, looks
up order status through a **sanitized tool over `data/orders.json`**, keeps
**multi-turn context**, treats all retrieved text and tool output as **untrusted
data**, cites its sources, abstains when information is insufficient, surfaces genuine
source conflicts, and never claims to have performed an action it cannot.

> The original take-home brief is preserved at [`docs/ASSIGNMENT.md`](docs/ASSIGNMENT.md).

---

## 1. Setup and run (from a clean clone)

Requires Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -e .

cp .env.example .env              # then paste your Gemini API key into .env
```

Run the interactive agent:

```bash
python -m aster_agent.cli          # chat; 'reset' clears the session, 'exit' quits
python -m aster_agent.cli --debug  # also stream a structured JSON trace per turn
```

Run the tests (offline; no API key needed):

```bash
python -m pytest                   # 60 tests, deterministic, no network
```

Run the evaluation suite (needs a Gemini key — see §5):

```bash
python -m aster_agent.evalsuite.runner            # all 22 cases
python -m aster_agent.evalsuite.runner --only visible   # 15 supplied cases
python -m aster_agent.evalsuite.runner --only extra     # 7 original cases
```

## 2. Environment variables

Copy `.env.example` to `.env` and fill in your key. **Never commit `.env`.**

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | — | Google Gemini API key (free tier works). Get one at https://aistudio.google.com/apikey |
| `GEMINI_MODEL` | `auto` | Chat / tool-calling model. `auto` asks the API which models the key can use and picks the best available (preferring generous-free-tier models like `gemini-2.0-flash`); pin an id to override. |
| `GEMINI_EMBED_MODEL` | `auto` | Embedding model (`auto` selects e.g. `gemini-embedding-001`) |
| `KB_TOP_K` | `5` | Passages retrieved per turn |
| `EMBED_BACKEND` | `gemini` | `gemini` (semantic) or `tfidf` (offline, deterministic, no key) |
| `GEMINI_MIN_INTERVAL_S` | `13` | Seconds between generate calls, to respect the free-tier rate limit (~5 req/min). Lower it if your key has more quota; `0` disables pacing. |

The deterministic parts of the system (ingestion, the TF-IDF retriever, and the
order-lookup tool) run with **no key at all**, which is what lets the whole test
suite run offline.

## 3. Model, embeddings, framework, storage

- **LLM:** Google **Gemini** (auto-selected, preferring high-free-quota `flash-lite`
  models; final results were produced on `gemini-3.5-flash-lite`) via the official
  `google-genai` SDK,
  with **manual function calling** (we execute and record tool calls ourselves rather
  than using automatic calling, so every tool call is observable and its arguments are
  assertable).
- **Embeddings:** Gemini **`gemini-embedding-001`**. A dependency-light, deterministic
  **hashed TF-IDF** embedder is included as an offline fallback so retrieval and most
  of the eval suite run with no key or network.
- **Framework:** none. Plain Python behind small interfaces (`LLMClient`, `Embedder`,
  `Tracer`) — the brief does not score framework choice, and this keeps the reliability
  logic explicit and testable.
- **Storage:** in-memory NumPy matrix; cosine similarity is a dot product over
  L2-normalized vectors. Gemini embeddings are cached to `.kb_cache/` keyed by a corpus
  fingerprint, so re-runs don't re-embed. No external vector database (per the brief).

## 4. Architecture

```
knowledge-base/*.md ─▶ ingestion ─▶ embeddings ─▶ KnowledgeBase (vector index)
   (front matter)      (chunk by      (Gemini or       │  precedence-aware search
                        heading)       TF-IDF)          ▼
                                                  SupportAgent ───────────▶ answer
data/orders.json ─▶ order_lookup tool (sanitized) ──▲   │  (+ sources, handoff)
                                                     │   ▼
                                              Gemini (function calling)
                                                         │
                                                    JsonTracer (per-turn trace)
```

Per turn the agent:

1. Builds a **conversation-aware retrieval query** (recent user turns + current) so
   follow-ups like *"What about Canada?"* retrieve the right passages.
2. Retrieves top-k passages, reranked by **document precedence**: raw cosine similarity
   is multiplied by an *authority weight* so **active/official** content outranks
   **superseded** or **draft / non-authoritative** content. Low-authority chunks stay
   retrievable (so the agent can say "the legacy doc is superseded") but never win by
   default. When two *active official* docs both score high, both are returned so the
   agent can **surface the conflict** instead of silently choosing.
3. Runs Gemini with the `order_lookup` tool available; executes any tool call, feeds the
   **sanitized** result back, and loops until a text answer is produced.
4. Parses machine-readable `Sources:` / `Handoff:` markers, **forces a handoff** when a
   tool result requires human review (exception / not-found), and returns a structured
   response plus a full trace.

**Key reliability decisions**

- **Untrusted data framing.** Retrieved passages and tool results are wrapped and
  labeled as untrusted in the prompt; the system prompt is the only instruction source.
  This neutralizes the injection payloads in `14-internal-content-migration-notes.md`
  and in the `ORD-1005` internal note.
- **Privacy by allow-list.** The order tool returns only an explicit set of
  customer-safe fields; PII and everything under `internal` can never leave the function.
  The tracer additionally scrubs internal keys as defense in depth.
- **Status precedence.** `status` is authoritative; stale carrier/ETA fields are
  suppressed for `cancelled`/`returned`, and a null ETA is reported as "unavailable"
  rather than invented.
- **No fabricated actions.** The agent can look up and explain only; it never claims a
  cancel/refund/replacement/address-change happened.

## 5. Running the evaluation

```bash
python -m aster_agent.evalsuite.runner        # writes evaluation/results.json
```

The harness runs each case in a **fresh, isolated session** and grades with
**deterministic, non-LLM assertions**: text include/exclude, required and
forbidden-as-authority sources, tool called/not-called and tool arguments, forbidden
disclosures, clarifying-question and abstention behavior, handoff, "must cite both
conflicting sources", and an **anti-fabrication check** (any date or tracking number in
an answer must be grounded in an actual tool result). Concept and injection-compliance
checks use curated keyword/regex rules (`evalsuite/concept_rules.py`) that tolerate
paraphrase. It reports **per-case and per-category** results.

- **15 supplied cases** in `evaluation/visible-cases.json` — all covered.
- **7 original cases** in `evaluation/extra-cases.json` — paraphrases, multi-turn
  combinations, and new scenarios (e.g. an injection *inside an order's internal note*).

> **Free-tier note:** Gemini's free tier caps `generate_content` **per model, per day**
> (e.g. `gemini-3.6-flash` is only 20/day). A full 22-case run makes ~30+ calls, so pick a
> `flash-lite` model (bigger free quota) — `GEMINI_MODEL=auto` does this for you — and, if a
> model gets rate-limited, switch to another (each id is a separate daily bucket).

## 6. Evaluation results

Grading is fully deterministic (see §5). Both runs below are real live runs.

**Baseline** — before the tool-call fix, on `gemini-3.6-flash`: **8/15 visible (53%)**.
Four of the seven failures were a single bug — Gemini's `thought_signature` requirement
crashed *every* tool-calling case with a 400 (see Bug 1); the rest were the grading /
behavior misses in Bugs 2–5.

**Final** — after the fixes, on `gemini-3.5-flash-lite`: **11/15 visible (73%)** and
**5/7 original (71%)** → **16/22 overall (73%)**. The tool-calling category recovered
fully (**0/3 → 3/3**).

| Category | Baseline (visible) | Final (visible) |
|---|---|---|
| retrieval | 2/2 | 1/2 |
| groundedness | 2/2 | 2/2 |
| conversation | 1/1 | 0/1 |
| multi-source-grounding | 0/1 | 1/1 |
| privacy | 1/1 | 1/1 |
| source-conflict | 1/1 | 1/1 |
| tool-use | 1/2 | 1/2 |
| tool-reliability | 0/3 (thought_signature 400) | **3/3** |
| prompt-security | 0/1 | 0/1 |
| abstention | 0/1 | 1/1 |
| **Overall** | **8/15 (53%)** | **11/15 (73%)** |

Original cases (`--only extra`), final: **5/7** — tool-reliability 2/2, retrieval 1/1,
multi-source-grounding 1/1, prompt-security 1/2, conversation 0/1.

**Remaining failures** are model-quality / judgment issues, not crashes, and are largely
tied to running a *lite* model on the free tier (the stronger `gemini-3.6-flash` passed
several of these before its 20/day quota was exhausted):
- `trailplus-return-window`, `valid-order-lookup`: the lite model paraphrased an
  exact-match string ("45 days" for "45 calendar days"; "in transit" for "shipped").
- `canada-multiturn`, `retrieved-prompt-injection`: the lite model dropped a required
  concept (didn't re-name "Canada" / didn't explicitly label the migration note
  non-authoritative), though it did *behave* correctly (ignored the injection).
- Two handoff judgment calls (`system-prompt-extraction` should hand off;
  `delivered-then-return-eligibility` shouldn't) — model-dependent.

The offline unit tests (`python -m pytest` — 67 passing) run deterministically with no
key and cover the order tool's safety properties, retrieval precedence, the agent
control flow, the grader, observability secret-safety, rate-limit/retry, model
auto-selection, and the Gemini tool-call replay.

## 7. Bug diary

**Bug 1 — Tool-calling crashed on Gemini 3.x (`thought_signature`).** *(discovered beyond
the visible-case wording, from the first live run.)*
- *Repro:* every tool-using case (`valid-order-lookup`, `cancelled-order-stale-eta`,
  `unknown-order`, `shipped-without-eta`) returned `400 INVALID_ARGUMENT: Function call is
  missing a thought_signature in functionCall parts`.
- *Root cause:* Gemini 3.x "thinking" models return an opaque `thought_signature` on the
  function-call part that must be echoed back verbatim on the follow-up turn. The tool loop
  *reconstructed* the call from name+args, dropping the signature.
- *Fix:* stash the original model `Content` on the `ToolCall` and replay it verbatim on the
  next turn instead of rebuilding it (`src/aster_agent/llm.py`).
- *Result:* the `tool-reliability` category went **0/3 → 3/3**.
- *Regression test:* `tests/test_llm_gemini.py::test_tool_call_raw_content_is_replayed_verbatim`.

**Bug 2 — Agent abstained correctly but the grader didn't recognize it.**
- *Repro:* `insufficient-information` failed the concept `the supplied information is
  insufficient` even though the agent answered "the provided information does not contain
  details… please contact a human."
- *Root cause:* the concept rule didn't include the "does not contain details" family of
  phrasings.
- *Fix:* broadened the concept rule (`concept_rules.py`); the case now passes.
- *Regression test:* `tests/test_evalsuite.py` concept coverage + the live case.

**Bug 3 — Agent omitted the 7-day damage-reporting window.**
- *Repro:* `final-sale-damaged-exception` failed the `report within 7 days` concept — the
  agent explained the damaged-item exception but never stated the reporting deadline.
- *Root cause:* nothing told the model to surface reporting deadlines/timeframes.
- *Fix:* system prompt now requires stating reporting deadlines (e.g. report damaged items
  within 7 calendar days) and preserving the KB's exact figures/units
  (`src/aster_agent/prompts.py`); the case now passes.
- *Regression test:* the `final-sale-damaged-exception` case (+ gold-answer validation).

**Bug 4 — Concept rule missed a hyphenated variant ("30-day" vs "30 day").**
- *Repro:* found *before spending API quota* by feeding hand-written correct answers
  through the grader (a `gold_check` script). "within the 30-day return window" failed the
  matching concept.
- *Root cause:* the rule listed `"30 day"` (space) but not `"30-day"` (hyphen); the
  normalizer unifies dashes but not hyphens.
- *Fix:* added hyphenated variants (`concept_rules.py`).
- *Regression test:* gold-answer validation passes for all 22 cases;
  `tests/test_evalsuite.py::test_concept_matching_canada`.

**Bug 5 — Flaky privacy assertion in the trace test (false positive on scores).**
- *Repro:* the observability PII test intermittently failed asserting `"82"` (a risk score)
  was absent — but `"82"` also appears inside retrieval **score** floats like `0.82xx`.
- *Root cause:* it matched a bare substring against the whole serialized trace.
- *Fix:* assert against the **structured, sanitized tool-result payload**, not the raw blob.
- *Regression test:* `test_trace_never_logs_pii_or_internal_on_order_lookup`, stable across
  repeated runs.

**Honorable mentions — model/quota churn (config + `ratelimit.py`).** `text-embedding-004`
and `gemini-2.5-flash`/`gemini-2.0-flash` returned 404 as Google retired ids; free-tier
`generate_content` is capped **per model, per day** (20/day for `gemini-3.6-flash`), which
429-crashed runs. Fixes: retry honoring the server `retryDelay` + pacing + fail-fast; and
**model auto-selection** (`model_select.py`) that queries the API and prefers high-quota
`flash-lite` models so a clean clone can run without hand-editing model ids. The eval
harness also treats any API error as a failed case, never a crashed run.

## 8. Known limitations / what I'd improve before production

- **Free-tier rate limits** make full live eval slow; a paid key or batched embeddings
  would remove the pacing.
- **Concept grading is keyword/regex based.** It's deterministic and paraphrase-tolerant
  by design, but new phrasings may need rule updates. A secondary (non-authoritative)
  LLM judge could be added alongside — never as the sole grader.
- **Chunking is heading-based**, which suits this small corpus; larger documents would
  benefit from size-bounded chunking with overlap.
- **Session memory is in-process**; production would need a session store and trimming.
- **Retrieval query for follow-ups** concatenates recent user turns — a lightweight query
  rewrite step would be more robust for long conversations.

## 9. AI coding tools used

- **Claude (Claude Code)** — used to scaffold the package, write the retrieval/tool/agent
  layers, design the deterministic eval harness, and diagnose the live-run failures.
- **Example of an AI suggestion that was wrong/incomplete:** the initial code defaulted to
  the embedding model `text-embedding-004` and chat model `gemini-2.5-flash`; both returned
  **404** on a current free-tier key (the models were renamed/retired to
  `gemini-embedding-001` and `gemini-3.6-flash`). This was only caught by running against
  the real API — a good reminder that model identifiers must be verified live, not assumed.

## 10. Demo video

▶ **[Watch the demo](docs/cometChatAgentAssignment.mp4)** (`docs/cometChatAgentAssignment.mp4`)

A screen recording demonstrating: a knowledge-base question with citations, an order
lookup, a multi-turn conversation (ship internationally → "what about Canada?"), a case
where the agent refuses to guess and recommends human help, and the evaluation suite
running.

<!-- Tip: to get an inline player instead of a download link, open this README in
GitHub's web editor and drag the .mp4 into it; GitHub rehosts it and embeds a player. -->

