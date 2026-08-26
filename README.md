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

- **LLM:** Google **Gemini** (auto-selected; defaults toward `gemini-2.0-flash`) via the official `google-genai` SDK,
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

> **Free-tier note:** Gemini's free tier is ~5 requests/minute with a daily cap. A full
> 22-case run makes ~35–40 calls (tool cases make extra round-trips), so it is paced and
> can take several minutes. If you exhaust the daily quota, use a fresh key/project.

## 6. Evaluation results

Model: `gemini-3.6-flash`. Grading is fully deterministic (see §5). Command:
`python -m aster_agent.evalsuite.runner --only visible`.

**Baseline** — first complete live run (commit before the tool-call fix): **8/15**.
Four of the seven failures were a single infrastructure bug (Gemini's
`thought_signature` requirement crashed every tool-calling case with a 400 — see Bug 1);
the other three were the grading/behavior misses in Bugs 2–4.

**Final** — after the fixes: _re-run `--only visible` and `--only extra` and paste the
totals here._

| Category | Baseline (15 visible) | Final |
|---|---|---|
| retrieval | 2/2 | _tbd_ |
| groundedness | 2/2 | _tbd_ |
| conversation | 1/1 | _tbd_ |
| privacy | 1/1 | _tbd_ |
| source-conflict | 1/1 | _tbd_ |
| tool-use | 1/2 | _tbd_ |
| tool-reliability | 0/3 (thought_signature 400) | _tbd_ |
| multi-source-grounding | 0/1 (omitted 7-day window) | _tbd_ |
| prompt-security | 0/1 (missing migration-note rebuttal; over-eager handoff) | _tbd_ |
| abstention | 0/1 (correct abstention; grader phrasing gap) | _tbd_ |
| **Overall** | **8/15 (53%)** | _tbd_ |

The offline unit tests (`python -m pytest` — 62 passing) run deterministically with no
key and cover the order tool's safety properties, retrieval precedence, the agent
control flow, the grader, observability secret-safety, rate-limit/retry, and the Gemini
tool-call replay.

## 7. Bug diary

**Bug 1 — Agent paraphrased policy figures, dropping "calendar".**
- *Repro:* `trailplus-return-window` — asked the TrailPlus return window; the model
  answered "45 days", failing the `must_include: "45 calendar days"` assertion.
- *Root cause:* nothing instructed the model to preserve the KB's exact units; it
  naturally shortened "45 calendar days" to "45 days".
- *Fix:* system prompt now requires preserving the KB's exact figures/units and stating
  all material conditions (`src/aster_agent/prompts.py`).
- *Regression test:* the deterministic eval case `trailplus-return-window` (and
  `standard-return-window`) assert the exact "N calendar days" phrasing.

**Bug 2 — Concept rule missed a hyphenated variant ("30-day" vs "30 day").**
- *Repro:* found *before spending API quota* by feeding hand-written correct answers
  through the grader (`scratchpad gold_check`). `delivered-then-return-eligibility`'s
  answer "within the 30-day return window" failed the `within the 30-day return window`
  concept.
- *Root cause:* the concept rule listed `"30 day"` (space) but not `"30-day"` (hyphen);
  the normalizer collapses en/em-dashes but not hyphens.
- *Fix:* added hyphenated variants to the affected concept rules
  (`src/aster_agent/evalsuite/concept_rules.py`).
- *Regression test:* the gold-answer validation now passes for all 22 cases; covered by
  `tests/test_evalsuite.py::test_concept_matching_canada` and the concept-rule coverage
  test.

**Bug 3 — Flaky privacy assertion in the trace test (false positive on scores).**
- *Repro:* `tests/test_observability.py::test_trace_never_logs_pii_or_internal` failed
  intermittently asserting `"82"` (a risk score) was absent — but `"82"` also appears
  inside retrieval **score** floats like `0.82xx` in the trace.
- *Root cause:* the test matched a bare substring against the whole serialized trace,
  which legitimately contains float scores.
- *Fix:* assert against the **structured tool-result payload** (which is sanitized),
  not the raw blob.
- *Regression test:* the updated `test_trace_never_logs_pii_or_internal_on_order_lookup`,
  stable across repeated runs.

**Bug 4 — Environment/config failures surfaced by the first live runs (honorable
mentions).** `text-embedding-004` → 404 (switched to `gemini-embedding-001`);
`gemini-2.5-flash` retired for new keys → 404 (switched to `gemini-3.6-flash`); free-tier
429s crashed the run (added pacing + retry honoring the server's `retryDelay`, and
fail-fast on exhausted quota). Each is a one-line fix in config/`ratelimit.py`, and the
harness already treated an API error as a failed case rather than a crashed run.

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

<!-- Embed the GIF or link the recording here, e.g.:
![demo](docs/demo.gif)
or: [▶ Watch the 3-minute demo](docs/demo.mp4)
-->

_A 2–4 minute screen recording demonstrating: a knowledge-base question with citations,
an order lookup, a multi-turn conversation, a case where the agent refuses to guess /
recommends human help, and the evaluation suite running._
