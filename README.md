# Canon

**Telem finds the sources. Canon keeps the facts.**

Canon is a shared memory of verified facts for AI agents, built on [Telem](https://telem.ai).
Agents ask Canon a factual question. Canon answers from the official source with an exact quote that it has checked
against the document, then remembers the fact so the next agent pays 0 searches for it.

## How it works

```
question
  │
  ├─▶ 1. check memory ── hit (cosine > 0.85, same entity, same years/quarters/versions, not expired)
  │                       └─▶ return the stored fact: 0 calls, from_memory = true
  │
  ├─▶ 2. classify ─────── MiniLM embeddings → Claude Haiku if unsure → keyword rules
  │
  ├─▶ 3. route to the official source via Telem (search with providers_include, then fetch)
  │        code     PyPI / npm registry JSON (+ GitHub release notes for "what changed")
  │        finance  sec_edgar → fetch the 10-Q / 8-K / 10-K
  │        medical  ncbi_pubmed → fetch the paper
  │        local    fetchlayer_google_maps
  │        general  exa + tavily; if they disagree, add brave + parallel and take the majority
  │
  ├─▶ 4. extract the fact + an exact quote (Claude Haiku; code needs no LLM: the JSON field is the quote)
  │
  ├─▶ 5. verify the quote is in the fetched document
  │        (normalized for whitespace, case and 28,046 vs 28046; one retry, else "unverified")
  │
  ├─▶ 6. save to memory (verified answers only)
  │        expiry: code 1 day · finance until next quarter end · medical 30 days · local and general 7 days
  │        facts with 3+ hits get a Telem monitor on their source domain
  │
  └─▶ 7. return {answer, quote, source, url, date, verified, status, calls_used vs 7, tokens_saved, …}
```

Finance and medical answers are also cross-checked against Tavily in the background. The disagreement is only
shown when the web contradicts the official source, and the official value wins. Memory stores no user identity:
only the public fact, its source and anonymous counts.

## Run

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env                       # TELEM_API_KEY, ANTHROPIC_API_KEY

.venv/bin/streamlit run app.py             # UI: answer card, live steps, web-only comparison, Memory tab
.venv/bin/uvicorn api:app --port 8000      # API: POST /answer, GET /facts, GET /stats
.venv/bin/python eval.py                   # eval → eval_results.md (moves facts.db aside first)
.venv/bin/python -m canon.canon "Latest version of next.js"   # CLI
```

```bash
curl -s localhost:8000/answer -H 'content-type: application/json' -d '{"question":"Latest version of pytorch"}'
```

| path | role |
|---|---|
| `canon/canon.py` | pipeline: `answer(question, source=None)` |
| `canon/classifier.py` | embedding → Haiku → rules classifier (fallback when top < 0.25 or margin < 0.10) |
| `canon/sources.py` | one adapter per route, all through the Telem SDK |
| `canon/verify.py` | Haiku extraction, quote verification, reconciliation |
| `canon/memory.py` | SQLite fact memory (`facts.db`, gitignored), expiry, monitors |
| `api.py`, `app.py`, `eval.py` | API, UI, eval (`questions.jsonl`) |
| `experiments/` | the scripts that shaped the design ([README](experiments/README.md)) |

## What the experiments found

- **Telem fans out to all 7 default providers on every search.** Its `auto_routing` ranks providers but still
  runs all of them. Routing has to happen in the client.
- **Specialists exist but are manual.** `sec_edgar`, `ncbi_pubmed` and `fetchlayer_google_maps` are in the
  catalog, but Telem never picks them on its own; you have to name them in `providers_include`.
- **Web engines are fairly independent.** exa and tavily share only about 7% of URLs, so they make the first pair.
  brave and parallel are the escalation pair.
- **Answer engines are right but cite secondary sources.** Through Telem only tavily returns an answer. It was
  right on 4/4 test questions, but on 3 of them none of its citations was a primary source.
- **The web is stale on versions.** Web pages lag the registries by weeks, which is why the registry wins.

## Eval

| mode | accuracy | avg calls | avg tokens the agent must read | avg latency | memory hit rate |
|---|---|---|---|---|---|
| Web-only (tavily) | 93% (14/15) | 1.0 | 2,690 | 3.8s | — |
| Telem default (all 7) | 87% (13/15) | 7.0 | 19,556 | 4.2s | — |
| Canon, first pass | 100% (15/15) | 3.0 | 44 | 7.4s | 0% |
| Canon, second pass (reworded) | 100% (15/15) | 0.5 | 44 | 0.4s | 80% |

These are 15 questions with known answers (6 code, 3 finance, 2 medical, 2 local, 2 general), then the same 15
reworded. Code ground truth is read live from PyPI and npm. "Tokens the agent must read" counts the result text
for web-only and Telem default; for Canon it counts the answer, quote, URL and date it returns. Per-question
results are in [eval_results.md](eval_results.md).

- **Web-only goes stale.** Asked for the latest pydantic, tavily said v2.13.4 (May 2026). Canon said 2.14.3
  (2026-11-05), straight from PyPI.
- **Memory pays off on rewording.** 12 of the 15 reworded questions came back from memory with 0 calls in about
  0.4s. The other 3 (langchain, react and fastapi rewordings) scored below the 0.85 similarity threshold, so they
  re-ran the full pipeline and still answered correctly.
