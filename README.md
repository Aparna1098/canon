# search-hack

Multi-provider web search using the [Telem](https://telem.ai) SDK, with [Tavily](https://tavily.com) available alongside.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env   # then add TELEM_API_KEY and TAVILY_API_KEY
```

## Run

```bash
.venv/bin/python search.py "your query"
```

Prints `by_provider` (status, result count, latency, top result per provider) and passes when 3+ providers return results.

## Experiments

| script | what it checks | output |
|---|---|---|
| `routing_test.py` | which providers run per tier and query (no routing) | stdout |
| `routing_test2.py` | `auto_routing` modes vs. the fixed fan-out, with reported cost | stdout |
| `provider_catalog.py` | full provider catalog, grouped by category (free, no searches) | `providers.csv` |
| `validate_router.py` | specialist access (A), web-engine overlap (B), sec_edgar + fetch + web end to end (C) | `validate_router_results.json` |

`python validate_router.py C [engine]` reruns only Test C (output: `validate_router_results_C.json`).

Findings so far: the server accepts only `auto_routing="accuracy"`, which ranks providers but still runs the
whole set; `max_routing_providers` is ignored. Only exa reports a dollar `usage.total`, so reported cost is a
lower bound. `sec_edgar` needs keyword queries and returns filing pointers only, so the design fetches the document.

## quorum: specialist router

```bash
.venv/bin/python quorum.py "What was Tesla's revenue in Q3 2026?"   # CLI
.venv/bin/streamlit run app.py                                      # UI
```

Needs `TELEM_API_KEY` and `ANTHROPIC_API_KEY` in `.env`. `QUORUM_MODEL` (default `claude-opus-5-5`) and
`QUORUM_EFFORT` (default `low`) override the LLM used for classify/extract/reconcile.

- finance → `sec_edgar`, medical → `ncbi_pubmed`, local → `fetchlayer_google_maps`: keyword search → fetch the
  matching document → extract the fact, checked in parallel against Tavily. Official source wins on conflict.
- general → exa + tavily, cluster the extracted claims; on conflict escalate to brave + parallel and resolve by quorum.
