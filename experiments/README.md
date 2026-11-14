# Experiments

Scripts used to learn how Telem behaves before building Canon. Run from the repo root:
`.venv/bin/python experiments/<script>.py` (validate_router.py reads `experiments/providers.csv`, so run it from here: `cd experiments`).

| script | what it proved |
|---|---|
| `search.py` | The Telem SDK works end to end; a default search fans out to 7 providers. |
| `routing_test.py` | Which providers run per tier. With no `providers_include`, Telem runs the same 7 every time. |
| `routing_test2.py` | Telem's `auto_routing` only accepts `"accuracy"`, which ranks providers but still runs all of them; `max_routing_providers` is ignored. Routing has to happen client-side. |
| `provider_catalog.py` | The full catalog (`providers.csv`): specialists such as `sec_edgar`, `ncbi_pubmed` and `fetchlayer_google_maps` exist, but are never chosen automatically. |
| `validate_router.py` A | Each specialist returns relevant results for its own kind of query. |
| `validate_router.py` B | Web engines overlap little: exa/tavily share ~7% of URLs, exa/parallel 35%. exa + tavily is the most independent first pair. |
| `validate_router.py` C | `sec_edgar` needs keyword queries and returns filing links, not figures, so the filing must be fetched. Tavily alone mixed the real $28.05B with an analyst estimate. |
| `validate_router.py` D | Through Telem only Tavily returns a direct answer. It was right on 4/4, but 0% of its citations were primary on 3/4. Perplexity surfaced the most primary sources (80–100% on code and medical). |

Outputs: `providers.csv`, `validate_router_results*.json`.
