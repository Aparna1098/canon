# search-hack: quorum

**quorum** answers factual questions by sending each one to the right specialist source, then checking the answer
against the open web. It is built on the [Telem](https://telem.ai) multi-provider search SDK.

By default Telem fans a query out to 7 providers. quorum does three things differently:

1. **Classify the question.** It labels each question as `finance`, `medical`, `local`, `code` or `general`.
2. **Use the authoritative source.** Finance questions go to SEC EDGAR, medical ones to PubMed and local ones to
   Google Maps. quorum fetches the matching document and pulls the fact out of it.
3. **Verify.** In parallel it runs a web check with Tavily. When the official source and the web disagree, the
   official source wins and the web value is shown struck through. General questions are settled by a vote across
   several search engines.

Each answer reports how many provider calls it made, compared with Telem's default of 7.

```
question ─▶ classifier.py ─┬─ finance │ medical │ local ─▶ keyword query ─▶ specialist search ─▶ fetch doc ─▶ extract fact ─┐
                           │                       └──────▶ tavily search (original question) ─▶ extract claim ──────────────┴─▶ reconcile
                           │                                                                                                   (official wins)
                           └─ general │ code ─▶ exa + tavily ─▶ extract & cluster claims ─▶ agree? ── yes ─▶ answer
                                                                                              └─ no ─▶ + brave + parallel ─▶ engine quorum
```

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env   # add TELEM_API_KEY and ANTHROPIC_API_KEY (TAVILY_API_KEY for search.py)
```

## Run

```bash
.venv/bin/streamlit run app.py                                         # web UI
.venv/bin/python quorum.py "What was Tesla's revenue in Q3 2026?"      # CLI, several questions allowed, --json for raw output
.venv/bin/python classifier.py                                         # classifier test table (10 queries)
.venv/bin/python classifier.py "urgent care clinics in Raleigh"        # classify one query
```

The UI shows the category, the classifier's method and confidence, the route taken, provider and LLM call counts,
and latency. It lists the sources with badges (**Official**, **Agreeing engines**, **Web**), puts a line through
values that lost, and has a full call log.

| env var | default | purpose |
|---|---|---|
| `QUORUM_MODEL` | `claude-opus-5-5` | LLM for parameter extraction, fact/claim extraction and reconciliation |
| `QUORUM_EFFORT` | `low` | effort level for those calls |

## How it works

### Classifier (`classifier.py`)
The classifier tries three methods in order, stopping at the first that gives a confident answer:
1. **Embeddings.** It embeds the question with `all-MiniLM-L6-v2` (local, CPU) and compares it with the average of
   8 example questions per category. This takes a few milliseconds and costs nothing.
2. **Claude Haiku.** If the top score is below 0.25, or less than 0.10 ahead of the runner-up, it asks
   `claude-haiku-5-5`. A strict tool schema limits the answer to the 5 labels.
3. **Keyword rules.** These run if there is no Anthropic key or the Haiku call fails.

### Specialist path (`quorum.py`)
| label | specialist | notes |
|---|---|---|
| finance | `sec_edgar` | needs both a company and a period, otherwise falls back to general |
| medical | `ncbi_pubmed` | |
| local | `fetchlayer_google_maps` | |

An LLM turns the question into a keyword query. quorum takes the specialist result that names the entity, fetches
the full document with Telem `fetch()` and extracts the value along with a verbatim supporting quote. The Tavily
check runs at the same time, and a reconcile step decides whether the two values agree (rounding and unit
differences count as agreement). If the specialist returns nothing or the document doesn't contain the answer,
quorum falls back to the general path.

### General path
Exa and Tavily run first because Test B found they overlap least (about 7% of URLs). quorum extracts one claim per
source and groups claims that are the same value (numbers within 0.5%). Each engine votes for the group holding most
of its claims. If the two engines disagree, or either finds nothing, quorum adds Brave and Parallel and the majority
of engines decides. `code` questions use this path too, because Telem has no code specialist.

## Experiments

These are the scripts used to work out how Telem actually behaves, which shaped the design above.

| script | what it checks | output |
|---|---|---|
| `search.py` | basic multi-provider search; passes when 3+ providers return results | stdout |
| `routing_test.py` | which providers run per tier and query (no routing) | stdout |
| `routing_test2.py` | `auto_routing` modes vs. the fixed fan-out, with reported cost | stdout |
| `provider_catalog.py` | full provider catalog, grouped by category (free, no searches) | `providers.csv` |
| `validate_router.py` | specialist access (A), web-engine overlap (B), sec_edgar + fetch + web end to end (C) | `validate_router_results.json` |

`python validate_router.py C [engine]` reruns only Test C (output: `validate_router_results_C.json`).

### Findings
- **Telem's built-in routing doesn't save calls.** The server accepts only `auto_routing="accuracy"`, which ranks
  providers but still runs all of them, and it ignores `max_routing_providers`. Routing has to happen on the
  client, which is why quorum exists.
- **Specialists work.** `sec_edgar`, `ncbi_pubmed`, `fetchlayer_reddit` and `fetchlayer_google_maps` all returned
  relevant results for their own kind of query (Test A).
- **Web engines overlap little.** URL overlap between pairs ranged from about 7% (exa/tavily) to 35% (exa/parallel),
  so a second engine adds real independent evidence (Test B).
- **`sec_edgar` needs keywords and returns only pointers to filings.** Given a plain-English question, it returned
  unrelated filings. Given `Tesla revenue Q3 2026 10-Q`, it found Tesla's 8-K exhibit. Either way it returns links,
  not figures, so quorum fetches the document itself. Tavily alone returned the correct $28.05B, but also the
  analyst estimate ($28.17B) and an unrelated older figure. That mix is why the web value is checked against the
  official one rather than trusted (Test C).
- **Reported cost is a lower bound.** Only exa reports a dollar `usage.total`; the other providers report credits
  or SKU counts.

## Files

| file | role |
|---|---|
| `quorum.py` | router, specialist and general paths, reconciliation, CLI |
| `classifier.py` | embedding → Haiku → rules question classifier |
| `app.py` | Streamlit UI |
| `search.py`, `routing_test*.py`, `provider_catalog.py`, `validate_router.py` | experiments (above) |
| `providers.csv`, `validate_router_results*.json` | experiment outputs |
