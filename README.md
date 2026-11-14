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
