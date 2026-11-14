"""Record real Canon runs for the hosted (static) demo page.

    .venv/bin/uvicorn api:app --port 8000      # in another terminal
    .venv/bin/python scripts/record_replay.py  # writes web/replay.json

Resets the fact memory, then asks the demo questions in order through GET /answer/stream with
compare=true, so the reworded Tesla question is a real memory hit. Each event keeps its arrival time.
"""
import json
import os
import sys
import time

import httpx

BASE = os.environ.get("CANON_URL", "http://localhost:8000")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web", "replay.json")
QUESTIONS = [
    "Latest pydantic version",
    "What was Tesla's revenue in Q3 2026?",
    "How much revenue did Tesla report for Q3 2026?",
    "What's new in the latest langchain release?",
]
DROP = {"events", "classifier", "memory", "monitor", "explanation"}  # not shown on the page


def stream(client, q):
    events, t0, name = [], time.perf_counter(), None
    with client.stream("GET", f"{BASE}/answer/stream", params={"q": q, "compare": "true"}, timeout=300) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: ") and name:
                data = json.loads(line[6:])
                if name == "result":
                    data = {k: v for k, v in data.items() if k not in DROP}
                    data["sources"] = [{k: s.get(k) for k in ("badge", "provider", "url", "value")}
                                       for s in data.get("sources", [])]
                events.append({"t": int((time.perf_counter() - t0) * 1000), "event": name, "data": data})
                name = None
    return events


def main():
    with httpx.Client() as c:
        c.post(f"{BASE}/memory/reset").raise_for_status()
        runs = []
        for q in QUESTIONS:
            ev = stream(c, q)
            res = next((e["data"] for e in ev if e["event"] == "result"), None)
            if not res or not res.get("verified"):
                sys.exit(f"'{q}' did not produce a verified result; not writing {OUT}")
            print(f"{q!r}: {res['answer'][:60]!r} from_memory={res['from_memory']} ({ev[-1]['t']} ms, {len(ev)} events)")
            runs.append({"question": q, "events": ev})
        ev_sum = c.get(f"{BASE}/eval").json()
    json.dump({"recorded_at": time.strftime("%Y-%m-%d %H:%M %Z"), "eval": ev_sum, "runs": runs},
              open(OUT, "w"), indent=1, default=str)
    print(f"wrote {OUT} ({os.path.getsize(OUT) // 1024} KB)")


if __name__ == "__main__":
    main()
