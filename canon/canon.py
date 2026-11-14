"""Canon pipeline: answer(question) -> dict.

CLI:  .venv/bin/python -m canon.canon "What was Tesla's revenue in Q3 2026?" ["another question" ...] [--json]
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from dataclasses import dataclass
from typing import Optional

from .sources import (DEFAULT_FANOUT, SPECIALISTS, _noop, code_path, extract_params, general_path,
                      specialist_path)
from .verify import Tracker, _ms


async def classify(question, tr):
    """Label via classifier.py (embedding -> haiku -> rules), off the event loop."""
    from . import classifier
    c = await asyncio.to_thread(classifier.classify, question)
    tr.add("classify", f"classifier:{c['method']}", c["latency_ms"])
    return c


@dataclass
class Route:
    category: str
    entity: Optional[str] = None
    period: Optional[str] = None
    keywords: Optional[str] = None
    period_assumed: bool = False


async def answer(question, on_step=None):
    """on_step(str) is called with a short progress message before each step."""
    step = on_step or _noop
    tr = Tracker()
    t0 = time.perf_counter()
    step("Classifying…")
    c = await classify(question, tr)
    step(f"Classified as {c['label']} via {c['method']} ({c['latency_ms']} ms)")
    route = Route(c["label"])
    note = ""
    if route.category in SPECIALISTS:
        step("Extracting query parameters…")
        p = await extract_params(question, route.category, tr)
        route.entity, route.period, route.keywords = p.entity, p.period, p.keywords
        if route.category == "finance" and not p.entity:
            note = " (finance needs a company name)"
            route.category = "general"
        elif route.category == "finance" and not p.period:
            # no period: assume the latest fiscal year and look for the most recent 10-K
            route.period, route.period_assumed = "latest fiscal year (assumed)", True
            kw = re.sub(r"\b10-Q\b", "", p.keywords or p.entity).strip()
            route.keywords = kw if "10-K" in kw else f"{kw} 10-K"
    if route.category in SPECIALISTS:
        out = await specialist_path(question, route, tr, on_step)
    elif route.category == "code":
        out = await code_path(question, tr, on_step)
    else:
        out = await general_path(question, tr, on_step)
        out["route"] += note
    step("Done")
    out.update({
        "question": question, "label": c["label"], "category": route.category,
        "classifier": {k: c[k] for k in ("method", "confidence", "margin", "latency_ms", "embedding_label")},
        "entity": route.entity, "period": route.period, "keywords": route.keywords,
        "provider_calls": tr.provider_calls, "llm_calls": tr.llm_calls,
        "default_calls": DEFAULT_FANOUT, "latency_ms": _ms(t0), "events": tr.events,
    })
    return out


# ---------------------------------------------------------------- CLI
def print_result(r):
    print("=" * 100)
    print(f"Q: {r['question']}")
    k = r["classifier"]
    print(f"label:    {r['label']} via {k['method']} ({k['latency_ms']} ms, conf {k['confidence']:.2f}, "
          f"margin {k['margin']:.3f})  -> category: {r['category']}")
    if r["entity"] or r["keywords"]:
        print(f"params:   entity: {r['entity']}  period: {r['period']}  keywords: {r['keywords']!r}")
    print(f"route:    {r['route']}")
    print(f"ANSWER:   {r['answer']}   [{r['status']}]")
    if r.get("explanation"):
        print(f"          {r['explanation']}")
    for s in r["sources"]:
        val = f"~~{s['value']}~~" if s["struck"] else (s["value"] or "")
        print(f"  [{s['badge']:16}] {s['provider']:24} {val:28} {s['date'] or '':12} {s['url'][:80]}")
    calls = ", ".join(f"{e['provider']}({e['ms']}ms{'' if e['ok'] else ' FAIL'})"
                      for e in r["events"] if e["kind"] in ("search", "fetch"))
    print(f"calls:    {r['provider_calls']} provider calls vs {r['default_calls']} default  |  "
          f"{r['llm_calls']} LLM calls  |  {r['latency_ms']} ms total")
    print(f"          {calls}")


async def _main(questions, as_json):
    results = []
    for q in questions:
        results.append(await answer(q))
        if not as_json:
            print_result(results[-1])
    if as_json:
        print(json.dumps(results, indent=1, default=str))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--json"]
    if not args:
        sys.exit(__doc__)
    asyncio.run(_main(args, "--json" in sys.argv))
