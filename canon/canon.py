"""Canon pipeline: answer(question, source=None) -> dict.

    check memory -> classify -> route to the official source via Telem -> extract fact + exact quote
    -> verify the quote is in the document -> save to memory (verified only) -> return

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

from . import memory
from .sources import (DEFAULT_FANOUT, SPECIALISTS, _noop, _text, code_path, extract_params, general_path,
                      specialist_path, telem)
from .verify import Tracker, _ms

ROUTES = ("code", "finance", "medical", "local", "general")


def _tok(s):
    """Rough token count (~4 characters per token)."""
    return len(s or "") // 4


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


def _public(r, question):
    """The return format shared by fresh answers and memory hits."""
    returned = _tok(" ".join(str(r.get(k) or "") for k in ("answer", "quote", "url", "date")))
    return {
        "question": question, "answer": r["answer"], "quote": r.get("quote"), "source": r.get("source"),
        "url": r.get("url"), "date": r.get("date"), "verified": bool(r.get("verified")), "status": r["status"],
        "route": r["route"], "category": r["category"], "entity": r.get("entity"),
        "from_memory": r.get("from_memory", False), "calls_used": r["calls_used"], "calls_default": DEFAULT_FANOUT,
        "tokens_read": r["tokens_read"], "tokens_returned": returned,
        "tokens_saved": max(0, r.get("tokens_basis", r["tokens_read"]) - returned),
        "latency_ms": r["latency_ms"], "classifier": r.get("classifier"),
        "sources": r.get("sources", []), "explanation": r.get("explanation", ""),
        "web_disagrees": r.get("web_disagrees", False), "less_precise": r.get("less_precise", False),
        "memory": r.get("memory"), "events": r.get("events", []), "monitor": r.get("monitor"),
    }


async def answer(question, source=None, on_step=None, use_memory=True):
    """Answer a factual question. `source` forces a route (code|finance|medical|local|general).
    on_step(str) is called with a short progress message before each step."""
    step = on_step or _noop
    tr = Tracker()
    t0 = time.perf_counter()

    emb = None
    if use_memory:
        step("Checking memory…")
        emb = await asyncio.to_thread(memory.embed, question)
        hit, cos = await asyncio.to_thread(memory.lookup, question, emb)
        if hit:
            step(f"Found in memory (cosine {cos:.2f}) — 0 calls")
            r = json.loads(hit["result_json"])
            r.update(from_memory=True, calls_used=0, tokens_read=0, tokens_basis=hit["tokens_read"],
                     latency_ms=_ms(t0), route=f"memory (fact #{hit['id']}, cosine {cos:.2f})",
                     memory={"id": hit["id"], "cosine": round(cos, 3), "verified_at": hit["verified_at"],
                             "expires_at": hit["expires_at"], "hit_count": hit["hit_count"] + 1},
                     classifier=None, events=[])
            out = _public(r, question)
            memory.record_hit(hit["id"], hit["calls_used"], out["tokens_saved"])
            if hit["hit_count"] + 1 >= MONITOR_AT and not hit["monitor_id"]:
                out["monitor"] = await watch(hit, step)
            step("Done")
            return out

    if source in ROUTES:
        c = {"label": source, "method": "forced", "confidence": 1.0, "margin": 1.0, "latency_ms": 0,
             "embedding_label": source}
    else:
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
    out.setdefault("entity", route.entity)
    out.update(category=route.category, calls_used=tr.provider_calls, tokens_read=tr.tokens_in,
               latency_ms=_ms(t0), events=tr.events,
               classifier={"label": c["label"], "method": c["method"], "confidence": c["confidence"],
                           "ms": c["latency_ms"]})
    res = _public(out, question)
    if use_memory and res["verified"]:
        stored = {k: v for k, v in res.items() if k not in ("events", "memory")}
        await asyncio.to_thread(memory.store, question, emb, res, json.dumps(stored, default=str))
        step("Saved to memory")
    elif use_memory:
        step("Not saved: unverified answers are never stored")
    step("Done")
    return res


MONITOR_AT = 3  # a fact this popular gets a Telem monitor on its source
MONITOR_FREQ = {"code": "1d", "finance": "1d", "medical": "7d", "local": "7d", "general": "7d"}


async def watch(fact, step=_noop):
    """Register a Telem monitor on a hot fact's source, so a change can invalidate it.
    Telem monitors watch a query (optionally restricted to domains), not a single URL."""
    from urllib.parse import urlparse
    domain = (urlparse(fact["source_url"] or "").hostname or "").removeprefix("www.")
    try:
        step(f"Hot fact ({MONITOR_AT}+ hits): registering a Telem monitor on {domain or 'the web'}…")
        m = await telem().monitors.create(
            f"Changes to: {fact['question']} (currently: {fact['answer']})",
            frequency=MONITOR_FREQ.get(fact["category"], "7d"), ttl="30d",
            include_domains=[domain] if domain else None)
        await asyncio.to_thread(memory.set_monitor, fact["id"], m.id)
        return {"id": m.id, "domain": domain, "agent_state": str(getattr(m, "agent_state", None)),
                "note": getattr(m, "agent_note", None)}
    except Exception as e:  # noqa: BLE001 - monitoring is best effort; the answer is already served
        return {"error": f"{type(e).__name__}: {str(e)[:160]}"}


async def web_only(question, provider="tavily"):
    """What a web-only agent sees: one answer engine, its answer and the text it would have to read.
    Shown for comparison only; not counted in Canon's totals."""
    t0 = time.perf_counter()
    r = await telem().search(question, providers_include=[provider], num_results=5,
                             fields=["url", "title", "summary", "excerpt", "answer"])
    run = (r.by_provider or [None])[0]
    results = run.results if run else []
    text = " ".join(_text(x.summary) + " " + _text(x.excerpt) for x in results)
    return {"provider": provider, "answer": (run.answer or "").strip() if run else "",
            "urls": [x.url for x in results], "tokens_read": _tok(text), "calls": 1, "latency_ms": _ms(t0)}


async def telem_default(question):
    """Telem's default fan-out (no providers_include): what the agent would read."""
    t0 = time.perf_counter()
    r = await telem().search(question, num_results=5, fields=["url", "title", "summary", "excerpt", "answer"])
    text = " ".join(_text(x.summary) + " " + _text(x.excerpt) for x in r.results)
    answers = [run.answer for run in r.by_provider or [] if run.answer]
    return {"answer": answers[0].strip() if answers else "", "calls": len(r.by_provider or []),
            "tokens_read": _tok(text), "latency_ms": _ms(t0), "texts": text}


# ---------------------------------------------------------------- CLI
def print_result(r):
    print("=" * 100)
    print(f"Q: {r['question']}")
    k = r["classifier"]
    if k:
        print(f"label:    {k['label']} via {k['method']} ({k['ms']} ms, conf {k['confidence']:.2f}) "
              f"-> category: {r['category']}")
    print(f"route:    {r['route']}")
    print(f"ANSWER:   {r['answer']}   [{r['status']}]{'  (from memory)' if r['from_memory'] else ''}")
    if r.get("quote"):
        print(f"quote:    \"{r['quote'][:200]}\"")
    print(f"source:   {r['source']}  {r['url']}  {r['date'] or ''}")
    print(f"calls:    {r['calls_used']} vs {r['calls_default']} default | tokens read {r['tokens_read']}, "
          f"returned {r['tokens_returned']}, saved {r['tokens_saved']} | {r['latency_ms']} ms")


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
