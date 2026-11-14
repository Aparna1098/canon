"""quorum: a specialist router on top of Telem.

    question -> classify (1 LLM call)
      finance | medical | local -> specialist path:
          keyword query -> specialist search (3) -> pick the entity's result -> Telem fetch()
          -> extract the fact from the document      } in parallel
          tavily search with the original question   }
          -> reconcile: agree = "verified"; disagree = official wins, web value struck through
      general -> exa + tavily in parallel -> extract claims -> cluster -> compare
          -> on conflict escalate to brave + parallel -> resolve by engine quorum

Every Telem provider run and fetch counts as a provider call; the summary compares
that with the 7 providers Telem runs by default.

CLI:  .venv/bin/python quorum.py "What was Tesla's revenue in Q3 2026?" ["another question" ...] [--json]
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Literal, Optional

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from telem import AsyncTelem, TelemError

load_dotenv()

MODEL = os.environ.get("QUORUM_MODEL", "claude-opus-5-5")
EFFORT = os.environ.get("QUORUM_EFFORT", "low")
DEFAULT_FANOUT = 7  # providers Telem runs when providers_include is omitted

SPECIALISTS = {
    "finance": "sec_edgar",
    "medical": "ncbi_pubmed",
    "local": "fetchlayer_google_maps",
}
WEB_CHECK = "tavily"                     # parallel web check on the specialist path
GENERAL_FIRST = ["exa", "tavily"]        # least-overlapping pair from validate_router.py Test B
GENERAL_ESCALATE = ["brave", "parallel"]
SEARCH_FIELDS = ["url", "title", "summary", "excerpt", "publish_date", "rank"]
DOC_CHARS = 60_000                       # document text passed to the extractor


# ---------------------------------------------------------------- LLM schemas
class Params(BaseModel):
    """Specialist query parameters, extracted only after classifier.py picks a specialist label."""
    entity: Optional[str] = Field(description="Company (finance), drug/condition (medical), or what+place (local); null if absent")
    period: Optional[str] = Field(description="Fiscal period for finance, e.g. 'Q3 2026' or 'FY2025'; null if absent or not finance")
    keywords: str = Field(description="Keyword query for the specialist index (no question words)")


class Fact(BaseModel):
    found: bool
    value: str = Field(description="The answer as a short human-readable value, e.g. '$28.05 billion'")
    normalized: str = Field(description="Canonical form for comparison: plain number in base units "
                                        "with unit (e.g. '28046000000 USD'), ISO date, or lowercase text")
    quote: str = Field(description="Verbatim snippet from the text that supports the value")
    source_index: int = Field(description="Index of the source the value came from, or -1")


class Claims(BaseModel):
    claims: list[Fact]


class Verdict(BaseModel):
    agree: bool = Field(description="True if the web value states the same fact as the official value")
    explanation: str


# ---------------------------------------------------------------- tracking
@dataclass
class Tracker:
    events: list = field(default_factory=list)

    def add(self, kind, provider, ms, ok=True, note=""):
        self.events.append({"kind": kind, "provider": provider, "ms": ms, "ok": ok, "note": note})

    @property
    def provider_calls(self):
        return sum(1 for e in self.events if e["kind"] in ("search", "fetch"))

    @property
    def llm_calls(self):
        # includes the classifier's Haiku fallback when it ran
        return sum(1 for e in self.events if e["kind"] == "llm" or e["provider"] == "classifier:haiku")


def _ms(t0):
    return int((time.perf_counter() - t0) * 1000)


def _text(v):
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return " ".join(_text(x) for x in v)
    if isinstance(v, dict):
        return " ".join(_text(x) for x in v.values())
    return str(v)


# ---------------------------------------------------------------- LLM
_llm = None


def llm():
    global _llm
    if _llm is None:
        _llm = anthropic.AsyncAnthropic()
    return _llm


async def ask(schema, system, user, tr: Tracker, label):
    t0 = time.perf_counter()
    resp = await llm().beta.messages.parse(
        model=MODEL,
        max_tokens=4000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",  # re-run on a fallback model if a safety classifier declines
        output_config={"effort": EFFORT},
        system=system,
        messages=[{"role": "user", "content": user}],
        output_format=schema,
    )
    tr.add("llm", f"{MODEL}:{label}", _ms(t0))
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"LLM declined the {label} step")
    return resp.parsed_output


PARAM_HINTS = {
    "finance": "SEC EDGAR full-text search. keywords: '<Company> <metric> <period> <10-Q|10-K>'. "
               "Resolve relative periods (e.g. 'last quarter') only if the question makes them unambiguous.",
    "medical": "PubMed. keywords: drug/intervention, condition and outcome terms.",
    "local": "Google Maps. keywords: '<what> <place>'.",
}


async def extract_params(question, label, tr):
    system = ("Extract query parameters for a specialist search index. " + PARAM_HINTS[label])
    return await ask(Params, system, question, tr, "params")


async def classify(question, tr):
    """Label via classifier.py (embedding -> haiku -> rules), off the event loop."""
    import classifier
    c = await asyncio.to_thread(classifier.classify, question)
    tr.add("classify", f"classifier:{c['method']}", c["latency_ms"])
    return c


def _sources_block(sources):
    return "\n\n".join(
        f"[{i}] {s['provider']} | {s['title']} | {s['date'] or 'n.d.'} | {s['url']}\n{s['text'][:3000]}"
        for i, s in enumerate(sources)
    )


async def extract_fact(question, doc, tr):
    system = ("Extract the single fact that answers the question from an official document. "
              "Use only the document. If it does not contain the answer, set found=false. "
              "Financial tables are often '$ in millions'; convert normalized values to base units.")
    user = f"Question: {question}\n\nDocument ({doc['url']}):\n{doc['text'][:DOC_CHARS]}"
    fact = await ask(Fact, system, user, tr, "extract-official")
    fact.source_index = 0
    return fact


async def extract_claims(question, sources, tr):
    system = ("For each numbered source, extract the value it gives as the answer to the question. "
              "Return one claim per source that answers it (skip sources that do not); "
              "source_index is the bracketed number. Use consistent units in 'normalized' across claims.")
    user = f"Question: {question}\n\nSources:\n{_sources_block(sources)}"
    out = await ask(Claims, system, user, tr, "extract-claims")
    return [c for c in out.claims if c.found and 0 <= c.source_index < len(sources)]


async def reconcile(question, official, web, tr):
    system = ("Decide whether a web value states the same fact as an official value. "
              "Treat rounding (e.g. $28.05B vs $28,046M) and unit differences as agreement; "
              "a different number, date or entity is disagreement.")
    user = (f"Question: {question}\nOfficial: {official.value} ({official.normalized})\n"
            f"Web: {web.value} ({web.normalized})")
    return await ask(Verdict, system, user, tr, "reconcile")


# ---------------------------------------------------------------- Telem
_telem = None


def telem():
    global _telem
    if _telem is None:
        _telem = AsyncTelem()
    return _telem


async def search(query, providers, num_results, tr):
    t0 = time.perf_counter()
    try:
        r = await telem().search(query, providers_include=providers, num_results=num_results,
                                 tier="default", fields=SEARCH_FIELDS)
    except TelemError as e:
        for p in providers:
            tr.add("search", p, _ms(t0), ok=False, note=str(e)[:120])
        return []
    wall = _ms(t0)
    sources = []
    for run in r.by_provider or []:
        tr.add("search", run.provider, run.latency_ms or wall, ok=run.status == "succeeded",
               note=_text(run.error)[:120])
        for res in run.results:
            sources.append({
                "provider": run.provider, "title": res.title or "", "url": res.url,
                "date": str(res.publish_date) if res.publish_date else None,
                "text": " ".join(filter(None, [_text(res.summary), _text(res.excerpt)])),
            })
    return sources


async def fetch(url, tr):
    t0 = time.perf_counter()
    try:
        r = await telem().fetch([url], inline_content=True, content_format="markdown",
                                inline_max_chars=DOC_CHARS)
    except TelemError as e:
        tr.add("fetch", "fetch", _ms(t0), ok=False, note=str(e)[:120])
        return None
    x = r.results[0]
    tr.add("fetch", f"fetch:{x.provider}", x.latency_ms or _ms(t0), ok=x.status == "succeeded",
           note=_text(x.error)[:120])
    return x.content if x.status == "succeeded" and x.content else None


# ---------------------------------------------------------------- clustering
_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def _same(a, b):
    """Same value: equal canonical strings, or numbers within 0.5%."""
    a, b = a.strip().lower(), b.strip().lower()
    if a == b:
        return True
    na, nb = _NUM.search(a.replace(",", "")), _NUM.search(b.replace(",", ""))
    if na and nb and not re.search(r"\d{4}-\d{2}", a + b):  # don't fuzz ISO dates
        x, y = float(na.group()), float(nb.group())
        unit_a, unit_b = a[na.end():].strip(), b[nb.end():].strip()
        return unit_a == unit_b and max(abs(x), abs(y)) > 0 and abs(x - y) / max(abs(x), abs(y)) < 0.005
    return False


def cluster(claims, sources):
    """Group claims by value; each cluster records the engines and sources behind it."""
    clusters = []
    for c in claims:
        for cl in clusters:
            if _same(cl["normalized"], c.normalized):
                break
        else:
            cl = {"normalized": c.normalized, "value": c.value, "claims": [], "engines": set()}
            clusters.append(cl)
        cl["claims"].append(c)
        cl["engines"].add(sources[c.source_index]["provider"])
    clusters.sort(key=lambda cl: (len(cl["engines"]), len(cl["claims"])), reverse=True)
    return clusters


def engine_votes(clusters, sources, engines):
    """Each engine votes for the cluster holding most of its claims (ties -> larger cluster).
    An engine with no claims votes None."""
    votes = {}
    for e in engines:
        counts = [(sum(1 for c in cl["claims"] if sources[c.source_index]["provider"] == e), -i)
                  for i, cl in enumerate(clusters)]
        best = max(counts, default=(0, 0))
        votes[e] = -best[1] if best[0] > 0 else None
    return votes


# ---------------------------------------------------------------- paths
def _source_row(s, badge, value=None, struck=False):
    return {"badge": badge, "provider": s["provider"], "title": s["title"], "url": s["url"],
            "date": s["date"], "value": value, "struck": struck}


async def specialist_path(question, cls, tr):
    specialist = SPECIALISTS[cls.category]

    async def official():
        hits = await search(cls.keywords, [specialist], 3, tr)
        if not hits:
            return None, None, None
        entity = (cls.entity or cls.keywords or "").lower()
        tokens = [t for t in re.findall(r"[a-z0-9]+", entity) if len(t) > 2] or [entity]
        pick = next((h for h in hits if any(t in (h["title"] + " " + h["text"]).lower() for t in tokens)), hits[0])
        content = await fetch(pick["url"], tr)
        doc = dict(pick, text=content or pick["text"], fetched=bool(content))
        fact = await extract_fact(question, doc, tr)
        return pick, doc, fact

    async def web():
        sources = await search(question, [WEB_CHECK], 3, tr)
        claims = await extract_claims(question, sources, tr) if sources else []
        return sources, claims

    (pick, doc, fact), (web_sources, web_claims) = await asyncio.gather(official(), web())

    if not fact or not fact.found:
        why = "no specialist results" if not pick else "fact not found in the specialist document"
        out = await general_path(question, tr)
        out["route"] = f"{specialist} -> fallback to general ({why})"
        return out

    official_row = _source_row(doc, "Official", fact.value)
    official_row["quote"] = fact.quote
    official_row["fetched"] = doc["fetched"]
    rows = [official_row]
    web_claim = web_claims[0] if web_claims else None
    verdict = None
    if web_claim:
        verdict = await reconcile(question, fact, web_claim, tr)
        ws = web_sources[web_claim.source_index]
        rows.append(_source_row(ws, "Web", web_claim.value, struck=not verdict.agree))
    status = ("verified: matches official source" if verdict and verdict.agree
              else "conflict: official source wins" if verdict
              else "official source only (no web value found)")
    return {"route": f"specialist: {specialist} -> fetch -> extract  ||  {WEB_CHECK}",
            "answer": fact.value, "status": status, "sources": rows,
            "explanation": verdict.explanation if verdict else ""}


async def general_path(question, tr):
    sources = await search(question, GENERAL_FIRST, 3, tr)
    claims = await extract_claims(question, sources, tr) if sources else []
    route = f"general: {' + '.join(GENERAL_FIRST)}"

    def resolve(claims, sources, engines):
        clusters = cluster(claims, sources)
        return clusters, engine_votes(clusters, sources, engines)

    clusters, votes = resolve(claims, sources, GENERAL_FIRST)
    # agreement = every first-round engine answered and all voted for the same cluster
    agreed = None not in votes.values() and len(set(votes.values())) == 1
    if not agreed:
        more = await search(question, GENERAL_ESCALATE, 3, tr)
        offset = len(sources)
        more_claims = await extract_claims(question, more, tr) if more else []
        for c in more_claims:
            c.source_index += offset
        sources, claims = sources + more, claims + more_claims
        engines = GENERAL_FIRST + GENERAL_ESCALATE
        clusters, votes = resolve(claims, sources, engines)
        route += f" -> conflict -> escalate {' + '.join(GENERAL_ESCALATE)}"

    if not clusters:
        return {"route": route, "answer": "no answer found", "status": "unresolved",
                "sources": [_source_row(s, "Web") for s in sources[:6]], "explanation": ""}

    tally = {}
    for e, i in votes.items():
        if i is not None:
            tally[i] = tally.get(i, 0) + 1
    win = max(tally, key=lambda i: (tally[i], len(clusters[i]["claims"]))) if tally else 0
    winner = clusters[win]
    n_eng = sum(1 for i in votes.values() if i == win)
    status = (f"agreed: {n_eng}/{len(votes)} engines" if agreed
              else f"resolved by quorum: {n_eng}/{len(votes)} engines")

    rows, seen = [], set()
    for cl_i, cl in enumerate(clusters):
        for c in cl["claims"]:
            s = sources[c.source_index]
            if s["url"] in seen:
                continue
            seen.add(s["url"])
            rows.append(_source_row(s, "Agreeing engines" if cl_i == win else "Web", c.value, struck=cl_i != win))
    rows.sort(key=lambda r: r["badge"] != "Agreeing engines")
    return {"route": route, "answer": winner["value"], "status": status, "sources": rows,
            "explanation": f"{len(clusters)} distinct value(s) across {len(claims)} claims"}


@dataclass
class Route:
    category: str
    entity: Optional[str] = None
    period: Optional[str] = None
    keywords: Optional[str] = None


async def answer(question):
    tr = Tracker()
    t0 = time.perf_counter()
    c = await classify(question, tr)
    route = Route(c["label"])
    note = ""
    if route.category in SPECIALISTS:
        p = await extract_params(question, route.category, tr)
        route.entity, route.period, route.keywords = p.entity, p.period, p.keywords
        if route.category == "finance" and not (p.entity and p.period):
            note = f" (finance needs company + period; got {p.entity!r}, {p.period!r})"
            route.category = "general"
    if route.category in SPECIALISTS:
        out = await specialist_path(question, route, tr)
    else:
        out = await general_path(question, tr)
        if c["label"] == "code":
            out["route"] += "  (no code specialist in Telem)"
        out["route"] += note
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
