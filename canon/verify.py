"""LLM extraction (Claude Haiku), reconciliation, and quote verification."""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

MODEL = os.environ.get("CANON_MODEL", "claude-haiku-5-5")
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
    tokens_in: int = 0  # ~tokens of source text read from Telem (search results + fetched documents)

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
    """One structured-output call to Claude Haiku; token usage is recorded on the tracker."""
    t0 = time.perf_counter()
    resp = await llm().messages.parse(model=MODEL, max_tokens=4000, system=system,
                                      messages=[{"role": "user", "content": user}], output_format=schema)
    tr.add("llm", f"{MODEL}:{label}", _ms(t0))
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"LLM declined the {label} step")
    return resp.parsed_output


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


# ---------------------------------------------------------------- quote verification
def _norm(s):
    """Lowercase, unify quotes/dashes, drop thousands separators and markdown, collapse whitespace."""
    s = (s or "").lower()
    s = s.translate(str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-",
                                   "\u2014": "-", "\u202f": " ", "\u00a0": " "}))
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)          # 28,046 -> 28046
    s = re.sub(r"[*_`#|>]+", " ", s)                    # markdown / table syntax
    return re.sub(r"\s+", " ", s).strip()


def quote_in(quote, text):
    """True if the quote appears in the document text after normalization. Long quotes may be
    stitched from a table row, so they pass when every 6-word window of them is present."""
    q, t = _norm(quote), _norm(text)
    if not q or len(q) < 4:
        return False
    if q in t:
        return True
    w = q.split()
    if len(w) < 8:
        return False
    windows = [" ".join(w[i:i + 6]) for i in range(0, len(w) - 5, 3)]
    return sum(x in t for x in windows) / len(windows) >= 0.8


async def extract_verified(question, doc, tr):
    """extract_fact, then check its quote is really in the document; retry once with a stricter prompt.
    Returns (fact, verified)."""
    fact = await extract_fact(question, doc, tr)
    if not fact.found:
        return fact, False
    if quote_in(fact.quote, doc["text"]):
        return fact, True
    tr.add("verify", "quote-not-found", 0, ok=False, note=fact.quote[:120])
    fact = await extract_fact(question + "\n(Copy the quote character for character from the document; "
                              "do not paraphrase or join lines.)", doc, tr)
    ok = fact.found and quote_in(fact.quote, doc["text"])
    tr.add("verify", "quote-retry", 0, ok=ok)
    return fact, ok


_YEAR = re.compile(r"^\D*((?:19|20)\d\d)\D*$")
_DATE = re.compile(r"((?:19|20)\d\d)-\d\d(?:-\d\d)?")


def less_precise(a, b):
    """'1998' vs '1998-11-20': consistent, one less precise. Returns True in that case."""
    for x, y in ((a, b), (b, a)):
        my, md = _YEAR.match((x or "").strip()), _DATE.search(y or "")
        if my and md and my.group(1) == md.group(1) and not _DATE.search(x or ""):
            return True
    return False
