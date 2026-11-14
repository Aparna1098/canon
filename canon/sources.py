"""One adapter per route, all searching and fetching through the Telem SDK.

    code     -> PyPI / npm registry JSON (+ GitHub release notes for "what changed" questions)
    finance  -> sec_edgar -> fetch filing -> extract   || tavily cross-check
    medical  -> ncbi_pubmed -> fetch paper -> extract  || tavily cross-check
    local    -> fetchlayer_google_maps -> extract      || tavily cross-check
    general  -> exa + tavily; on disagreement escalate to brave + parallel and resolve by engine majority
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Literal, Optional

from pydantic import BaseModel, Field
from telem import AsyncTelem, TelemError

from .verify import (DOC_CHARS, Fact, Params, Tracker, _ms, _text, ask, extract_claims, extract_fact,
                     reconcile)

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
FILING_CHARS = 1_500_000                 # how much of an SEC filing to fetch before excerpting

METRIC_TERMS = {
    "revenue": ["total revenues", "total revenue", "net sales", "total net sales", "revenues"],
    "sales": ["net sales", "total net sales", "total revenues"],
    "income": ["net income", "income from operations", "operating income"],
    "profit": ["net income", "gross profit", "operating income"],
    "earnings": ["net income", "earnings per share", "diluted"],
    "eps": ["earnings per share", "diluted"],
    "margin": ["gross margin", "operating margin", "gross profit"],
    "cash": ["cash flow", "net cash provided by operating activities", "free cash flow"],
    "debt": ["long-term debt", "total debt", "borrowings"],
}


def relevant_excerpt(text, question, budget=DOC_CHARS, window=1200):
    """For long filings: the opening plus windows around the question's metric terms, within budget."""
    ql = question.lower()
    terms = {t for k, v in METRIC_TERMS.items() if k in ql for t in v} or {"total revenues", "net sales", "net income"}
    spans = []
    for t in terms:
        for m in re.finditer(re.escape(t), text, re.I):
            spans.append((max(0, m.start() - window), min(len(text), m.end() + window)))
    spans.sort()
    merged = []
    for s, e in spans:  # merge overlapping windows
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    parts, used = [text[:4000]], 4000
    for s, e in merged:
        if used + (e - s) > budget:
            break
        parts.append(text[s:e])
        used += e - s
    return "\n…\n".join(parts)


PARAM_HINTS = {
    "finance": "SEC EDGAR full-text search. keywords: '<Company> <metric> <period> <10-Q|10-K>'. "
               "Resolve relative periods (e.g. 'last quarter') only if the question makes them unambiguous.",
    "medical": "PubMed. keywords: drug/intervention, condition and outcome terms.",
    "local": "Google Maps. keywords: '<what> <place>'.",
}


async def extract_params(question, label, tr):
    system = ("Extract query parameters for a specialist search index. " + PARAM_HINTS[label])
    return await ask(Params, system, question, tr, "params")


# ---------------------------------------------------------------- Telem
_telem = None


def telem():
    global _telem
    if _telem is None:
        _telem = AsyncTelem()
    return _telem


async def _search_once(query, providers, num_results, tr, attempt):
    """One Telem search. Returns {provider: [sources]}; every provider asked gets a key."""
    t0 = time.perf_counter()
    tag = " (retry)" if attempt else ""
    try:
        r = await telem().search(query, providers_include=providers, num_results=num_results,
                                 tier="default", fields=SEARCH_FIELDS)
    except TelemError as e:
        for p in providers:
            tr.add("search", p + tag, _ms(t0), ok=False, note=str(e)[:120])
        return {p: [] for p in providers}
    wall = _ms(t0)
    out = {p: [] for p in providers}
    for run in r.by_provider or []:
        ok = run.status == "succeeded" and bool(run.results)
        note = _text(run.error)[:120] or ("" if run.results else "0 results")
        tr.add("search", run.provider + tag, run.latency_ms or wall, ok=ok, note=note)
        out[run.provider] = [{
            "provider": run.provider, "title": res.title or "", "url": res.url,
            "date": str(res.publish_date) if res.publish_date else None,
            "text": " ".join(filter(None, [_text(res.summary), _text(res.excerpt)])),
        } for res in run.results]
    return out


async def search(query, providers, num_results, tr):
    """Search; any provider that returns 0 results (or errors) is retried once before giving up."""
    by_prov = await _search_once(query, providers, num_results, tr, attempt=0)
    empty = [p for p in providers if not by_prov.get(p)]
    if empty:
        by_prov.update({p: v for p, v in (await _search_once(query, empty, num_results, tr, attempt=1)).items() if v})
    return [s for p in providers for s in by_prov.get(p, [])]


async def fetch(url, tr, fmt="markdown", max_chars=DOC_CHARS, no_cache=False):
    t0 = time.perf_counter()
    try:
        r = await telem().fetch([url], inline_content=True, content_format=fmt,
                                inline_max_chars=max_chars, no_cache=no_cache or None)
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


SPECIALIST_LABELS = {"finance": ("SEC", "filing"), "medical": ("PubMed", "paper"), "local": ("Google Maps", "place")}


def _noop(_):
    pass


async def specialist_path(question, cls, tr, on_step=None):
    step = on_step or _noop
    specialist = SPECIALISTS[cls.category]
    src_name, doc_name = SPECIALIST_LABELS[cls.category]

    assumed = getattr(cls, "period_assumed", False)

    entity = (cls.entity or cls.keywords or "").lower()
    tokens = [t for t in re.findall(r"[a-z0-9]+", entity) if len(t) > 2] or [entity]

    def is_entity(h):
        return any(t in (h["title"] + " " + h["text"]).lower() for t in tokens)

    async def latest_10k():
        """sec_edgar ranks by relevance, so '<Company> 10-K' returns 2000s filings; adding the year
        finds the recent ones. Returns the newest primary 10-K, or None if none is under 18 months old."""
        short = (cls.entity or "").split()[0] if cls.entity else cls.keywords
        year = time.gmtime().tm_year
        for y in (year, year - 1):
            hits = [h for h in await search(f"{short} 10-K {y}", [specialist], 5, tr) if is_entity(h)]
            annual = [h for h in hits if re.search(r"\b10-K\b[^·]*\((?:FORM )?10-K\)", h["text"])]
            if annual:
                newest = max(annual, key=lambda h: h["date"] or "")
                cutoff = time.strftime("%Y-%m-%d", time.gmtime(time.time() - 548 * 86400))
                return newest if (newest["date"] or "") >= cutoff else None
        return None

    async def official():
        if assumed:
            step(f"Searching {src_name} for the latest 10-K…")
            pick = await latest_10k()
            if not pick:
                return None, None, None
        else:
            step(f"Searching {src_name}…")
            hits = await search(cls.keywords, [specialist], 3, tr)
            if not hits:
                return None, None, None
            pick = next((h for h in hits if is_entity(h)), hits[0])
        step(f"Fetching {doc_name}…")
        big = cls.category == "finance"  # 10-K/10-Q text runs to hundreds of pages
        content = await fetch(pick["url"], tr, max_chars=FILING_CHARS if big else DOC_CHARS)
        if content and len(content) > DOC_CHARS:
            content = relevant_excerpt(content, question)
        doc = dict(pick, text=content or pick["text"], fetched=bool(content))
        q = (f"{question} (No period was given: use the most recent fiscal year reported in this filing, "
             f"and name that fiscal year in the value.)") if assumed else question
        step(f"Extracting the fact from the {doc_name}…")
        fact = await extract_fact(q, doc, tr)
        return pick, doc, fact

    async def web():
        step("Cross-checking with Tavily…")
        sources = await search(question, [WEB_CHECK], 3, tr)
        claims = await extract_claims(question, sources, tr) if sources else []
        return sources, claims

    (pick, doc, fact), (web_sources, web_claims) = await asyncio.gather(official(), web())

    if not fact or not fact.found:
        why = "no specialist results" if not pick else "fact not found in the specialist document"
        step(f"{src_name}: {why} — falling back to web search…")
        out = await general_path(question, tr, on_step)
        out["route"] = f"{specialist} -> fallback to general ({why})"
        return out

    official_row = _source_row(doc, "Official", fact.value)
    official_row["quote"] = fact.quote
    official_row["fetched"] = doc["fetched"]
    rows = [official_row]
    web_claim = web_claims[0] if web_claims else None
    verdict = None
    if web_claim:
        step("Comparing official and web values…")
        verdict = await reconcile(question, fact, web_claim, tr)
        ws = web_sources[web_claim.source_index]
        rows.append(_source_row(ws, "Web", web_claim.value, struck=not verdict.agree))
    status = ("verified: matches official source" if verdict and verdict.agree
              else "conflict: official source wins" if verdict
              else "official source only (no web value found)")
    return {"route": f"specialist: {specialist}{' (latest 10-K, period assumed)' if assumed else ''}"
                     f" -> fetch -> extract  ||  {WEB_CHECK}",
            "answer": fact.value, "status": status, "sources": rows,
            "explanation": verdict.explanation if verdict else ""}


# ---------------------------------------------------------------- code route
# alias -> (registry package name, ecosystem). Bare "next" is left out: too common a word.
KNOWN_PACKAGES = {
    "pytorch": ("torch", "pypi"), "torch": ("torch", "pypi"), "langchain": ("langchain", "pypi"),
    "pydantic": ("pydantic", "pypi"), "django": ("django", "pypi"), "fastapi": ("fastapi", "pypi"),
    "flask": ("flask", "pypi"), "numpy": ("numpy", "pypi"), "pandas": ("pandas", "pypi"),
    "requests": ("requests", "pypi"), "transformers": ("transformers", "pypi"),
    "scikit-learn": ("scikit-learn", "pypi"), "sklearn": ("scikit-learn", "pypi"),
    "tensorflow": ("tensorflow", "pypi"), "streamlit": ("streamlit", "pypi"), "polars": ("polars", "pypi"),
    "sqlalchemy": ("sqlalchemy", "pypi"), "pytest": ("pytest", "pypi"),
    "next.js": ("next", "npm"), "nextjs": ("next", "npm"), "react": ("react", "npm"), "vue": ("vue", "npm"),
    "svelte": ("svelte", "npm"), "express": ("express", "npm"), "typescript": ("typescript", "npm"),
    "vite": ("vite", "npm"), "angular": ("@angular/core", "npm"), "tailwind": ("tailwindcss", "npm"),
    "tailwindcss": ("tailwindcss", "npm"), "eslint": ("eslint", "npm"), "webpack": ("webpack", "npm"),
    "prisma": ("prisma", "npm"), "zod": ("zod", "npm"),
}
INSTALL_PATTERNS = [
    (r"\bpip3? install\s+([A-Za-z0-9_.\-]+)", "pypi"),
    (r"\b(?:npm (?:i|install|add)|yarn add|pnpm add)\s+(@?[\w.\-/]+)", "npm"),
    (r"(?<![\w@])(@[\w\-]+/[\w.\-]+)", "npm"),
]
CHANGES_RE = re.compile(r"\b(breaking|what'?s new|changes?|changelog|release notes|migrat\w*|deprecat\w*|"
                        r"new features?)\b", re.I)


class PackageGuess(BaseModel):
    package: Optional[str] = Field(description="Registry package name, e.g. 'torch' or 'next'; null if none")
    ecosystem: Literal["pypi", "npm", "unknown"]


class ReleaseNotes(BaseModel):
    summary: str = Field(description="2-3 sentence answer to the question from the release notes")
    breaking: list[str] = Field(description="Breaking changes or removals listed in the notes; empty if none")


async def identify_package(question, tr):
    """(package, ecosystem, method): install commands -> known list -> Haiku if still unclear."""
    for pat, eco in INSTALL_PATTERNS:
        m = re.search(pat, question, re.I)
        if m:
            return m.group(1).lower(), eco, "regex"
    ql = question.lower()
    for alias in sorted(KNOWN_PACKAGES, key=len, reverse=True):
        if re.search(rf"(?<![\w.\-]){re.escape(alias)}(?![\w\-])", ql):
            pkg, eco = KNOWN_PACKAGES[alias]
            return pkg, eco, "known-list"
    g = await ask(PackageGuess, "Identify the PyPI or npm package the question is about.", question, tr,
                  "package")
    if g.package and g.ecosystem != "unknown":
        return g.package.lower(), g.ecosystem, "haiku"
    return None, None, None


def _json_text(t):
    t = (t or "").strip()
    if t.startswith("```"):  # some fetch providers wrap JSON in a code fence
        t = t[t.find("{"):].rstrip("`").strip()
    return t


def _gh_repo(u):
    m = re.search(r"github\.com[/:]([\w.\-]+)/([\w.\-]+?)(?:\.git)?(?:[/#?\s]|$)", u or "")
    return f"{m.group(1)}/{m.group(2)}" if m else None


async def pypi_latest(pkg, tr):
    t = _json_text(await fetch(f"https://pypi.org/pypi/{pkg}/json", tr, fmt="text", max_chars=200_000, no_cache=True))
    m = re.search(r'"version"\s*:\s*"([^"]+)"', t)  # info.version; the doc is truncated after info
    if not m:
        return None
    ver = m.group(1)
    candidates = []
    pu = re.search(r'"project_urls"\s*:\s*(\{[^{}]*\})', t)
    if pu:
        try:
            urls = json.loads(pu.group(1))
            candidates += [v for k, v in urls.items() if re.search(r"source|repo|github|code", k, re.I)]
            candidates += list(urls.values())
        except json.JSONDecodeError:
            pass
    hp = re.search(r'"home_page"\s*:\s*"([^"]*)"', t)
    if hp:
        candidates.append(hp.group(1))
    repo = next((r for r in map(_gh_repo, candidates) if r), None)
    api = f"https://pypi.org/pypi/{pkg}/{ver}/json"
    vt = _json_text(await fetch(api, tr, fmt="text", max_chars=400_000, no_cache=True))
    times = re.findall(r'"upload_time_iso_8601"\s*:\s*"([^"]+)"', vt)
    return {"registry": "PyPI", "version": ver, "date": min(times)[:10] if times else None, "repo": repo,
            "url": f"https://pypi.org/project/{pkg}/{ver}/"}


async def npm_latest(pkg, tr):
    # Version from dist-tags: Telem fetch served a 2.5-month-stale copy of /<pkg>/latest
    # (next 16.3.4 vs 16.4.6) even with no_cache, while dist-tags came back current.
    t = _json_text(await fetch(f"https://registry.npmjs.org/-/package/{pkg}/dist-tags", tr, fmt="text",
                               no_cache=True))
    try:
        ver = json.loads(t)["latest"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    # The version's own document gives the repository. npm has no small endpoint with the publish
    # time and the full document is too large to fetch whole, so the date comes from GitHub.
    vt = _json_text(await fetch(f"https://registry.npmjs.org/{pkg}/{ver}", tr, fmt="text", max_chars=200_000))
    try:
        repo = _gh_repo(_text(json.loads(vt).get("repository")))
    except json.JSONDecodeError:
        repo = _gh_repo(vt)
    return {"registry": "npm", "version": ver, "date": None, "repo": repo,
            "url": f"https://www.npmjs.com/package/{pkg}/v/{ver}"}


def _vkey(v):
    """Sort key for version strings; pre-releases sort below their release."""
    parts = re.findall(r"\d+", _v(v).split("-")[0].split("a")[0].split("b")[0].split("rc")[0])
    pre = bool(re.search(r"(a|b|rc|alpha|beta|dev|canary|-)", _v(v)))
    return tuple(int(p) for p in parts[:4]) + ((0,) if pre else (1,))


def same_version(a, b):
    """'2.15' == '2.15.0' == 'v2.15.0'; a pre-release never equals its final release."""
    def parts(v):  # numeric parts without trailing zeros, plus the release/pre-release flag
        *nums, final = _vkey(v)
        while nums and nums[-1] == 0:
            nums.pop()
        return nums, final

    (na, fa), (nb, fb) = parts(a), parts(b)
    return bool(_vkey(a)[:-1]) and na == nb and fa == fb


def best_web_claim(claims):
    """The web's best answer: the most-cited version, ties broken by the highest version."""
    if not claims:
        return None
    counts = {}
    for c in claims:
        counts.setdefault(_v(c.normalized) or _v(c.value), []).append(c)
    key = max(counts, key=lambda k: (len(counts[k]), _vkey(k)))
    return counts[key][0]


async def github_release(repo, pkg, ver, tr):
    from urllib.parse import quote
    for tag in (f"v{ver}", ver, f"{pkg}=={ver}", f"{pkg}@{ver}"):
        url = f"https://github.com/{repo}/releases/tag/{quote(tag, safe='')}"
        t = await fetch(url, tr, max_chars=30_000)
        if t and ver in t[:3000] and "page not found" not in t[:500].lower():
            m = re.search(r"Published:\s*(\d{4}-\d\d-\d\d)", t) or re.search(r"\b(20\d\d-\d\d-\d\d)\b", t)
            return {"url": url, "tag": tag, "text": t, "date": m.group(1) if m else None}
    return None


def _v(s):
    return (s or "").strip().lower().lstrip("v")


async def code_path(question, tr, on_step=None):
    step = on_step or _noop
    step("Identifying package…")
    pkg, eco, how = await identify_package(question, tr)
    if not pkg:
        out = await general_path(question, tr, on_step)
        out["route"] += "  (code: no package identified)"
        return out
    wants_changes = bool(CHANGES_RE.search(question))

    async def official():
        step(f"Fetching {'PyPI' if eco == 'pypi' else 'npm'} registry…")
        info = await (pypi_latest if eco == "pypi" else npm_latest)(pkg, tr)
        rel = None
        if info and info["repo"] and (wants_changes or not info["date"]):
            step("Fetching GitHub release…")
            rel = await github_release(info["repo"], pkg, info["version"], tr)
            if rel and not info["date"]:
                info["date"] = rel["date"]
        notes = None
        if rel and wants_changes:
            step("Reading release notes…")
            notes = await ask(ReleaseNotes, "Answer the question using only these release notes.",
                              f"Question: {question}\n\nRelease notes ({rel['url']}):\n{rel['text'][:DOC_CHARS]}",
                              tr, "release-notes")
        return info, rel, notes

    async def web():
        step("Cross-checking with Tavily…")
        sources = await search(question, [WEB_CHECK], 3, tr)
        if not sources:
            return sources, []
        claims = await extract_claims(f"What is the latest released version of the {pkg} package? "
                                      f"(value = the version number only)", sources, tr)
        return sources, claims

    (info, rel, notes), (web_sources, web_claims) = await asyncio.gather(official(), web())
    if not info:
        out = await general_path(question, tr, on_step)
        out["route"] += f"  (code: {eco} lookup for '{pkg}' failed)"
        return out

    rows = [{"badge": "Official", "provider": info["registry"], "title": f"{pkg} {info['version']} on {info['registry']}",
             "url": info["url"], "date": info["date"], "value": info["version"], "struck": False}]
    if rel:
        rows.append({"badge": "Official", "provider": "GitHub release", "title": f"{info['repo']} {rel['tag']}",
                     "url": rel["url"], "date": rel["date"], "value": "release notes", "struck": False})
    web_claim = best_web_claim(web_claims)
    agree = None
    if web_claim:
        ws = web_sources[web_claim.source_index]
        agree = same_version(web_claim.normalized, info["version"]) or same_version(web_claim.value, info["version"])
        rows.append(_source_row(ws, "Web", web_claim.value, struck=not agree))

    answer = f"{pkg} {info['version']}" + (f" (released {info['date']})" if info["date"] else "")
    if notes:
        answer += f" — {notes.summary}"
    status = ("verified: matches official source" if agree
              else "conflict: official source wins" if agree is False
              else "official source only (no web value found)")
    route = f"code: {pkg} ({eco}, via {how}) -> {info['registry']}" + (" + GitHub release" if rel else "") + f"  ||  {WEB_CHECK}"
    return {"route": route, "answer": answer, "status": status, "sources": rows,
            "explanation": ("Breaking: " + "; ".join(notes.breaking)) if notes and notes.breaking else "",
            "package": {"name": pkg, "ecosystem": eco, "method": how, "official": info["version"],
                        "official_date": info["date"], "web": web_claim.value if web_claim else None,
                        "web_url": web_sources[web_claim.source_index]["url"] if web_claim else None,
                        "web_date": web_sources[web_claim.source_index]["date"] if web_claim else None}}


async def general_path(question, tr, on_step=None):
    step = on_step or _noop
    step(f"Searching {' + '.join(GENERAL_FIRST)}…")
    sources = await search(question, GENERAL_FIRST, 3, tr)
    step("Extracting and comparing claims…")
    claims = await extract_claims(question, sources, tr) if sources else []
    route = f"general: {' + '.join(GENERAL_FIRST)}"

    def resolve(claims, sources, engines):
        clusters = cluster(claims, sources)
        return clusters, engine_votes(clusters, sources, engines)

    clusters, votes = resolve(claims, sources, GENERAL_FIRST)
    # agreement = every first-round engine answered and all voted for the same cluster
    agreed = None not in votes.values() and len(set(votes.values())) == 1
    if not agreed:
        step(f"Engines disagree — escalating to {' + '.join(GENERAL_ESCALATE)}…")
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
              else f"resolved by majority: {n_eng}/{len(votes)} engines")

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
