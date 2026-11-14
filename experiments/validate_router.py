"""Validate the specialist-router idea: specialist access/quality (A), web-engine
independence (B), the 2-call design end to end (C), and answer-provider correctness and
citation quality (D).

Budget guard: aborts if any single call reports cost > $0.10 or the reported total > $1.
Caveat: most providers report no dollar cost, so the guard only sees what is reported."""
import asyncio
import csv
import json
import re
import sys
import time
import urllib.request
from itertools import combinations
from urllib.parse import urlparse

from dotenv import load_dotenv
from telem import AsyncTelem, TelemError

load_dotenv()

MAX_CALL_COST = 0.10
MAX_TOTAL_COST = 1.00
FIELDS = ["url", "title", "summary", "excerpt", "publish_date", "rank", "usage"]

spent = 0.0
unreported_calls = 0
log = []


class BudgetExceeded(Exception):
    pass


def run_cost(run):
    u = run.usage
    t = u.get("total") if isinstance(u, dict) else None
    return float(t) if isinstance(t, (int, float)) else None


async def call(c, query, providers, num_results, fields=FIELDS, tier="default"):
    """One Telem search with budget accounting. Returns (response|None, error|None, wall_ms)."""
    global spent, unreported_calls
    t0 = time.perf_counter()
    try:
        r = await c.search(query, providers_include=providers, num_results=num_results, fields=fields, tier=tier)
    except TelemError as e:
        return None, f"{type(e).__name__}: {e}", int((time.perf_counter() - t0) * 1000)
    wall = int((time.perf_counter() - t0) * 1000)
    cost = 0.0
    for run in r.by_provider or []:
        rc = run_cost(run)
        if rc is None:
            unreported_calls += 1
        else:
            cost += rc
            if rc > MAX_CALL_COST:
                raise BudgetExceeded(f"{run.provider} reported ${rc:.4f} > ${MAX_CALL_COST} on '{query}'")
    spent += cost
    log.append({"query": query, "providers": providers, "cost": cost,
                "runs": [{"provider": x.provider, "status": x.status, "usage": x.usage, "error": x.error,
                          "latency_ms": x.latency_ms, "n": len(x.results)} for x in r.by_provider or []]})
    if spent > MAX_TOTAL_COST:
        raise BudgetExceeded(f"total reported spend ${spent:.4f} > ${MAX_TOTAL_COST}")
    return r, None, wall


def groups_from_csv():
    with open(__import__("os").path.join(__import__("os").path.dirname(__file__), "providers.csv")) as f:
        rows = list(csv.DictReader(f))
    g = {}
    for r in rows:
        g.setdefault(r["category"], []).append(r["name"])
    return {r["name"]: r["category"] for r in rows}, g


# ---------------------------------------------------------------- Test A
SPECIALISTS = [
    ("sec_edgar", "Tesla revenue Q3 2026 10-Q"),
    ("ncbi_pubmed", "GLP-1 agonists cardiovascular outcomes"),
    ("fetchlayer_reddit", "cursor vs claude code opinions"),
    ("fetchlayer_google_maps", "best ramen in Sunnyvale"),
]
# Fallbacks within the same providers.csv group, most comparable first.
FALLBACK_PREF = {
    "sec_edgar": ["coresignal_company", "tianyancha"],
    "ncbi_pubmed": ["ctgov_studies"],
    "fetchlayer_reddit": ["fetchlayer_x", "xcom", "socialdatax_x", "apify_quora", "apify_substack"],
    "fetchlayer_google_maps": ["outscraper_gmaps", "outscraper_greviews", "outscraper_tripadvisor"],
}


async def test_a(c):
    print("=" * 100 + "\nTEST A: specialist access & quality (num_results=3)\n" + "=" * 100)
    cat_of, groups = groups_from_csv()
    out = {}
    for primary, q in SPECIALISTS:
        group = groups.get(cat_of.get(primary, ""), [])
        chain = [primary] + [p for p in FALLBACK_PREF[primary] if p in group]
        attempts = []
        for prov in chain:
            r, err, wall = await call(c, q, [prov], 3)
            run = (r.by_provider or [None])[0] if r else None
            if err or not run or run.status != "succeeded" or not run.results:
                why = err or (run and (run.error or f"status={run.status}, 0 results")) or "no run"
                print(f"\n{prov:24} | FAILED: {str(why)[:150]} | {wall} ms")
                attempts.append({"provider": prov, "ok": False, "error": str(why)})
                continue
            print(f"\n{prov:24} | {run.status} | provider {run.latency_ms} ms (wall {wall} ms) | "
                  f"{len(run.results)} results | usage={json.dumps(run.usage)} | q='{q}'")
            for res in run.results[:3]:
                print(f"    - {(res.title or '(no title)')[:85]}\n      {res.url}")
            attempts.append({"provider": prov, "ok": True, "latency_ms": run.latency_ms, "wall_ms": wall,
                             "n": len(run.results), "usage": run.usage,
                             "titles": [x.title for x in run.results[:3]], "urls": [x.url for x in run.results[:3]]})
            break
        out[primary] = attempts
    return out


# ---------------------------------------------------------------- Test B
ENGINES = ["exa", "brave", "tavily", "parallel"]
GENERAL_QUERIES = ["Tesla Q3 2026 revenue", "when was the ISS launched", "latest pytorch version"]


def norm_url(u):
    p = urlparse(u or "")
    return (p.netloc.lower().removeprefix("www.") + p.path.rstrip("/")).lower()


def domain(u):
    return urlparse(u or "").netloc.lower().removeprefix("www.")


def overlap(a, b):
    """Jaccard overlap, as a percentage."""
    return 100.0 * len(a & b) / len(a | b) if (a | b) else 0.0


async def test_b(c):
    print("\n" + "=" * 100 + "\nTEST B: web engine independence (num_results=5)\n" + "=" * 100)
    urls = {e: set() for e in ENGINES}
    doms = {e: set() for e in ENGINES}
    per_query = {}
    for q in GENERAL_QUERIES:
        r, err, _ = await call(c, q, ENGINES, 5)
        if err:
            print(f"ERROR on '{q}': {err}")
            continue
        per_query[q] = {}
        for run in r.by_provider or []:
            u = {norm_url(x.url) for x in run.results}
            d = {domain(x.url) for x in run.results}
            per_query[q][run.provider] = (u, d)
            urls[run.provider] |= {(q, x) for x in u}
            doms[run.provider] |= {(q, x) for x in d}
        print(f"  '{q}': " + ", ".join(f"{p}={len(v[0])}" for p, v in per_query[q].items()))

    print(f"\n{'pair':20} | {'URL overlap':>11} | {'domain overlap':>14}   (Jaccard, pooled over {len(per_query)} queries)")
    print("-" * 60)
    pairs = []
    for a, b in combinations(ENGINES, 2):
        uo, do = overlap(urls[a], urls[b]), overlap(doms[a], doms[b])
        pairs.append((a, b, uo, do))
        print(f"{a + ' / ' + b:20} | {uo:10.0f}% | {do:13.0f}%")
    return pairs


# ---------------------------------------------------------------- Test C
REVENUE_RE = re.compile(
    r"total revenues?[^$\d]{0,80}\$?\s?([\d,]{4,}(?:\.\d+)?)\s*(million|billion)?", re.I)
MONEY_RE = re.compile(r"\$\s?([\d.,]+)\s*(billion|million|B)\b", re.I)


def as_text(v):
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return " ".join(as_text(x) for x in v)
    if isinstance(v, dict):
        return " ".join(as_text(x) for x in v.values())
    return str(v)


def extract_numbers(results, rx):
    found = []
    for res in results:
        text = " ".join(as_text(x) for x in [res.title, res.summary, res.excerpt, getattr(res, "full_content", None)])
        for m in rx.finditer(text):
            start = max(0, m.start() - 60)
            found.append((m.group(0).strip(), text[start:m.end() + 40].replace("\n", " "), res.url))
    return found


QUARTER_RE = re.compile(r"\bQ[1-4]-20\d\d\b")


async def fetch_filing_revenue(c, url, quarter="Q3-2026"):
    """Step 3 of the design: sec_edgar returns only a filing pointer, so fetch the document
    and read `quarter`'s column from the 'Total revenues' row ($ in millions)."""
    t0 = time.perf_counter()
    try:
        r = await c.fetch([url], inline_content=True, content_format="markdown")
    except TelemError as e:
        return {"url": url, "error": f"{type(e).__name__}: {e}"}
    wall = int((time.perf_counter() - t0) * 1000)
    x = r.results[0]
    text = x.content or ""
    out = {"url": url, "status": x.status, "provider": x.provider, "wall_ms": wall,
           "chars": len(text), "truncated": x.content_truncated, "error": x.error}
    header = QUARTER_RE.findall(text[:5000])
    # the first header row lists the quarters in column order, e.g. Q3-2025 ... Q3-2026
    cols = []
    for q in header:
        if q in cols:
            break
        cols.append(q)
    m = re.search(r"Total revenues\s+([\d,]+(?:\s+[\d,]+)*)", text)
    if m and quarter in cols:
        values = m.group(1).split()
        idx = cols.index(quarter)
        if idx < len(values):
            millions = int(values[idx].replace(",", ""))
            out.update(quarter=quarter, revenue_musd=millions, columns=cols, row=values)
    return out


async def test_c(c, web_engine):
    q = "What was Tesla's revenue in Q3 2026?"
    # sec_edgar is a keyword index: the natural-language question returned unrelated filers,
    # so the 2-call design also tries a keyword rewrite (what a router would send it).
    q_kw = "Tesla revenue Q3 2026 10-Q"
    print("\n" + "=" * 100 + f"\nTEST C: 2-call design on '{q}'  (sec_edgar + {web_engine})\n" + "=" * 100)
    fields = FIELDS + ["full_content", "answer"]
    (r_sec, e_sec, w_sec), (r_kw, e_kw, w_kw), (r_web, e_web, w_web) = await asyncio.gather(
        call(c, q, ["sec_edgar"], 3, fields=fields, tier="extended"),
        call(c, q_kw, ["sec_edgar"], 3, fields=fields, tier="extended"),
        call(c, q, [web_engine], 3, fields=fields, tier="extended"),
    )
    out = {}
    for name, r, e, w, rx in [("sec_edgar (NL question)", r_sec, e_sec, w_sec, REVENUE_RE),
                              ("sec_edgar (keywords)", r_kw, e_kw, w_kw, REVENUE_RE),
                              (web_engine, r_web, e_web, w_web, MONEY_RE)]:
        print(f"\n--- {name} (wall {w} ms) ---")
        if e or not r or not r.by_provider:
            print(f"ERROR: {e}")
            out[name] = {"error": e}
            continue
        run = r.by_provider[0]
        print(f"status={run.status} results={len(run.results)} usage={json.dumps(run.usage)}")
        if run.answer:
            print(f"answer: {str(run.answer)[:300]}")
        for res in run.results[:3]:
            print(f"  - {(res.title or '')[:80]} | {res.publish_date} | {res.url}")
        nums = extract_numbers(run.results, rx)
        for hit, ctx, url in nums[:4]:
            print(f"    number: {hit!r}\n      ctx: ...{ctx[:200]}...\n      src: {url}")
        out[name] = {"status": run.status, "n": len(run.results), "numbers": nums[:4],
                     "urls": [x.url for x in run.results[:3]], "answer": run.answer, "wall_ms": w}

    # Step 3: fetch the top Tesla filing from the keyword sec_edgar call and read the official number.
    print("\n--- fetch: official number from the filing ---")
    filing_url = None
    if r_kw and r_kw.by_provider:
        filing_url = next((x.url for x in r_kw.by_provider[0].results if "tesla" in (x.title or "").lower()), None)
    if not filing_url:
        print("no Tesla filing found by sec_edgar; skipping fetch")
        out["filing"] = {"error": "no Tesla filing in sec_edgar results"}
        return out
    filing = await fetch_filing_revenue(c, filing_url)
    out["filing"] = filing
    print(f"url: {filing_url}\nstatus={filing.get('status')} via {filing.get('provider')} | wall {filing.get('wall_ms')} ms | "
          f"{filing.get('chars')} chars (truncated={filing.get('truncated')})")
    if "revenue_musd" in filing:
        print(f"OFFICIAL {filing['quarter']} total revenues: ${filing['revenue_musd']:,}M "
              f"(= ${filing['revenue_musd'] / 1000:.2f}B)   columns={filing['columns']}")
    else:
        print(f"could not parse 'Total revenues' row: {filing.get('error')}")

    # Compare the web engine's figure to the filing.
    official = filing.get("revenue_musd")
    web_nums = [h for h, _, _ in out.get(web_engine, {}).get("numbers", [])]
    if official and web_nums:
        top = MONEY_RE.search(web_nums[0])
        val = float(top.group(1).replace(",", "")) * (1000 if top.group(2).lower().startswith("b") else 1)
        diff = abs(val - official) / official
        out["agreement"] = {"web_musd": val, "official_musd": official, "rel_diff": diff}
        print(f"\nside by side:  {web_engine} = ${val / 1000:.2f}B  |  filing = ${official / 1000:.2f}B  "
              f"|  diff {diff:.2%}  -> {'MATCH' if diff < 0.005 else 'MISMATCH'}")
    return out


# ---------------------------------------------------------------- Test D
# Answer providers: what they answer, whether it's right, and how much of what they cite is a primary source.
# Probed 2026-11-14: through Telem only tavily fills `answer` (when the `answer` field is requested);
# perplexity and linkup return search results only (raw payload = {id?, results}). The URLs a provider
# returns are treated as its citations; providers with no answer are still scored on those URLs.
ANSWER_PROVIDERS = ["perplexity", "linkup", "tavily"]
ANSWER_FIELDS = ["url", "title", "summary", "answer", "usage"]
TESLA_Q3_2026_MUSD = 28046  # "Total revenues" row of Tesla's Q3 2026 8-K exhibit 99.1 (Test C)

# PRIMARY = the original publisher of the fact. Everything else (news, blogs, forums, issue trackers of
# other projects, aggregators, video) is SECONDARY. Entries are host suffixes or host+path prefixes.
PRIMARY_ALWAYS = ["sec.gov", "nih.gov", "pypi.org", "npmjs.com", "docs.python.org", "peps.python.org"]


def _latest_pypi(pkg):
    import ssl
    try:  # python.org builds ship without a CA bundle; certifi comes in with requests/httpx
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    with urllib.request.urlopen(f"https://pypi.org/pypi/{pkg}/json", timeout=20, context=ctx) as r:
        return json.load(r)["info"]["version"]


def _money_musd(text):
    """Dollar amounts in text, in $ millions, in order of appearance."""
    out = []
    for m in re.finditer(r"\$\s?(\d[\d,]*(?:\.\d+)?)\s*(billion|bn|b|million|mn|m)\b", text, re.I):
        v = float(m.group(1).replace(",", ""))
        out.append(v * 1000 if m.group(2).lower().startswith("b") else v)
    return out


def check_tesla(ans, _):
    vals = _money_musd(ans)
    if not vals:
        return False, "no dollar figure"
    v = vals[0]  # the headline figure; later ones are often estimates or prior periods
    return abs(v - TESLA_Q3_2026_MUSD) / TESLA_Q3_2026_MUSD < 0.002, f"${v / 1000:.2f}B vs ${TESLA_Q3_2026_MUSD / 1000:.2f}B"


def check_pytorch(ans, latest):
    m = re.search(r"\b(\d+\.\d+(?:\.\d+)?)\b", ans)
    if not m:
        return False, "no version"
    v, want = m.group(1), latest
    return v in (want, want.removesuffix(".0")), f"{v} vs PyPI {want}"


def check_semaglutide(ans, _):
    a = ans.lower()
    neg = re.search(r"\b(no|not|does not|doesn't|did not)\b[^.]{0,40}\b(reduc|lower|benefit)", a)
    pos = re.search(r"\b(yes|reduc\w*|lower\w*)\b", a)
    return bool(pos and not neg), "expects yes (SELECT trial: ~20% fewer MACE)"


def check_utcnow(ans, _):
    a = ans.lower()
    return ("deprecat" in a and "3.12" in a and "not deprecated" not in a), "expects yes, since 3.12"


ANSWER_QUESTIONS = [  # (question, extra primary sources for this question, checker, label for expected primary)
    ("What was Tesla's revenue in Q3 2026?", ["tesla.com"], check_tesla, "sec.gov"),
    ("Latest version of pytorch", ["pytorch.org", "github.com/pytorch/"], check_pytorch, "pypi.org"),
    # peer-reviewed journals publish the trial itself; the regulator's approval notice is also first-hand
    ("Does semaglutide reduce cardiovascular events in obesity?",
     ["nejm.org", "thelancet.com", "jamanetwork.com", "jacc.org", "ahajournals.org", "fda.gov"],
     check_semaglutide, "pubmed/NEJM"),
    ("Is datetime.utcnow deprecated in Python?", [], check_utcnow, "docs.python.org"),
]


def label_url(url, extra):
    p = urlparse(url or "")
    host = (p.hostname or "").removeprefix("www.")
    hp = host + p.path
    for d in PRIMARY_ALWAYS + extra:
        if "/" in d:
            if hp.startswith(d):
                return "PRIMARY"
        elif host == d or host.endswith("." + d):
            return "PRIMARY"
    return "SECONDARY"


async def test_d(c):
    print("=" * 100 + f"\nTEST D: answer providers ({', '.join(ANSWER_PROVIDERS)}): correctness and citation quality\n"
          + "=" * 100)
    try:
        latest_torch = _latest_pypi("torch")
    except Exception as e:  # noqa: BLE001 - ground truth unavailable; the row is still printed
        latest_torch = None
        print(f"could not read PyPI for torch: {e}")
    print(f"ground truth: Tesla Q3 2026 revenue ${TESLA_Q3_2026_MUSD:,}M (8-K) | torch latest on PyPI: {latest_torch}")

    rows = []
    for q, extra, check, expected_primary in ANSWER_QUESTIONS:
        r, err, wall = await call(c, q, ANSWER_PROVIDERS, 5, fields=ANSWER_FIELDS)
        print(f"\n--- {q}   (expected primary: {expected_primary}; wall {wall} ms)")
        runs = {x.provider: x for x in (r.by_provider if r else [])}
        for prov in ANSWER_PROVIDERS:
            run = runs.get(prov)
            if err or not run or run.status != "succeeded":
                why = err or (run and (run.error or run.status)) or "no run"
                print(f"  {prov:11} FAILED: {str(why)[:150]}")
                rows.append({"question": q, "provider": prov, "error": str(why)})
                continue
            ans = (run.answer or "").strip()
            cites = [{"url": x.url, "label": label_url(x.url, extra)} for x in run.results]
            n_pri = sum(1 for x in cites if x["label"] == "PRIMARY")
            pct = 100 * n_pri / len(cites) if cites else None
            if not ans:
                correct, why = None, "no answer returned"
            elif check is check_pytorch and not latest_torch:
                correct, why = None, "PyPI unavailable"
            else:
                correct, why = check(ans, latest_torch)
            print(f"  {prov:11} answer: {ans[:160] or '(none)'}")
            print(f"  {'':11} correct: {'n/a' if correct is None else 'YES' if correct else 'NO'} ({why}) | "
                  f"primary {n_pri}/{len(cites)}")
            for x in cites:
                print(f"  {'':11}   {x['label']:9} {x['url'][:110]}")
            rows.append({"question": q, "provider": prov, "answer": ans or None, "correct": correct,
                         "check": why, "citations": cites, "primary_pct": pct,
                         "expected_primary_cited": n_pri > 0, "usage": run.usage})

    # Table: question | provider | answer | correct? | % of citations PRIMARY
    print(f"\n{'question':44} | {'provider':10} | {'answer':52} | {'correct?':8} | % primary")
    print("-" * 134)
    for x in rows:
        if "error" in x:
            print(f"{x['question'][:44]:44} | {x['provider']:10} | {'ERROR: ' + x['error'][:45]:52} | {'':8} |")
            continue
        ans = (x["answer"] or "(no answer)").replace("\n", " ")[:52]
        ok = "n/a" if x["correct"] is None else "yes" if x["correct"] else "NO"
        pct = "n/a" if x["primary_pct"] is None else f"{x['primary_pct']:.0f}%"
        print(f"{x['question'][:44]:44} | {x['provider']:10} | {ans:52} | {ok:8} | {pct:>5} ({sum(1 for y in x['citations'] if y['label'] == 'PRIMARY')}/{len(x['citations'])})")
    return rows


async def main():
    c = AsyncTelem()
    results = {}
    if len(sys.argv) > 1 and sys.argv[1] == "D":  # rerun only Test D: python validate_router.py D
        results["D"] = await test_d(c)
        print(f"\n[budget] reported spend ${spent:.4f}; provider runs with no cost reported: {unreported_calls}")
        with open("validate_router_results_D.json", "w") as f:
            json.dump({"results": results, "log": log, "spent": spent}, f, default=str, indent=1)
        return
    if len(sys.argv) > 1 and sys.argv[1] == "C":  # rerun only Test C: python validate_router.py C [engine]
        results["C"] = await test_c(c, sys.argv[2] if len(sys.argv) > 2 else "brave")
        print(f"\n[budget] reported spend ${spent:.4f}; provider runs with no cost reported: {unreported_calls}")
        with open("validate_router_results_C.json", "w") as f:
            json.dump({"results": results, "log": log, "spent": spent}, f, default=str, indent=1)
        return
    try:
        results["A"] = await test_a(c)
        print(f"\n[budget] reported spend so far ${spent:.4f}; calls with no cost reported: {unreported_calls}")
        results["B"] = await test_b(c)
        print(f"\n[budget] reported spend so far ${spent:.4f}; calls with no cost reported: {unreported_calls}")
        pairs = results["B"]
        if pairs:
            a, b, _, _ = min(pairs, key=lambda p: (p[2], p[3]))
            web = b if a == "exa" else a  # prefer a non-exa engine of the least-overlapping pair
            print(f"\nlowest-overlap pair: {a}/{b} -> web engine for Test C: {web}")
        else:
            web = "brave"
        results["C"] = await test_c(c, web)
        results["D"] = await test_d(c)
    except BudgetExceeded as e:
        print(f"\n!!! BUDGET STOP: {e}")
    print(f"\n[budget] FINAL reported spend ${spent:.4f}; provider runs with no cost reported: {unreported_calls}")
    with open("validate_router_results.json", "w") as f:
        json.dump({"results": results, "log": log, "spent": spent}, f, default=str, indent=1)


asyncio.run(main())
