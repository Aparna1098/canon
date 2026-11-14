"""Eval: web-only (tavily) vs Telem default (all 7) vs Canon first pass vs Canon second pass (reworded).

    .venv/bin/python eval.py            # fresh memory (facts.db is moved aside), writes eval_results.md + eval_results.json

Ground truth: code versions are read live from PyPI / npm over direct HTTPS (independent of Telem);
the others are fixed patterns in questions.jsonl (Tesla figures from the 8-K revenue row).
"""
import asyncio
import json
import os
import re
import ssl
import statistics
import time
import urllib.request

from canon import canon, memory

HERE = os.path.dirname(os.path.abspath(__file__))


def _ctx():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def registry_latest(spec):
    eco, pkg = spec.split(":", 1)
    url = (f"https://pypi.org/pypi/{pkg}/json" if eco == "pypi"
           else f"https://registry.npmjs.org/-/package/{pkg}/dist-tags")
    with urllib.request.urlopen(url, timeout=20, context=_ctx()) as r:
        d = json.load(r)
    return d["info"]["version"] if eco == "pypi" else d["latest"]


def expected_patterns(q):
    if "registry" in q:
        v = q["truth"]
        short = v[:-2] if v.endswith(".0") else None
        pats = [rf"(?<![\d.]){re.escape(v)}(?!\.?\d)"]  # "16.4.6." ends a sentence; "16.4.60" is different
        if short:
            pats.append(rf"(?<![\d.]){re.escape(short)}(?!\.?\d)")
        return pats
    return q["expect"]


def correct(answer, q):
    a = re.sub(r"[\u202f\u00a0\u2009]", " ", (answer or "").lower())
    if not a or a == "no answer found":
        return False
    if q["category"] == "medical" and re.search(r"\b(no|not|does not|doesn't)\b[^.]{0,30}\b(reduc|lower)", a):
        return False
    return any(re.search(p, a) for p in expected_patterns(q))


async def run_canon(q, text):
    t0 = time.perf_counter()
    try:
        r = await canon.answer(text)
    except Exception as e:  # noqa: BLE001
        return {"answer": f"ERROR {type(e).__name__}: {e}", "calls": 0, "tokens": 0,
                "latency_ms": int((time.perf_counter() - t0) * 1000), "from_memory": False, "verified": False}
    return {"answer": r["answer"], "calls": r["calls_used"], "tokens": r["tokens_returned"],
            "latency_ms": r["latency_ms"], "from_memory": r["from_memory"], "verified": r["verified"],
            "route": r["route"], "url": r["url"]}


async def run_web(q):
    try:
        w = await canon.web_only(q["question"])
        return {"answer": w["answer"], "calls": 1, "tokens": w["tokens_read"], "latency_ms": w["latency_ms"],
                "urls": w["urls"][:3]}
    except Exception as e:  # noqa: BLE001
        return {"answer": f"ERROR {e}", "calls": 1, "tokens": 0, "latency_ms": 0}


async def run_default(q):
    try:
        d = await canon.telem_default(q["question"])
        return {"answer": d["answer"], "calls": d["calls"], "tokens": d["tokens_read"], "latency_ms": d["latency_ms"]}
    except Exception as e:  # noqa: BLE001
        return {"answer": f"ERROR {e}", "calls": 7, "tokens": 0, "latency_ms": 0}


def summarize(rows, key, qs):
    xs = [r[key] for r in rows]
    return {"accuracy": sum(correct(x["answer"], q) for x, q in zip(xs, qs)) / len(qs),
            "calls": statistics.mean(x["calls"] for x in xs),
            "tokens": statistics.mean(x["tokens"] for x in xs),
            "latency": statistics.mean(x["latency_ms"] for x in xs) / 1000,
            "hit_rate": (sum(bool(x.get("from_memory")) for x in xs) / len(xs)) if "canon" in key else None}


async def main():
    qs = [json.loads(line) for line in open(os.path.join(HERE, "questions.jsonl"))]
    for q in qs:
        if "registry" in q:
            q["truth"] = registry_latest(q["registry"])
    if os.path.exists(memory.DB_PATH):  # start from an empty memory
        os.replace(memory.DB_PATH, memory.DB_PATH + ".bak")

    rows = []
    for q in qs:  # pass 1: web-only and Telem default run alongside Canon (they don't touch memory)
        c, w, d = await asyncio.gather(run_canon(q, q["question"]), run_web(q), run_default(q))
        rows.append({"q": q, "web": w, "default": d, "canon1": c})
        print(f"[1] {q['id']:2} {q['category']:8} canon={c['answer'][:60]!r} ({'ok' if correct(c['answer'], q) else 'WRONG'}) "
              f"| web={w['answer'][:50]!r} ({'ok' if correct(w['answer'], q) else 'WRONG'})", flush=True)
    for row in rows:  # pass 2: reworded, memory warm
        q = row["q"]
        row["canon2"] = await run_canon(q, q["reworded"])
        c = row["canon2"]
        print(f"[2] {q['id']:2} mem={c['from_memory']} canon={c['answer'][:60]!r} "
              f"({'ok' if correct(c['answer'], q) else 'WRONG'})", flush=True)

    report(rows)


def report(rows):
    modes = [("web", "Web-only (tavily)"), ("default", "Telem default (all 7)"),
             ("canon1", "Canon, first pass"), ("canon2", "Canon, second pass (reworded)")]
    S = {k: summarize(rows, k, [r["q"] for r in rows]) for k, _ in modes}
    md = ["# Canon eval", "",
          f"15 questions (6 code, 3 finance, 2 medical, 2 local, 2 general), then the same 15 reworded. "
          f"Run {time.strftime('%Y-%m-%d %H:%M %Z')}.", "",
          "| mode | accuracy | avg calls | avg tokens the agent must read | avg latency | memory hit rate |",
          "|---|---|---|---|---|---|"]
    for k, name in modes:
        s = S[k]
        md.append(f"| {name} | {s['accuracy']:.0%} ({round(s['accuracy'] * 15)}/15) | {s['calls']:.1f} | "
                  f"{s['tokens']:,.0f} | {s['latency']:.1f}s | {'—' if s['hit_rate'] is None else format(s['hit_rate'], '.0%')} |")
    md += ["", "Tokens: web-only and Telem default = the result text the agent reads; Canon = the answer, quote, "
           "URL and date it returns (0 calls on a memory hit). Telem default accuracy uses its `answer` field "
           "(tavily is the only answer provider in the default set).", "",
           "## Per question", "",
           "| # | category | question | truth | web-only (tavily) | Canon pass 1 | Canon pass 2 (reworded) |",
           "|---|---|---|---|---|---|---|"]
    for r in rows:
        q = r["q"]

        def cell(x):
            a = (x["answer"] or "(no answer)").replace("|", "/").replace("\n", " ")
            return f"{'✅' if correct(x['answer'], q) else '❌'} {a[:70]}" + (" *(memory)*" if x.get("from_memory") else "")
        truth = q.get("truth") or " / ".join(q["expect"]).replace("|", "/")
        md.append(f"| {q['id']} | {q['category']} | {q['question']} | {truth} | {cell(r['web'])} | "
                  f"{cell(r['canon1'])} | {cell(r['canon2'])} |")
    open(os.path.join(HERE, "eval_results.md"), "w").write("\n".join(md) + "\n")
    json.dump({"summary": S, "rows": rows}, open(os.path.join(HERE, "eval_results.json"), "w"), indent=1, default=str)
    print("\n".join(md[:10]))


if __name__ == "__main__":
    import sys
    if "--rescore" in sys.argv:  # rebuild eval_results.md from eval_results.json without new calls
        report(json.load(open(os.path.join(HERE, "eval_results.json")))["rows"])
    else:
        asyncio.run(main())
