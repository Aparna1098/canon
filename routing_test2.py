# routing_test2.py: compare Telem auto_routing modes against the fixed fan-out baseline.
# Modes come from the SDK docstring (client.py: "accuracy", "latency", "search_cost");
# "cost" is included as a probe so the server's rejection message shows valid values.
import asyncio

from dotenv import load_dotenv
from telem import AsyncTelem, TelemError

load_dotenv()

QUERIES = [
    "When was the ISS launched?",
    "latest stable version of pytorch",
    "Anthropic latest funding round amount",
    "best ramen in Sunnyvale",
    "python asyncio gather vs TaskGroup",
]
# (label, auto_routing, max_routing_providers). None = baseline, no routing.
MODES = [
    ("none (all)", None, None),
    ("accuracy", "accuracy", None),
    ("accuracy/max3", "accuracy", 3),
    ("latency", "latency", None),
    ("search_cost", "search_cost", None),
    ("cost", "cost", None),
]
TIER = "default"
NUM_RESULTS = 3
# `usage` is not in the default tier's field set; request it explicitly.
FIELDS = ["url", "title", "summary", "excerpt", "publish_date", "rank", "usage"]


def usage_total(run):
    """Dollar total if the provider reports one, else None (most providers report nothing;
    parallel reports SKU counts as a list, not dollars)."""
    u = run.usage
    t = u.get("total") if isinstance(u, dict) else None
    return float(t) if isinstance(t, (int, float)) else None


def routed_providers(resp):
    """Providers Telem's router chose (from the raw `routing` block), if exposed."""
    raw = resp.raw if isinstance(resp.raw, dict) else {}
    routing = raw.get("routing") or []
    if routing and isinstance(routing, list):
        return routing[0].get("providers")
    return None


async def run_query(c, mode, max_prov, q):
    kwargs = {"tier": TIER, "num_results": NUM_RESULTS, "fields": FIELDS}
    if mode:
        kwargs["auto_routing"] = mode
    if max_prov:
        kwargs["max_routing_providers"] = max_prov
    try:
        r = await c.search(q, **kwargs)
    except TelemError as e:
        return {"error": f"{type(e).__name__}: {e}"}
    runs = r.by_provider or []
    return {
        "ran": sorted({run.provider for run in runs}),
        "ok": sorted({run.provider for run in runs if run.results}),
        "cost": sum(usage_total(run) or 0.0 for run in runs),
        "reporting": sorted(run.provider for run in runs if usage_total(run) is not None),
        "per_provider": {run.provider: usage_total(run) for run in runs},
        "routing": routed_providers(r),
        "warnings": [w.get("code") for run in runs for w in (run.warnings or []) if isinstance(w, dict)],
    }


async def main():
    c = AsyncTelem()
    summary = {}
    provider_cost = {}  # baseline cost per provider, summed over queries

    print(f"{'mode':13} | {'#':>2} | {'cost':>7} | {'providers that ran (router order if routed)':58} | query")
    print("-" * 135)
    for label, mode, max_prov in MODES:
        # Probe with the first query; skip the rest if the mode is rejected.
        first = await run_query(c, mode, max_prov, QUERIES[0])
        if "error" in first:
            print(f"{label:13} | REJECTED: {first['error'][:110]}")
            summary[label] = {"error": first["error"]}
            continue
        rows = [first] + list(await asyncio.gather(*(run_query(c, mode, max_prov, q) for q in QUERIES[1:])))
        for q, row in zip(QUERIES, rows):
            if "error" in row:
                print(f"{label:13} | ERR {row['error'][:100]} | {q}")
                continue
            shown = row["routing"] or row["ran"]
            print(f"{label:13} | {len(row['ran']):>2} | {row['cost']:7.4f} | {','.join(shown):58} | {q}")
            if mode is None:
                for p, v in row["per_provider"].items():
                    if v is not None:
                        provider_cost[p] = provider_cost.get(p, 0.0) + v
        good = [r for r in rows if "error" not in r]
        summary[label] = {
            "queries": len(good),
            "avg_providers": sum(len(r["ran"]) for r in good) / max(len(good), 1),
            "total": sum(r["cost"] for r in good),
            "routing_exposed": any(r["routing"] for r in good),
            "reporting": sorted({p for r in good for p in r["reporting"]}),
            "providers": sorted({p for r in good for p in r["ran"]}),
        }

    print("\n=== summary (tier=default, num_results=3) ===")
    base = summary.get("none (all)", {}).get("total")
    print(f"{'mode':13} | {'queries':>7} | {'avg #prov':>9} | {'total cost':>10} | {'per query':>9} | {'vs base':>7} | routing block | cost reported by")
    print("-" * 115)
    grand = 0.0
    for label, s in summary.items():
        if "error" in s:
            print(f"{label:13} | rejected: {s['error'][:60]}")
            continue
        per_q = s["total"] / max(s["queries"], 1)
        rel = f"{s['total'] / base:.0%}" if base else "-"
        grand += s["total"]
        cov = f"{len(s['reporting'])}/{len(s['providers'])} ({','.join(s['reporting']) or '-'})"
        print(f"{label:13} | {s['queries']:>7} | {s['avg_providers']:>9.1f} | {s['total']:>10.4f} | {per_q:>9.4f} | {rel:>7} | "
              f"{'yes' if s['routing_exposed'] else 'no':13} | {cov}")
    print(f"\nreported spend this run (sum of usage.total): ${grand:.4f}")
    print("NOTE: usage.total covers only providers that report a dollar total; it is a lower bound, not Telem's bill.")

    if provider_cost:
        print("\n=== baseline reported cost per provider (5 queries) ===")
        for p, v in sorted(provider_cost.items(), key=lambda kv: -kv[1]):
            print(f"{p:24} ${v:.4f}")


asyncio.run(main())
