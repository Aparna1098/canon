# routing_test.py: does Telem pick providers per query, or fan out to a fixed set?
import asyncio

from dotenv import load_dotenv
from telem import AsyncTelem   # use the same import as your working test script

load_dotenv()  # pick up TELEM_API_KEY from .env


def g(o, k, d=None):
    return o.get(k, d) if isinstance(o, dict) else getattr(o, k, d)

QUERIES = [
    "When was the ISS launched?",                      # simple fact
    "latest stable version of pytorch",               # technical, recent
    "Anthropic latest funding round amount",          # news / finance
    "best ramen in Sunnyvale",                        # local
    "python asyncio gather vs TaskGroup",             # code
]
TIERS = ["minimalist", "default", "extended", "max"]

async def main():
    c = AsyncTelem()

    # 1. What does Telem say about each provider?
    provs = c.providers
    provs = await provs() if callable(provs) else provs
    print("=== provider config ===")
    for p in (provs or []):
        print(f"{g(p,'name'):10} default={g(p,'active_by_default')} tiers={g(p,'tiers')}")

    # 2. Which providers actually run, per tier and per query?
    print("\n=== who ran ===")
    for tier in TIERS:
        for q in QUERIES:
            r = await c.search(query=q, tier=tier)          # no providers_include
            ran = sorted({g(x, 'provider') for x in (g(r, 'by_provider') or [])}
                         | {g(x, 'provider') for x in (g(r, 'results') or [])})
            print(f"{tier:10} | {len(ran)} | {','.join(ran):55} | {q}")

    # 3. Is any cost data exposed? Dump one raw run.
    print("\n=== one by_provider entry (look for cost fields) ===")
    print(g(r, 'by_provider')[0] if g(r, 'by_provider') else r)

asyncio.run(main())
