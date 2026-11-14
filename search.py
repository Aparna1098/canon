"""Run a Telem search across the default provider set and print by_provider."""

import sys

from dotenv import load_dotenv
from telem import Telem

MIN_PROVIDERS = 3


def main() -> int:
    load_dotenv()  # picks up TELEM_API_KEY from .env
    query = " ".join(sys.argv[1:]) or "when was the International Space Station launched"

    client = Telem()
    resp = client.search(query, num_results=3)

    print(f"query:  {query}")
    print(f"status: {resp.status}\n")
    print(f"{'provider':<12} {'status':<12} {'results':>7} {'latency_ms':>10}  top result")
    print("-" * 90)
    for run in resp.by_provider:
        top = run.results[0].title or run.results[0].url if run.results else (run.error or "")
        latency = run.latency_ms if run.latency_ms is not None else "-"
        print(f"{run.provider:<12} {run.status:<12} {len(run.results):>7} {latency:>10}  {str(top)[:45]}")

    ok = sorted({r.provider for r in resp.by_provider if r.results})
    print(f"\nproviders with results: {len(ok)} -> {', '.join(ok)}")
    if len(ok) < MIN_PROVIDERS:
        print(f"FAIL: expected {MIN_PROVIDERS}+ providers")
        return 1
    print(f"PASS: {len(ok)} providers returned results")
    return 0


if __name__ == "__main__":
    sys.exit(main())
