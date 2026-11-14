"""Dump Telem's provider catalog to providers.csv and print it grouped by apparent category.
Free: only calls GET /v1/preprocessors, no searches."""
import asyncio
import csv

from dotenv import load_dotenv
from telem import AsyncTelem

load_dotenv()

# Category is inferred from provider names; the catalog has no category/description for these.
RULES = [
    ("utility (not search)", lambda n: n in {"dummy", "url_list", "web_fetch"}),
    ("maps/local/reviews", lambda n: any(s in n for s in ("gmaps", "google_maps", "greviews", "tripadvisor", "airbnb", "trustpilot"))),
    ("academic/papers/science", lambda n: n in {"ncbi_pubmed", "ctgov_studies"}),
    ("patents/IP", lambda n: n in {"patsnap", "uspto_applications"}),
    ("finance/filings/companies", lambda n: n in {"sec_edgar", "tianyancha", "coresignal_company"}),
    ("people/professional", lambda n: n in {"coresignal_people", "apify_linkedin", "fetchlayer_linkedin_ads"}),
    ("shopping/apps/products", lambda n: n in {"fetchlayer_amazon", "fetchlayer_playstore", "fetchlayer_producthunt", "apify_g2"}),
    ("social: reddit/x/forums", lambda n: n in {"fetchlayer_reddit", "fetchlayer_x", "socialdatax_x", "xcom", "apify_quora", "apify_substack"}),
    ("social: other (western)", lambda n: n in {"apify_facebook", "fetchlayer_facebook_ads", "apify_pinterest", "apify_twitch",
                                                 "fetchlayer_tiktok", "socialdatax_tiktok", "socialdatax_instagram",
                                                 "socialdatax_youtube", "tikhub_telegram", "tikhub_lemon8"}),
    ("social: chinese platforms", lambda n: any(s in n for s in ("bilibili", "douyin", "kuaishou", "toutiao", "wechat", "weibo", "xhs", "zhihu"))),
    ("web scrape/extract", lambda n: n in {"firecrawl", "diffbot", "olostep"}),
    ("SERP (Google results)", lambda n: n in {"serpapi", "serper"}),
    ("general web search", lambda n: n in {"brave", "ceramic", "exa", "parallel", "seltz", "tavily", "you",
                                           "linkup", "perplexity", "tinyfish", "keenable"}),
]


def category(name):
    return next((cat for cat, rule in RULES if rule(name)), "other")


async def main():
    provs = await AsyncTelem().providers()
    rows = []
    for p in provs:
        d = p.model_dump()
        d["tiers"] = ",".join(d.get("tiers") or [])
        d["category"] = category(d["name"])
        rows.append(d)

    cols = ["name", "category", "type", "active_by_default", "auto_routed", "normalized", "tiers", "method", "url", "description"]
    cols += [k for k in rows[0] if k not in cols]  # keep any fields a newer SDK adds
    with open("providers.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    order = [cat for cat, _ in RULES] + ["other"]
    print(f"{len(rows)} providers -> providers.csv   (D=on by default, R=auto-routable, N=normalized)\n")
    for cat in order:
        group = sorted((r for r in rows if r["category"] == cat), key=lambda r: r["name"])
        if not group:
            continue
        print(f"## {cat} ({len(group)})")
        for r in group:
            flags = ("D" if r["active_by_default"] else "-") + ("R" if r["auto_routed"] else "-") + ("N" if r["normalized"] else "-")
            print(f"   {flags}  {r['name']}")
    n = sum(r["normalized"] for r in rows)
    ar = [r["name"] for r in rows if r["auto_routed"]]
    print(f"\nnormalized: {n}/{len(rows)}   auto-routable: {len(ar)} -> {', '.join(sorted(ar))}")


asyncio.run(main())
