"""Query classifier: embeddings first, Claude Haiku when unsure, keyword rules as a last resort.

    classify(q) -> {label, confidence, margin, method: embedding|haiku|rules, latency_ms, scores}

1. Embed q with all-MiniLM-L6-v2, take cosine similarity to each category's centroid
   (8 examples per category, embedded once at startup).
2. If top < MIN_TOP (0.25) or margin (top - second) < MIN_MARGIN (0.10) -> ask claude-haiku-5-5 through a
   strict tool whose `label` is an enum of the 5 categories, so the output is constrained.
3. If there's no ANTHROPIC_API_KEY (or the call fails) -> keyword rules.

CLI:  .venv/bin/python classifier.py            # runs the 10-query test table
      .venv/bin/python classifier.py "query"    # classify one query
"""
from __future__ import annotations

import os
import re
import sys
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
from dotenv import load_dotenv

load_dotenv()

LABELS = ["code", "finance", "medical", "local", "general"]
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HAIKU_MODEL = "claude-haiku-5-5"
MIN_TOP = 0.25
MIN_MARGIN = 0.10

EXAMPLES = {
    # what coding agents ask: versions, breaking changes, errors, API usage, deprecations
    "code": [
        "latest stable version of pytorch",
        "breaking changes in pydantic v2 migration from v1",
        "TypeError: 'NoneType' object is not subscriptable in pandas groupby",
        "how to use the OpenAI responses API streaming in python",
        "is datetime.utcnow deprecated in python 3.12",
        "react 19 useEffect cleanup behavior change",
        "npm ERR! ERESOLVE unable to resolve dependency tree",
        "what replaced the deprecated next.js getServerSideProps in app router",
    ],
    "finance": [
        "What was Tesla's revenue in Q3 2026?",
        "Apple net income fiscal year 2025 10-K",
        "Microsoft operating margin last quarter",
        "Nvidia data center revenue Q2 FY2027",
        "Amazon free cash flow 2025 annual report",
        "how much debt does Ford have on its balance sheet",
        "Meta earnings per share Q1 2026",
        "Alphabet capital expenditures guidance 2026",
    ],
    "medical": [
        "Does semaglutide reduce cardiovascular events in obesity?",
        "side effects of metformin in elderly patients",
        "efficacy of statins for primary prevention of heart disease",
        "first-line treatment for community acquired pneumonia in adults",
        "clinical trial results for lecanemab in early Alzheimer's",
        "is intermittent fasting effective for type 2 diabetes",
        "risk of myocarditis after mRNA vaccination",
        "SGLT2 inhibitors outcomes in heart failure with preserved ejection fraction",
    ],
    "local": [
        "best ramen in Sunnyvale",
        "coffee shops open late near downtown Palo Alto",
        "dentist accepting new patients in Cary NC",
        "highest rated pizza in Brooklyn",
        "vegan restaurants near me in Austin",
        "hardware store open on Sunday in San Jose",
        "kid friendly parks in Seattle",
        "good sushi places in Mountain View with parking",
    ],
    "general": [
        "When was the ISS launched?",
        "who wrote the novel Middlemarch",
        "how does a heat pump work",
        "what is the population of Canada",
        "history of the Byzantine empire",
        "why is the sky blue",
        "tallest mountain in Africa",
        "what causes the northern lights",
    ],
}

_model = None
_centroids = None


def _load():
    """Load the model and embed the examples once (lazy, so importing is cheap)."""
    global _model, _centroids
    if _model is None:
        from sentence_transformers import SentenceTransformer
        # CPU: the MPS backend crashes in some macOS sandboxes, and MiniLM is fast on CPU.
        _model = SentenceTransformer(EMBED_MODEL, device="cpu")
        cents = []
        for label in LABELS:
            e = _model.encode(EXAMPLES[label], normalize_embeddings=True)
            c = e.mean(axis=0)
            cents.append(c / np.linalg.norm(c))
        _centroids = np.vstack(cents)
    return _model, _centroids


def embed_scores(q):
    model, cents = _load()
    v = model.encode([q], normalize_embeddings=True)[0]
    return dict(zip(LABELS, (cents @ v).tolist()))


# ---------------------------------------------------------------- Haiku fallback
LABEL_TOOL = {
    "name": "set_label",
    "description": "Record the category of the search query.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {"label": {"type": "string", "enum": LABELS}},
        "required": ["label"],
        "additionalProperties": False,
    },
}
HAIKU_SYSTEM = (
    "Classify a search query into exactly one category and record it with the set_label tool.\n"
    "code: software versions, breaking changes, error messages, API usage, deprecations.\n"
    "finance: a company's financial figures, filings or earnings.\n"
    "medical: clinical, drug, disease or biomedical research questions.\n"
    "local: places or businesses in a specific location.\n"
    "general: anything else."
)


def haiku_label(q):
    """Returns a label, or None if there's no key or the call fails."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    import anthropic

    client = anthropic.Anthropic(max_retries=1, timeout=20)
    kwargs = dict(model=HAIKU_MODEL, max_tokens=256, system=HAIKU_SYSTEM, tools=[LABEL_TOOL],
                  messages=[{"role": "user", "content": q}])
    try:
        try:
            r = client.messages.create(tool_choice={"type": "tool", "name": "set_label"}, **kwargs)
        except anthropic.BadRequestError:
            # some models reject forced tool choice; auto + strict schema still constrains the label
            r = client.messages.create(tool_choice={"type": "auto"}, **kwargs)
    except (anthropic.APIStatusError, anthropic.APIConnectionError):
        return None
    for block in r.content:
        if block.type == "tool_use" and block.name == "set_label":
            label = block.input.get("label")
            return label if label in LABELS else None
    return None


# ---------------------------------------------------------------- keyword rules
RULES = [
    ("code", r"\b(error|exception|traceback|deprecat\w*|breaking change|version|api|sdk|npm|pip|python|"
             r"javascript|typescript|react|node|rust|java|library|package|module|import|install|migrat\w*)\b"),
    ("finance", r"\b(revenue|earnings|eps|10-k|10-q|net income|profit|margin|cash flow|guidance|"
                r"q[1-4]\s?(fy)?\d{2,4}|fiscal|stock|shares|dividend|balance sheet)\b"),
    ("medical", r"\b(drug|trial|patients?|disease|treatment|therapy|symptoms?|side effects?|dose|"
                r"vaccin\w*|cancer|diabetes|cardio\w*|clinical|efficacy)\b"),
    ("local", r"\b(near me|nearby|best .+ in|restaurants?|cafe|coffee shop|open (late|now|on)|"
              r"dentist|clinics?|urgent care|store|shop|parks?)\b"),
]


def rules_label(q):
    ql = q.lower()
    hits = [(label, len(re.findall(rx, ql))) for label, rx in RULES]
    best = max(hits, key=lambda h: h[1])
    return best[0] if best[1] > 0 else "general"


# ---------------------------------------------------------------- entry point
def classify(q):
    t0 = time.perf_counter()
    scores = embed_scores(q)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    (top_label, top), (_, second) = ranked[0], ranked[1]
    margin = top - second
    out = {"label": top_label, "confidence": round(top, 3), "margin": round(margin, 3),
           "method": "embedding", "embedding_label": top_label,
           "scores": {k: round(v, 3) for k, v in scores.items()}}
    if top < MIN_TOP or margin < MIN_MARGIN:
        label = haiku_label(q)
        if label:
            out.update(label=label, method="haiku")
        else:
            out.update(label=rules_label(q), method="rules")
    out["latency_ms"] = int((time.perf_counter() - t0) * 1000)
    return out


# ---------------------------------------------------------------- test
TESTS = [  # (query, expected label, note)
    ("pydantic v2 ValidationError field required after upgrading from v1", "code", ""),
    ("what's new and deprecated in Django 5.2", "code", ""),
    ("Netflix operating income Q2 2026", "finance", ""),
    ("how much revenue did Costco make last fiscal year", "finance", ""),
    ("does aspirin prevent colorectal cancer", "medical", ""),
    ("best treatment for migraine with aura", "medical", ""),
    ("thai food open late in Cupertino", "local", ""),
    ("urgent care clinics in Raleigh", "local", "ambiguous: local vs medical"),
    ("what is the speed of light", "general", ""),
    ("Apple stock price impact of the new M5 chip", "finance", "ambiguous: finance vs general/tech"),
]


def run_tests():
    t0 = time.perf_counter()
    _load()
    print(f"model + 40 examples loaded in {(time.perf_counter() - t0) * 1000:.0f} ms (one-time)\n")
    print(f"{'query':62} | {'expected':8} | {'label':8} | {'method':9} | {'conf':>5} | {'margin':>6} | {'ms':>5} | ok")
    print("-" * 125)
    correct = 0
    for q, expected, note in TESTS:
        r = classify(q)
        ok = r["label"] == expected
        correct += ok
        parts = ([note] if note else []) + ([f"embedding said {r['embedding_label']}"] if r["method"] != "embedding" else [])
        tag = f"  ({'; '.join(parts)})" if parts else ""
        print(f"{q[:62]:62} | {expected:8} | {r['label']:8} | {r['method']:9} | {r['confidence']:5.2f} | "
              f"{r['margin']:6.3f} | {r['latency_ms']:5} | {'✓' if ok else '✗'}{tag}")
    print(f"\n{correct}/{len(TESTS)} correct")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        import json
        print(json.dumps(classify(" ".join(sys.argv[1:])), indent=1))
    else:
        run_tests()
