"""Canon HTTP API and demo page.   Run:  .venv/bin/uvicorn api:app --port 8000   then open http://localhost:8000

GET  /                                   -> demo page (web/index.html)
POST /answer  {"question": "...", "source": null, "compare": false}  -> Canon answer (+ "web_only" if compare)
GET  /answer/stream?q=...&compare=true   -> Server-Sent Events: one "step" event per pipeline step,
                                            "web_only" (if compare), then "result" (or "failure")
GET  /facts                              -> every fact in memory
GET  /stats                              -> facts stored, memory hits, searches saved, tokens saved
GET  /eval                               -> eval summary (eval_results.json)
POST /memory/reset                       -> clear the fact memory (for clean demos)
"""
import asyncio
import json
import os
from contextlib import asynccontextmanager
from typing import Literal, Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from canon import canon, classifier, memory

HERE = os.path.dirname(os.path.abspath(__file__))


@asynccontextmanager
async def lifespan(_):
    classifier._load()  # warm up the embedding model once at startup
    yield


app = FastAPI(title="Canon", description="Telem finds the sources. Canon keeps the facts.", lifespan=lifespan)


class Ask(BaseModel):
    question: str
    source: Optional[Literal["code", "finance", "medical", "local", "general"]] = None
    compare: bool = False


def _clean(r):
    r = dict(r)
    r.pop("events", None)
    return r


def _friendly(e):
    """A short message for the page; never a stack trace."""
    s = str(e)
    if "api_key" in s.lower() or "authentication" in s.lower() or "401" in s:
        return "An API key is missing or invalid. Check TELEM_API_KEY and ANTHROPIC_API_KEY in .env."
    if "timeout" in s.lower() or "timed out" in s.lower():
        return "A source took too long to respond. Try again."
    return f"Something went wrong while answering ({type(e).__name__}). Try again or rephrase the question."


# Canon reports progress as short messages; the page wants typed steps.
STEP_KINDS = [
    ("Checking memory", "memory_check"), ("Found in memory", "memory_hit"),
    ("Classif", "route"), ("Extracting query", "route"), ("Identifying package", "route"),
    ("Searching", "telem_search"), ("Cross-checking", "telem_search"), ("Engines disagree", "telem_search"),
    ("Fetching", "telem_fetch"),
    ("Extracting", "extract"), ("Reading release notes", "extract"),
    ("Verifying", "verify"), ("Quote not found", "verify"),
    ("Saved", "saved"), ("Not saved", "saved"), ("Hot fact", "saved"),
]


def step_kind(msg):
    return next((k for prefix, k in STEP_KINDS if msg.startswith(prefix)), "info")


def step_label(msg, kind):
    label = msg.rstrip("…").rstrip(".")
    if kind in ("telem_search", "telem_fetch") and "Telem" not in label:
        label += " via Telem"
    return label


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "web", "index.html"))


@app.post("/answer")
async def answer(body: Ask):
    q = body.question.strip()
    try:
        if body.compare:
            r, w = await asyncio.gather(canon.answer(q, source=body.source), canon.web_only(q))
            return {**_clean(r), "web_only": w}
        return _clean(await canon.answer(q, source=body.source))
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": _friendly(e)}, status_code=502)


@app.get("/answer/stream")
async def answer_stream(q: str, compare: bool = False):
    queue: asyncio.Queue = asyncio.Queue()

    def on_step(msg):
        if msg == "Done":
            return
        kind = step_kind(msg)
        queue.put_nowait(("step", {"kind": kind, "label": step_label(msg, kind)}))

    async def run_canon():
        try:
            r = await canon.answer(q.strip(), on_step=on_step)
            queue.put_nowait(("result", _clean(r)))
        except Exception as e:  # noqa: BLE001
            queue.put_nowait(("failure", {"message": _friendly(e)}))

    async def run_web():
        try:
            queue.put_nowait(("web_only", await canon.web_only(q.strip())))
        except Exception as e:  # noqa: BLE001
            queue.put_nowait(("web_only", {"error": _friendly(e)}))

    async def events():
        tasks = [asyncio.create_task(run_canon())] + ([asyncio.create_task(run_web())] if compare else [])
        pending = {"result"} | ({"web_only"} if compare else set())
        try:
            while pending:
                name, data = await queue.get()
                if name == "failure":
                    pending.discard("result")
                pending.discard(name)
                yield f"event: {name}\ndata: {json.dumps(data, default=str)}\n\n"
            yield "event: done\ndata: {}\n\n"
        finally:
            for t in tasks:
                t.cancel()

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/facts")
def facts():
    return memory.all_facts()


@app.get("/stats")
def stats():
    return memory.stats()


@app.get("/eval")
def eval_summary():
    path = os.path.join(HERE, "eval_results.json")
    if not os.path.exists(path):
        return JSONResponse({"error": "no eval results yet: run eval.py"}, status_code=404)
    return json.load(open(path))["summary"]


@app.post("/memory/reset")
def memory_reset():
    return {"cleared": memory.reset()}
