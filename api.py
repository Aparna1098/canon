"""Canon HTTP API.   Run:  .venv/bin/uvicorn api:app --port 8000

POST /answer  {"question": "...", "source": null}   -> Canon answer (see canon.canon.answer)
GET  /facts                                        -> every fact in memory
GET  /stats                                        -> facts stored, memory hits, searches saved, tokens saved
"""
from contextlib import asynccontextmanager
from typing import Literal, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from canon import canon, classifier, memory


@asynccontextmanager
async def lifespan(_):
    classifier._load()  # warm up the embedding model once at startup
    yield


app = FastAPI(title="Canon", description="Telem finds the sources. Canon keeps the facts.", lifespan=lifespan)


class Ask(BaseModel):
    question: str
    source: Optional[Literal["code", "finance", "medical", "local", "general"]] = None


@app.post("/answer")
async def answer(body: Ask):
    r = await canon.answer(body.question.strip(), source=body.source)
    r.pop("events", None)
    return r


@app.get("/facts")
def facts():
    return [{k: v for k, v in f.items()} for f in memory.all_facts()]


@app.get("/stats")
def stats():
    return memory.stats()
