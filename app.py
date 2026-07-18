"""app.py — v2 Phase C. Thin FastAPI skin over the existing RAG pipeline.

/ask  reuses rag.answer() — no reimplementation, so the endpoint inherits the
      exact behavior the eval already measured (grounding, DOI citations,
      refusal-when-absent).
/health  checks DB connectivity + that the embedding model is loaded.

The embedding model loads ONCE at startup (lifespan), not per request.
"""
from contextlib import asynccontextmanager

import psycopg
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import rag
from settings import DB_DSN, TOP_K


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the SentenceTransformer once so the first /ask doesn't eat the load,
    # and concurrent first-requests don't race the lazy global.
    rag.get_model()
    yield


app = FastAPI(title="Springer_RAG", version="v2", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    k: int = Field(TOP_K, ge=1, le=20)
    hybrid: bool = False


class Citation(BaseModel):
    doi: str
    title: str
    section_title: str


class RetrievedChunk(BaseModel):
    doi: str
    section_title: str
    text: str
    score: float | None = None


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    retrieved: list[RetrievedChunk]
    retriever: str


@app.get("/health")
def health():
    db_ok = False
    try:
        conn = psycopg.connect(DB_DSN, connect_timeout=3)
        cur = conn.cursor()
        cur.execute("SELECT count(*) FROM chunks;")
        n_chunks = cur.fetchone()[0]
        conn.close()
        db_ok = True
    except Exception as e:
        n_chunks, db_err = None, str(e)

    model_ok = rag._model is not None

    status = "ok" if (db_ok and model_ok) else "degraded"
    body = {"status": status, "db": db_ok, "model_loaded": model_ok,
            "chunks": n_chunks}
    if status != "ok":
        # 503 so orchestrators (and you) can tell healthy from not at a glance.
        raise HTTPException(status_code=503, detail=body)
    return body


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    retriever = rag.retrieve_hybrid if req.hybrid else rag.retrieve
    try:
        answer_text, hits = rag.answer(req.question, k=req.k, retriever=retriever)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"pipeline error: {e}")

    # De-dupe citations by DOI, preserve retrieval order.
    seen, citations = set(), []
    for h in hits:
        if h["doi"] not in seen:
            seen.add(h["doi"])
            citations.append(Citation(doi=h["doi"], title=h["title"],
                                      section_title=h["section_title"]))

    retrieved = [
        RetrievedChunk(
            doi=h["doi"], section_title=h["section_title"], text=h["text"],
            score=h.get("cosine_sim") if not req.hybrid else h.get("rrf_score"),
        )
        for h in hits
    ]
    return AskResponse(
        question=req.question, answer=answer_text, citations=citations,
        retrieved=retrieved, retriever="hybrid" if req.hybrid else "vector",
    )
