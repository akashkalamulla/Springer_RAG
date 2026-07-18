# Task: v2 Phase C — FastAPI serving layer + config extraction

**Scope: one new `app.py`, one new `settings.py`, a `.env`, and mechanical
edits to point existing files at settings.** Wrap the *existing, already-measured*
pipeline in a thin HTTP API. Do NOT change retrieval logic, chunking, embedding,
the schema, the eval, or the scraper. Do NOT reimplement `answer()` — the API
calls it. No UI, no auth, no streaming (see **Non-goals**).

**Why now:** every retrieval metric is saturated (Phase B confirmed the corpus,
not the retriever, is the ceiling), so the next real portfolio value is turning
"scripts an evaluator runs" into "a service that round-trips a request." The API
is a thin skin over code that's already correct and measured.

**Definition of done:**
- `docker compose up` brings up Postgres **and** the API.
- `GET /health` returns 200 with DB + model status (503 if either is down).
- `POST /ask` returns a grounded answer with DOI citations, reusing `answer()`.
- Refusals (out-of-corpus questions) return a normal 200 with the refusal text —
  a refusal is a valid answer, not an error.
- No DB credential or model name is hardcoded outside `settings.py`.

---

## Preconditions — check ALL of these first. If any fail, STOP and report back.

1. **Phase B is done and committed.** `rag.py` has `retrieve`,
   `retrieve_hybrid`, and `answer(query, k, retriever=...)`. `eval.py` runs.
2. **Docker is running**; Postgres holds the chunk set (1422) with embeddings
   intact. The API talks to this same DB.
3. **`.env` already holds `OPENAI_API_KEY`** (from Phase A). Phase C adds DB +
   model settings to the same file.
4. **venv active.** Install the two new deps:
   ```
   pip install fastapi "uvicorn[standard]"
   ```
5. **Chunk-count reconciliation (do NOT skip).** Baseline says 1422 chunks; v1/M3
   said 1331. Before `/health` reports a live count, know why it changed. Quick
   check:
   ```
   git log --oneline -- build_index.py
   docker compose exec postgres psql -U raguser -d ragdb -c "SELECT section_title='Abstract' AS is_abstract, count(*) FROM chunks GROUP BY 1;"
   ```
   If the abstract chunks (~91) account for the 1331→1422 jump, that's the
   expected cause (abstract indexing added in v1). Note it; do not "fix" it.

---

## Steps

### 1. Create `settings.py` — one source of truth

```python
"""settings.py — central config. Everything that was hardcoded across
build_index.py / rag.py / migrate_add_tsv.py lives here, sourced from .env."""
import os
from dotenv import load_dotenv

load_dotenv()

DB_DSN = os.environ.get(
    "DB_DSN",
    "host=localhost dbname=ragdb user=raguser password=ragpass port=5432",
)
EMBED_MODEL = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")
EMBED_DIM = int(os.environ.get("EMBED_DIM", "384"))
GEN_MODEL = os.environ.get("GEN_MODEL", "gpt-4o-mini")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-4o")
TOP_K = int(os.environ.get("TOP_K", "5"))
RRF_K = int(os.environ.get("RRF_K", "60"))
```

Add to `.env` (keep the existing `OPENAI_API_KEY` line):
```
DB_DSN=host=localhost dbname=ragdb user=raguser password=ragpass port=5432
EMBED_MODEL=all-MiniLM-L6-v2
GEN_MODEL=gpt-4o-mini
JUDGE_MODEL=gpt-4o
```

> **Docker note:** when the API runs *inside* compose, `localhost` won't reach
> the Postgres container — the host is the service name `postgres`. Step 5
> handles this with a compose-level `DB_DSN` env override; the `.env` default
> above is for running `app.py` on the host during development.

### 2. Point existing files at settings (mechanical, no logic change)

In `rag.py`, `build_index.py`, and `migrate_add_tsv.py`, replace the hardcoded
constants with imports. Example for `rag.py`:

Find:
```python
DB_DSN = "host=localhost dbname=ragdb user=raguser password=ragpass port=5432"
EMBED_MODEL = "all-MiniLM-L6-v2"   # MUST match build_index.py
TOP_K = 5
```
Replace:
```python
from settings import DB_DSN, EMBED_MODEL, TOP_K, RRF_K
```
(and delete the now-duplicated `RRF_K = 60` line added in Phase B — it comes from
settings now).

Do the equivalent in `build_index.py` (`DB_DSN`, `EMBED_MODEL`, `EMBED_DIM`) and
`migrate_add_tsv.py` (`DB_DSN`). **Do not change any behavior** — the values are
identical; you're only removing duplication. Re-run one quick check after:
```
python -c "from rag import retrieve; print(len(retrieve('influenza vaccination', 3)))"
```
Prints `3` → imports resolved, pipeline still works.

> Leave `eval.py`'s `JUDGE_MODEL` and `llm.py`'s `MODEL` as-is if you prefer a
> smaller diff; they already read cleanly. Optional: point them at
> `settings.GEN_MODEL` / `settings.JUDGE_MODEL` for full centralization.

### 3. Create `app.py` — the API

```python
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
```

> **Refusal is a 200, not an error.** When the question is out-of-corpus,
> `answer()` returns the generator's refusal text and whatever chunks were
> nearest. `/ask` returns that as a normal 200 with the refusal in `answer` —
> the pipeline's honesty behavior is the product, not a failure to handle.

### 4. Add the API to `docker-compose.yml`

Add an `api` service alongside `postgres`. Create a `Dockerfile`:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Add to `docker-compose.yml` under `services:` (peer of `postgres`):
```yaml
  api:
    build: .
    container_name: rag_api
    depends_on:
      - postgres
    environment:
      # inside compose, reach Postgres by service name, not localhost
      DB_DSN: "host=postgres dbname=ragdb user=raguser password=ragpass port=5432"
    env_file:
      - .env            # brings in OPENAI_API_KEY and the rest
    ports:
      - "8000:8000"
```

> `requirements.txt` must exist for the Docker build. If it doesn't yet, generate
> it now (this is a Phase E polish item pulled forward because the build needs
> it): `pip freeze > requirements.txt`, then trim to the real deps
> (fastapi, uvicorn, psycopg[binary], pgvector, sentence-transformers, openai,
> python-dotenv, requests, beautifulsoup4, pymupdf4llm). Do NOT ship a 200-line
> pip freeze.

---

## Verify (all must pass)

**Local (host) first — fastest loop:**
```
uvicorn app:app --reload
```
Then in another shell:
```
curl -s http://localhost:8000/health
# -> {"status":"ok","db":true,"model_loaded":true,"chunks":1422}

curl -s -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
  -d "{\"question\": \"what reduces heat stress in hospital workers?\"}"
# -> 200 with answer, citations[], retrieved[], "retriever":"vector"
```

**Refusal check — must be a 200, not a 4xx/5xx:**
```
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"what was the efficacy of the mRNA-1273 booster?\"}"
# -> 200  (body's answer field says the info isn't in the context)
```

**Hybrid path reachable:**
```
curl -s -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
  -d "{\"question\": \"tobacco use among men\", \"hybrid\": true}" | \
  python -c "import sys,json; print(json.load(sys.stdin)['retriever'])"
# -> hybrid
```

**Full stack via compose:**
```
docker compose up -d --build
curl -s http://localhost:8000/health
```
`/health` returns ok with the chunk count from the containerized DB.

**Interactive docs** (free with FastAPI): open `http://localhost:8000/docs` —
this is the demoable artifact for interviews.

---

## If it breaks

- **`/health` 503, db false, "connection refused"** running in compose → the API
  used `localhost` instead of `postgres`. Confirm the compose `DB_DSN` env
  override is present and that `settings.DB_DSN` reads the env var.
- **`/health` model_loaded false** → lifespan didn't run or `rag.get_model()`
  failed; check startup logs. On first ever run the model downloads (~90MB) —
  allow a minute, it's cached after.
- **`/ask` 502 pipeline error** → the same error `python rag.py "..."` would
  throw; debug there, not in the API. The endpoint is a thin wrapper.
- **`KeyError` building citations/retrieved** → a hybrid row lacks `cosine_sim`
  or a vector row lacks `rrf_score`; the `.get(...)` guards handle this — if it
  still throws, a row is missing `doi`/`title`/`section_title`, which means
  `retrieve_hybrid`'s `meta` carried a malformed row.
- **Docker build slow / huge** → `sentence-transformers` + torch is a large
  image; expected. Do not optimize the image in this phase.

---

## Non-goals (explicitly out of scope for Phase C)

- **No UI.** A minimal frontend is optional Phase C.5. The API + `/docs` is the
  demoable slice.
- **No auth, no rate limiting, no streaming.** Real concerns, but scope creep
  here. Note them in the README roadmap; don't build them now.
- **No new retrieval work.** Hybrid is exposed via the `hybrid` flag exactly as
  Phase B left it. Adopting/tuning it is deferred to post-Phase-D.
- **No corpus growth.** That's Phase D.

---

## Do NOT

- Do NOT reimplement retrieval or generation in `app.py`. It calls `rag.answer()`.
  If you find yourself copying pipeline logic into the endpoint, stop.
- Do NOT return an error status for a refusal. Out-of-corpus → 200 with refusal
  text. Breaking this hides the pipeline's best behavior.
- Do NOT load the embedding model per-request. It loads once in lifespan.
- Do NOT hardcode credentials in `app.py`. Everything comes from `settings`.
- Do NOT ship a raw `pip freeze` as requirements.txt — trim to real deps.

---

## The gate (read this)

Phase C is done when a request **round-trips**, not when the code exists:
`docker compose up`, `/health` returns ok, `/ask` returns a grounded, cited
answer, and an out-of-corpus question returns a 200 refusal. The endpoint must
reuse `rag.answer()` unchanged — the whole point is that the API inherits the
behavior the eval already measured. If `app.py` grows its own retrieval or
generation logic, it's no longer the same measured pipeline, and the metrics
stop describing what the service actually does. Thin skin over measured code —
that's the milestone.
