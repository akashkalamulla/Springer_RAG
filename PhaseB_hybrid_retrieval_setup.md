# Task: v2 Phase B — Hybrid BM25 + vector retrieval (RRF), measured against baseline

**Scope: `rag.py` + a one-time DB migration + one `eval.py` flag.** Add
Postgres full-text (BM25-style) retrieval alongside the existing vector
retrieval, fuse the two rankings with Reciprocal Rank Fusion (RRF), and expose
it behind a flag so the eval can A/B vector-only vs hybrid on the *same* golden
sets. Do NOT re-chunk, do NOT re-embed, do NOT grow the corpus (that's Phase D),
do NOT touch the scraper.

**Why hybrid and not reranking:** the known weakness is exact terms/numbers that
MiniLM misses semantically (same gap abstract-indexing addressed). BM25 catches
lexical matches vector search drops; RRF fuses without tuning. It stays entirely
in Postgres — no new model or service — so the Phase C serving path stays light.
Cross-encoder reranking is deferred to v2.5 (post-Phase-D): on a 30-article
corpus, top-5 already contains the answer, so reranking mostly reorders correct
results and its lift wouldn't register against this baseline.

**Definition of done:**
- `chunks` has a `tsv` full-text column + GIN index, added by migration with
  **zero** change to row count or the 384-dim embeddings.
- `rag.py` has `retrieve_bm25()` and `retrieve_hybrid()`; `retrieve_hybrid()`
  returns the **same row shape** as `retrieve()` (same dict keys), so `eval.py`
  and `answer()` need no downstream changes.
- `eval.py` accepts `--hybrid` and runs every existing metric through
  `retrieve_hybrid()` instead of `retrieve()`.
- A delta table (`v2_phaseB_delta.md`) compares vector-only vs hybrid across
  EASY / HARD / faithfulness / refusal.

---

## Preconditions — check ALL of these first. If any fail, STOP and report back.

1. **`v2_baseline.md` exists** with vector-only numbers. This is the thing Phase
   B is measured against; without it there is no before/after. If missing, STOP
   — run `python eval.py` on the current (vector-only) pipeline first and save it.
2. **Docker is running** and Postgres holds the full chunk set:
   ```
   docker compose exec postgres psql -U raguser -d ragdb -c "SELECT count(*) FROM chunks;"
   ```
   Record this number — the migration must not change it.
3. **`rag.py` has the current `retrieve(query, k)`** returning dict rows with keys
   `chunk_id, doi, title, section_title, text, cited_refs, cosine_sim`. The
   hybrid function must preserve these keys.
4. **`eval.py` is the Phase-A version** (has `--retrieval-only`, EASY/HARD/REFUSAL
   blocks, `eval_retrieval(rows, max_k)`).
5. **venv active**, `psycopg[binary]`, `pgvector`, `sentence-transformers` present.
6. **`OPENAI_API_KEY` set** — needed only for the faithfulness/refusal halves of
   the final comparison, not the retrieval delta.

> **Baseline caveats carried in (read once).** Two known smudges ride into this
> phase from Phase A and Akash chose to proceed anyway: (a) hard-set Q8 is
> under-specified (fits two food-decision studies), which is why HARD hit@1 is
> 0.967 not 1.0 — if hybrid moves that row, part of the movement may be the fuzzy
> question, not the retriever; (b) the refusal set is all obvious off-topic, so
> REFUSAL=1.0 is untested on adversarial near-misses. Neither blocks Phase B.
> Both mean: read the *per-row* HARD deltas, don't just trust the aggregate.

---

## Steps

### 1. DB migration — add the full-text column (one-time, no re-embed)

Create `migrate_add_tsv.py`:

```python
"""One-time migration: add a full-text (tsvector) column + GIN index to chunks.
Idempotent. Does NOT touch embeddings or row count — pure additive column."""
import psycopg

DB_DSN = "host=localhost dbname=ragdb user=raguser password=ragpass port=5432"

conn = psycopg.connect(DB_DSN)
cur = conn.cursor()

before = cur.execute("SELECT count(*) FROM chunks;") or cur.fetchone()[0]

# Generated column: Postgres keeps tsv in sync with text automatically.
cur.execute("""
    ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('english', coalesce(text, ''))) STORED;
""")
cur.execute("CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);")
conn.commit()

cur.execute("SELECT count(*) FROM chunks;")
after = cur.fetchone()[0]
cur.execute("SELECT count(*) FROM chunks WHERE embedding IS NOT NULL;")
emb = cur.fetchone()[0]
conn.close()
print(f"chunks before={before} after={after} (must be equal), embeddings intact={emb}")
```

Run it:
```
python migrate_add_tsv.py
```
`before` and `after` MUST be equal, and `embeddings intact` MUST equal the row
count. If they differ, STOP — something rebuilt the table.

> Note: `build_index.py` does `TRUNCATE ... CASCADE` and recreates `chunks` on
> every run. The `tsv` column uses `ADD COLUMN IF NOT EXISTS` on a **generated**
> column, so it survives re-inserts — but if you re-run `build_index.py` after
> this, re-run this migration too (or fold the column into the `CREATE TABLE` in
> `store_chunks`). For Phase B, do NOT re-run `build_index.py`; the current index
> is the baseline's index.

### 2. Add BM25 + hybrid retrieval to `rag.py`

After the existing `retrieve()` function, insert:

```python
RRF_K = 60   # standard RRF constant; do not tune before seeing the honest default


def retrieve_bm25(query, k=TOP_K):
    """Lexical top-k via Postgres full-text (ts_rank over the GIN index). Same
    row shape as retrieve(), with bm25_rank in place of cosine_sim."""
    conn = psycopg.connect(DB_DSN)
    cur = conn.cursor()
    cur.execute("""
        SELECT c.chunk_id, c.doi, a.title, c.section_title, c.text, c.cited_refs,
               ts_rank(c.tsv, plainto_tsquery('english', %s)) AS bm25_rank
        FROM chunks c
        JOIN articles a ON a.doi = c.doi
        WHERE c.tsv @@ plainto_tsquery('english', %s)
        ORDER BY bm25_rank DESC
        LIMIT %s;
    """, (query, query, k))
    cols = [d.name for d in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    conn.close()
    return rows


def retrieve_hybrid(query, k=TOP_K, pool=20):
    """Reciprocal Rank Fusion of vector and BM25 rankings.

    Pull a wider pool from each retriever (default 20), score each chunk by
    sum(1 / (RRF_K + rank)) across the lists it appears in, return top-k. Row
    shape matches retrieve(): keeps chunk_id/doi/title/section_title/text/
    cited_refs, plus rrf_score. Downstream (answer(), eval) is unchanged.
    """
    vec = retrieve(query, pool)
    bm = retrieve_bm25(query, pool)

    scores, meta = {}, {}
    for lst in (vec, bm):
        for rank, row in enumerate(lst, start=1):
            cid = row["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
            meta.setdefault(cid, row)   # first occurrence carries the fields

    fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
    out = []
    for cid, s in fused:
        row = dict(meta[cid])
        row["rrf_score"] = round(s, 6)
        out.append(row)
    return out
```

### 3. Let `answer()` optionally use hybrid (one-line, non-breaking)

Change the signature of `answer()`:
```python
def answer(query, k=TOP_K):
    hits = retrieve(query, k)
```
to:
```python
def answer(query, k=TOP_K, retriever=retrieve):
    hits = retriever(query, k)
```
Default stays `retrieve`, so nothing existing changes. The eval passes
`retrieve_hybrid` when `--hybrid` is set.

### 4. Add the `--hybrid` flag to `eval.py`

At the top, import the hybrid retriever:
```python
from rag import retrieve, answer
```
→
```python
from rag import retrieve, retrieve_hybrid, answer
```

Change `eval_retrieval` to take a retriever:
```python
def eval_retrieval(rows, max_k=5):
    ...
        dois = [h["doi"] for h in retrieve(r["question"], max_k)]
```
→
```python
def eval_retrieval(rows, max_k=5, retriever=retrieve):
    ...
        dois = [h["doi"] for h in retriever(r["question"], max_k)]
```

Change `eval_faithfulness` and `eval_refusal` to thread a retriever into
`answer()`:
```python
def eval_faithfulness(rows):
    ...
            answer_text, hits = answer(r["question"])
```
→
```python
def eval_faithfulness(rows, retriever=retrieve):
    ...
            answer_text, hits = answer(r["question"], retriever=retriever)
```
(same one-line change inside `eval_refusal`: `answer(r["question"], retriever=retriever)`)

In `main()`, add the flag and select the retriever once:
```python
    ap.add_argument("--retrieval-only", action="store_true", ...)
```
→
```python
    ap.add_argument("--retrieval-only", action="store_true",
                    help="skip faithfulness + refusal (no LLM)")
    ap.add_argument("--hybrid", action="store_true",
                    help="use RRF hybrid (BM25 + vector) retrieval")
    args = ap.parse_args()
    R = retrieve_hybrid if args.hybrid else retrieve
    tag = "HYBRID" if args.hybrid else "VECTOR"
    print(f"[retriever: {tag}]")
```
Then pass `R` through: `eval_retrieval(easy, retriever=R)`,
`eval_retrieval(hard, retriever=R)`, `eval_faithfulness(easy, retriever=R)`,
`eval_refusal(refuse, retriever=R)`.

---

## Verify (all must pass)

- `python migrate_add_tsv.py` reports before == after and embeddings intact.
- Sanity-check BM25 alone works:
  ```
  python -c "from rag import retrieve_bm25; print([h['doi'] for h in retrieve_bm25('influenza vaccination children', 3)])"
  ```
  Returns DOIs, no error.
- `python eval.py --retrieval-only` (vector) and
  `python eval.py --retrieval-only --hybrid` both run and print their retriever
  tag. Compare the two EASY/HARD blocks.
- Full comparison, both ways:
  ```
  python eval.py            > run_vector.txt
  python eval.py --hybrid   > run_hybrid.txt
  ```

---

## The deliverable — `v2_phaseB_delta.md`

Build a table from the two runs. This is the point of the phase — a *measured*
before/after, not a claim.

```
| Metric              | Vector-only | Hybrid (RRF) | Δ    |
|---------------------|-------------|--------------|------|
| EASY  hit@1         | 1.0         | ?            |      |
| EASY  mrr           | 1.0         | ?            |      |
| HARD  hit@1         | 0.967       | ?            |      |
| HARD  hit@3         | 1.0         | ?            |      |
| HARD  mrr           | 0.983       | ?            |      |
| FAITHFULNESS (easy) | 1.0         | ?            |      |
| REFUSAL             | 1.0         | ?            |      |
```

Then write 2-3 honest sentences under it. Expected shape:
- EASY has no room to move (already 1.0) — flat is correct, not a failure.
- HARD hit@1 is where fusion should help — that's the number to watch.
- Check the **per-row HARD deltas**, not just the aggregate: confirm any HARD
  improvement isn't only the fuzzy Q8 row flipping. If Q8 is the only mover, the
  retriever gain is unproven; say so.

**If hybrid does NOT beat vector-only on HARD:** that is a real, reportable
result. Document it, keep `retrieve` as the default in `answer()`, and note in
the README that hybrid was tested and did not improve on this corpus. A measured
null result is a stronger portfolio story than an unmeasured "improvement."
Do NOT tune `RRF_K` or `pool` to manufacture a positive delta before recording
the honest default-parameter result first.

---

## If it breaks

- **`column "tsv" does not exist`** in retrieve_bm25 → migration didn't run or
  `build_index.py` rebuilt the table after it. Re-run `migrate_add_tsv.py`.
- **BM25 returns nothing** for a query with obvious keywords → the `WHERE tsv @@
  ...` filtered everything; check `plainto_tsquery('english', 'your terms')`
  returns lexemes, and that `text` actually contains them.
- **Hybrid row missing a key** downstream (`KeyError` in answer/eval) → the
  `meta.setdefault` row came from BM25 and lacks `cosine_sim`, or vice versa.
  That's fine for eval (only `doi` is read) and for `answer()` (reads `doi`,
  `section_title`, `text`); if some other code reads `cosine_sim`, guard it with
  `.get()`.
- **Hybrid slower than expected** → `pool=20` runs two queries + fusion; that's
  expected. Do not add caching in this phase.

---

## Do NOT

- Do NOT re-run `build_index.py`, re-chunk, or re-embed. The baseline's index is
  the index under test.
- Do NOT grow the corpus — that's Phase D, and doing it now would mean the delta
  was measured on two different corpora.
- Do NOT tune `RRF_K`, `pool`, or the tsquery config to chase a number before
  the honest default-parameter comparison is recorded.
- Do NOT change chunk size, embedding model, or vector dimension.
- Do NOT delete `run_vector.txt` / `run_hybrid.txt` — they are the evidence
  behind the delta table.

---

## The gate (read this)

Phase B has exactly one deliverable: a **measured** delta between vector-only and
hybrid retrieval, on the identical golden sets, recorded in `v2_phaseB_delta.md`
with the raw run files behind it. The code is scaffolding; the delta is the
milestone.

The honest outcome — improvement, flat, or regression — is whatever the numbers
say. The one failure mode this phase can't tolerate is an *unmeasured* claim:
shipping hybrid because it "should" be better without running both sides. Run
both, record both, keep whichever wins, and write down why.
