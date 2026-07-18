"""
rag.py — v1, M4 (retrieve). Semantic search over the chunks table.

Embeds the query with the SAME model used to build the index (must match, or the
vectors live in different spaces and cosine is meaningless), runs cosine top-k in
pgvector, and joins articles for the citation (title + DOI + cited_refs).

M5 (generate) gets added on top of this file next.

Usage:  python rag.py "your question here"
"""

import sys
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

from llm import generate
from settings import DB_DSN, EMBED_MODEL, TOP_K, RRF_K

PROMPT_SYSTEM = (
    "You are a research assistant. Answer the question using ONLY the "
    "provided context passages. If the context does not contain the answer, "
    "say so plainly — do not use outside knowledge. Cite the DOI of every "
    "passage you rely on, in square brackets like [10.1186/...]."
)

_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def retrieve(query, k=TOP_K):
    """Embed the query, cosine top-k against chunks, join articles for citation.

    The query vector is cast ::vector explicitly. In `embedding <=> %s` there is
    no column-type context for the parameter, so a bare list would be read as
    double precision[] and `vector <=> double precision[]` has no operator. The
    cast forces the vector interpretation. (On the store side the column type
    supplies that context, so no cast is needed there.)
    """
    qv = get_model().encode([query])[0].tolist()
    conn = psycopg.connect(DB_DSN)
    register_vector(conn)
    cur = conn.cursor()
    cur.execute("""
        SELECT c.chunk_id, c.doi, a.title, c.section_title, c.text, c.cited_refs,
               1 - (c.embedding <=> %s::vector) AS cosine_sim
        FROM chunks c
        JOIN articles a ON a.doi = c.doi
        ORDER BY c.embedding <=> %s::vector
        LIMIT %s;
    """, (qv, qv, k))
    cols = [d.name for d in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    conn.close()
    return rows


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


def build_prompt(query, hits):
    blocks = []
    for h in hits:
        blocks.append(f"[DOI: {h['doi']} | {h['section_title']}]\n{h['text']}")
    context = "\n\n".join(blocks)
    return f"CONTEXT:\n{context}\n\nQUESTION: {query}\n\nANSWER:"


def answer(query, k=TOP_K, retriever=retrieve):
    """Retrieve, then generate a grounded answer. Returns (text, hits)."""
    hits = retriever(query, k)
    text = generate(build_prompt(query, hits), system=PROMPT_SYSTEM)
    return text, hits


def main():
    if len(sys.argv) < 2:
        raise SystemExit('usage: python rag.py "your question here"')
    query = sys.argv[1]
    hits = retrieve(query)
    print(f"\nQUERY: {query}\n")
    for i, h in enumerate(hits, 1):
        print(f"[{i}] sim={h['cosine_sim']:.3f} | {h['section_title']} | doi={h['doi']}")
        print(f"    {h['title'][:75]}")
        if h["cited_refs"]:
            print(f"    cites: {h['cited_refs']}")
        print(f"    {h['text'][:200]}...\n")

    ans, _ = answer(query)
    print("\n=== ANSWER ===\n")
    print(ans)


if __name__ == "__main__":
    main()
