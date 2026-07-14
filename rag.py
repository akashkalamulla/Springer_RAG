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

DB_DSN = "host=localhost dbname=ragdb user=raguser password=ragpass port=5432"
EMBED_MODEL = "all-MiniLM-L6-v2"   # MUST match build_index.py
TOP_K = 5

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


if __name__ == "__main__":
    main()
