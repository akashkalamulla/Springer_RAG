"""
build_index.py — v1, milestone M1 (chunk only).

Reads the scraper's JSON, chunks the `body` field, writes chunks.jsonl next to
the source. M2 (embed) and M3 (store) get added on top of this same file later —
each at a clean seam, no rewrite.

Two decisions are baked in here because they're the parts that are easy to get
wrong and are the actual learning:

1. SUBSECTION RECURSION. In this corpus 341 of 344 subsections carry paragraphs —
   i.e. most of the body lives in subsections, not top-level `paragraphs`. Walk
   only `section["paragraphs"]` and you drop ~90% of the text silently.

2. WINDOW-AWARE CHUNK SIZE. all-MiniLM-L6-v2 (the M2 embedding model) truncates
   input at 256 tokens. A 500-token chunk gets half-ignored in the vector. So we
   target ~200 tokens/chunk — under the 256 ceiling with headroom.

Usage:  python build_index.py path/to/journal.json
"""

import json
import re
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer
import psycopg
from pgvector.psycopg import register_vector

TARGET_TOKENS = 200            # stay under MiniLM's 256-token window
MODEL_WINDOW = 256             # all-MiniLM-L6-v2 hard input limit
EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM = 384                # frozen: this is the pgvector column width in M3
DB_DSN = "host=localhost dbname=ragdb user=raguser password=ragpass port=5432"              # frozen: this is the pgvector column width in M3


def approx_tokens(text: str) -> int:
    # rough words->tokens; good enough to size chunks, not for billing
    return int(len(text.split()) * 1.3)


def sentence_split(text: str):
    return re.split(r"(?<=[.!?])\s+", text)


def iter_units(article: dict):
    """Yield (section_title, text, cited_refs) units, each already <= the window.

    Walks top-level paragraphs AND subsection paragraphs. Any paragraph that on
    its own exceeds TARGET_TOKENS is sentence-split so no single unit blows the
    embedding window.
    """
    def emit(paras, sect):
        for p in paras:
            text = (p.get("text") or "").strip()
            if not text:
                continue
            refs = p.get("cited_refs", []) or []
            if approx_tokens(text) <= TARGET_TOKENS:
                yield sect, text, refs
            else:
                for sent in sentence_split(text):
                    if sent.strip():
                        yield sect, sent.strip(), refs

    for section in article.get("body", []):
        title = section.get("title", "") or ""
        yield from emit(section.get("paragraphs", []), title)
        for sub in section.get("subsections", []):
            heading = sub.get("heading", "")
            sect = f"{title} — {heading}" if heading else title
            yield from emit(sub.get("paragraphs", []), sect)


def chunk_article(article: dict):
    """Pack units into ~TARGET_TOKENS chunks. Flush at section boundaries so
    section_title stays clean, and flush when the next unit would overflow."""
    doi = article.get("doi", "") or ""
    chunks, buf, refs, cur_title = [], [], [], None

    def flush():
        nonlocal buf, refs, cur_title
        if buf:
            chunks.append({
                "doi": doi,
                "section_title": cur_title or "",
                "text": " ".join(buf).strip(),
                "cited_refs": list(dict.fromkeys(refs)),   # dedupe, keep order
            })
        buf, refs, cur_title = [], [], None

    for title, text, rf in iter_units(article):
        if cur_title is not None and title != cur_title:
            flush()
        if buf and approx_tokens(" ".join(buf + [text])) > TARGET_TOKENS:
            flush()
        cur_title = title
        buf.append(text)
        refs += rf
    flush()
    return chunks

def embed_chunks(all_chunks):
    """M2: embed each chunk's text with MiniLM. Returns an (N, 384) array,
    row-aligned to all_chunks. Vectors are unit-normalized by the model, so
    cosine == dot product downstream."""
    import time
    texts = [c["text"] for c in all_chunks]
    t = time.time()
    model = SentenceTransformer(EMBED_MODEL)
    vecs = model.encode(texts, batch_size=64, show_progress_bar=False)
    assert vecs.shape[1] == EMBED_DIM, f"expected {EMBED_DIM}-dim, got {vecs.shape[1]}"
    print(f"embedded {vecs.shape[0]} chunks -> dim {vecs.shape[1]} in {time.time()-t:.1f}s")
    return vecs

def store_chunks(articles, all_chunks, vecs):
    """M3: write articles + chunks(+vectors) into Postgres. Clean rebuild each run
    (TRUNCATE then reload) — correct for a single-journal v1 corpus.

    Note: chunks.doi is a FK into articles.doi. Safe here because empty_doi=0 in
    this corpus, so every chunk's DOI has a matching article row. A journal with
    missing DOIs would trip the FK — which is the right moment to handle missing
    identifiers properly, not paper over now.
    """
    conn = psycopg.connect(DB_DSN)
    cur = conn.cursor()
    # The vector extension must exist BEFORE register_vector() looks up the
    # `vector` type, or a cold database fails with "vector type not found".
    # Create + commit the extension first, THEN register the type adapter.
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()
    register_vector(conn)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            doi TEXT PRIMARY KEY, title TEXT, journal TEXT, authors JSONB);
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            doi TEXT REFERENCES articles(doi),
            section_title TEXT, text TEXT, cited_refs JSONB,
            embedding vector(384));
    """)
    cur.execute("TRUNCATE chunks, articles CASCADE;")

    seen = set()
    for a in articles:
        doi = a.get("doi", "")
        if not doi or doi in seen:
            continue
        seen.add(doi)
        cur.execute(
            "INSERT INTO articles (doi, title, journal, authors) VALUES (%s, %s, %s, %s)",
            (doi, a.get("english_title", ""), a.get("journal_title", ""),
             json.dumps(a.get("authors", []))))

    for c, v in zip(all_chunks, vecs):
        cur.execute(
            "INSERT INTO chunks (chunk_id, doi, section_title, text, cited_refs, embedding) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (c["chunk_id"], c["doi"], c["section_title"], c["text"],
             json.dumps(c["cited_refs"]), v.tolist()))   # list, not raw ndarray

    cur.execute("CREATE INDEX IF NOT EXISTS chunks_embedding_idx "
                "ON chunks USING hnsw (embedding vector_cosine_ops);")
    conn.commit()
    cur.execute("SELECT count(*) FROM articles;"); n_art = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM chunks;");   n_chunk = cur.fetchone()[0]
    conn.close()
    print(f"stored in Postgres: {n_art} articles, {n_chunk} chunks")


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: python build_index.py path/to/journal.json")

    src = Path(sys.argv[1])
    articles = json.loads(src.read_text(encoding="utf-8"))

    all_chunks = []
    for a in articles:
        cs = chunk_article(a)
        for i, c in enumerate(cs):
            c["chunk_id"] = f"{c['doi']}::{i}"   # stable, DOI-scoped
        all_chunks += cs

    out = src.parent / "chunks.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # --- eyeball stats: this is the M1 "done" check ---
    tok = [approx_tokens(c["text"]) for c in all_chunks]
    tok_sorted = sorted(tok)
    print(f"articles:            {len(articles)}")
    print(f"chunks:              {len(all_chunks)}")
    print(f"chunk tokens min/med/max: {min(tok)}/{tok_sorted[len(tok)//2]}/{max(tok)}")
    print(f"chunks over window ({MODEL_WINDOW}), will truncate: {sum(1 for t in tok if t > MODEL_WINDOW)}")
    print(f"chunks carrying cited_refs: {sum(1 for c in all_chunks if c['cited_refs'])}")
    print(f"written: {out}")

    # --- M2: embed (vectors held in memory; M3 writes them to Postgres) ---
    vecs = embed_chunks(all_chunks)
    print(f"vectors ready: {vecs.shape[0]} x {vecs.shape[1]}")

    # --- M3: store into Postgres + pgvector ---
    store_chunks(articles, all_chunks, vecs)
    
if __name__ == "__main__":
    main()
