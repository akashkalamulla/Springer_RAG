"""One-time migration: add a full-text (tsvector) column + GIN index to chunks.
Idempotent. Does NOT touch embeddings or row count — pure additive column."""
import psycopg

DB_DSN = "host=localhost dbname=ragdb user=raguser password=ragpass port=5432"

conn = psycopg.connect(DB_DSN)
cur = conn.cursor()

cur.execute("SELECT count(*) FROM chunks;")
before = cur.fetchone()[0]

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
