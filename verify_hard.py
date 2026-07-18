# verify_hard.py — one-off. Each hard question next to its article's title + abstract.
import json, psycopg
rows = [json.loads(l) for l in open("golden_hard.jsonl", encoding="utf-8") if l.strip()]
conn = psycopg.connect("host=localhost dbname=ragdb user=raguser password=ragpass port=5432")
cur = conn.cursor()
for i, r in enumerate(rows, 1):
    cur.execute("SELECT title FROM articles WHERE doi=%s", (r["answer_doi"],))
    row = cur.fetchone()
    title = row[0] if row else "!!! DOI NOT IN CORPUS"
    cur.execute("SELECT text FROM chunks WHERE doi=%s AND section_title='Abstract' ORDER BY chunk_id LIMIT 1", (r["answer_doi"],))
    ab = cur.fetchone()
    abstract = (ab[0][:300] + "...") if ab else "(no abstract chunk)"
    print(f"\n{'='*80}\nQ{i}: {r['question']}\nDOI: {r['answer_doi']}\nTITLE: {title}\nABSTRACT: {abstract}")
conn.close()