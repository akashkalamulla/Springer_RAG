# rank_check.py — one-off. Shows where each hard question's answer_doi ranked.
import json
from rag import retrieve

rows = [json.loads(l) for l in open("golden_hard.jsonl", encoding="utf-8") if l.strip()]
for r in rows:
    dois = [h["doi"] for h in retrieve(r["question"], 5)]
    tgt = r["answer_doi"]
    rank = dois.index(tgt) + 1 if tgt in dois else "MISS"
    flag = "  <-- below #1" if rank not in (1, "MISS") else ""
    print(f"rank={rank}{flag}  {r['question'][:70]}")