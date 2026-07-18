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
