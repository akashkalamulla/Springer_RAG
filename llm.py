"""
llm.py — M5 (generate). Thin wrapper around the Gemini API.

Loads GEMINI_API_KEY from .env and exposes generate(prompt, system=...), used by
rag.py to turn retrieved chunks into a grounded answer.
"""

import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

MODEL = "gemini-3.5-flash"

_client = None


def get_client():
    global _client
    if _client is None:
        api_key = os.environ["GEMINI_API_KEY"]
        _client = genai.Client(api_key=api_key)
    return _client


def generate(prompt, system=None, temperature=0.2, model=None, retries=4):
    config = types.GenerateContentConfig(system_instruction=system, temperature=temperature)
    client = get_client()
    for attempt in range(retries):
        try:
            resp = client.models.generate_content(
                model=model or MODEL, contents=prompt, config=config)
            return (resp.text or "").strip()
        except Exception as e:
            is_rate_limit = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e).upper()
            if is_rate_limit and attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
