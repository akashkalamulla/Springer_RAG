"""
llm.py — M5 (generate). Thin wrapper around the OpenAI API.

Loads OPENAI_API_KEY from .env and exposes generate(prompt, system=..., model=...),
used by rag.py to turn retrieved chunks into a grounded answer. The generator and
the eval's faithfulness judge pass different model names through the same call.
"""
import os
import time

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, InternalServerError, APIConnectionError

load_dotenv()

MODEL = "gpt-4o-mini"

_client = None


def get_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client



def generate(prompt, system=None, temperature=0.2, model=None, retries=4):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    client = get_client()
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model or MODEL,
                messages=messages,
                temperature=temperature,
            )
            return (resp.choices[0].message.content or "").strip()
        except (RateLimitError, InternalServerError, APIConnectionError):
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
