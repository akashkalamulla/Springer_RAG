"""
llm.py — M5 (generate). Thin wrapper around the Gemini API.

Loads GEMINI_API_KEY from .env and exposes generate(prompt, system=...), used by
rag.py to turn retrieved chunks into a grounded answer.
"""

import os

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


def generate(prompt, system=None):
    """Send prompt (+ optional system instruction) to Gemini, return response text."""
    config = types.GenerateContentConfig(system_instruction=system) if system else None
    response = get_client().models.generate_content(
        model=MODEL,
        contents=prompt,
        config=config,
    )
    return response.text
