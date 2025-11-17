
from __future__ import annotations
import os
from dotenv import load_dotenv

load_dotenv()

PG_HOST = os.getenv("PG_HOST", "")
PG_PORT = os.getenv("PG_PORT", "")
PG_DB   = os.getenv("PG_DB",   "")
PG_USER = os.getenv("PG_USER", "")
PG_PASS = os.getenv("PG_PASSWORD", "")

OPENAI_MODEL     = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE  = float(os.getenv("LLM_TEMPERATURE", "0.6"))

def resolve_openai_key() -> str:
    try:
        import streamlit as st  # type: ignore
        return st.secrets.get("OPENAI_API_KEY", "")  # type: ignore[attr-defined]
    except Exception:
        return os.getenv("OPENAI_API_KEY", "") or ""
