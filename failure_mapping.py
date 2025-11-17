
from __future__ import annotations
from typing import List
import pandas as pd

from .db import q
from .config import OPENAI_MODEL, resolve_openai_key, LLM_TEMPERATURE
from .utils_logging import log_agent
from .schema_meta import TABLES_META

OPENAI_API_KEY = resolve_openai_key()

# Load failure types from DB if possible
def load_failure_types() -> List[str]:
    try:
        # try from dim_fault
        cols = [c["name"] for c in TABLES_META.get("dim_fault", {}).get("columns", [])]
        if "failure_type" in cols:
            df = q("SELECT DISTINCT failure_type FROM public.dim_fault ORDER BY failure_type", {})
            return [str(x) for x in df["failure_type"].dropna().tolist()]
    except Exception as e:
        log_agent("load_failure_types failed", error=str(e))
    return []


FAILURE_TYPES: List[str] = load_failure_types()
FAILURE_TYPES_TEXT = "\n".join(f"- {ft}" for ft in FAILURE_TYPES) or "- (none)"

MAP_FAILURE_SYS = f"""You map a natural-language question about bus failures
to a list of failure_type values taken ONLY from the list below.

Valid failure_type values (dim_fault.failure_type):
{FAILURE_TYPES_TEXT}

Rules:
- User may speak Hebrew or English, with typos and slang
  (e.g. "מזגן", "AC", "קירור", "בלמים", "גיר", "חשמל").
- Return 0 or more items from the list above.
- If the question talks about failures in general (not a specific system),
  return an empty list [].
Output STRICT JSON, no text around it:
{{"failure_types": ["...","..."]}}"""


def _safe_json_loads(s: str):
    import json, re
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    try:
        return json.loads(s)
    except Exception:
        return None


def map_failure_types_from_query(user_text: str) -> list[str]:
    if not OPENAI_API_KEY or not FAILURE_TYPES:
        return []

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": MAP_FAILURE_SYS},
                {"role": "user", "content": user_text},
            ],
        )
        raw = resp.choices[0].message.content or ""
        data = _safe_json_loads(raw) or {}
        cand = data.get("failure_types", []) or []

        result: list[str] = []
        for c in cand:
            c_low = str(c).strip().lower()
            for real in FAILURE_TYPES:
                if real.lower() == c_low:
                    result.append(real)
                    break

        uniq = list(dict.fromkeys(result))
        log_agent("Mapped failure types", query=user_text, mapped=uniq)
        return uniq
    except Exception as e:
        log_agent("map_failure_types_from_query failed", error=str(e))
        return []
