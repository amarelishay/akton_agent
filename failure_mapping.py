
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
# =========================
#  Likely Fault Mapping
# =========================
import re
from difflib import SequenceMatcher

# הערכים הקנוניים בעמודה likely_fault
LIKELY_FAULT_CANONICAL = [
    "Unknown",
    "Engine",
    "Transmission/Suspension",
    "Cooling/Engine",
    "Part wear",
    "Brake",
]

# מילון של מילים נרדפות (עברית + אנגלית + שגיאות נפוצות) לכל קטגוריה
LIKELY_FAULT_SYNONYMS = {
    "Engine": [
        "engine", "motor", "engine failure", "engine problem",
        "מנוע", "מנועים",
    ],
    "Cooling/Engine": [
        "cooling", "ac", "a/c", "aircon", "air conditioner",
        "air conditioning", "hvac", "cooler", "cooling system",
        "מזגן", "מזוג", "קירור", "קירור מנוע", "מערכת קירור",
    ],
    "Brake": [
        "brake", "brakes", "braking", "abs", "brake system",
        "בלם", "בלמים", "מערכת בלימה",
    ],
    "Transmission/Suspension": [
        "transmission", "gearbox", "gear", "shift",
        "suspension", "shock", "shocks",
        "גיר", "תמסורת", "תיבת הילוכים", "מתלים", "בולמים",
    ],
    "Part wear": [
        "wear", "part wear", "wearing",
        "שחיקה", "בלאי", "שחיקת חלקים",
    ],
    "Unknown": [],
}


def _tokenize_for_faults(text: str) -> list[str]:
    """פירוק פשוט של הטקסט למילים, לעבודה עם fuzzy matching."""
    return re.findall(r"[א-תA-Za-z]+", (text or "").lower())


def map_likely_faults_from_query(user_text: str) -> list[str]:
    """
    מקבל שאלה בשפה חופשית (עברית/אנגלית, כולל שגיאות),
    ומחזיר רשימת קטגוריות likely_fault קנוניות מתאימות.
    לדוגמה:
      'מזגן'  → ['Cooling/Engine']
      'בלמים' → ['Brake']
      'מזגן ובלמים' → ['Cooling/Engine', 'Brake']
    """
    text = (user_text or "").lower()
    tokens = _tokenize_for_faults(text)

    mapped: list[str] = []

    for canon, synonyms in LIKELY_FAULT_SYNONYMS.items():
        found = False

        # 1. התאמה ישירה – מילה נרדפת שהיא substring בשאלה
        for s in synonyms:
            if s.lower() in text:
                mapped.append(canon)
                found = True
                break
        if found:
            continue

        # 2. fuzzy matching – מאפשר שגיאות כתיב
        for s in synonyms:
            s_low = s.lower()
            for tok in tokens:
                if len(tok) < 3:
                    continue
                ratio = SequenceMatcher(None, tok, s_low).ratio()
                if ratio >= 0.8:
                    mapped.append(canon)
                    found = True
                    break
            if found:
                break

    # הסרה של כפולים, שמירה על סדר
    seen = set()
    uniq: list[str] = []
    for m in mapped:
        if m not in seen:
            seen.add(m)
            uniq.append(m)

    try:
        log_agent("Mapped likely_fault from query", query=user_text, mapped=uniq)
    except Exception:
        pass

    return uniq
