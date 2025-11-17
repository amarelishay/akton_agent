
from __future__ import annotations
import re
from typing import Any, Dict
from datetime import date

from .time_range import parse_natural_range

HE_NUMS = {
    "אחת": 1, "אחד": 1, "שניים": 2, "שתיים": 2,
    "שלושה": 3, "שלוש": 3, "ארבעה": 4, "ארבע": 4,
    "חמישה": 5, "חמשת": 5, "חמש": 5,
    "שישה": 6, "שש": 6, "שבעה": 7, "שבע": 7,
    "שמונה": 8, "תשעה": 9, "תשע": 9, "עשרה": 10, "עשר": 10,
}

INTENTS = {
    "WHO_AT_RISK_TODAY": re.compile(r"(מי|איזה)\s+(אוטובוס|אוטובוסים)?.*בסיכון(\s*היום)?\??", re.IGNORECASE),
    "BUS_STATUS": re.compile(r"(?:\bBUS[_\-\s]*|\bאוטובוס\s*)(\d{1,3})\b", re.IGNORECASE),
    "MOST_REPLACED_PARTS": re.compile(r"(איזה|מהם)\s+חלק(ים)?\s+(שהוחלפו|הוחלפו)\s+הכי\s+הרבה", re.IGNORECASE),
    "WHAT_HAPPENED_LAST_DAYS": re.compile(r"(מה\sקרה\sבשבועיים|מה\sקרה\sב\d+\s*יום)", re.IGNORECASE),
    "BUS_MOST_FAILURES": re.compile(r"(הכי הרבה תקלות|הכי הרבה\s+תקלות)", re.IGNORECASE),
    "HIGHEST_RISK_N": re.compile(
        r"(?:חמש(?:ת)?|ארבע(?:ה)?|שלוש(?:ה)?|שתיים|שניים|עשר(?:ה)?|\d+)\s+האוטובוסים?\s+.*בסיכון\s+(?:הכי|הגבוה(?:ה)?(?:\sביותר)?)\s*(?:היום)?",
        re.IGNORECASE,
    ),
    "TOP_LIST": re.compile(r"(top|טופ)\s*(\d+)?", re.IGNORECASE),
    "ANY_NATURAL_RANGE": re.compile(
        r"(מה\sקרה\b.*)|\b(last|past)\b|\bמאז\b|\bמ[- ]\d|\bעד\b|\bשבוע האחרון\b|\bחודש האחרון\b|\bשנה האחרונה\b",
        re.IGNORECASE,
    ),
}


def normalize_bus_id(text: str) -> str | None:
    m = re.search(r"\bbus[_\-\s]*(\d{1,3})\b", text, re.IGNORECASE) or re.search(
        r"\bאוטובוס\s*(\d{1,3})\b", text, re.IGNORECASE
    )
    return f"BUS_{int(m.group(1)):03d}" if m else None


def extract_top_n(text: str, default_n: int) -> int:
    m = re.search(r"\b(\d+)\b", text)
    if m:
        return max(1, int(m.group(1)))
    for w, n in HE_NUMS.items():
        if re.search(rf"\b{w}\b", text):
            return n
    return default_n


def detect_intents(text: str, today: date, default_top_limit: int) -> Dict[str, Any]:
    t = text.strip()
    out: Dict[str, Any] = {}
    for name, rx in INTENTS.items():
        m = rx.search(t)
        out[name] = (m.groups() if (m and m.groups()) else True) if m else False

    if not out.get("TOP_LIST") and not out.get("HIGHEST_RISK_N"):
        if re.search(r"בסיכון\s+(?:הכי|הגבוה(?:ה)?(?:\sביותר)?)", t):
            out["TOP_LIST"] = True

    out["RESOLVED_RANGE"] = parse_natural_range(t, today)
    out["DAYS"] = None
    m = re.search(r"(top|טופ)\s*(\d+)", t, re.IGNORECASE)
    if m:
        out["TOP_N"] = int(m.group(2))
    out["TOP_N_TEXT"] = extract_top_n(t, default_top_limit)
    out["BUS_ID"] = normalize_bus_id(t)
    return out
