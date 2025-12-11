from __future__ import annotations
import re
from difflib import SequenceMatcher
from typing import Any, Dict

# ======================================================
# Normalization Helpers
# ======================================================

HEB_FINALS = str.maketrans({
    'ם': 'מ', 'ן': 'נ', 'ף': 'פ', 'ך': 'כ', 'ץ': 'צ',
})

def normalize_text(t: str) -> str:
    """ Normalize Hebrew/English text, collapse finals, remove punctuation. """
    if not t:
        return ""
    t = t.lower().strip()
    t = t.translate(HEB_FINALS)
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^\w\sא-ת]", " ", t)
    return t


# ======================================================
# Utility: String similarity
# ======================================================

def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


# ======================================================
# Golden Examples (for high-level intents)
# ======================================================

GOLDEN_EXAMPLES = {
    "WHO_AT_RISK_TODAY": [
        "מי בסיכון היום",
        "מי בסיכון",
        "מי בסיכון כעת",
        "אילו אוטובוסים בסיכון",
        "רשימת סיכונים להיום",
        "who is at risk today",
        "high risk buses",
        "which buses are at risk",
    ],

    "PERIOD_SUMMARY": [
        # שבועי
        "מה קרה בשבוע האחרון",
        "מה קרה בשבועיים האחרונים",
        "סיכום שבוע",
        "סיכום שבועיים",
        "תמונת מצב שבוע",
        "summary of last week",
        "status last week",
        "what happened this week",
        "מה קרה השבוע",

        # חודשי – חדש
        "סיכום חודשי",
        "סיכום חודש",
        "מה קרה בחודש האחרון",
        "summary of last month",
        "status last month",
        "monthly summary",
    ],
}


# ======================================================
# Days extraction
# ======================================================

def extract_days(text: str) -> int:
    """ Extract number of days from Hebrew/English natural expressions. """
    t = normalize_text(text)

    if "שבועיים" in t or "שבועים" in t:
        return 14
    if "חודשיים" in t or "חדשיים" in t:
        return 60
    if "יומיים" in t:
        return 2

    m = re.search(r"(\d+)\s*(?:יום|ימים|day|days)", t)
    if m:
        return int(m.group(1))

    m = re.search(r"(\d+)\s*(?:שבוע|שבועות|week|weeks)", t)
    if m:
        return int(m.group(1)) * 7

    m = re.search(r"(\d+)\s*(?:חודש|חודשים|month|months)", t)
    if m:
        return int(m.group(1)) * 30

    # מילים בלי מספר
    if "שבוע" in t:
        return 7
    if "חודש" in t or "חודשי" in t:
        return 30
    if "יום" in t:
        return 1

    # ברירת מחדל
    return 14


# ======================================================
# BUS ID Detection
# ======================================================

def detect_bus_id(text_raw: str) -> str | None:
    """
    Detects:
    - "bus 32", "bus32"
    - "אוטובוס 32"
    - "רכב 32"
    - Or just "32" in context
    """
    patterns = [
        r"(?:bus|אוטובוס|רכב)[^\d]*(\d{1,4})",
        r"\b(\d{1,4})\b"
    ]
    for p in patterns:
        m = re.search(p, text_raw, re.IGNORECASE)
        if m:
            return m.group(1)
    return None


# ======================================================
# Main Intent Engine
# ======================================================

def detect_intents(text: str, today: Any, default_top_limit: int) -> Dict[str, Any]:

    t_raw = text or ""
    t = normalize_text(t_raw)

    out: Dict[str, Any] = {
        "WHO_AT_RISK_TODAY": False,
        "PERIOD_SUMMARY": False,
        "BUS_LOCATION": False,
        "BUS_ID": None,
        "TOP_N": default_top_limit,
        "DAYS": 14,
    }

    # -----------------------------
    # Detect BUS ID
    # -----------------------------
    bus_id = detect_bus_id(t_raw)
    if bus_id:
        out["BUS_ID"] = bus_id

    # -----------------------------
    # BUS LOCATION — robust matching
    # -----------------------------
    location_keywords = [
        "איפה", "באיזה אזור", "באיזה מרחב",
        "צפון", "דרום",
        "north", "south", "region", "location",
    ]

    location_error_patterns = [
        r"דרו?מ", r"דרום", r"בדרו?מ", r"הדרום", r"הבדרום",
        r"צפו?נ", r"צפון", r"בצפו?נ", r"הצפון",
    ]

    found_location = False

    if any(k in t for k in location_keywords):
        found_location = True

    for pat in location_error_patterns:
        if re.search(pat, t):
            found_location = True
            break

    if found_location and bus_id:
        out["BUS_LOCATION"] = True
        return out

    # -----------------------------
    # Golden Intents (WHO_AT_RISK, PERIOD_SUMMARY)
    # -----------------------------
    best_score = 0.0
    best_intent = None
    threshold = 0.75

    for intent, examples in GOLDEN_EXAMPLES.items():
        for ex in examples:
            score = _sim(t, normalize_text(ex))
            if score > best_score:
                best_score = score
                best_intent = intent

    if best_score >= threshold:
        out[best_intent] = True
        if best_intent == "PERIOD_SUMMARY":
            out["DAYS"] = extract_days(t_raw)
        return out

    # -----------------------------
    # Heuristic fallback for PERIOD_SUMMARY
    # -----------------------------
    # דוגמה: "תן לי סיכום חודשי", "תן לי סיכום שבועי", "תן סיכום של 3 חודשים"
    summary_triggers = ["סיכום", "תמונת מצב", "summary", "status"]
    period_words = ["יום", "ימים", "שבוע", "שבועות", "חודש", "חודשים",
                    "day", "days", "week", "weeks", "month", "months"]

    if any(w in t for w in summary_triggers) and any(w in t for w in period_words):
        out["PERIOD_SUMMARY"] = True
        out["DAYS"] = extract_days(t_raw)
        return out

    # -----------------------------
    # Default fallback: no intent detected
    # -----------------------------
    return out
