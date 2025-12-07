from __future__ import annotations
from difflib import SequenceMatcher
from typing import Any, Dict
import re

# -------------------------
# Normalization Helpers
# -------------------------

# ממיר אותיות סופיות → רגילות כדי לא לפספס התאמות
HEB_FINALS = str.maketrans({
    'ם': 'מ',
    'ן': 'נ',
    'ף': 'פ',
    'ך': 'כ',
    'ץ': 'צ',
})

def normalize_text(t: str) -> str:
    if not t:
        return ""
    t = t.lower().strip()
    t = t.translate(HEB_FINALS)
    t = re.sub(r"\s+", " ", t)  # רווח אחד בלבד
    t = re.sub(r"[^\w\sא-ת]", "", t)  # מסיר סימנים מיותרים
    return t


# ----------------------------
# GOLDEN QUESTIONS (Expanded)
# ----------------------------

GOLDEN_EXAMPLES = {
    "WHO_AT_RISK_TODAY": [
        "מי בסיכון היום",
        "מי בסיכון",
        "אילו אוטובוסים בסיכון",
        "רשימת סיכונים להיום",
        "מי בסכון",          # שגיאות נפוצות
        "מי בסקון",
        "מי בסכיון",
        "מי בסיכן",
        "who is at risk today",
        "high risk buses",
        "which buses are at risk",
    ],

    "BUS_STATUS": [
        # מטופל ע"י Regex, לא צריך דוגמאות
    ],

    "PERIOD_SUMMARY": [
        "מה קרה בשבוע האחרון",
        "מה קרה בשבועיים האחרונים",
        "סיכום שבועיים אחרונים",
        "סכם לי את השבוע",
        "סכם לי את התקופה",
        "סיכום שבוע",
        "summary of last week",
        "what happened this week",
        "status last week",
        "מה קרה השבוע",
        "מה היה השבוע",
        "תמונת מצב שבועית",
        "סכון שבוע",   # שגיאות נפוצות
    ],
}


# ----------------------------
# Similarity Function
# ----------------------------

def _get_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


# ----------------------------
# Main Intent Detector
# ----------------------------

def detect_intents(text: str, today: Any, default_top_limit: int) -> Dict[str, Any]:

    # Normalize aggressively
    t_raw = text or ""
    t = normalize_text(t_raw)

    out: Dict[str, Any] = {
        "WHO_AT_RISK_TODAY": False,
        "PERIOD_SUMMARY": False,
        "BUS_ID": None,
        "TOP_N": default_top_limit,
        "DAYS": 14
    }

    # -----------------------------------------------------
    # 1. זיהוי BUS ID עם Regex (מדויק ותמיד תקף)
    # -----------------------------------------------------
    m_bus = re.search(r"(?:bus|אוטובוס)[_\-\s]*0*(\d{1,3})", t_raw, re.IGNORECASE)
    if m_bus:
        out["BUS_ID"] = f"BUS_{int(m_bus.group(1)):03d}"
        return out

    # -----------------------------------------------------
    # 2. Golden Questions Detection (Fuzzy)
    # -----------------------------------------------------

    threshold = 0.75            # הורדנו מ-0.9 כדי לתפוס שגיאות כתיב
    best_score = 0.0
    best_intent = None

    for intent, examples in GOLDEN_EXAMPLES.items():
        for ex in examples:
            ex_norm = normalize_text(ex)
            score = _get_similarity(t, ex_norm)
            if score > best_score:
                best_score = score
                best_intent = intent

    if best_score >= threshold:
        out[best_intent] = True

        # חילוץ ימים מתוך השאלה: "17 ימים", "3 day", "5 days"
        if best_intent == "PERIOD_SUMMARY":
            m_days = re.search(r"(\d+)\s*(?:יום|ימים|day|days)", t)
            if m_days:
                out["DAYS"] = int(m_days.group(1))

        return out

    # -----------------------------------------------------
    # סיום: אם לא זוהה שום Golden Intent → חוזרים לסוכן
    # -----------------------------------------------------
    return out
