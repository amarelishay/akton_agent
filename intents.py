from __future__ import annotations
from difflib import SequenceMatcher
from typing import Any, Dict
import re

# --- שאלות זהב (Hardcoded Triggers) ---
# רק שאלות שדומות לאלו ב-90% יפעילו את הדשבורדים הקבועים.
GOLDEN_EXAMPLES = {
    "WHO_AT_RISK_TODAY": [
        "מי בסיכון היום",
        "מי בסיכון",
        "אילו אוטובוסים בסיכון",
        "רשימת סיכונים להיום",
        "Who is at risk today",
        "High risk buses",
        "Which buses are at risk"
    ],

    "BUS_STATUS": [
        # יטופל בנפרד ע"י Regex כי זה כולל מספר משתנה
    ],

    # אך ורק סיכומים כלליים בזמן מוגדר מראש
    "PERIOD_SUMMARY": [
        "מה קרה בשבוע האחרון",
        "מה קרה בשבועיים האחרונים",
        "סיכום שבועיים אחרונים",
        "סכם לי את השבוע",  # <--- הוספנו את זה
        "סכם לי את התקופה",  # <--- ואת זה
        "summary of last week"
    ]
}


def _get_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def detect_intents(text: str, today: Any, default_top_limit: int) -> Dict[str, Any]:
    t = text.strip()
    out: Dict[str, Any] = {
        "WHO_AT_RISK_TODAY": False,
        "PERIOD_SUMMARY": False,
        "BUS_ID": None,
        "TOP_N": default_top_limit,
        "DAYS": 14
    }

    # 1. זיהוי אוטובוס ספציפי (Regex קשיח - תמיד תופס)
    # תופס: BUS 17, אוטובוס 100, bus_050
    m_bus = re.search(r"(?:BUS|אוטובוס)[_\-\s]*(\d{1,3})", t, re.IGNORECASE)
    if m_bus:
        out["BUS_ID"] = f"BUS_{int(m_bus.group(1)):03d}"
        return out

        # 2. זיהוי שאלות זהב (Strict Matching)
    # סף 0.9 אומר "כמעט זהה".
    threshold = 0.9

    best_score = 0.0
    best_intent = None

    for intent, examples in GOLDEN_EXAMPLES.items():
        for ex in examples:
            score = _get_similarity(t, ex)
            if score > best_score:
                best_score = score
                best_intent = intent

    if best_score >= threshold:
        out[best_intent] = True

        # אם זוהה סיכום, ננסה לחלץ ימים (אם המשתמש שינה את המספר)
        if best_intent == "PERIOD_SUMMARY":
            m_days = re.search(r"(\d+)\s*(?:יום|ימים|day|days)", t)
            if m_days:
                out["DAYS"] = int(m_days.group(1))

    return out