from __future__ import annotations
from difflib import SequenceMatcher
from typing import Any, Dict, Tuple
import re

# --- רשימת שאלות הזהב (Golden Examples) ---
# המפתחות תואמים ללוגיקה ב-app_streamlit.py
# ככל שיש יותר דוגמאות מגוונות, הזיהוי יהיה מדויק יותר (Fuzzy Matching).

GOLDEN_EXAMPLES = {
    "WHO_AT_RISK_TODAY": [
        # עברית
        "מי בסיכון היום",
        "אילו אוטובוסים בסיכון",
        "מי בסיכון",
        "מי הכי מסוכן היום",
        "תראה לי אוטובוסים בסיכון גבוה",
        "רשימת סיכונים להיום",
        "מי עומד להתקלקל",
        "איזה אוטובוסים בבעיה",
        "דוח סיכונים יומי",
        "סיכון יומי",

        # אנגלית
        "Who is at risk today",
        "Which buses are at risk",
        "High risk buses",
        "Today's risk report",
        "Show me risky buses",
        "Predictive failures today"
    ],

    "PERIOD_SUMMARY": [
        # עברית - וריאציות של סיכום
        "מה קרה בשבוע האחרון",
        "סיכום שבועיים אחרונים",
        "מה קרה לאחרונה",
        "תמונת מצב שבועית",
        "סיכום תקופה",
        "דוח מסכם",
        "מה היה בשבוע שעבר",
        "מה קרה בשבועיים האחרונים",
        "מה קרה בשבוע האחרון",
        "סכם לי את השבוע האחרון",
        "סכם לי את החודש האחרון"
        "איך היו הביצועים לאחרונה",
        "סיכום אירועים אחרונים",
        "סטטוס שבועי",

        # אנגלית
        "summary of last week",
        "what happened recently",
        "last 2 weeks summary",
        "period overview",
        "recent events",
        "weekly report"
    ],

    # זיהוי אוטובוס ספציפי נעשה בעיקר ע"י Regex, אבל הדוגמאות עוזרות לדיוק
    "BUS_STATUS": [
        "מה המצב של אוטובוס",
        "תבדוק את אוטובוס",
        "סטטוס BUS",
        "פרטים על אוטובוס",
        "Check bus status",
        "Bus details"
    ]
}


def _get_similarity(a: str, b: str) -> float:
    """מחשב דמיון בין מחרוזות (0.0 עד 1.0)"""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def detect_intents(text: str, today: Any, default_top_limit: int) -> Dict[str, Any]:
    """
    מזהה כוונות אך ורק אם הן דומות מאוד לדוגמאות הזהב.
    כל השאר יחזיר False וילך ל-LLM.
    """
    t = text.strip()
    out: Dict[str, Any] = {
        "WHO_AT_RISK_TODAY": False,
        "PERIOD_SUMMARY": False,
        "BUS_ID": None,
        "TOP_N": default_top_limit,
        "DAYS": 14  # ברירת מחדל לסיכום
    }

    # 1. זיהוי אוטובוס ספציפי (Regex עדיף למספרים)
    # תופס: "BUS 123", "אוטובוס 55", "bus_001"
    m_bus = re.search(r"(?:BUS|אוטובוס)[_\-\s]*(\d{1,3})", t, re.IGNORECASE)
    if m_bus:
        out["BUS_ID"] = f"BUS_{int(m_bus.group(1)):03d}"
        return out  # נחזיר מיד כדי לא לבלבל עם כוונות אחרות

    # 2. בדיקת דמיון לשאר השאלות הקבועות (Fuzzy Matching)
    # סף הדמיון: 0.65 (קצת יותר גמיש כדי לתפוס וריאציות קלות)
    threshold = 0.65

    best_score = 0.0
    best_intent = None

    for intent, examples in GOLDEN_EXAMPLES.items():
        if intent == "BUS_STATUS": continue  # טופל כבר למעלה

        for ex in examples:
            score = _get_similarity(t, ex)
            if score > best_score:
                best_score = score
                best_intent = intent

    if best_score >= threshold:
        out[best_intent] = True

    # 3. זיהוי פרמטרים משלימים (למקרה שנפלנו על PERIOD_SUMMARY)
    # מנסה לחלץ ימים אם המשתמש כתב "סיכום 30 ימים"
    m_days = re.search(r"(\d+)\s*(?:יום|ימים|day|days)", t)
    if m_days:
        out["DAYS"] = int(m_days.group(1))

    return out