from __future__ import annotations
import re
from typing import Optional
import pandas as pd
from .config import OPENAI_MODEL, LLM_TEMPERATURE, resolve_openai_key

OPENAI_API_KEY = resolve_openai_key()

LIKELY_FAULT_MAP_HE = {
    "Cooling/Engine": "במערכת הקירור או המנוע (רדיאטור, משאבת מים, צנרת, תרמוסטט)",
    "Transmission/Suspension": "במערכת התמסורת או המתלים (גיר, קלאץ', בולמים)",
    "Engine": "במנוע (הצתה/דלק/חיישנים)",
    "Brakes": "במערכת הבלמים",
    "Electrical": "במערכת החשמל/חיווט",
    "Unknown": "נדרש בירור — אין דפוס ברור",
}


def humanize_reason_he(reason: Optional[str]) -> str:
    """
    ממיר רשימת גורמים טכניים (כמו speed_std_7d↑) להסבר אנושי בעברית.
    """
    r = (reason or "").strip()
    if not r or r.lower() == "none":
        return "אין גורם בולט ספציפי"

    # מיפוי משופר - תומך גם ב (+) וגם ב (↑)
    mapping = [
        (r"speed_std_7d[\+↑]?", "תנודתיות גבוהה במהירות בשבוע האחרון"),
        (r"speed_delta[\+↑]?", "שינויי מהירות חדים"),
        (r"temp_delta[\+↑]?", "קפיצות חום חריגות"),
        (r"part_km_since_event[\+↑]?", "מרחק גדול מאז טיפול/החלפה אחרונים"),
        (r"mileage_growth[\+↑]?", "עלייה חריגה בקילומטראז'"),
        (r"engine_growth[\+↑]?", "עלייה מצטברת בשעות מנוע"),
        (r"no standout factors", "אין גורם בולט ספציפי"),
    ]

    for pat, rep in mapping:
        # שימוש ב-IGNORECASE כדי לתפוס אותיות גדולות/קטנות
        r = re.sub(pat, rep, r, flags=re.IGNORECASE)

    # ניקוי רעשים טכניים כמו (0.5σ) או (0.4)
    r = re.sub(r"\(\d+(\.\d+)?[σ]?\)", "", r)

    # ניקוי רווחים כפולים ופסיקים מיותרים
    r = re.sub(r"\s{2,}", " ", r).replace(" ,", ", ").strip().strip(",")

    return r or "אין גורם בולט ספציפי"


def where_from_likely_fault(likely_fault: Optional[str]) -> str:
    val = (likely_fault or "").strip()
    # טיפול במקרה שהמודל מחזיר רשימה או מחרוזת מלוכלכת
    for k, v in LIKELY_FAULT_MAP_HE.items():
        if k.lower() in val.lower():
            return v
    return LIKELY_FAULT_MAP_HE["Unknown"]


def paraphrase_he(text: str) -> str:
    """
    משתמש ב-LLM כדי לשכתב טקסטים גנריים לעברית טבעית יותר.
    """
    if not OPENAI_API_KEY:
        return text
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=LLM_TEMPERATURE,
            messages=[
                {"role": "user", "content": "שכתב לעברית טבעית וברורה. שמור על המשמעות המקורית.\n---\n" + text}
            ],
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return text


def pretty_bus_id(bus_id: str | None) -> str:
    """
    מציג BUS_032 כ-'BUS 32'.
    """
    s = str(bus_id or "").strip()
    m = re.match(r"^BUS[_\-\s]*0*(\d+)$", s, re.IGNORECASE)
    if m:
        return f"BUS {int(m.group(1))}"
    return s


def add_row_explanation(df: pd.DataFrame, prob_col: str) -> pd.DataFrame:
    """
    מוסיף עמודת הסבר מילולי מלא (explanation_he) לכל שורה בטבלה.
    """
    if df.empty:
        return df

    df = df.copy()

    # 1. מילוי עמודות עזר בעברית אם הן חסרות
    if "reason_he" not in df.columns and "failure_reason" in df.columns:
        df["reason_he"] = df["failure_reason"].apply(humanize_reason_he)

    if "where_he" not in df.columns and "likely_fault" in df.columns:
        df["where_he"] = df["likely_fault"].apply(where_from_likely_fault)

    # שימוש ב-LLM רק אם הטבלה קטנה (לחסכון בזמן ועלויות)
    use_llm = bool(OPENAI_API_KEY) and len(df) <= 15

    def _build_explanation(row):
        # שליפת נתונים
        p = row.get(prob_col)
        p_txt = f"{p:.0%}" if isinstance(p, (int, float)) else "N/A"

        bus_name = pretty_bus_id(row.get("bus_id"))
        date_str = row.get("d", row.get("date", ""))

        reasons = row.get("reason_he", "סיבות טכניות")
        system = row.get("where_he", "מערכת לא ידועה")

        # בניית המשפט
        base_text = (
            f"לאוטובוס {bus_name} יש הסתברות של {p_txt} לתקלה ב-{date_str}. "
            f"הסימנים המקדימים הם {reasons}, "
            f"והחשד הוא לתקלה {system}."
        )

        # אם הרשימה קצרה, נבקש מה-LLM לנסח יפה יותר
        if use_llm:
            return paraphrase_he(base_text)

        return base_text

    df["explanation_he"] = df.apply(_build_explanation, axis=1)
    return df