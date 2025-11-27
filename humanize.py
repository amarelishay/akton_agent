from __future__ import annotations
import re
from typing import Optional
import pandas as pd
from .config import OPENAI_MODEL, LLM_TEMPERATURE, resolve_openai_key

OPENAI_API_KEY = resolve_openai_key()

LIKELY_FAULT_MAP_HE = {
    "Cooling/Engine": "במערכת הקירור או המנוע",
    "Transmission/Suspension": "במערכת התמסורת או המתלים",
    "Engine": "במנוע",
    "Brakes": "במערכת הבלמים",
    "Electrical": "במערכת החשמל",
    "Unknown": "תקלה כללית",
}


def humanize_reason_he(reason: Optional[str]) -> str:
    """
    ממיר רשימת גורמים טכניים להסבר אנושי וקריא בעברית.
    """
    r = (reason or "").strip()
    if not r or r.lower() == "none" or r == "nan":
        return "ללא גורמים חריגים"

    # סדר המיפוי חשוב (ביטויים ספציפיים קודם)
    mapping = [
        # === מונחים חדשים שנוספו ===
        (r"part_wear_pct[\+↑]?", "אחוז שחיקת חלקים גבוה"),
        (r"part_wear[\+↑]?", "שחיקת חלקים"),

        # === טמפרטורה ===
        (r"temp_std_7d[\+↑]?", "חוסר יציבות בטמפרטורת מנוע"),
        (r"temp_delta[\+↑]?", "קפיצות טמפרטורה חדות"),
        (r"temp_mean[\+↑]?", "התחממות ממוצעת"),

        # === מהירות ===
        (r"speed_std_7d[\+↑]?", "נהיגה לא יציבה"),
        (r"speed_delta[\+↑]?", "שינויי מהירות חריגים"),

        # === שימוש ובלאי ===
        (r"part_km_since_event[\+↑]?", "זמן רב מאז טיפול אחרון"),
        (r"mileage_growth[\+↑]?", "עלייה חריגה בקילומטראז'"),
        (r"engine_growth[\+↑]?", "צבירת שעות מנוע"),

        # === ניקוי כללי ===
        (r"no standout factors", "ללא גורמים בולטים"),
    ]

    for pat, rep in mapping:
        r = re.sub(pat, rep, r, flags=re.IGNORECASE)

    # ניקוי רעשים טכניים: (0.5σ), ↑, +
    r = re.sub(r"\(\d+(\.\d+)?[σ]?\)", "", r)
    r = re.sub(r"[\↑\+]", "", r)

    # סידור פסיקים ורווחים
    r = re.sub(r"\s{2,}", " ", r)
    r = r.replace(" ,", ",").replace(",,", ",").strip().strip(",")

    return r


def where_from_likely_fault(likely_fault: Optional[str]) -> str:
    val = (likely_fault or "").strip()
    for k, v in LIKELY_FAULT_MAP_HE.items():
        if k.lower() in val.lower():
            return v
    return "במערכת לא מזוהה"


def paraphrase_he(text: str) -> str:
    """
    משתמש ב-LLM כדי ללטש ניסוחים אם יש מפתח API.
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
                {"role": "system", "content": "אתה מנהל צי רכב. שכתב את המשפט לעברית מקצועית וקצרה."},
                {"role": "user", "content": text}
            ],
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return text


def pretty_bus_id(bus_id: str | None) -> str:
    """
    מתקן תצוגת שם אוטובוס (במקום קו -> אוטובוס).
    """
    s = str(bus_id or "").strip()
    # מזהה BUS_028 או BUS 28
    m = re.match(r"^BUS[_\-\s]*0*(\d+)$", s, re.IGNORECASE)
    if m:
        return f"אוטובוס {int(m.group(1))}"  # התיקון שביקשת
    return s


def add_row_explanation(df: pd.DataFrame, prob_col: str) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()

    # מילוי עמודות עזר
    if "reason_he" not in df.columns and "failure_reason" in df.columns:
        df["reason_he"] = df["failure_reason"].apply(humanize_reason_he)

    if "where_he" not in df.columns and "likely_fault" in df.columns:
        df["where_he"] = df["likely_fault"].apply(where_from_likely_fault)

    # שימוש ב-LLM רק לטבלאות קטנות (עד 5 שורות) כדי לא להאט
    use_llm = bool(OPENAI_API_KEY) and len(df) <= 5

    def _build_explanation(row):
        p = row.get(prob_col)
        p_txt = f"{p:.0%}" if isinstance(p, (int, float)) else "?%"

        bus_name = pretty_bus_id(row.get("bus_id"))
        date_raw = row.get("d", row.get("date", ""))
        date_str = str(date_raw)[:10]  # חיתוך שעה אם יש

        reasons = row.get("reason_he", "")
        system = row.get("where_he", "")

        # הפורמט המתוקן
        base_text = (
            f"ל-{bus_name} יש הסתברות של {p_txt} לתקלה ב-{date_str}. "
            f"החשד הוא לתקלה {system}. "
            f"גורמים חריגים שזוהו: {reasons}."
        )

        if use_llm:
            return paraphrase_he(base_text)

        return base_text

    df["explanation_he"] = df.apply(_build_explanation, axis=1)
    return df