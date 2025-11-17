
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
    "Electrical": "במערכת החשמל/חיווט)",
    "Unknown": "נדרש בירור — אין דפוס ברור",
}


def humanize_reason_he(reason: Optional[str]) -> str:
    r = (reason or "").strip()
    if not r:
        return "אין גורם בולט ספציפי"

    mapping = [
        (r"speed_std_7d\+", "תנודתיות גבוהה במהירות בשבוע האחרון"),
        (r"speed_delta\+", "שינויי מהירות חדים"),
        (r"temp_delta\+", "קפיצות חום חריגות"),
        (r"part_km_since_event\+", "מרחק גדול מאז טיפול/החלפה אחרונים"),
        (r"mileage_growth\+", "עלייה חריגה בקילומטראז'"),
        (r"engine_growth\+", "עלייה מצטברת בשעות מנוע"),
        (r"no standout factors", "אין גורם בולט ספציפי"),
    ]
    for pat, rep in mapping:
        r = re.sub(pat, rep, r)

    r = re.sub(r"\(\d+(\.\d+)?σ?\)", "", r)
    r = re.sub(r"\s{2,}", " ", r).replace(" ,", ", ").strip().rstrip(",")
    return r or "אין גורם בולט ספציפי"


def where_from_likely_fault(likely_fault: Optional[str]) -> str:
    return LIKELY_FAULT_MAP_HE.get((likely_fault or "").strip(), LIKELY_FAULT_MAP_HE["Unknown"])


def paraphrase_he(text: str) -> str:
    if not OPENAI_API_KEY:
        return text
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=LLM_TEMPERATURE,
            messages=[
                {"role": "user", "content": "שכתב לעברית טבעית וברורה. הדגש הסתברות.\n---\n" + text}
            ],
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return text


def add_row_explanation(df: pd.DataFrame, prob_col: str) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    if "reason_he" not in df and "failure_reason" in df.columns:
        df["reason_he"] = df["failure_reason"].apply(humanize_reason_he)
    if "where_he" not in df and "likely_fault" in df.columns:
        df["where_he"] = df["likely_fault"].apply(where_from_likely_fault)

    def _ex(r):
        p = r.get(prob_col, None)
        p_txt = f"{p:.0%}" if isinstance(p, (int, float)) else "N/A"
        bus = r.get("bus_id", "")
        d = r.get("d", r.get("date", ""))
        base = (
            f"אוטובוס {bus} בתאריך {d}: הסתברות לשבוע הקרוב {p_txt}. "
            f"גורמים בולטים: {r.get('reason_he', '')}. "
            f"איפה עלולה להיות התקלה: {r.get('where_he', '')}."
        )
        return paraphrase_he(base)

    df["explanation_he"] = df.apply(_ex, axis=1)
    return df
