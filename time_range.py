
from __future__ import annotations
import datetime as dt
import re
from datetime import date
from typing import Optional, Tuple

HE_UNITS = {
    "יום": "days", "ימים": "days",
    "שבוע": "weeks", "שבועות": "weeks",
    "חודש": "months", "חודשים": "months",
    "שנה": "years", "שנים": "years",
}
EN_UNITS = {
    "day": "days", "days": "days",
    "week": "weeks", "weeks": "weeks",
    "month": "months", "months": "months",
    "year": "years", "years": "years",
}


def _shift_months(d: date, months: int) -> date:
    y = d.year + (d.month - 1 + months) // 12
    m = (d.month - 1 + months) % 12 + 1
    max_day = [31, 29 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 28,
               31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m - 1]
    day = min(d.day, max_day)
    return date(y, m, day)


def _range_last_n(today: date, n: int, unit: str) -> Tuple[date, date]:
    if unit == "days":
        return today - dt.timedelta(days=n - 1), today
    if unit == "weeks":
        return today - dt.timedelta(days=7 * n - 1), today
    if unit == "months":
        start = _shift_months(today.replace(day=1), -(n - 1))
        return start, today
    if unit == "years":
        return date(today.year - n + 1, 1, 1), today
    return today, today


def _normalize_date_token(tok: str) -> Optional[date]:
    tok = tok.strip().replace(".", "/").replace("-", "/")
    for fmt in ("%d/%m/%Y", "%Y/%m/%d", "%d/%m/%y"):
        try:
            return dt.datetime.strptime(tok, fmt).date()
        except Exception:
            pass
    return None


def parse_natural_range(user_text: str, today: date) -> Optional[Tuple[date, date, str]]:
    t = user_text.strip()

    if re.search(r"\bהיום\b|\btoday\b", t, re.IGNORECASE):
        return today, today, "היום"
    if re.search(r"\bאתמול\b|\byesterday\b", t, re.IGNORECASE):
        d = today - dt.timedelta(days=1)
        return d, d, "אתמול"

    if re.search(r"\bבשבוע האחרון\b|\blast week\b", t, re.IGNORECASE):
        s, e = _range_last_n(today, 1, "weeks")
        return s, e, "בשבוע האחרון"
    if re.search(r"\bבחודש האחרון\b|\blast month\b", t, re.IGNORECASE):
        s, e = _range_last_n(today, 1, "months")
        return s, e, "בחודש האחרון"

    m = re.search(r"(?:מ|-|from)\s*([0-9./-]{6,10})\s*(?:עד|ל|to)\s*([0-9./-]{6,10})", t, re.IGNORECASE)
    if m:
        d1 = _normalize_date_token(m.group(1))
        d2 = _normalize_date_token(m.group(2))
        if d1 and d2:
            s, e = (d1, d2) if d1 <= d2 else (d2, d1)
            return s, e, f"{s}–{e}"

    # ברירת מחדל: כל השנה עד היום
    start = date(today.year, 1, 1)
    return start, today, "תקלות אוטובוסים"
