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
    max_day = [
        31,
        29 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 28,
        31, 30, 31, 30, 31, 31, 30, 31, 30, 31
    ][m - 1]
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
    t = (user_text or "").strip()
    if not t:
        return None

    t_norm = t.lower()

    # 🔹 שנים מפורשות: "בשנת 2023", "שנת 2023", או סתם "2023" בתוך המשפט
    m_year = re.search(r"(?:בשנת|שנת)\s+(20\d{2})", t_norm)
    if not m_year:
        m_year = re.search(r"\b(20\d{2})\b", t_norm)
    if m_year:
        year = int(m_year.group(1))
        start = date(year, 1, 1)
        end = date(year, 12, 31)
        title = f"שנת {year}"
        return start, end, title

    # היום / אתמול
    if re.search(r"\bהיום\b|\btoday\b", t, re.IGNORECASE):
        return today, today, "היום"
    if re.search(r"\bאתמול\b|\byesterday\b", t, re.IGNORECASE):
        d = today - dt.timedelta(days=1)
        return d, d, "אתמול"

    # "ב17 ימים האחרונים" / "last 17 days"
    m = re.search(
        r"[ב]?(?P<n>\d+)\s*"
        r"(?P<u>יום|ימים|day|days|שבוע|שבועות|week|weeks|חודש|חודשים|month|months|שנה|שנים|year|years)"
        r"\s*(האחרונ(?:ים|ות|ה)?|האחרונה)?",
        t,
        re.IGNORECASE,
    )
    if m:
        try:
            n = int(m.group("n"))
        except ValueError:
            n = 1

        unit_token = m.group("u").lower()
        unit = HE_UNITS.get(unit_token) or EN_UNITS.get(unit_token) or "days"

        s, e = _range_last_n(today, n, unit)

        if unit == "days":
            label = f"ב{n} הימים האחרונים"
        elif unit == "weeks":
            label = f"ב{n} השבועות האחרונים"
        elif unit == "months":
            label = f"ב{n} החודשים האחרונים"
        else:
            label = f"ב{n} השנים האחרונות"

        return s, e, label

    # "בשבוע האחרון" / "last week"
    if re.search(r"\bבשבוע האחרון\b|\blast week\b", t, re.IGNORECASE):
        s, e = _range_last_n(today, 1, "weeks")
        return s, e, "בשבוע האחרון"

    # "בחודש האחרון" / "last month"
    if re.search(r"\bבחודש האחרון\b|\blast month\b", t, re.IGNORECASE):
        s, e = _range_last_n(today, 1, "months")
        return s, e, "בחודש האחרון"

    # "בשנה האחרונה" / "last year"
    if re.search(r"\bבשנה האחרונה\b|\blast year\b", t, re.IGNORECASE):
        s, e = _range_last_n(today, 1, "years")
        return s, e, "בשנה האחרונה"

    # תאריך יחיד מפורש: "5/12/2024", "05-12-23" וכו
    m_date = re.search(r"(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})", t)
    if m_date:
        d = _normalize_date_token(m_date.group(1))
        if d:
            label = d.strftime("%d/%m/%Y")
            return d, d, label

    return None
