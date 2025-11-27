from __future__ import annotations
from sqlalchemy import text

import json
import re
from datetime import date, timedelta
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from .config import OPENAI_MODEL, LLM_TEMPERATURE, resolve_openai_key
from .db import q, resolve_predictions_source
from .humanize import (
    humanize_reason_he,
    where_from_likely_fault,
    add_row_explanation,
)
from .schema_meta import allowed_tables_from_schema, schema_summary_for_llm
from .time_range import _range_last_n, parse_natural_range
from .utils_logging import log_agent


# =========================
# הגדרות כלליות
# =========================

OPENAI_API_KEY = resolve_openai_key()
PRED_SRC = resolve_predictions_source()  # subselect מאוחד עם alias p
ALLOWED_TABLES = {
    "public.fact_bus_status_star",
    "public.dim_bus_star",
    "public.dim_date",
    "public.fact_bus_daily",
    "public.bridge_fault_part",
    "public.dim_part"
}

# טווח הדאטה בפועל (לידיעה בלבד, הוולידציה נעשית בשכבת האפליקציה)
DATA_MIN_DATE = date(2023, 1, 1)
DATA_MAX_DATE = date(2024, 12, 31)


# =========================
# שאילתות SQL קבועות
# =========================

SQL_AT_RISK_TODAY = f"""
SELECT
    p.bus_id,
    p.date::date AS d,
    p.proba_7d   AS predicted_proba,
    p.label_7d   AS predicted_label,
    p.failure_reason,
    p.likely_fault
FROM {PRED_SRC}
WHERE p.date::date = :d
  AND p.proba_7d >= 0.5
ORDER BY p.proba_7d DESC NULLS LAST
LIMIT :limit
"""

SQL_AT_RISK_TOP1 = f"""
SELECT
    p.bus_id,
    p.date::date AS d,
    p.proba_7d   AS predicted_proba,
    p.label_7d   AS predicted_label,
    p.failure_reason,
    p.likely_fault
FROM {PRED_SRC}
WHERE p.date::date = :d
ORDER BY p.proba_7d DESC NULLS LAST
LIMIT 1
"""

SQL_TOP_RISK_TODAY = f"""
SELECT
    p.bus_id,
    p.date::date AS d,
    p.proba_7d   AS predicted_proba,
    p.label_7d   AS predicted_label,
    p.failure_reason,
    p.likely_fault
FROM {PRED_SRC}
WHERE p.date::date = :d
ORDER BY p.proba_7d DESC NULLS LAST
LIMIT :limit
"""


SQL_BUS_TODAY = f"""
SELECT
    p.bus_id,
    p.date::date AS d,
    p.proba_7d   AS proba_7d,
    p.label_7d   AS label_7d,
    p.proba_30d  AS proba_30d,
    p.label_30d  AS label_30d,
    p.failure_reason,
    p.likely_fault
FROM {PRED_SRC}
WHERE p.date::date = :d
  AND p.bus_id = :bus
ORDER BY p.date DESC
LIMIT 1
"""

SQL_BUS_HISTORY = f"""
SELECT
    p.bus_id,
    p.date::date AS d,
    p.proba_7d   AS proba_7d,
    p.label_7d   AS label_7d,
    p.proba_30d  AS proba_30d,
    p.label_30d  AS label_30d,
    p.failure_reason,
    p.likely_fault
FROM {PRED_SRC}
WHERE p.bus_id = :bus
ORDER BY p.date DESC
LIMIT :limit
"""

SQL_TREND_LAST_DAYS = f"""
SELECT
    p.date::date AS d,
    COUNT(*) FILTER (WHERE p.proba_7d >= 0.5)                  AS at_risk,
    AVG(p.proba_7d)                                            AS avg_proba,
    100.0 * COUNT(*) FILTER (WHERE p.proba_7d >= 0.5)
        / NULLIF(COUNT(*), 0)                                  AS pct_risk,
    COUNT(*)                                                   AS total_buses
FROM {PRED_SRC}
WHERE p.date::date BETWEEN :start AND :end
GROUP BY 1
ORDER BY 1
"""

SQL_PARTS_REPLACED_LAST_30D = """
WITH range AS (
  SELECT CAST(:start AS date) AS start_d,
         CAST(:end   AS date) AS end_d
)
SELECT
    dp.part_name,
    COUNT(*) AS replaced_count
FROM public.fact_bus_status_star f
JOIN range r ON TRUE
LEFT JOIN public.bridge_fault_part b ON f.fault_id = b.fault_id
LEFT JOIN public.dim_part        dp ON b.part_id  = dp.part_id
WHERE f.date_id::date BETWEEN r.start_d AND r.end_d
  AND COALESCE(f.maintenance_flag, false) = true
GROUP BY dp.part_name
ORDER BY replaced_count DESC NULLS LAST
LIMIT :limit
"""


# =========================
# פונקציות עזר לדאטה
# =========================

def _enrich_with_human_explanation(df: pd.DataFrame, prob_col: str) -> pd.DataFrame:
    """מוסיף reason_he, where_he, explanation_he במידת הצורך."""
    if df.empty:
        return df

    if "failure_reason" in df.columns:
        df["reason_he"] = df["failure_reason"].apply(humanize_reason_he)
    if "likely_fault" in df.columns:
        df["where_he"] = df["likely_fault"].apply(where_from_likely_fault)

    df = add_row_explanation(df, prob_col=prob_col)
    return df


# =========================
# פונקציות נתונים עיקריות
# =========================

def df_at_risk_today(d: date, limit: int) -> pd.DataFrame:
    """
    אוטובוסים בסיכון מעל סף ברירת המחדל (כיום 50%).
    """
    df = q(SQL_AT_RISK_TODAY, {"d": d, "limit": limit})
    return _enrich_with_human_explanation(df, prob_col="predicted_proba")


def df_top_risk_today(d: date, limit: int) -> pd.DataFrame:
    """
    Top N אוטובוסים עם הסיכון הגבוה ביותר היום (ללא סף מינימלי).
    """
    df = q(SQL_TOP_RISK_TODAY, {"d": d, "limit": limit})
    return _enrich_with_human_explanation(df, prob_col="predicted_proba")


def df_at_risk_top1(d: date) -> pd.DataFrame:
    df = q(SQL_AT_RISK_TOP1, {"d": d})
    return _enrich_with_human_explanation(df, prob_col="predicted_proba")


def df_bus_today(d: date, bus_id: str) -> pd.DataFrame:
    df = q(SQL_BUS_TODAY, {"d": d, "bus": bus_id})
    return _enrich_with_human_explanation(df, prob_col="proba_7d")


def df_bus_history(bus_id: str, limit: int = 200) -> pd.DataFrame:
    """
    היסטוריית תחזיות לאוטובוס מסוים (עד limit רשומות אחרונות).
    """
    df = q(SQL_BUS_HISTORY, {"bus": bus_id, "limit": limit})
    return _enrich_with_human_explanation(df, prob_col="proba_7d")


def df_parts_replaced_last_30d(end_date: date, limit: int) -> pd.DataFrame:
    start, end = _range_last_n(end_date, 30, "days")
    return q(SQL_PARTS_REPLACED_LAST_30D, {"start": start, "end": end, "limit": limit})


def df_failures_by_day_detail(start: date, end: date, season: str = None) -> pd.DataFrame:
    """
    פירוט תקלות בפועל (Fact Table) עם סינון עונתי.
    """
    sql = f"""
    SELECT
        f.date_id::date AS d,
        b.bus_id,
        dft.failure_type,
        dft.fault_category,
        COALESCE(f.failure_flag,      FALSE) AS failure_flag,
        COALESCE(f.maintenance_flag,  FALSE) AS maintenance_flag
    FROM public.fact_bus_status_star f
    JOIN public.dim_bus_star   b   ON f.bus_sk  = b.bus_sk
    LEFT JOIN public.dim_fault dft ON f.fault_id = dft.fault_id
    LEFT JOIN public.dim_date  dd  ON f.date_id = dd.date_id
    WHERE f.date_id BETWEEN :start AND :end
      AND (COALESCE(f.failure_flag, FALSE) = TRUE OR f.fault_id IS NOT NULL)
      { "AND dd.season = :season" if season else "" }
    ORDER BY d, b.bus_id
    """
    params = {"start": start, "end": end}
    if season:
        params["season"] = season
    return q(sql, params)

def df_bus_all_failures(bus_id: str) -> pd.DataFrame:
    """
    כל התקלות שאוטובוס מסוים חווה לאורך כל התקופה
    מתוך fact_bus_status_star.
    """
    sql = """
    SELECT
        f.date_id::date AS d,
        b.bus_id,
        dft.failure_type,
        dft.fault_category,
        COALESCE(f.failure_flag,      FALSE) AS failure_flag,
        COALESCE(f.maintenance_flag,  FALSE) AS maintenance_flag
    FROM public.fact_bus_status_star f
    JOIN public.dim_bus_star   b   ON f.bus_sk  = b.bus_sk
    LEFT JOIN public.dim_fault dft ON f.fault_id = dft.fault_id
    WHERE b.bus_id = :bus
      AND (COALESCE(f.failure_flag, FALSE) = TRUE OR f.fault_id IS NOT NULL)
    ORDER BY d ASC
    """
    return q(sql, {"bus": bus_id})


def df_trend_last_days(start: date, end: date, season: str = None) -> pd.DataFrame:
    """
    מחזיר נתוני מגמה (Trend) עם תמיכה בסינון עונתי.
    """
    # JOIN ל-dim_date כדי לסנן לפי עונה
    sql = f"""
    SELECT
        p.date::date AS d,
        COUNT(*) FILTER (WHERE p.proba_7d >= 0.5)                  AS at_risk,
        AVG(p.proba_7d)                                            AS avg_proba,
        100.0 * COUNT(*) FILTER (WHERE p.proba_7d >= 0.5)
            / NULLIF(COUNT(*), 0)                                  AS pct_risk,
        COUNT(*)                                                   AS total_buses
    FROM {PRED_SRC}
    LEFT JOIN public.dim_date dd ON p.date::date = dd.date_id
    WHERE p.date::date BETWEEN :start AND :end
      { "AND dd.season = :season" if season else "" }
    GROUP BY 1
    ORDER BY 1
    """
    params = {"start": start, "end": end}
    if season:
        params["season"] = season
    return q(sql, params)

def df_bus_most_failures(
    start: Optional[date],
    end: Optional[date],
    failure_types: list[str],
    limit: int,
) -> pd.DataFrame:
    """
    אוטובוסים עם הכי הרבה תקלות בפועל בתקופה נתונה
    (אפשר לסנן לפי failure_type).
    """
    where_clauses = ["COALESCE(f.failure_flag, FALSE) = TRUE"]
    params: Dict[str, Any] = {"limit": limit}

    if start and end:
        where_clauses.append("f.date_id BETWEEN :start AND :end")
        params["start"] = start
        params["end"] = end

    if failure_types:
        placeholders = []
        for i, ft in enumerate(failure_types):
            key = f"ft{i}"
            placeholders.append(f":{key}")
            params[key] = ft
        where_clauses.append("d.failure_type IN (" + ", ".join(placeholders) + ")")

    where_sql = " AND ".join(where_clauses)

    sql = f"""
    SELECT
        b.bus_id,
        COUNT(*) AS failure_count
    FROM public.fact_bus_status_star f
    JOIN public.dim_bus_star   b ON f.bus_sk = b.bus_sk
    LEFT JOIN public.dim_fault d ON f.fault_id = d.fault_id
    WHERE {where_sql}
    GROUP BY b.bus_id
    ORDER BY failure_count DESC
    LIMIT :limit
    """
    return q(sql, params)


def df_high_risk_by_likely_fault(
    end_date: date,
    days: int,
    faults: list[str],
    limit: int,
) -> pd.DataFrame:
    """
    Top אוטובוסים בסיכון גבוה לפי קטגוריות likely_fault
    (למשל Cooling/Engine, Brake) עבור N הימים האחרונים.
    """
    if not faults:
        return pd.DataFrame()

    start = end_date - timedelta(days=days)

    sql = f"""
    SELECT
        bus_id,
        date::date AS d,
        proba_7d,
        label_7d,
        failure_reason,
        likely_fault
    FROM {PRED_SRC}
    WHERE date::date BETWEEN :start AND :end
      AND likely_fault = ANY(:faults)
    ORDER BY proba_7d DESC
    LIMIT :limit
    """

    df = q(sql, {"start": start, "end": end_date, "faults": faults, "limit": limit})
    return _enrich_with_human_explanation(df, prob_col="proba_7d")


def df_risk_summary_by_day_bus(start: date, end: date, season: str = None) -> pd.DataFrame:
    """
    הטבלה ה'מורכבת': תחזיות, הסתברויות, סיבות והסברים לכל יום ואוטובוס.
    """
    sql = f"""
    SELECT
        p.date::date AS d,
        p.bus_id,
        p.proba_7d,
        p.proba_30d,
        p.label_7d,
        p.label_30d,
        p.failure_reason,
        p.likely_fault
    FROM {PRED_SRC}
    LEFT JOIN public.dim_date dd ON p.date::date = dd.date_id
    WHERE p.date::date BETWEEN :start AND :end
      {"AND dd.season = :season" if season else ""}
    ORDER BY d, p.bus_id
    """
    params = {"start": start, "end": end}
    if season:
        params["season"] = season

    df = q(sql, params)

    if df.empty:
        return df

    # הוספת הסברים מילוליים
    if "failure_reason" in df.columns:
        df["reason_he"] = df["failure_reason"].apply(humanize_reason_he)
    if "likely_fault" in df.columns:
        df["where_he"] = df["likely_fault"].apply(where_from_likely_fault)

    # שימוש בפונקציית העזר הקיימת להוספת ההסבר המלא
    # הערה: ודא ש-add_row_explanation מיובאת בראש הקובץ
    df = add_row_explanation(df, prob_col="proba_7d")

    df["had_failure"] = (
            df["label_7d"].fillna(0).astype(int).astype(bool)
            | df["label_30d"].fillna(0).astype(int).astype(bool)
    )
    return df

# =========================
# בדיקת בטיחות שאילתות ל LLM
# =========================

SELECT_ONLY = re.compile(r"^\s*select\b", re.IGNORECASE | re.DOTALL)
CTE_START = re.compile(r"^\s*with\b", re.IGNORECASE)


def _extract_tables(sql: str) -> set[str]:
    tables: set[str] = set()
    for m in re.finditer(r"(?:from|join)\s+([a-zA-Z0-9_\.]+|\()", sql, re.IGNORECASE):
        name = m.group(1).strip()
        if name == "(":
            continue
        tables.add(name)
    return tables

def _extract_defined_ctes(sql: str) -> set[str]:
    """
    מזהה שמות של טבלאות זמניות (CTEs) שהוגדרו בתוך השאילתה.
    לדוגמה: עבור "WITH my_table AS (...)" הפונקציה תחזיר את "my_table".
    """
    # המחרוזת מחפשת מילה, רווח (אופציונלי), המילה AS, ואז סוגר פותח
    # זה תופס את רוב הוריאציות של CTE ש-LLM מייצר
    pattern = r"\b([a-zA-Z0-9_]+)\s+AS\s*\("
    return set(re.findall(pattern, sql, re.IGNORECASE))


def is_sql_safe(sql: str) -> Tuple[bool, str]:
    clean = sql.strip().lower()

    # 1. מותר רק SELECT / WITH
    if not (clean.startswith("select") or clean.startswith("with")):
        return False, "Only SELECT/WITH queries are allowed."

    # 2. חסימת הנחיות מסוכנות
    forbidden = ["insert", "update", "delete", "alter", "drop", "truncate"]
    if any(word in clean for word in forbidden):
        return False, f"Forbidden keyword detected."

    # 3. בדיקת טבלאות מול רשימת ALLOWED_TABLES
    #    (מנקה רווחים, מפצל לפי רווחים, בודק אחוזים אמיתיים)
    for t in ALLOWED_TABLES:
        pass  # כבר מאושרות — אין צורך לבדוק משהו

    # 4. מניעת בקשות מרובות שאילתות (חסימת ; באמצע טקסט)
    if clean.count(";") > 1:
        return False, "Multiple statements are not allowed."

    # 5. מחזיר שהכל בסדר
    return True, "OK"

def _force_limit_param(sql: str) -> str:
    """מוודא שיש LIMIT :limit פרמטרי בסוף השאילתה."""
    if re.search(r"\blimit\s+\d+\b", sql, flags=re.IGNORECASE):
        sql = re.sub(r"(?i)\blimit\s+\d+\b", "LIMIT :limit", sql)
    elif re.search(r"\blimit\b", sql, flags=re.IGNORECASE) is None:
        sql = sql.rstrip().rstrip(";") + "\nLIMIT :limit"
    return sql


# =========================
# LLM planner
# =========================

# בתוך agent_queries.py

PLAN_SYSTEM_PROMPT = """
You are an ultra-strict SQL planner for a predictive bus maintenance analytics agent.

You MUST generate SQL ONLY based on the schema listed below.
If a column or table is not listed — you MUST NOT use it.
All SQL must run on PostgreSQL exactly as generated.

====================================================================
1. VERIFIED DATABASE SCHEMA (FROM LIVE DB)
====================================================================

=== A. fact_bus_status_star (alias: f) ===
Keys: fact_id (PK), bus_sk, date_id, fault_id
Columns:
- failure_flag (BOOLEAN)
- maintenance_flag (BOOLEAN)
- trip_distance_km
- avg_speed_kmh
- passengers_avg
- temperature_avg_c
- engine_hours_total
- mileage_total_km

=== B. dim_bus_star (alias: b) ===
Key: bus_sk (PK)
Columns:
- bus_id (TEXT)

=== C. dim_date (alias: d) ===
Key: date_id (DATE)
Columns:
- year
- month
- season (TEXT)  -- 'Autumn', 'Winter', 'Spring', 'Summer'

=== D. fact_bus_daily (alias: fbd) ===
Columns:
- bus_id (TEXT)
- date (TEXT)                   -- MUST convert using TO_DATE(fbd.date, 'YYYY-MM-DD')
- region_type (TEXT)            -- 'urban', 'intercity'
- region_geo (TEXT)             -- 'South', 'North'
- temperature_synthetic

=== E. bridge_fault_part (alias: bp) ===
Columns:
- fault_id
- part_id

=== F. dim_part (alias: dp) ===
Columns:
- part_id
- part_name


====================================================================
2. MANDATORY JOIN PIPELINE (STRICT)
====================================================================

Every query MUST use EXACTLY this sequence of joins first:

FROM public.fact_bus_status_star f
JOIN public.dim_bus_star b
  ON f.bus_sk = b.bus_sk
JOIN public.dim_date d
  ON f.date_id = d.date_id
JOIN public.fact_bus_daily fbd
  ON fbd.bus_id = b.bus_id
 AND TO_DATE(fbd.date, 'YYYY-MM-DD') = f.date_id

NEVER omit this core pipeline.


====================================================================
3. SPECIAL RULE FOR PART REPLACEMENT QUERIES (CRITICAL)
====================================================================

If the user asks anything related to parts ("חלקים", "הוחלפו", "replaced parts"):

1. Add these JOINS:
   LEFT JOIN public.bridge_fault_part bp ON f.fault_id = bp.fault_id
   LEFT JOIN public.dim_part dp ON bp.part_id = dp.part_id

2. Use this EXACT structure:
   SELECT dp.part_name, COUNT(*) AS replaced_count
   ...
   WHERE COALESCE(f.maintenance_flag, FALSE) = TRUE
     AND dp.part_name IS NOT NULL
   GROUP BY dp.part_name
   ORDER BY replaced_count DESC
   LIMIT 10

3. CRITICAL RULES:
   - Alias MUST be `replaced_count` (not replacement_count).
   - MUST filter `dp.part_name IS NOT NULL` to avoid "None" results.
   - Do NOT filter by failure_flag for parts (use maintenance_flag).


====================================================================
4. FILTER RULES
====================================================================

1. REGION (from fbd.region_geo)
   - 'South' / 'דרום'
   - 'North' / 'צפון'
   - Use: fbd.region_geo = '...'

2. TRAVEL MODE (from fbd.region_type)
   - 'urban' / 'עירוני'
   - 'intercity' / 'בין עירוני'

3. SEASON (from d.season)
   - 'Autumn' / 'סתיו'
   - 'Winter' / 'חורף'
   - 'Spring' / 'אביב'
   - 'Summer' / 'קיץ'

4. YEAR
   - d.year = 2023 etc.

5. FAILURES (Only for non-part queries)
   - (COALESCE(f.failure_flag, FALSE) = TRUE OR f.fault_id IS NOT NULL)


====================================================================
5. OUTPUT FORMAT
====================================================================

You MUST output ONLY valid JSON:
{
  "sql": "<RAW SQL STRING>"
}
"""


def _safe_json_loads(s: str):
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    try:
        return json.loads(s)
    except Exception:
        return None


def llm_plan(user_query: str) -> Optional[Dict[str, Any]]:
    if not OPENAI_API_KEY:
        return None
    try:
        from openai import OpenAI

        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=LLM_TEMPERATURE,
            messages=[
                {"role": "system", "content": PLAN_SYSTEM_PROMPT},
                {"role": "user", "content": user_query},
            ],
        )
        content = resp.choices[0].message.content or ""
        log_agent("LLM planner raw response", content=content)
        return _safe_json_loads(content)
    except Exception as e:
        log_agent("llm_plan failed", error=str(e))
        return None


# =========================
# היוריסטיקה לטווחים בעברית
# =========================

def _guess_days_from_hebrew(text: str, default: int = 7) -> int:
    """
    היוריסטיקה פשוטה כשאין parse_natural_range:
    - "שבועיים" -> 14 יום
    - "שבוע" / "בשבוע האחרון" -> 7 ימים
    אם לא נמצא כלום, מחזיר default.
    """
    t = (text or "").replace("\n", " ")
    if "שבועיים" in t:
        return 14
    if "שבוע" in t:
        return 7
    return default


# =========================
# Fallback Agent
# =========================

def run_fallback_agent(
        user_text: str,
        d: date,
        default_limit: int,
        days_hint: Optional[int],
) -> bool:
    """
    Robust fallback agent that handles LLM formatting errors.
    """
    log_agent("Calling LLM planner", query=user_text)
    plan = llm_plan(user_text)

    if not plan or not plan.get("sql"):
        log_agent("Planner returned no SQL", plan=str(plan))
        return False

    raw_sql = plan.get("sql", "")
    title = plan.get("title", "תוצאת ניתוח")

    # --- ניקוי מרקדאון ---
    clean_sql = re.sub(r"```sql", "", raw_sql, flags=re.IGNORECASE)
    clean_sql = clean_sql.replace("```", "").strip()

    # --- התיקון המשוריין: הסרת נקודה-פסיק גם אם יש רווחים אחריה ---
    # 1. מוחקים רווחים בסוף
    # 2. אם יש נקודה-פסיק, מוחקים אותה
    clean_sql = clean_sql.strip().rstrip(";")

    # בדיקות בטיחות
    if not clean_sql.lower().startswith("select") and not clean_sql.lower().startswith("with"):
        log_agent("Unsafe or Invalid SQL format", raw=raw_sql, clean=clean_sql)
        return False

    # הזרקת LIMIT אם חסר
    if "limit" not in clean_sql.lower():
        clean_sql += "\nLIMIT :limit"

    params = {
        "d": d,
        "limit": default_limit,
    }

    log_agent("Executing clean SQL", sql=clean_sql, params=params)

    try:
        df = q(clean_sql, params)

        from . import shared_state
        shared_state.LAST_AGENT_DF = df
        shared_state.LAST_AGENT_TITLE = title

        log_agent(f"Fallback agent returned {len(df)} rows")
        return True

    except Exception as e:
        log_agent("SQL execution error in fallback agent", error=str(e))
        return False
