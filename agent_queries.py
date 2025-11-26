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
ALLOWED_TABLES = allowed_tables_from_schema()

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
    s = sql.strip()
    low = s.lower()

    # בדיקה בסיסית: חייב להתחיל ב-SELECT או WITH
    if not (SELECT_ONLY.match(low) or CTE_START.match(low)):
        return False, "Only SELECT (or CTE starting with WITH) is allowed."

    forbidden = [
        "insert", "update", "delete", "drop", "alter", "truncate",
        "create", "grant", "revoke", "copy", "vacuum",
    ]
    if any(re.search(rf"\b{kw}\b", low) for kw in forbidden):
        return False, "Write or DDL keywords are not allowed."

    # 1. שליפת כל הטבלאות שהשאילתה מנסה לגשת אליהן
    used_tables = _extract_tables(low)

    # 2. שליפת השמות שהוגדרו זמנית בתוך השאילתה (CTE)
    defined_ctes = _extract_defined_ctes(low)

    # 3. סינון: אנו בודקים רק טבלאות שלא הוגדרו כ-CTE
    tables_to_validate = used_tables - defined_ctes

    for t in tables_to_validate:
        if t in ("p",):  # התעלמות מ-alias ידועים נוספים אם יש
            continue

        # ניקוי סכמה כפולה אם המשתמש כתב public.public.table (קורה לפעמים עם LLM)
        t_clean = t.replace("public.", "")
        t_norm = f"public.{t_clean}"  # תמיד מנרמלים ל-public בשביל הבדיקה

        # בדיקה מול הרשימה המאושרת
        if t_norm not in ALLOWED_TABLES:
            return False, f"Table {t_norm} is not in allowed tables (detected CTEs: {defined_ctes})."

    return True, ""

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
Keys: fact_id (PK), bus_sk, date_id
Columns:
- fault_id
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
- quarter
- day
- day_of_week
- dow_name
- season (TEXT)  -- 'Autumn', 'Winter', 'Spring', 'Summer'

=== D. fact_bus_daily (alias: fbd) ===
(Verified from information_schema.columns)
Columns:
- bus_id (TEXT)
- date (TEXT)                   -- MUST convert using TO_DATE(fbd.date, 'YYYY-MM-DD')
- region_type (TEXT)            -- Travel mode: 'urban', 'intercity'
- trip_distance_km
- avg_speed_kmh
- passengers_avg
- temperature_avg_c
- engine_hours_total
- mileage_total_km
- failure_type
- maintenance_date
- failure_flag
- maintenance_flag
- season
- region_geo (TEXT)             -- Geography: 'South', 'North'
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
SPECIAL RULE FOR PART REPLACEMENT QUERIES
====================================================================

If the user asks any question about:
- "איזה חלקים הוחלפו"
- "most replaced parts"
- "parts replaced"
- "מה הוחלף"
- "parts failures"

You MUST use the following join sequence IN ADDITION to the mandatory joins:

LEFT JOIN public.bridge_fault_part bp
  ON f.fault_id = bp.fault_id

LEFT JOIN public.dim_part dp
  ON bp.part_id = dp.part_id

And when grouping:
- ALWAYS group by dp.part_name
- NEVER group only by fault_id

====================================================================
2. MANDATORY JOIN PIPELINE (NEVER MODIFY)
====================================================================

Always use EXACTLY this join sequence:

FROM public.fact_bus_status_star f
JOIN public.dim_bus_star b
  ON f.bus_sk = b.bus_sk
JOIN public.dim_date d
  ON f.date_id = d.date_id
JOIN public.fact_bus_daily fbd
  ON fbd.bus_id = b.bus_id
 AND TO_DATE(fbd.date, 'YYYY-MM-DD') = f.date_id

NEVER omit the TO_DATE conversion.
NEVER join different tables unless explicitly required.

====================================================================
3. CANONICAL VALUE MAPPINGS (STRICT)
====================================================================

=== A. GEOGRAPHY (from fbd.region_geo) ===
Valid values:
- 'South'
- 'North'

User intent mapping:
- "דרום" / "south" → fbd.region_geo = 'South'
- "צפון" / "north" → fbd.region_geo = 'North'

=== B. TRAVEL MODE (from fbd.region_type) ===
Valid values:
- 'urban'
- 'intercity'

User intent:
- "עירוני" / "urban" → fbd.region_type = 'urban'
- "בין עירוני" / "intercity" → fbd.region_type = 'intercity'

=== C. SEASONS (from dim_date.season) ===
Valid values:
- 'Autumn'
- 'Winter'
- 'Spring'
- 'Summer'

=== D. FAILURE DEFINITION ===
A failure is:
COALESCE(f.failure_flag, FALSE) = TRUE OR f.fault_id IS NOT NULL

====================================================================
4. SQL PLANNING RULES
====================================================================

1. If user specifies REGION → filter on fbd.region_geo.
2. If user specifies TRAVEL MODE → filter on fbd.region_type.
3. If user specifies SEASON → use d.season = '<value>'.
4. If user specifies YEAR → use d.year = <value>.
5. COUNT failures ONLY using the FAILURE DEFINITION above.
6. NEVER infer date ranges unless user explicitly asks.
7. NEVER guess columns that aren’t in this document.

====================================================================
5. OUTPUT FORMAT REQUIREMENTS
====================================================================

You MUST output ONLY valid JSON in the following format:

{
  "sql": "<RAW SQL STRING>"
}

NO comments.
NO explanations.
NO markdown.
SQL string only.

====================================================================
6. SAFETY RULES
====================================================================

- NEVER invent columns.
- NEVER use region_type instead of region_geo.
- NEVER use date directly — always TO_DATE(fbd.date, 'YYYY-MM-DD').
- NEVER remove any of the mandatory joins.
- NEVER join predictions tables unless explicitly requested.
- SQL must be runnable exactly as-is.

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

def run_fallback_agent(sql: str, params: dict, engine, logger=None):
    """
    Safe fallback agent:
    - Ensures SELECT only
    - Removes trailing semicolon
    - Adds LIMIT safely
    - Handles missing logger gracefully
    - Executes SQL safely and returns DataFrame
    """
    try:
        # ---------------------------------------------------
        # 1. הבטחת לוגר תקין
        # ---------------------------------------------------
        if logger is None:
            class DummyLogger:
                def info(self, *a, **k): pass
                def error(self, *a, **k): pass
            logger = DummyLogger()

        clean_sql = (sql or "").strip()
        lower_sql = clean_sql.lower()

        # ---------------------------------------------------
        # 2. מוודא שהשאילתה היא SELECT בלבד
        # ---------------------------------------------------
        if not (lower_sql.startswith("select") or lower_sql.startswith("with")):
            raise ValueError("Fallback agent: only SELECT queries are allowed.")

        # ---------------------------------------------------
        # 3. מסיר נקודה פסיק בסוף
        # ---------------------------------------------------
        if clean_sql.endswith(";"):
            clean_sql = clean_sql[:-1].strip()

        # ---------------------------------------------------
        # 4. מוסיף LIMIT :limit אם אין
        # ---------------------------------------------------
        if "limit" not in lower_sql:
            clean_sql += "\nLIMIT :limit"

        logger.info("[FALLBACK] Executing SQL:\n%s", clean_sql)
        logger.info("[FALLBACK] Params: %s", params)

        # ---------------------------------------------------
        # 5. הרצת SQL בצורה בטוחה
        # ---------------------------------------------------
        from sqlalchemy import text
        import pandas as pd

        with engine.connect() as conn:
            result = conn.execute(text(clean_sql), params)
            rows = result.fetchall()
            cols = result.keys()

        df = pd.DataFrame(rows, columns=cols)

        logger.info("[FALLBACK] Query OK. rows=%d", len(df))
        return df

    except Exception as e:
        logger.error("[FALLBACK ERROR] %s", e, exc_info=True)
        raise

