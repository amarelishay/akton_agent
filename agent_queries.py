from __future__ import annotations

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
    df = q(SQL_AT_RISK_TODAY, {"d": d, "limit": limit})
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


def df_failures_by_day_detail(start: date, end: date) -> pd.DataFrame:
    """
    מחזיר לכל יום אילו אוטובוסים חוו תקלות בפועל ומה סוג התקלה.
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
    WHERE f.date_id BETWEEN :start AND :end
      AND (COALESCE(f.failure_flag, FALSE) = TRUE OR f.fault_id IS NOT NULL)
    ORDER BY d, b.bus_id
    """
    return q(sql, {"start": start, "end": end})


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


def df_trend_last_days(start: date, end: date) -> pd.DataFrame:
    return q(SQL_TREND_LAST_DAYS, {"start": start, "end": end})


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


def df_risk_summary_by_day_bus(start: date, end: date) -> pd.DataFrame:
    """
    סיכום תחזיות לפי יום ואוטובוס:
    תאריך, אוטובוס, proba_7d, proba_30d, label_7d/30d, הסבר בעברית.
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
    WHERE p.date::date BETWEEN :start AND :end
    ORDER BY d, p.bus_id
    """
    df = q(sql, {"start": start, "end": end})

    if df.empty:
        return df

    df = _enrich_with_human_explanation(df, prob_col="proba_7d")

    # דגל אם בפועל היתה תקלה באחד האופקים
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


def is_sql_safe(sql: str) -> Tuple[bool, str]:
    s = sql.strip()
    low = s.lower()

    if not (SELECT_ONLY.match(low) or CTE_START.match(low)):
        return False, "Only SELECT (or CTE starting with WITH) is allowed."

    forbidden = [
        "insert",
        "update",
        "delete",
        "drop",
        "alter",
        "truncate",
        "create",
        "grant",
        "revoke",
        "copy",
        "vacuum",
    ]
    if any(re.search(rf"\b{kw}\b", low) for kw in forbidden):
        return False, "Write or DDL keywords are not allowed."

    used = _extract_tables(low)
    for t in used:
        if t in ("p",):
            continue
        t_norm = t if "." in t else f"public.{t}"
        if t_norm not in ALLOWED_TABLES:
            return False, f"Table {t_norm} is not in allowed tables."

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

PLAN_SYSTEM_PROMPT = f"""
You are a cautious SQL planner for a predictive bus maintenance app.

You MUST base all queries ONLY on the schema listed below.
If the schema does not contain the requested data, you MUST NOT invent columns or tables.

=== Logical data model (PostgreSQL) ===

Key tables:

1. public.predictions_for_powerbi
   - One row per (bus_id, date).
   - Columns:
     - bus_id (text)
     - date (text, castable to date)
     - proba_7d, label_7d  –  7-day failure prediction
     - proba_30d, label_30d – 30-day failure prediction
     - failure_reason (text)
     - likely_fault (text)
   - Use this when the user asks about predicted risk / probability / labels.

2. public.dim_bus_star
   - One row per bus.
   - Columns:
     - bus_sk (PK, bigint)
     - bus_id (text)
     - avg_daily_distance (double precision)
     - avg_daily_speed (double precision)
     - avg_passengers (double precision)
     - max_engine_hours (double precision)
     - max_mileage_total (bigint)  -- total lifetime mileage per bus
     - total_failures (bigint)
   - Use this when the user asks about:
     - “buses with the highest mileage / קילומטראז' הכי גבוה”
     - “buses with most failures overall”
     - distribution of bus-level features.

3. public.fact_bus_status_star
   - One row per bus per date.
   - Columns:
     - fact_id (PK)
     - bus_sk (FK -> dim_bus_star.bus_sk)
     - date_id (date, FK -> dim_date.date_id)
     - fault_id (FK -> dim_fault.fault_id, may be NULL)
     - trip_distance_km (double precision)   -- distance for that day
     - mileage_total_km (bigint)             -- cumulative mileage up to that day
     - avg_speed_kmh, passengers_avg, temperature_avg_c, engine_hours_total
     - maintenance_flag (boolean)
     - failure_flag (boolean)
   - Use this for:
     - daily distance / mileage by date
     - failures and maintenance over time
     - aggregations by date range.

4. public.fact_bus_daily
   - One row per (bus_id, date) in a denormalized raw form.
   - Columns:
     - bus_id (text)
     - date (text)
     - trip_distance_km, avg_speed_kmh, passengers_avg
     - engine_hours_total, mileage_total_km
     - failure_type, maintenance_date, failure_flag, maintenance_flag, season
   - You may use this instead of fact_bus_status_star when bus_id (not bus_sk) is more convenient.

5. public.dim_fault
   - Fault/failure dimension:
     - fault_id (PK)
     - failure_type (text)
     - fault_category (text)
     - severity (text)
     - cost metrics (avg_repair_cost_usd, avg_labor_hours, avg_parts_cost_usd, avg_repair_frequency_per_10k_km)

6. public.dim_part, public.bridge_fault_part
   - Parts and mapping between faults and parts.

7. public.dim_date
   - Date dimension (date_id, year, month, day, season, etc.)

8. public.vw_preds_costed_final, public.ml_costed_predictions(_physical), public.ml_predictions_daily(_physical)
   - Additional prediction / cost outputs. Only use columns that exist in the schema:
     bus_id, date, predicted_proba, predicted_label, failure_type, expected_cost_usd, etc.

=== VERY IMPORTANT RULES ===

- You MUST NOT invent columns or tables.
  - Example of valid mileage columns:
    - dim_bus_star.max_mileage_total       (total lifetime mileage per bus)
    - fact_bus_status_star.trip_distance_km, fact_bus_status_star.mileage_total_km
    - fact_bus_daily.trip_distance_km, fact_bus_daily.mileage_total_km
  - DO NOT use columns like "mileage" or "distance" if they are not exactly present in the schema.

- When the user asks:
  - “Top buses by highest mileage / הקילומטראז' הכי גבוה”:
    Prefer:
      SELECT bus_id, max_mileage_total AS total_mileage_km
      FROM public.dim_bus_star
      ORDER BY max_mileage_total DESC
      LIMIT :limit;

  - “Daily distance / mileage over a period”:
    You may use:
      public.fact_bus_status_star (join dim_bus_star for bus_id if needed)
      or public.fact_bus_daily (directly by bus_id).

- When working with predictions:
  - Always use the unified predictions subselect aliased as p (injected via {schema_summary_for_llm()} and PRED_SRC).
  - Typical columns: p.bus_id, p.date, p.proba_7d, p.label_7d, p.proba_30d, p.label_30d, p.failure_reason, p.likely_fault.

=== SQL Constraints ===

- Output ONLY a JSON object with keys:
  action, sql, needs_date, needs_range, needs_limit, title.
- The SQL MUST be a read-only PostgreSQL SELECT (CTE allowed), ending with LIMIT (it will be replaced with :limit).
- For “today” filters, use: p.date::date = :d or date_id = :d where appropriate.
- For date ranges, use BETWEEN :start AND :end on date/date_id.
- Always fully qualify tables with schema when possible (public.table_name).

Return JSON ONLY, no markdown fences.
""".strip()


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
    סוכן פולבק:
    טקסט חופשי -> תכנון שאילתה עם LLM -> בדיקות בטיחות -> הרצת SQL.
    התוצאה נשמרת ב shared_state.LAST_AGENT_DF / LAST_AGENT_TITLE.
    """
    # ננסה קודם להבין טווח טבעי (למשל "ב 17 הימים האחרונים")
    nat_rng = parse_natural_range(user_text, d)

    log_agent("Calling LLM planner", query=user_text)
    plan = llm_plan(user_text)
    if not plan or not plan.get("sql"):
        log_agent("Planner returned no SQL", plan=str(plan))
        return False

    sql = plan.get("sql", "")

    # 1. מחיקה של CTE בעייתי מהסוג:
    #    WITH p AS (SELECT * FROM vw_preds_costed_final) ...
    sql = re.sub(
        r"^\s*WITH\s+p\s+AS\s*\(SELECT\s+\*\s+FROM\s+vw_preds_costed_final\s*\)\s*",
        "",
        sql,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # 2. נורמליזציה של טבלת התחזיות
    sql = re.sub(r"\bvw_preds_costed_final\b", PRED_SRC, sql, flags=re.IGNORECASE)
    sql = sql.replace("FROM public.predictions_for_powerbi p", f"FROM {PRED_SRC}")
    sql = sql.replace("FROM predictions_for_powerbi p", f"FROM {PRED_SRC}")

    # מקרים שבהם ה LLM שם סתם "FROM p"
    if " FROM p" in sql or re.search(r"\bfrom\s+p\b", sql, re.IGNORECASE):
        sql = re.sub(r"\bfrom\s+p\b", f"FROM {PRED_SRC}", sql, flags=re.IGNORECASE)

    # 3. אם צריך טווח, נעגן את CURRENT_DATE לתאריך הסימולציה (:d)
    if plan.get("needs_range") and "CURRENT_DATE" in sql:
        sql = sql.replace("CURRENT_DATE", ":d")

    # 4. נורמליזציה של שמות עמודות
    replacements = {
        "p.predicted_proba": "p.proba_7d",
        "predicted_proba": "proba_7d",
        "p.predicted_label": "p.label_7d",
        "predicted_label": "label_7d",
        "p.failure_type_canon": "p.failure_reason",
        "failure_type_canon": "failure_reason",
    }
    for src, dst in replacements.items():
        sql = sql.replace(src, dst)

    # 5. הזרקת LIMIT פרמטרי
    sql = _force_limit_param(sql)

    ok, reason = is_sql_safe(sql)
    if not ok:
        log_agent("Unsafe SQL from planner", reason=reason, sql=sql)
        return False

    # ---------------- פרמטרים ----------------
    params: Dict[str, Any] = {}

    # טווח תאריכים
    needs_range = bool(plan.get("needs_range", False)) or (":start" in sql and ":end" in sql)
    if needs_range:
        if nat_rng:
            start, end, _ = nat_rng
        else:
            # אם אין טווח טבעי מפורש, ננסה:
            # 1. days_hint שמגיע מה intents
            # 2. היוריסטיקה למילים "שבועיים" / "שבוע"
            base_days = days_hint or _guess_days_from_hebrew(user_text, default=7)
            start, end = _range_last_n(d, base_days, "days")

        params["start"] = start
        params["end"] = end

    # תאריך סימולציה יחיד
    if bool(plan.get("needs_date", False)) or (":d" in sql):
        params["d"] = d

    # LIMIT:
    #   ברירת מחדל – מה sidebar
    #   אבל אם זו שאילתת טווח (שבוע/שבועיים/last days) נגדיל ל 500 לפחות.
    effective_limit = default_limit
    if needs_range:
        txt = user_text or ""
        if (
            "שבוע" in txt  # כולל "שבועיים" ו"שבוע האחרון"
            or "שבועיים" in txt
            or re.search(r"\b(last|past)\b", txt, re.IGNORECASE)
            or "ימים האחרונים" in txt
        ):
            effective_limit = max(default_limit, 500)

    params["limit"] = max(1, min(2000, effective_limit))

    log_agent("Final SQL plan", sql=sql, params=params)

    # ---------------- הרצת השאילתה ----------------
    try:
        df = q(sql, params)
    except Exception as e:
        log_agent("SQL error in fallback agent", error=str(e))
        # נחזיר True כדי שהשכבה העליונה תוכל להסביר שהבקשה נכשלה
        return True

    # ---------------- פוסט פרוססינג ----------------
    if "failure_reason" in df.columns:
        df["reason_he"] = df["failure_reason"].apply(humanize_reason_he)
    if "likely_fault" in df.columns:
        df["where_he"] = df["likely_fault"].apply(where_from_likely_fault)

    prob_col = None
    if "proba_7d" in df.columns:
        prob_col = "proba_7d"
    elif "predicted_proba" in df.columns:
        prob_col = "predicted_proba"

    if prob_col:
        df = add_row_explanation(df, prob_col=prob_col)

    from .utils_logging import logger
    logger.info("Fallback agent returned %d rows", len(df))

    from . import shared_state
    shared_state.LAST_AGENT_DF = df
    shared_state.LAST_AGENT_TITLE = plan.get("title") or "תוצאה (Agent)"

    return True
