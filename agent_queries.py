import re

from __future__ import annotations
from datetime import date
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from .db import q, resolve_predictions_source
from .humanize import humanize_reason_he, where_from_likely_fault, add_row_explanation
from .failure_mapping import map_failure_types_from_query
from .time_range import _range_last_n
from .utils_logging import log_agent
from .schema_meta import allowed_tables_from_schema, schema_summary_for_llm
from .config import OPENAI_MODEL, LLM_TEMPERATURE, resolve_openai_key

OPENAI_API_KEY = resolve_openai_key()
PRED_SRC = resolve_predictions_source()
ALLOWED_TABLES = allowed_tables_from_schema()

SQL_AT_RISK_TODAY = f"""
SELECT p.bus_id, p.date::date AS d, p.proba_7d AS predicted_proba, p.label_7d AS predicted_label,
       p.failure_reason, p.likely_fault
FROM {PRED_SRC}
WHERE p.date::date = :d AND p.proba_7d >= 0.5
ORDER BY p.proba_7d DESC NULLS LAST
LIMIT :limit
"""

SQL_AT_RISK_TOP1 = f"""
SELECT p.bus_id, p.date::date AS d, p.proba_7d AS predicted_proba, p.label_7d AS predicted_label,
       p.failure_reason, p.likely_fault
FROM {PRED_SRC}
WHERE p.date::date = :d
ORDER BY p.proba_7d DESC NULLS LAST
LIMIT 1
"""

SQL_BUS_TODAY = f"""
SELECT p.bus_id, p.date::date AS d,
       p.proba_7d  AS proba_7d,  p.label_7d  AS label_7d,
       p.proba_30d AS proba_30d, p.label_30d AS label_30d,
       p.failure_reason, p.likely_fault
FROM {PRED_SRC}
WHERE p.date::date = :d AND p.bus_id = :bus
ORDER BY p.date DESC
LIMIT 1
"""

SQL_TREND_LAST_DAYS = f"""
SELECT p.date::date AS d,
       COUNT(*) FILTER (WHERE p.proba_7d >= 0.5) AS at_risk,
       AVG(p.proba_7d)  AS avg_proba,
       100.0*COUNT(*) FILTER (WHERE p.proba_7d >= 0.5)/NULLIF(COUNT(*),0) AS pct_risk,
       COUNT(*) AS total_buses
FROM {PRED_SRC}
WHERE p.date::date BETWEEN :start AND :end
GROUP BY 1
ORDER BY 1
"""

SQL_PARTS_REPLACED_LAST_30D = """
WITH range AS (
  SELECT CAST(:start AS date) AS start_d, CAST(:end AS date) AS end_d
)
SELECT dp.part_name, COUNT(*) AS replaced_count
FROM public.fact_bus_status_star f
JOIN range r ON TRUE
LEFT JOIN public.bridge_fault_part b ON f.fault_id = b.fault_id
LEFT JOIN public.dim_part dp        ON b.part_id  = dp.part_id
WHERE (f.date_id::date) BETWEEN r.start_d AND r.end_d
  AND COALESCE(f.maintenance_flag, false) = true
GROUP BY dp.part_name
ORDER BY replaced_count DESC NULLS LAST
LIMIT :limit
"""


def df_at_risk_today(d: date, limit: int) -> pd.DataFrame:
    df = q(SQL_AT_RISK_TODAY, {"d": d, "limit": limit})
    if not df.empty:
        if "failure_reason" in df.columns:
            df["reason_he"] = df["failure_reason"].apply(humanize_reason_he)
        if "likely_fault" in df.columns:
            df["where_he"] = df["likely_fault"].apply(where_from_likely_fault)
        df = add_row_explanation(df, prob_col="predicted_proba")
    return df


def df_at_risk_top1(d: date) -> pd.DataFrame:
    df = q(SQL_AT_RISK_TOP1, {"d": d})
    if not df.empty:
        if "failure_reason" in df.columns:
            df["reason_he"] = df["failure_reason"].apply(humanize_reason_he)
        if "likely_fault" in df.columns:
            df["where_he"] = df["likely_fault"].apply(where_from_likely_fault)
        df = add_row_explanation(df, prob_col="predicted_proba")
    return df


def df_bus_today(d: date, bus_id: str) -> pd.DataFrame:
    df = q(SQL_BUS_TODAY, {"d": d, "bus": bus_id})
    if not df.empty:
        if "failure_reason" in df.columns:
            df["reason_he"] = df["failure_reason"].apply(humanize_reason_he)
        if "likely_fault" in df.columns:
            df["where_he"] = df["likely_fault"].apply(where_from_likely_fault)
        df = add_row_explanation(df, prob_col="proba_7d")
    return df


def df_parts_replaced_last_30d(end_date: date, limit: int) -> pd.DataFrame:
    start, end = _range_last_n(end_date, 30, "days")
    return q(SQL_PARTS_REPLACED_LAST_30D, {"start": start, "end": end, "limit": limit})


def df_trend_last_days(start: date, end: date) -> pd.DataFrame:
    return q(SQL_TREND_LAST_DAYS, {"start": start, "end": end})


def df_bus_most_failures(
    start: Optional[date],
    end: Optional[date],
    failure_types: list[str],
    limit: int,
) -> pd.DataFrame:
    where_clauses = ["COALESCE(f.failure_flag, FALSE) = TRUE"]
    params: Dict[str, Any] = {"limit": limit}

    if start and end:
        where_clauses.append("f.date_id BETWEEN :start AND :end")
        params["start"] = start
        params["end"] = end

    if failure_types:
        ph_list = []
        for i, ft in enumerate(failure_types):
            key = f"ft{i}"
            ph_list.append(f":{key}")
            params[key] = ft
        where_clauses.append("d.failure_type IN (" + ", ".join(ph_list) + ")")

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
    """"

    return q(sql, params)


# =========== Fallback Agent (general SQL) ===========

SELECT_ONLY = re.compile(r"^\s*select\b", re.IGNORECASE | re.DOTALL)
CTE_START   = re.compile(r"^\s*with\b",   re.IGNORECASE)


def _extract_tables(sql: str) -> set[str]:
    import re
    tables: set[str] = set()
    for m in re.finditer(r"(?:from|join)\s+([a-zA-Z0-9_\.]+|\()", sql, re.IGNORECASE):
        name = m.group(1).strip()
        if name == "(":
            continue
        tables.add(name)
    return tables


def is_sql_safe(sql: str) -> tuple[bool, str]:
    import re
    s = sql.strip()
    low = s.lower()
    if not (SELECT_ONLY.match(low) or CTE_START.match(low)):
        return False, "Only SELECT/CTE SELECT are allowed."
    forbidden = ["insert", "update", "delete", "drop", "alter", "truncate", "create", "grant", "revoke", "copy", "vacuum"]
    if any(re.search(rf"\b{kw}\b", low) for kw in forbidden):
        return False, "Write/DDL keywords are not allowed."
    used = _extract_tables(low)
    for t in used:
        if t in ("p",):
            continue
        t_norm = t if "." in t else f"public.{t}"
        if t_norm not in ALLOWED_TABLES:
            return False, f"Table {t} not allowed."
    return True, ""


def _force_limit_param(sql: str) -> str:
    import re
    if re.search(r"\blimit\s+\d+\b", sql, flags=re.IGNORECASE):
        sql = re.sub(r"(?i)\blimit\s+\d+\b", "LIMIT :limit", sql)
    elif re.search(r"\blimit\b", sql, flags=re.IGNORECASE) is None:
        sql = sql.rstrip().rstrip(";") + "\nLIMIT :limit"
    return sql


PLAN_SYSTEM_PROMPT = f"""You are a cautious SQL planner for a predictive bus-maintenance app.

Database schema (PostgreSQL):
{schema_summary_for_llm()}

Constraints:
- Output ONLY a JSON object with keys: action, sql, needs_date, needs_range, needs_limit, title.
- The SQL MUST be a read-only PostgreSQL SELECT (CTE allowed), ending with LIMIT (it will be replaced to :limit).
- Always select FROM the unified subselect aliased as p when using predictions (it already normalizes prediction tables),
  or join allowed tables explicitly.
Filters:
- "Today": add WHERE p.date::date = :d
- "Last-X-days": add WHERE p.date::date BETWEEN :start AND :end
Columns:
- Prefer: p.bus_id, p.date::date AS d, p.proba_7d AS predicted_proba, p.label_7d AS predicted_label, p.failure_reason, p.likely_fault
Return JSON ONLY, no markdown fences."""


def _safe_json_loads(s: str):
    import json, re
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
            temperature=0.2,
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


def run_fallback_agent(user_text: str, d: date, default_limit: int, days_hint: Optional[int]) -> bool:
    from .time_range import parse_natural_range
    import re

    nat_rng = parse_natural_range(user_text, d)
    log_agent("Calling LLM planner", query=user_text)
    plan = llm_plan(user_text)
    if not plan or plan.get("action") != "sql":
        return False

    sql = (plan.get("sql", "")
           .replace("FROM public.predictions_for_powerbi p", f"FROM {PRED_SRC}")
           .replace("FROM predictions_for_powerbi p", f"FROM {PRED_SRC}"))
    if " FROM p" in sql or re.search(r"\bfrom\s+p\b", sql, re.IGNORECASE):
        sql = re.sub(r"\bfrom\s+p\b", f"FROM {PRED_SRC}", sql, flags=re.IGNORECASE)

    sql = _force_limit_param(sql)
    ok, reason = is_sql_safe(sql)
    if not ok:
        log_agent("Unsafe SQL from planner", reason=reason, sql=sql)
        return False

    params: Dict[str, Any] = {}
    if bool(plan.get("needs_range", False)) or (":start" in sql and ":end" in sql):
        if nat_rng:
            start, end, _ = nat_rng
        else:
            days = days_hint or 7
            start, end = _range_last_n(d, days, "days")
        params.update({"start": start, "end": end})
    if bool(plan.get("needs_date", False)) or (":d" in sql):
        params["d"] = d
    params["limit"] = max(1, min(200, default_limit))

    log_agent("Final SQL plan", sql=sql, params=params)

    try:
        df = q(sql, params)
    except Exception as e:
        log_agent("SQL error in fallback agent", error=str(e))
        return True

    if "failure_reason" in df.columns:
        df["reason_he"] = df["failure_reason"].apply(humanize_reason_he)
    if "likely_fault" in df.columns:
        df["where_he"] = df["likely_fault"].apply(where_from_likely_fault)
    prob_col = "proba_7d" if "proba_7d" in df.columns else ("predicted_proba" if "predicted_proba" in df.columns else None)
    if prob_col:
        df = add_row_explanation(df, prob_col=prob_col)

    # החזרה של ה-DataFrame תטופל ע"י השכבה העליונה (app_streamlit)
    from .utils_logging import logger
    logger.info("Fallback agent returned %d rows", len(df))
    from . import shared_state
    shared_state.LAST_AGENT_DF = df
    shared_state.LAST_AGENT_TITLE = plan.get("title") or "תוצאה (Agent)"
    return True
