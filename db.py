
from __future__ import annotations
from typing import Any, Dict, Set
from sqlalchemy import create_engine, text
import pandas as pd

from .config import PG_HOST, PG_PORT, PG_DB, PG_USER, PG_PASS
from .utils_logging import log_sql, log_agent

CONN_URL = f"postgresql+psycopg2://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}/{PG_DB}"
engine   = create_engine(CONN_URL, pool_pre_ping=True)


def q(sql: str, params: Dict[str, Any]) -> pd.DataFrame:
    log_sql("Running query", sql, params)
    with engine.begin() as conn:
        df = pd.read_sql(text(sql), conn, params=params)
    log_sql(f"Success. rows={len(df)}")
    return df


def list_existing_tables(schema: str = "public") -> Set[str]:
    sql = """
        SELECT table_schema||'.'||table_name AS fqtn
        FROM information_schema.tables
        WHERE table_schema = :schema
    """
    df = q(sql, {"schema": schema})
    return set(df["fqtn"].tolist())


def resolve_predictions_source() -> str:
    candidates = [
        "public.predictions_for_powerbi",
        "public.vw_preds_costed_final",
        "public.ml_predictions_daily_physical",
        "public.ml_predictions_daily",
    ]
    existing = list_existing_tables()
    chosen = None
    for tab in candidates:
        if tab in existing:
            chosen = tab
            break
    if not chosen:
        chosen = "public.predictions_for_powerbi"

    log_agent("Using predictions source table", table=chosen)

    t = chosen.lower()
    if t.endswith("predictions_for_powerbi"):
        sub = f"""
        (SELECT
            bus_id,
            (date)::date AS date,
            proba_7d      AS proba_7d,
            label_7d      AS label_7d,
            proba_30d     AS proba_30d,
            label_30d     AS label_30d,
            failure_reason,
            likely_fault
         FROM {chosen})"""
    elif t.endswith("vw_preds_costed_final"):
        sub = f"""
        (SELECT
            bus_id,
            date::date      AS date,
            predicted_proba AS proba_7d,
            predicted_label AS label_7d,
            NULL::double precision AS proba_30d,
            NULL::smallint         AS label_30d,
            NULL::text             AS failure_reason,
            failure_type_canon     AS likely_fault
         FROM {chosen})"""
    elif t.endswith("ml_predictions_daily_physical") or t.endswith("ml_predictions_daily"):
        sub = f"""
        (SELECT
            bus_id,
            date::date      AS date,
            predicted_proba AS proba_7d,
            predicted_label AS label_7d,
            NULL::double precision AS proba_30d,
            NULL::smallint         AS label_30d,
            NULL::text             AS failure_reason,
            NULL::text             AS likely_fault
         FROM {chosen})"""
    else:
        sub = """(SELECT NULL::text AS bus_id, NULL::date AS date, NULL::double precision AS proba_7d,
                           NULL::smallint AS label_7d, NULL::double precision AS proba_30d,
                           NULL::smallint AS label_30d, NULL::text AS failure_reason,
                           NULL::text AS likely_fault WHERE 1=0)"""

    return sub + " AS p"
