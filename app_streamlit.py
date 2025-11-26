from __future__ import annotations

import uuid
import datetime as dt
from datetime import date, timedelta
from typing import Any, Tuple, Optional
import re as _re2

import pandas as pd
import streamlit as st
from sqlalchemy import text  # <--- תוספת קריטית להרצת שאילתות ישירות
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

# --- ייבוא המודולים הפנימיים ---
from .db import engine, q  # <--- הייבוא המתוקן של המנוע
from .config import resolve_openai_key
from .humanize import paraphrase_he, pretty_bus_id
from .intents import detect_intents
from .agent_queries import (
    df_at_risk_today,
    df_top_risk_today,
    df_bus_today,
    df_bus_history,
    df_parts_replaced_last_30d,
    df_trend_last_days,
    df_bus_most_failures,
    df_bus_all_failures,
    df_failures_by_day_detail,
    df_risk_summary_by_day_bus,
    df_high_risk_by_likely_fault,
    run_fallback_agent,
)
from .failure_mapping import (
    map_failure_types_from_query,
    map_likely_faults_from_query,
)
from .utils_logging import log_agent
from . import shared_state
from .db_chat import (
    create_conversation,
    list_conversations,
    load_messages,
    save_message,
    update_conversation_title,
    generate_chat_title,
)

# -------------------------------------------------
# הגדרות בסיס
# -------------------------------------------------

st.set_page_config(page_title="🚌 Agent", layout="wide")
OPENAI_API_KEY = resolve_openai_key()
SIM_MIN_DATE = date(2023, 1, 1)
SIM_MAX_DATE = date(2024, 12, 31)


# -------------------------------------------------
# כלי עזר UI
# -------------------------------------------------

class fancy_spinner:
    def __init__(self, msg: str = "מעבד..."):
        self.msg = msg
        self.placeholder = st.empty()
        self._ctx = None

    def __enter__(self):
        self.placeholder.markdown(f"🌀 **{self.msg}**")
        self._ctx = st.spinner(self.msg)
        self._ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._ctx: self._ctx.__exit__(exc_type, exc, tb)
        self.placeholder.empty()


def _is_risk_query(text: str) -> bool:
    t = (text or "").lower()
    return any(w in t for w in ["סיכון", "risk", "probability", "chance", "סיכוי"])


def _extract_days_from_query(text: str, default: int = 30) -> int:
    m = _re2.search(r"(\d+)\s*(יום|ימים|day|days)", (text or ""))
    return int(m.group(1)) if m else default


def _has_hebrew(text: str) -> bool:
    return bool(_re2.search(r"[\u0590-\u05FF]", text or ""))


def _is_total_failures_query(text: str) -> bool:
    patterns = [r"כמה\s+תקלות", r"סה\"?כ\s+תקלות", r"total\s+failures"]
    return any(_re2.search(p, text.lower()) for p in patterns)


# -------------------------------------------------
# ✨ Table Prettifier (המשפך לטבלאות) ✨
# -------------------------------------------------

def _prettify_table(df: pd.DataFrame, query_text: str = "") -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    is_he = _has_hebrew(query_text)

    # נרמול שמות עמודות טכניים לשמות אחידים
    rename_map_tech = {
        "d": "date",
        "predicted_proba": "prob",
        "proba_7d": "prob",
        "likely_fault": "system",
        "failure_reason": "reason",
        "bus_id": "bus"
    }
    out.rename(columns=rename_map_tech, inplace=True)

    # המרת הסתברות לאחוזים
    if "prob" in out.columns:
        out["prob"] = out["prob"].apply(lambda x: f"{x:.0%}" if isinstance(x, (float, int)) else x)

    # הגדרת כותרות לפי שפה
    if is_he:
        final_cols_map = {
            "bus": "אוטובוס",
            "date": "תאריך",
            "prob": "הסתברות לתקלה (%)",
            "where_he": "מערכת חשודה",
            "system": "מערכת חשודה (Tech)",
            "explanation_he": "פירוט הסיכון",
            "reason_he": "גורמים",
            "had_failure": "הייתה תקלה?",
            "part_name": "שם החלק",
            "replaced_count": "כמות החלפות"
        }
        priority_order = ["bus", "date", "prob", "where_he", "explanation_he", "part_name", "replaced_count"]
    else:
        final_cols_map = {
            "bus": "Bus ID",
            "date": "Date",
            "prob": "Failure Probability (%)",
            "system": "Suspected System",
            "reason": "Risk Factors",
            "explanation_he": "Details (Hebrew)",
            "had_failure": "Failed?",
            "part_name": "Part Name",
            "replaced_count": "Replacement Count"
        }
        priority_order = ["bus", "date", "prob", "system", "reason", "part_name", "replaced_count"]

    # סינון וסידור העמודות
    existing_cols = [c for c in priority_order if c in out.columns]
    other_cols = [c for c in out.columns if c not in existing_cols and c in final_cols_map]
    final_selection = existing_cols + other_cols

    if not final_selection:
        out.rename(columns=final_cols_map, inplace=True)
        return out

    out = out[final_selection]
    out.rename(columns=final_cols_map, inplace=True)

    return out


# -------------------------------------------------
# HTML Formatting for KPIs
# -------------------------------------------------

def _generate_kpi_html(title: str, metrics: list[tuple[str, str, str]]) -> str:
    cards_html = ""
    card_style = "background-color:#fff; border:1px solid #ddd; border-radius:8px; padding:10px; flex:1; min-width:110px; text-align:center; margin:4px; box-shadow:0 1px 2px rgba(0,0,0,0.05);"

    for label, value, icon in metrics:
        cards_html += f"<div style='{card_style}'><div style='font-size:22px; margin-bottom:4px;'>{icon}</div><div style='font-size:12px; color:#666;'>{label}</div><div style='font-size:20px; font-weight:bold; color:#333;'>{value}</div></div>"

    return f"<div style='direction:rtl; margin-bottom:15px;'><div style='font-weight:bold; margin-bottom:8px; color:#444;'>📊 {title}</div><div style='display:flex; flex-wrap:wrap; gap:8px;'>{cards_html}</div></div>"


# -------------------------------------------------
# ניהול תאריכים ועונות 🧠
# -------------------------------------------------

def _extract_season_name(text: str) -> str | None:
    t = text.lower()
    if "חורף" in t or "winter" in t: return "Winter"
    if "קיץ" in t or "summer" in t: return "Summer"
    if "סתיו" in t or "fall" in t or "autumn" in t: return "Autumn"
    if "אביב" in t or "spring" in t: return "Spring"
    return None


def _extract_year(text: str) -> int | None:
    m = _re2.search(r"\b(202[0-9])\b", text)
    return int(m.group(1)) if m else None


def resolve_period_dates(query: str, today: date, intents: dict) -> Tuple[date, date, str, str | None]:
    season = _extract_season_name(query)
    year = _extract_year(query)

    if season:
        target_year = year if year else today.year
        start = date(target_year, 1, 1)
        end = date(target_year, 12, 31)
        he_seasons = {"Winter": "חורף", "Summer": "קיץ", "Fall": "סתיו", "Autumn": "סתיו", "Spring": "אביב"}
        title = f"{he_seasons.get(season, season)} {target_year}"
        return start, end, title, season

    rng = intents.get("RESOLVED_RANGE")
    if rng:
        return rng[0], rng[1], rng[2], None

    query_lower = query.lower()
    if "יומי" in query_lower or "היום" in query_lower or "daily" in query_lower or "today" in query_lower:
        days = 1
    else:
        days = intents.get("DAYS")
        if not days:
            days = _extract_days_from_query(query, default=14)

    end = today
    start = today - timedelta(days=days - 1)
    title = f"סיכום יומי ({today})" if days == 1 else f"{days} הימים האחרונים"

    return start, end, title, None


# -------------------------------------------------
# מנוע הדשבורד
# -------------------------------------------------

def render_period_dashboard(start: date, end: date, title: str, query_text: str, season_val: str = None) -> None:
    with fancy_spinner(f"מנתח נתונים עבור: {title}..."):
        trend = df_trend_last_days(start, end, season=season_val)
        risk = df_risk_summary_by_day_bus(start, end, season=season_val)
        detail = df_failures_by_day_detail(start, end, season=season_val)

    if trend.empty and risk.empty and detail.empty:
        say(f"לא נמצאו נתונים בטווח: {title}.")
        return

    n_buses = risk["bus_id"].nunique() if not risk.empty else 0
    n_high_risk = risk.loc[risk["proba_7d"] >= 0.5, "bus_id"].nunique() if not risk.empty else 0
    n_failures = len(detail) if not detail.empty else 0
    n_preds = len(risk) if not risk.empty else 0

    metrics_data = [
        ("אוטובוסים פעילים", str(n_buses), "🚌"),
        ("בסיכון גבוה", str(n_high_risk), "⚠️"),
        ("תקלות בפועל", str(n_failures), "🛠️"),
        ("רשומות תחזית", str(n_preds), "📉"),
    ]
    kpi_html = _generate_kpi_html(title, metrics_data)

    trend_txt = ""
    if not trend.empty:
        first, last = trend.iloc[0]["pct_risk"], trend.iloc[-1]["pct_risk"]
        direction = "עלייה" if (last - first) >= 0 else "ירידה"
        trend_txt = paraphrase_he(f"במהלך התקופה נראית {direction} בסיכון הממוצע (מ-{first:.1f}% ל-{last:.1f}%).")

    say(f"{kpi_html}\n\n{trend_txt}")

    if not trend.empty:
        add_table(f"מגמות סיכון ({title})", trend, query_text)

    if not risk.empty:
        add_table(f"פירוט סיכונים ותחזיות ({title})", risk.sort_values("proba_7d", ascending=False).head(100),
                  query_text)

    if not detail.empty:
        render_failures_matrix(detail, title="פירוט תקלות בפועל")


# -------------------------------------------------
# ניהול Session
# -------------------------------------------------
def get_user_id():
    if "user_id" not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())
    return st.session_state.user_id


def init_chat_session():
    if "current_chat_id" not in st.session_state:
        st.session_state.current_chat_id = None
    if "chat" not in st.session_state:
        st.session_state.chat = []
    if "date" not in st.session_state:
        st.session_state.date = dt.date(2024, 12, 30)
    if "top_limit" not in st.session_state:
        st.session_state.top_limit = 10


def load_chat_history(chat_id):
    st.session_state.current_chat_id = chat_id
    st.session_state.chat = load_messages(chat_id)


def start_new_chat():
    st.session_state.current_chat_id = None
    st.session_state.chat = []


# -------------------------------------------------
# UI Helpers
# -------------------------------------------------
def append_chat_message(role: str, text: str) -> dict:
    msg = {
        "id": str(uuid.uuid4()),
        "role": role,
        "text": text,
        "ts": dt.datetime.now().strftime("%H:%M"),
        "tables": [],
    }
    st.session_state.chat.append(msg)
    if st.session_state.current_chat_id:
        save_message(st.session_state.current_chat_id, role, text)
    return msg


def say(text: str) -> None:
    append_chat_message("assistant", text)


def add_table(title: str, df: pd.DataFrame, query_text: str = "") -> None:
    if not st.session_state.chat: return

    # הפעלת העיצוב האוטומטי
    clean_df = _prettify_table(df, query_text)

    last = st.session_state.chat[-1]
    t_obj = {"id": str(uuid.uuid4()), "title": title, "df": clean_df.copy()}
    last.setdefault("tables", []).append(t_obj)
    if st.session_state.current_chat_id:
        save_message(st.session_state.current_chat_id, "assistant", "", tables=[t_obj])


def render_chat_message(msg: dict[str, Any]) -> None:
    role = msg.get("role", "assistant")
    text = msg.get("text", "")
    ts = msg.get("ts", "")
    tables = msg.get("tables", []) or []

    is_he = _has_hebrew(text)
    direction = "rtl" if is_he else "ltr"

    if role == "user":
        align = "right"
        bg_color = "#d1e7dd"
        border_color = "#0f5132"
        label = "את/ה"
        icon = "🧑‍💻"
    else:
        align = "left"
        bg_color = "#f8f9fa"
        border_color = "#6c757d"
        label = "Agent"
        icon = "🤖"

    if not text and not tables: return

    if text:
        html = f"""
        <div style="display: flex; justify-content: {align}; margin: 4px 0;">
          <div style="
              max-width: 90%;
              background-color: {bg_color};
              border: 1px solid {border_color};
              border-radius: 12px;
              padding: 8px 10px;
              font-size: 0.95rem;
              direction: {direction};
              text-align: {'right' if direction == 'rtl' else 'left'};
              box-shadow: 0 1px 2px rgba(0,0,0,0.05);
          ">
            <div style="font-weight: 600; margin-bottom: 4px;">
              {icon} {label}
            </div>
            <div>{text}</div>
            <div style="
                font-size: 0.75rem;
                color: #6c757d;
                margin-top: 4px;
                text-align: {'left' if direction == 'rtl' else 'right'};
            ">
              {ts}
            </div>
          </div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)

    for t in tables:
        st.markdown(f"**📊 {t['title']}**")
        st.dataframe(
            t["df"],
            use_container_width=True,
            height=320,
            key=f"df_{t['id']}",
        )
        csv = t["df"].to_csv(index=False).encode("utf-8")
        safe_name = f"{t['title'].replace(' ', '_')}.csv"
        st.download_button(
            "⬇️ הורד CSV",
            csv,
            file_name=safe_name,
            key=f"dl_{t['id']}",
        )


def render_failures_matrix(detail: pd.DataFrame, title: str) -> None:
    if detail.empty: return
    df = detail.copy()
    df["date"] = df["d"].astype(str)
    st.markdown(f"#### {title}")
    gb = GridOptionsBuilder.from_dataframe(df[["date", "bus_id", "failure_type", "fault_category"]])
    gb.configure_default_column(sortable=True, filter=True, resizable=True)
    AgGrid(df, gridOptions=gb.build(), height=350, fit_columns_on_grid_load=True)


# -------------------------------------------------
# ROUTING LOGIC
# -------------------------------------------------

def _is_specific_question(query: str) -> bool:
    t = query.lower()
    triggers = ["מי", "איזה", "אילו", "כמה", "למה", "תן לי", "רשימת", "who", "which", "how many", "why", "list", "top",
                "most", "worst"]
    summary_triggers = ["סיכום", "מה קרה", "תמונת מצב", "summary", "status", "overview", "what happened"]
    is_specific = any(w in t for w in triggers)
    is_summary = any(w in t for w in summary_triggers)
    return is_specific and not is_summary


def answer(query: str) -> None:
    log_agent("Processing user query", query=query)
    today: date = st.session_state.date
    intents = detect_intents(query, today, st.session_state.top_limit)
    log_agent("Detected intents", **intents)

    # 1. זיהוי שאלות ספציפיות (עוקף דשבורד)
    if _is_specific_question(query):
        pass  # Go to Fallback LLM

    # 2. דשבורד תקופתי
    elif (
            _extract_season_name(query) is not None or
            _extract_year(query) is not None or
            intents.get("ANY_NATURAL_RANGE") or
            intents.get("WHAT_HAPPENED_LAST_DAYS") or
            "מה קרה" in query or "סיכום" in query
    ):
        start, end, title, season_val = resolve_period_dates(query, today, intents)
        render_period_dashboard(start, end, title, query, season_val)
        return

    # 3. שאלות קשיחות
    if intents.get("WHO_AT_RISK_TODAY"):
        with fancy_spinner("טוען דוח סיכונים יומי..."):
            df = df_at_risk_today(today, intents.get("TOP_N", 10))
            say(f"נמצאו {len(df)} אוטובוסים בסיכון גבוה היום ({today}).")
            add_table(f"סיכונים - {today}", df, query)
        return

    if intents.get("BUS_ID") and not ("היסטוריה" in query or "history" in query):
        bus_id = intents["BUS_ID"]
        with fancy_spinner(f"בודק סטטוס {bus_id}..."):
            df = df_bus_today(today, bus_id)
            if not df.empty:
                say(paraphrase_he(f"סטטוס עדכני: {df.iloc[0]['explanation_he']}"))
                add_table(f"סטטוס {bus_id}", df, query)
            else:
                say(f"אין נתונים להיום עבור {bus_id}")
        return

    # 4. Fallback Agent (LLM)
    with fancy_spinner("🤖 הסוכן מנתח את הבקשה..."):
        used = run_fallback_agent(query, today, st.session_state.top_limit, None)
        if used and shared_state.LAST_AGENT_DF is not None:
            df = shared_state.LAST_AGENT_DF
            if df.empty:
                say("הסוכן הבין את השאלה אך לא מצא נתונים מתאימים.")
            else:
                say("הנה הנתונים שמצאתי:")
                add_table(shared_state.LAST_AGENT_TITLE or "תוצאה (Agent)", df, query)
        else:
            say("לא הצלחתי למצוא תשובה מתאימה.")


# -------------------------------------------------
# MAIN
# -------------------------------------------------

def main():
    user_id = get_user_id()
    init_chat_session()

    with st.sidebar:
        st.title("🗄️ היסטוריה")
        if st.button("➕ חדש", use_container_width=True):
            start_new_chat()
            st.rerun()
        st.markdown("---")
        for c in list_conversations(user_id):
            lbl = f"🔹 {c['title']}" if c['id'] == st.session_state.current_chat_id else c['title']
            if st.button(lbl, key=c['id'], use_container_width=True):
                load_chat_history(c['id'])
                st.rerun()
        st.markdown("---")
        st.session_state.date = st.date_input(
            "תאריך סימולציה",
            value=st.session_state.date,
            min_value=SIM_MIN_DATE,  # חוסם תאריכים לפני 2023
            max_value=SIM_MAX_DATE  # חוסם תאריכים אחרי 2024
        )
    if st.session_state.current_chat_id is None:
        st.subheader("👋 שלום! איך אפשר לעזור בניהול הצי?")

    for m in st.session_state.chat:
        render_chat_message(m)

    q = st.chat_input("נסה: 'מה קרה בחורף 2023?', 'סיכום שבועיים אחרונים'...", key="main_input")
    if q:
        msg_obj = append_chat_message("user", q)
        render_chat_message(msg_obj)

        if st.session_state.current_chat_id is None:
            new_id = create_conversation(user_id, generate_chat_title(q))
            st.session_state.current_chat_id = new_id
            save_message(new_id, "user", q)

        answer(q)
        st.rerun()


if __name__ == "__main__":
    main()