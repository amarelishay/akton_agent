from __future__ import annotations

import re
import uuid
import datetime as dt
from datetime import date, timedelta
from typing import Any, Tuple, Optional
import re as _re2

import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

# --- ייבוא מודולים ---
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


def _extract_time_period(text: str) -> int:
    """
    מחלץ טווח זמן גמיש מתוך טקסט בעברית/אנגלית.
    כולל תמיכה בצורות זוגיות: יומיים, שבועיים, חודשיים, שנתיים.
    מחזיר כמות ימים.
    """
    t = (text or "").lower().strip()

    # --- ניקוי עברית מלוכלכת (אותיות סופיות) ---
    t = t.translate(str.maketrans({
        'ם': 'מ',
        'ן': 'נ',
        'ך': 'כ',
        'ף': 'פ',
        'ץ': 'צ',
    }))

    # ======================================================
    # 1. צורות זוגיות (Duels) — עדיפות עליונה
    # ======================================================
    if "יומיים" in t or "יוםיים" in t:
        return 2

    if "שבועיים" in t or "שבועין" in t or "שבוים" in t:
        return 14

    if "חודשיים" in t or "חדשיים" in t:
        return 60

    if "שנתיים" in t:
        return 730  # 365 * 2

    # ======================================================
    # 2. מספר + יחידת זמן
    # ======================================================

    # שנים
    m_year = re.search(r"(\d+)\s*(?:שנה|שנים|year|years)", t)
    if m_year:
        return int(m_year.group(1)) * 365

    # חודשים
    m_month = re.search(r"(\d+)\s*(?:חודש|חודשים|month|months)", t)
    if m_month:
        return int(m_month.group(1)) * 30

    # שבועות
    m_week = re.search(r"(\d+)\s*(?:שבוע|שבועות|week|weeks)", t)
    if m_week:
        return int(m_week.group(1)) * 7

    # ימים
    m_day = re.search(r"(\d+)\s*(?:יום|ימים|day|days)", t)
    if m_day:
        return int(m_day.group(1))

    # ======================================================
    # 3. יחידת זמן ללא מספר (מילה כללית)
    # ======================================================
    # שנה
    if "שנה" in t or "year" in t:
        return 365

    # חודש
    if "חודש" in t or "month" in t:
        return 30

    # שבוע
    if "שבוע" in t or "week" in t:
        return 7

    # יום
    if "יום" in t or "day" in t:
        return 1

    # ======================================================
    # 4. ברירת מחדל
    # ======================================================
    return 14

def _extract_days_from_query(text: str, default: int = 30) -> int:
    return _extract_time_period(text)


def _has_hebrew(text: str) -> bool:
    return bool(_re2.search(r"[\u0590-\u05FF]", text or ""))


# -------------------------------------------------
# ✨ Table Prettifier (עיצוב טבלאות)
# -------------------------------------------------

def _prettify_table(df: pd.DataFrame, query_text: str = "") -> pd.DataFrame:
    if df.empty: return df

    # 1. מחיקת עמודות כפולות (Safety Net)
    out = df.loc[:, ~df.columns.duplicated()].copy()

    is_he = _has_hebrew(query_text)

    rename_map_tech = {
        "d": "date", "predicted_proba": "prob", "proba_7d": "prob",
        "likely_fault": "system", "failure_reason": "reason", "bus_id": "bus",
        "replacement_count": "replaced_count"
    }
    # שינוי שמות רק למה שקיים
    existing_renames = {k: v for k, v in rename_map_tech.items() if k in out.columns}
    out.rename(columns=existing_renames, inplace=True)

    if "prob" in out.columns:
        out["prob"] = out["prob"].apply(lambda x: f"{x:.0%}" if isinstance(x, (float, int)) else x)

    if is_he:
        final_cols_map = {
            "bus": "אוטובוס", "date": "תאריך", "prob": "הסתברות לתקלה (%)",
            "where_he": "מערכת חשודה", "system": "מערכת (Tech)",
            "explanation_he": "פירוט הסיכון", "reason_he": "גורמים",
            "had_failure": "תקלה בפועל", "failure_count": "מספר תקלות",
            "part_name": "שם החלק", "replaced_count": "כמות החלפות"
        }
        priority = ["bus", "date", "prob", "where_he", "explanation_he", "failure_count", "part_name", "replaced_count"]
    else:
        final_cols_map = {
            "bus": "Bus ID", "date": "Date", "prob": "Probability (%)",
            "system": "System", "reason": "Factors", "explanation_he": "Details",
            "failure_count": "Failures", "part_name": "Part", "replaced_count": "Count"
        }
        priority = ["bus", "date", "prob", "system", "reason", "failure_count", "part_name", "replaced_count"]

    existing = [c for c in priority if c in out.columns]
    others = [c for c in out.columns if c not in existing and c in final_cols_map]

    final_sel = existing + others
    if final_sel:
        out = out[final_sel]
        out.rename(columns=final_cols_map, inplace=True)
    else:
        out.rename(columns=final_cols_map, inplace=True)

    return out


# -------------------------------------------------
# HTML KPIs
# -------------------------------------------------

def _generate_kpi_html(title: str, metrics: list[tuple[str, str, str]]) -> str:
    cards = ""
    style = "background:#fff;border:1px solid #e0e0e0;border-radius:8px;padding:12px;flex:1;min-width:100px;text-align:center;box-shadow:0 1px 2px rgba(0,0,0,0.05);margin:4px;"
    for lbl, val, ico in metrics:
        cards += f"<div style='{style}'><div style='font-size:20px;margin-bottom:4px;'>{ico}</div><div style='font-size:12px;color:#666;'>{lbl}</div><div style='font-size:18px;font-weight:bold;color:#2c3e50;'>{val}</div></div>"
    return f"<div style='direction:rtl;margin-bottom:10px;'><div style='font-weight:bold;margin-bottom:8px;color:#333;'>📊 {title}</div><div style='display:flex;flex-wrap:wrap;gap:8px;'>{cards}</div></div>"


# -------------------------------------------------
# ניהול תאריכים ועונות
# -------------------------------------------------

def _extract_season_name(text: str) -> str | None:
    t = text.lower()
    if "חורף" in t or "winter" in t: return "Winter"
    if "קיץ" in t or "summer" in t: return "Summer"
    if "סתיו" in t or "fall" in t or "autumn" in t: return "Autumn"  # מותאם ל-DB
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

    # חישוב מתמטי לפי טווח הזמן המבוקש
    days = intents.get("DAYS") or _extract_time_period(query)

    end = today
    start = today - timedelta(days=days)

    title = f"סיכום {days} ימים אחרונים"
    if days == 1: title = f"סיכום יומי ({today})"
    if days == 30: title = "סיכום חודש אחרון"
    if days == 365: title = "סיכום שנה אחרונה"

    return start, end, title, None


# -------------------------------------------------
# מנוע הדשבורד
# -------------------------------------------------

def render_period_dashboard(start: date, end: date, title: str, query_text: str, season_val: str = None) -> None:
    with fancy_spinner(f"מנתח נתונים עבור: {title}..."):
        # כאן התיקון החשוב: העברת season_val לפונקציות ה-SQL
        trend = df_trend_last_days(start, end, season=season_val)
        risk = df_risk_summary_by_day_bus(start, end, season=season_val)
        detail = df_failures_by_day_detail(start, end, season=season_val)

    if trend.empty and risk.empty and detail.empty:
        say(f"לא נמצאו נתונים בטווח: {title}.")
        return

    n_buses = risk["bus_id"].nunique() if not risk.empty else 0
    n_high = risk.loc[risk["proba_7d"] >= 0.5, "bus_id"].nunique() if not risk.empty else 0
    n_fail = len(detail) if not detail.empty else 0
    n_rec = len(risk) if not risk.empty else 0

    kpi_html = _generate_kpi_html(title, [
        ("אוטובוסים", str(n_buses), "🚌"),
        ("בסיכון גבוה", str(n_high), "⚠️"),
        ("תקלות", str(n_fail), "🛠️"),
        ("רשומות", str(n_rec), "📉")
    ])

    trend_txt = ""
    if not trend.empty:
        f_val, l_val = trend.iloc[0]["pct_risk"], trend.iloc[-1]["pct_risk"]
        direction = "עלייה" if (l_val - f_val) >= 0 else "ירידה"
        trend_txt = paraphrase_he(f"זוהתה {direction} ברמת הסיכון מ-{f_val:.1f}% ל-{l_val:.1f}%.")

    say(f"{kpi_html}\n\n{trend_txt}")

    if not trend.empty: add_table(f"מגמות סיכון ({title})", trend, query_text)
    if not risk.empty: add_table(f"פירוט סיכונים ({title})", risk.sort_values("proba_7d", ascending=False).head(100),
                                 query_text)
    if not detail.empty: render_failures_matrix(detail, "פירוט תקלות")


# -------------------------------------------------
# Routing Logic (The Brain) 🧠
# -------------------------------------------------

def _is_specific_question(query: str) -> bool:
    t = query.lower()
    # מילות מפתח לשאלות ספציפיות
    triggers = ["מי", "איזה", "אילו", "כמה", "למה", "תן לי", "רשימת", "who", "which", "how many", "why", "list", "top",
                "most", "worst"]
    # מילות מפתח לסיכום
    summary_triggers = ["סיכום", "מה קרה", "תמונת מצב", "summary", "status", "overview", "what happened"]

    is_specific = any(w in t for w in triggers)
    is_summary = any(w in t for w in summary_triggers)

    # אם זה ספציפי וגם לא סיכום -> זה ספציפי
    return is_specific and not is_summary


def answer(query: str) -> None:
    log_agent("User Query", query=query)
    today = st.session_state.date
    intents = detect_intents(query, today, st.session_state.top_limit)
    log_agent("Detected Intents", **intents)

    # --- עדיפות 1: ישויות ספציפיות (כדי שלא ייבלעו ע"י סיכום כללי) ---

    # 1. כרטיס אוטובוס (אלא אם זו שאלת היסטוריה מורכבת שתלך ל-LLM)
    if intents.get("BUS_ID"):
        # אם המשתמש ביקש היסטוריה/תקלות ספציפיות, נדלג ישר ל-LLM
        if "היסטוריה" in query or "history" in query or "תקלות" in query:
            pass  # Go to Fallback LLM (למטה)
        else:
            # אחרת, נציג כרטיס ביקור מהיר
            bus_id = intents["BUS_ID"]
            with fancy_spinner(f"שולף כרטיס ל-{bus_id}..."):
                df = df_bus_today(today, bus_id)
                if not df.empty:
                    say(paraphrase_he(f"סטטוס עדכני: {df.iloc[0]['explanation_he']}"))
                    add_table(f"סטטוס {bus_id}", df, query)
                else:
                    say(f"אין נתונים להיום עבור {bus_id}")
            return

    # 2. דף הבית (מי בסיכון)
    if intents.get("WHO_AT_RISK_TODAY"):
        with fancy_spinner("טוען דוח סיכונים יומי..."):
            df = df_at_risk_today(today, intents.get("TOP_N", 10))
            say(f"נמצאו {len(df)} אוטובוסים בסיכון גבוה היום ({today}).")
            add_table(f"סיכונים - {today}", df, query)
        return

    # --- עדיפות 2: שאלות חכמות ספציפיות (עוקף דשבורד) ---
    # אם שואלים "כמה תקלות היו...", זה לא סיכום כללי אלא שאילתה מדויקת
    if _is_specific_question(query):
        pass  # Go to Fallback LLM directly

    # --- עדיפות 3: דשבורד תקופתי (עונות, טווחים, מה קרה, סיכום) ---
    # רק אם לא זיהינו משהו ספציפי יותר, נניח שהמשתמש רוצה סיכום כללי
    elif (
            _extract_season_name(query) is not None or
            _extract_year(query) is not None or
            intents.get("ANY_NATURAL_RANGE") or
            intents.get("WHAT_HAPPENED_LAST_DAYS") or
            "מה קרה" in query or
            "סיכום" in query or
            "סכם" in query
    ):
        start, end, title, season_val = resolve_period_dates(query, today, intents)
        render_period_dashboard(start, end, title, query, season_val)
        return

    # --- עדיפות 4: סוכן חכם (כל השאר) ---
    with fancy_spinner("🤖 הסוכן מנתח את הבקשה..."):
        used = run_fallback_agent(query, today, st.session_state.top_limit, None)

        if used and shared_state.LAST_AGENT_DF is not None:
            df = shared_state.LAST_AGENT_DF
            if df.empty:
                say("הסוכן הבין את השאלה אך לא נמצאו נתונים תואמים.")
            else:
                say(f"הנה התוצאות שמצאתי ({len(df)} שורות):")
                add_table(shared_state.LAST_AGENT_TITLE or "תוצאות", df, query)
        else:
            say("לא הצלחתי למצוא תשובה. נסה לנסח אחרת.")

# -------------------------------------------------
# Session & UI Helpers
# -------------------------------------------------

def get_user_id():
    if "user_id" not in st.session_state: st.session_state.user_id = str(uuid.uuid4())
    return st.session_state.user_id


def init_chat_session():
    if "current_chat_id" not in st.session_state: st.session_state.current_chat_id = None
    if "chat" not in st.session_state: st.session_state.chat = []
    if "date" not in st.session_state: st.session_state.date = dt.date(2024, 12, 30)
    if "top_limit" not in st.session_state: st.session_state.top_limit = 10


def load_chat_history(chat_id):
    st.session_state.current_chat_id = chat_id
    st.session_state.chat = load_messages(chat_id)


def start_new_chat():
    st.session_state.current_chat_id = None
    st.session_state.chat = []


def append_chat_message(role: str, text: str) -> dict:
    msg = {"id": str(uuid.uuid4()), "role": role, "text": text, "ts": dt.datetime.now().strftime("%H:%M"), "tables": []}
    st.session_state.chat.append(msg)
    if st.session_state.current_chat_id:
        save_message(st.session_state.current_chat_id, role, text)
    return msg


def say(text: str) -> None:
    append_chat_message("assistant", text)


def add_table(title: str, df: pd.DataFrame, query_text: str = "") -> None:
    if not st.session_state.chat: return
    clean_df = _prettify_table(df, query_text)
    last = st.session_state.chat[-1]
    t_obj = {"id": str(uuid.uuid4()), "title": title, "df": clean_df.copy()}
    last.setdefault("tables", []).append(t_obj)
    if st.session_state.current_chat_id:
        save_message(st.session_state.current_chat_id, "assistant", "", tables=[t_obj])


def render_chat_message(msg: dict):
    role = msg.get("role", "assistant")
    text = msg.get("text", "")
    tables = msg.get("tables", []) or []

    if not text and not tables: return

    align = "right" if role == "user" else "left"
    bg = "#d1e7dd" if role == "user" else "#f8f9fa"
    icon = "🧑‍💻" if role == "user" else "🤖"

    if text:
        st.markdown(
            f"<div style='display:flex;justify-content:{align};margin:5px 0;'><div style='background:{bg};padding:10px;border-radius:10px;max-width:85%;direction:rtl;text-align:right;'><b>{icon}</b> {text}</div></div>",
            unsafe_allow_html=True)

    for t in tables:
        st.markdown(f"**📊 {t['title']}**")
        st.dataframe(t["df"], width=None, use_container_width=True, height=320, key=f"df_{t['id']}")
        csv = t["df"].to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ CSV", csv, file_name=f"table.csv", key=f"dl_{t['id']}")


def render_failures_matrix(detail: pd.DataFrame, title: str) -> None:
    if detail.empty: return
    df = detail.copy()
    df["date"] = df["d"].astype(str)
    st.markdown(f"#### {title}")
    gb = GridOptionsBuilder.from_dataframe(df[["date", "bus_id", "failure_type", "fault_category"]])
    gb.configure_default_column(sortable=True, filter=True, resizable=True)
    AgGrid(df, gridOptions=gb.build(), height=350, fit_columns_on_grid_load=True)


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
        # התיקון הקריטי: הגבלת התאריכים ל-2023-2024 כפי שביקשת
        st.session_state.date = st.date_input("תאריך סימולציה", value=st.session_state.date, min_value=SIM_MIN_DATE,
                                              max_value=SIM_MAX_DATE)
        st.session_state.top_limit = st.number_input("Top N", value=st.session_state.top_limit, min_value=1,
                                                     max_value=500)

    if st.session_state.current_chat_id is None:
        st.subheader("👋 שלום! איך אפשר לעזור בניהול הצי?")

    for m in st.session_state.chat:
        render_chat_message(m)

    q = st.chat_input("שאל משהו...", key="main_input")
    if q:
        msg = append_chat_message("user", q)
        render_chat_message(msg)

        if st.session_state.current_chat_id is None:
            new_id = create_conversation(user_id, generate_chat_title(q))
            st.session_state.current_chat_id = new_id
            save_message(new_id, "user", q)

        answer(q)
        st.rerun()


if __name__ == "__main__":
    main()