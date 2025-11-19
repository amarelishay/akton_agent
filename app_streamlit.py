from __future__ import annotations

import uuid
import datetime as dt
from datetime import date, timedelta
from typing import Any
import re as _re2

import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

# --- מודולים קיימים ---
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

# --- מודול חדש לניהול צ'אט ---
# ודא שיצרת את הקובץ db_chat.py באותה תיקייה!
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

st.set_page_config(
    page_title="🚌 תחזוקה חכמה — Agent",
    page_icon="🚌",
    layout="wide",
)

OPENAI_API_KEY = resolve_openai_key()

SIM_MIN_DATE = date(2023, 1, 1)
SIM_MAX_DATE = date(2024, 12, 31)


# -------------------------------------------------
# כלי עזר UI
# -------------------------------------------------

class fancy_spinner:
    def __init__(self, msg: str = "מעבד את הבקשה..."):
        self.msg = msg
        self.placeholder = st.empty()
        self._spinner_ctx = None

    def __enter__(self):
        self.placeholder.markdown(f"🌀 **{self.msg}**")
        self._spinner_ctx = st.spinner(self.msg)
        self._spinner_ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._spinner_ctx is not None:
            self._spinner_ctx.__exit__(exc_type, exc, tb)
        self.placeholder.empty()


def _is_risk_query(text: str) -> bool:
    t = (text or "").lower()
    risk_words = ["סיכון", "בסיכון", "high risk", "risk", "probability", "chance", "סיכוי"]
    return any(w in t for w in risk_words)


def _extract_days_from_query(text: str, default: int = 30) -> int:
    import re as _re
    m = _re.search(r"(\d+)\s*(יום|ימים|day|days)", (text or ""))
    if m:
        try:
            n = int(m.group(1))
            return max(1, min(365, n))
        except Exception:
            pass
    return default


def _guess_days_hebrew(text: str, default: int = 14) -> int:
    t = (text or "").replace("?", "").replace("!", "").strip()
    if "שבועיים" in t:
        return 14
    if "שבוע" in t and "שבועיים" not in t:
        return 7
    return default


def _has_hebrew(text: str) -> bool:
    return bool(_re2.search(r"[\u0590-\u05FF]", text or ""))


def _is_total_failures_query(text: str) -> bool:
    t = (text or "").lower()
    patterns = [
        r"כמה\s+תקלות",
        r"מספר\s+התקלות",
        r"סה\"?כ\s+תקלות",
        r"total\s+failures",
        r"how\s+many\s+failures",
    ]
    return any(_re2.search(p, t) for p in patterns)


# -------------------------------------------------
# ניהול Session ומצב אורח (Guest Mode)
# -------------------------------------------------

def get_user_id():
    """מייצר או שולף מזהה אורח ייחודי ושומר ב-Session"""
    if "user_id" not in st.session_state:
        # כאן נוצר ה-UUID למשתמש האורח
        st.session_state.user_id = str(uuid.uuid4())
    return st.session_state.user_id


def init_chat_session():
    if "current_chat_id" not in st.session_state:
        st.session_state.current_chat_id = None  # None = שיחה חדשה שטרם נשמרה
    if "chat" not in st.session_state:
        st.session_state.chat = []

    # אתחול הגדרות גלובליות אם חסרות
    if "date" not in st.session_state:
        st.session_state.date = date(2024, 12, 30)
    if "top_limit" not in st.session_state:
        st.session_state.top_limit = 10


def load_chat_history(chat_id):
    """טוען הודעות מה-DB לתוך ה-Session"""
    st.session_state.current_chat_id = chat_id
    st.session_state.chat = load_messages(chat_id)


def start_new_chat():
    """מאפס את המסך לשיחה חדשה"""
    st.session_state.current_chat_id = None
    st.session_state.chat = []


# -------------------------------------------------
# פונקציות צ'אט מעודכנות (עם שמירה ל-DB)
# -------------------------------------------------

def append_chat_message(role: str, text: str) -> dict[str, Any]:
    """מוסיף הודעה ל-Session ושומר ל-DB אם יש שיחה פעילה"""
    msg = {
        "id": str(uuid.uuid4()),
        "role": role,
        "text": text,
        "ts": dt.datetime.now().strftime("%H:%M"),
        "tables": [],
    }
    st.session_state.chat.append(msg)

    # שמירה ל-DB: רק אם כבר יש ID לשיחה.
    # אם השיחה חדשה (None), השמירה תתבצע בפונקציה main אחרי שנקבע שם.
    if st.session_state.current_chat_id:
        save_message(st.session_state.current_chat_id, role, text)

    return msg


def say(text: str) -> None:
    """פונקציית עזר לקיצור"""
    append_chat_message("assistant", text)


def add_table(title: str, df: pd.DataFrame) -> None:
    """
    מוסיף טבלה להודעה האחרונה ב-UI, ושומר אותה ל-DB.
    """
    if not st.session_state.chat:
        return

    last_agent = None
    for m in reversed(st.session_state.chat):
        if m.get("role") == "assistant":
            last_agent = m
            break

    if last_agent is None:
        return

    # 1. עדכון ב-UI
    tables = last_agent.setdefault("tables", [])
    table_obj = {
        "id": str(uuid.uuid4()),
        "title": title,
        "df": df.copy(),
    }
    tables.append(table_obj)

    # 2. שמירה ל-DB (Persistency)
    # כדי לפשט, נשמור את הטבלה כ"הודעת מערכת" נפרדת או נעדכן.
    # בחרתי לשמור כהודעה נוספת מאחורי הקלעים עם התוכן הטבלאי,
    # כדי להבטיח שהיא תיטען מחדש.
    if st.session_state.current_chat_id:
        # אופציה: שולחים שוב את ההודעה האחרונה עם הטבלאות,
        # או שומרים רשומה ייעודית. ב-db_chat.py יש תמיכה ב-tables.
        # נבצע "תיקון" פשוט: נשמור רשומה עם טקסט ריק שמכילה את הטבלה.
        save_message(st.session_state.current_chat_id, "assistant", "", tables=[table_obj])


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

    # דילוג על הודעות טכניות ריקות שנועדו רק לשמירת טבלאות
    if not text and not tables:
        return

    # אם יש טקסט, נציג אותו
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

    # אם יש טבלאות, נציג אותן מתחת להודעה
    if role == "assistant":
        for t in tables:
            st.markdown(f"**📊 {t['title']}**")
            st.dataframe(
                t["df"],
                use_container_width=True,  # <--- התיקון: במקום width=None
                height=320,
                key=f"df_{t['id']}",  # מפתח ייחודי לסטרימליט
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
    if detail.empty:
        return

    df = detail.copy()
    df["date"] = df["d"].astype(str)
    df_display = df[
        [
            "date",
            "bus_id",
            "failure_type",
            "fault_category",
            "failure_flag",
            "maintenance_flag",
        ]
    ].sort_values(by=["date", "bus_id"])

    st.markdown(f"#### {title}")

    gb = GridOptionsBuilder.from_dataframe(df_display)
    gb.configure_default_column(sortable=True, filter=True, resizable=True)
    gb.configure_column("failure_flag", header_name="failure_flag")
    gb.configure_column("maintenance_flag", header_name="maintenance_flag")
    grid_options = gb.build()

    AgGrid(
        df_display,
        gridOptions=grid_options,
        update_mode=GridUpdateMode.NO_UPDATE,
        enable_enterprise_modules=False,
        fit_columns_on_grid_load=True,
        height=420,
    )


# -------------------------------------------------
# לוגיקה עסקית (Business Logic) - נשמר במלואו
# -------------------------------------------------

def handle_period_question(query: str, today: date, intents: dict[str, Any]) -> None:
    rng = intents.get("RESOLVED_RANGE")

    if rng:
        start, end, title = rng
    else:
        days = intents.get("DAYS")
        if not days:
            days = _extract_days_from_query(query, default=14)
            days = _guess_days_hebrew(query, default=days)
        end = today
        start = today - timedelta(days=days - 1)
        title = f"{days} הימים האחרונים"

    if end > today:
        say(
            f"הטווח שביקשת ({title}) כולל תאריכים אחרי תאריך הסימולציה ({today}). "
            f"כרגע הסימולציה מוגדרת עד {today} בלבד."
        )
        return

    if start < SIM_MIN_DATE or end > SIM_MAX_DATE:
        say(
            f"הטווח שביקשת ({title}) חורג מטווח הנתונים שבמערכת. "
            f"כרגע יש נתונים רק בין {SIM_MIN_DATE} ל־{SIM_MAX_DATE}."
        )
        return

    with fancy_spinner(f"מחשב סיכום לתקופה: {title}..."):
        trend = df_trend_last_days(start, end)
        risk = df_risk_summary_by_day_bus(start, end)
        detail = df_failures_by_day_detail(start, end)

    if trend.empty and risk.empty and detail.empty:
        say(f"לא נמצאו נתונים בטווח: {title}.")
        return

    n_buses = risk["bus_id"].nunique() if not risk.empty else 0
    n_buses_high_risk = (
        risk.loc[risk["proba_7d"] >= 0.5, "bus_id"].nunique() if not risk.empty else 0
    )
    total_preds = len(risk) if not risk.empty else 0
    total_failures = len(detail) if not detail.empty else 0

    st.markdown("#### 📊 סיכום כללי לתקופה")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("אוטובוסים שונים במערכת", n_buses)
    c2.metric("אוטובוסים שהיו בסיכון גבוה", n_buses_high_risk)
    c3.metric("סה\"כ רשומות תחזית", total_preds)
    c4.metric("סה\"כ תקלות בפועל", total_failures)

    summary_text = None
    if not trend.empty:
        first, last = trend.iloc[0]["pct_risk"], trend.iloc[-1]["pct_risk"]
        delta = last - first
        direction = "עלייה" if delta >= 0 else "ירידה"
        summary_text = paraphrase_he(
            f"במהלך {title} נראית {direction} בשיעור האוטובוסים בסיכון: "
            f"מ־{first:.1f}% ל־{last:.1f}%."
        )
    elif not (risk.empty and detail.empty):
        summary_text = f"הצגתי סיכום תקלות וסיכונים עבור {title}."

    if summary_text:
        say(summary_text)

    if not trend.empty:
        add_table(f"מגמות סיכון ({title})", trend)

    if not risk.empty:
        cols = [
            "d",
            "bus_id",
            "proba_7d",
            "proba_30d",
            "had_failure",
            "where_he",
            "reason_he",
            "explanation_he",
        ]
        cols = [c for c in cols if c in risk.columns]
        risk_sorted = risk.sort_values("proba_7d", ascending=False)
        add_table(
            f"סיכון לשבוע לכל יום ואוטובוס — Top 100 ({title})",
            risk_sorted[cols].head(100),
        )

    if not detail.empty:
        render_failures_matrix(detail, title="פירוט תקלות בפועל לפי יום ואוטובוס")


def answer(query: str) -> None:
    today: date = st.session_state.date
    intents = detect_intents(query, today, st.session_state.top_limit)
    log_agent("Detected intents", **intents)
    top_n = intents.get("TOP_N", st.session_state.top_limit)

    # 1. שאלה על סך כל התקלות עד כה
    if _is_total_failures_query(query):
        with fancy_spinner("סופר את כל התקלות במערכת עד תאריך הסימולציה..."):
            detail = df_failures_by_day_detail(SIM_MIN_DATE, today)

        total_failures = len(detail)

        if total_failures == 0:
            say(
                f"לא נמצאו תקלות מתועדות במערכת בין {SIM_MIN_DATE} לבין {today}."
            )
        else:
            say(
                paraphrase_he(
                    f"נמצאו בסך הכול {total_failures} תקלות מתועדות בכל התקופה "
                    f"עד תאריך הסימולציה {today}."
                )
            )
            # סיכום לפי אוטובוס
            summary_bus = (
                detail.groupby("bus_id")
                .size()
                .reset_index(name="failures_count")
                .sort_values("failures_count", ascending=False)
            )
            add_table(
                f"סך תקלות לפי אוטובוס עד {today}",
                summary_bus,
            )
        return

    fault_cats = map_likely_faults_from_query(query)
    if fault_cats and _is_risk_query(query):
        days = _extract_days_from_query(query, default=30)
        with fancy_spinner(
                f"מחשב את האוטובוסים עם הסיכון הגבוה ביותר לתקלות בקטגוריות {', '.join(fault_cats)} "
                f"ב-{days} הימים האחרונים..."
        ):
            df = df_high_risk_by_likely_fault(today, days, fault_cats, top_n)

        if df.empty:
            say(
                "לא נמצאו אוטובוסים עם סיכון משמעותי לתקלות בקטגוריות "
                f"{', '.join(fault_cats)} ב-{days} הימים האחרונים בדאטה."
            )
        else:
            say(
                f"הנה האוטובוסים עם הסיכון הגבוה ביותר לתקלות בקטגוריות "
                f"{', '.join(fault_cats)} ב-{days} הימים האחרונים."
            )
            title = (
                f"Top {top_n} אוטובוסים בסיכון גבוה "
                f"({', '.join(fault_cats)}) ב-{days} הימים האחרונים"
            )
            add_table(title, df)
        return

    if intents.get("WHO_AT_RISK_TODAY"):
        with fancy_spinner("מביא את האוטובוסים בסיכון היום..."):
            df = df_at_risk_today(today, top_n)
        if df.empty:
            say(f"לא נמצאו אוטובוסים עם סיכון ≥ 50% בתאריך {today}.")
        else:
            say("אלה האוטובוסים שנמצאים היום בסיכון גבוה (≥50%).")
            add_table(
                f"{today} — מי בסיכון היום (≥50%)",
                df[
                    [
                        "bus_id",
                        "d",
                        "predicted_proba",
                        "predicted_label",
                        "failure_reason",
                        "reason_he",
                        "likely_fault",
                        "where_he",
                        "explanation_he",
                    ]
                ],
            )
        return

    if intents.get("BUS_ALL_FAILURES") and intents.get("BUS_ID"):
        bus_id = intents["BUS_ID"]
        nice_id = pretty_bus_id(bus_id)

        with fancy_spinner(f"מביא את כל התקלות של {nice_id} בכל התקופה..."):
            df_hist = df_bus_all_failures(bus_id)

        if df_hist.empty:
            say(f"לא מצאתי תקלות מתועדות עבור {nice_id} בכל התקופה.")
        else:
            say(f"הנה כל התקלות של {nice_id} בכל התקופה.")
            add_table(
                f"כל התקלות של {nice_id}",
                df_hist[
                    [
                        "d",
                        "bus_id",
                        "failure_type",
                        "fault_category",
                        "failure_flag",
                        "maintenance_flag",
                    ]
                ],
            )
        return

    if intents.get("BUS_ID"):
        bus_id = intents["BUS_ID"]
        nice_id = pretty_bus_id(bus_id)

        want_all_failures_model = bool(
            _re2.search(r"כל\s+התקלות|כל\s+התקולות|all\s+failures", query, _re2.IGNORECASE)
        )

        if want_all_failures_model:
            with fancy_spinner(f"מביא היסטוריה של תחזיות/תקלות עבור {nice_id}..."):
                dfb = df_bus_history(bus_id, limit=200)

            if dfb.empty:
                say(f"לא נמצאו נתוני תחזיות היסטוריים עבור {nice_id}.")
            else:
                say(f"הנה היסטוריית תחזיות/סיכונים עבור {nice_id}.")
                add_table(
                    f"{nice_id} — היסטוריית תחזיות",
                    dfb[
                        [
                            "bus_id",
                            "d",
                            "proba_7d",
                            "label_7d",
                            "proba_30d",
                            "label_30d",
                            "failure_reason",
                            "reason_he",
                            "likely_fault",
                            "where_he",
                            "explanation_he",
                        ]
                    ],
                )
            return

        with fancy_spinner(f"מחשב סיכון עבור {nice_id}..."):
            dfb = df_bus_today(today, bus_id)

        if dfb.empty:
            say(f"לא נמצאו נתונים עבור {nice_id} בתאריך {today}.")
        else:
            r = dfb.iloc[0]
            msg = paraphrase_he(
                f"**{nice_id} — {today}**: p7={r.proba_7d:.3f}"
                + (f", p30={r.proba_30d:.3f}" if pd.notnull(r.proba_30d) else "")
                + f". {r.explanation_he}"
            )
            say(msg)
            add_table(
                f"{nice_id} — פירוט {today}",
                dfb[
                    [
                        "bus_id",
                        "d",
                        "proba_7d",
                        "label_7d",
                        "proba_30d",
                        "label_30d",
                        "failure_reason",
                        "reason_he",
                        "likely_fault",
                        "where_he",
                        "explanation_he",
                    ]
                ],
            )
        return

    if intents.get("MOST_REPLACED_PARTS"):
        with fancy_spinner("סורק החלפות חלקים בחודש האחרון..."):
            dfp = df_parts_replaced_last_30d(today, 20)
        if dfp.empty:
            say("לא נמצאו החלפות חלקים בחודש האחרון.")
        else:
            top = dfp.iloc[0]
            say(
                paraphrase_he(
                    f'החלק שהוחלף הכי הרבה בחודש האחרון הוא {top.part_name} '
                    f'(סה\"כ {int(top.replaced_count)} החלפות). הצגתי גם טבלה מלאה.'
                )
            )
            add_table(
                f"החלקים שהוחלפו הכי הרבה — 30 יום אחרונים עד {today}",
                dfp,
            )
        return

    if intents.get("BUS_MOST_FAILURES"):
        rng = intents.get("RESOLVED_RANGE")
        if rng:
            start, end, title = rng
        else:
            start = end = None
            title = "כל התקופה"

        ft_list = map_failure_types_from_query(query)

        with fancy_spinner("מחשב את האוטובוסים עם הכי הרבה תקלות בפועל..."):
            df = df_bus_most_failures(start, end, ft_list, top_n)

        if df.empty:
            msg = "לא נמצאו תקלות בפועל"
            if ft_list:
                msg += f" עבור סוגי התקלה: {', '.join(ft_list)}"
            if rng:
                msg += f" בטווח {title}"
            say(msg + ".")
        else:
            top_row = df.iloc[0]
            say(
                paraphrase_he(
                    f"האוטובוס עם הכי הרבה תקלות הוא {top_row.bus_id} "
                    f"עם {int(top_row.failure_count)} תקלות מתועדות בתקופה {title}. "
                    f"הצגתי גם טבלת Top {top_n}."
                )
            )
            title_str = "אוטובוסים עם הכי הרבה תקלות"
            if ft_list:
                title_str += f" ({', '.join(ft_list)})"
            title_str += f" - {title}"
            add_table(title_str, df)
        return

    if intents.get("ANY_NATURAL_RANGE") or intents.get("WHAT_HAPPENED_LAST_DAYS"):
        handle_period_question(query, today, intents)
        return

    # Top N היום (סיכון גבוה ביותר)
    if intents.get("TOP_LIST") or intents.get("HIGHEST_RISK_N"):
        n = intents.get("TOP_N", intents.get("TOP_N_TEXT", st.session_state.top_limit))
        with fancy_spinner(f"מחשב Top {n} לסיכון היום (ללא סף מינימלי)..."):
            df = df_top_risk_today(today, n)
        if df.empty:
            say(f"לא נמצאו נתונים ל־{today}.")
        else:
            add_table(
                f"Top {n} Highest Risk — {today}",
                df[
                    [
                        "bus_id",
                        "d",
                        "predicted_proba",
                        "predicted_label",
                        "failure_reason",
                        "reason_he",
                        "likely_fault",
                        "where_he",
                        "explanation_he",
                    ]
                ],
            )
            say("הצגתי את האוטובוסים עם הסיכון הגבוה ביותר היום.")
        return

    with fancy_spinner("🤖🌀 הסוכן חושב ומרכיב שאילתה..."):
        used = run_fallback_agent(
            query,
            today,
            st.session_state.top_limit,
            intents.get("DAYS"),
        )

    if used and shared_state.LAST_AGENT_DF is not None:
        df = shared_state.LAST_AGENT_DF
        if df.empty:
            say(
                "ניסיתי לפענח את הבקשה בעזרת הסוכן, "
                "אבל לא נמצאו נתונים שמתאימים לקריטריונים."
            )
        else:
            say("פענחתי את הבקשה בעזרת הסוכן והצגתי טבלה מתאימה.")
            add_table(
                shared_state.LAST_AGENT_TITLE or "תוצאה (Agent)",
                df,
            )
        return

    say(
        "לא זיהיתי בקשה. נסה: “מי בסיכון היום?”, “BUS 17”, "
        "“מה קרה בשבוע האחרון?”, “Top 10”, "
        "או “לאילו אוטובוסים יש סיכון גבוה לתקלות במזגן ובבלמים?”."
    )


# -------------------------------------------------
# Main / Layout
# -------------------------------------------------

def main() -> None:
    user_id = get_user_id()
    init_chat_session()

    # --- Sidebar: היסטוריה והגדרות ---
    with st.sidebar:
        st.title("🗄️ היסטוריית שיחות")
        if st.button("➕ שיחה חדשה", use_container_width=True):
            start_new_chat()
            st.rerun()

        st.markdown("---")

        # רשימת השיחות של המשתמש
        my_chats = list_conversations(user_id)
        for c in my_chats:
            label = c['title']
            if c['id'] == st.session_state.current_chat_id:
                label = f"🔹 {label}"

            if st.button(label, key=c['id'], use_container_width=True):
                load_chat_history(c['id'])
                st.rerun()

        st.markdown("---")

        # הגדרות
        st.subheader("הגדרות")
        st.session_state.date = st.date_input(
            "📅 תאריך סימולציה:",
            value=st.session_state.date,
            min_value=date(2023, 1, 1),
            max_value=date(2024, 12, 31),
        )
        st.session_state.top_limit = st.number_input(
            "Top N / Limit:",
            min_value=1,
            max_value=10000,
            value=st.session_state.top_limit,
            step=1,
        )

        st.markdown("---")
        if st.button("🧹 נקה תצוגה (מקומי)", use_container_width=True):
            st.session_state.chat = []
            st.success("נוקתה התצוגה.")

    # --- Main Chat Area ---

    # מחקנו מכאן את החלק של עריכת הכותרת ("שם השיחה" + "שמור")

    if st.session_state.current_chat_id is None:
        st.markdown("### 🚌 התחל שיחה חדשה")

    # תצוגת היסטוריית ההודעות הקיימות
    for m in st.session_state.chat:
        render_chat_message(m)

    # קלט המשתמש
    user_msg = st.chat_input(
        "שאלה (עברית/English)...",
        key="agent_input"
    )

    if user_msg:
        # 1. שמירה ב-State + DB
        msg_obj = append_chat_message("user", user_msg)

        # --- התיקון לבעיה מס' 2 ---
        # אנחנו מציירים את ההודעה החדשה מיד, ידנית,
        # כדי שהמשתמש יראה אותה בזמן שהסוכן חושב
        render_chat_message(msg_obj)
        # ---------------------------

        # 2. טיפול בשיחה חדשה (יצירת ID וכותרת)
        if st.session_state.current_chat_id is None:
            auto_title = generate_chat_title(user_msg)
            new_id = create_conversation(user_id, auto_title)
            st.session_state.current_chat_id = new_id
            # שומרים רטרואקטיבית את ההודעה הראשונה ל-DB
            save_message(new_id, "user", user_msg)

        # 3. חישוב התשובה (בזמן הזה ההודעה של המשתמש כבר מוצגת)
        with fancy_spinner("מבצע את הבקשה..."):
            answer(user_msg)

        # 4. רענון סופי להצגת התשובה
        st.rerun()


if __name__ == "__main__":
    main()