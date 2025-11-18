from __future__ import annotations

import uuid
from datetime import date as dt_date, timedelta, date
from typing import Any
import datetime as dt  # ← להוסיף שורה זו

import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

from .config import resolve_openai_key
from .humanize import paraphrase_he, pretty_bus_id
from .intents import detect_intents
from .agent_queries import (
    df_at_risk_today,
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

# -------------------------------------------------
# הגדרות בסיס ל-Streamlit
# -------------------------------------------------

st.set_page_config(
    page_title="🚌 תחזוקה חכמה — Agent",
    page_icon="🚌",
    layout="wide",
)

OPENAI_API_KEY = resolve_openai_key()

# טווח הדאטה בפועל (סימולציה)
SIM_MIN_DATE = dt_date(2023, 1, 1)
SIM_MAX_DATE = dt_date(2024, 12, 31)


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
    """
    מחלץ מספר ימים מהשאלה ('11 ימים', '11 יום', '11 days').
    אם לא מוצא – מחזיר default.
    """
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
    """
    עדכון מהיר למילים 'שבועיים' ו'שבוע' בשאלות טבעיות.
    """
    t = (text or "").replace("?", "").replace("!", "").strip()
    if "שבועיים" in t:
        return 14
    if "שבוע" in t and "שבועיים" not in t:
        return 7
    return default


def add_table(title: str, df: pd.DataFrame) -> None:
    """
    מוסיף טבלה להודעת ה-Agent האחרונה בצ'אט.
    """
    if "chat" not in st.session_state or not st.session_state.chat:
        return

    last_msg = st.session_state.chat[-1]
    if last_msg.get("role") != "assistant":
        # אם מסיבה כלשהי ההודעה האחרונה היא של המשתמש – לא נצרף
        return

    tables = last_msg.setdefault("tables", [])
    tables.append(
        {
            "id": str(uuid.uuid4()),
            "title": title,
            "df": df.copy(),
        }
    )


def render_all_tables() -> None:
    """מציג את כל הטבלאות שנשמרו + כפתורי הורדה."""
    for item in st.session_state.get("tables_store", []):
        st.markdown(f"**{item['title']}**")
        st.dataframe(
            item["df"],
            use_container_width=True,
            height=320,
            key=f"df_{item['id']}",
        )
        csv = item["df"].to_csv(index=False).encode("utf-8")
        safe_name = f"{item['title'].replace(' ', '_')}.csv"
        st.download_button(
            "⬇️ הורד CSV",
            csv,
            file_name=safe_name,
            key=f"dl_{item['id']}",
        )


def render_failures_matrix(detail: pd.DataFrame, title: str) -> None:
    """
    טבלה מסודרת של תקלות בפועל:
    date, bus, failure_type, fault_category, failure_flag, maintenance_flag.
    עם st-aggrid.
    """
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


import re as _re2


def _has_hebrew(text: str) -> bool:
    return bool(_re2.search(r"[\u0590-\u05FF]", text or ""))


def render_chat_message(msg: dict[str, Any]) -> None:
    """
    מציג הודעת צ'אט כבועה, כולל שעה מתחת לטקסט
    וטבלאות שקשורות להודעה (אם יש).
    """
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

    # בועת טקסט
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

    # טבלאות ששייכות להודעת Agent זו
    if role == "assistant":
        for t in tables:
            st.markdown(f"**{t['title']}**")
            st.dataframe(
                t["df"],
                width="stretch",
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
def append_chat_message(role: str, text: str) -> dict[str, Any]:
    """יוצר אובייקט הודעה עם שעה ושומר אותו ב-session_state."""
    if "chat" not in st.session_state:
        st.session_state.chat: list[dict[str, Any]] = []
    msg = {
        "id": str(uuid.uuid4()),
        "role": role,
        "text": text,
        "ts": dt.datetime.now().strftime("%H:%M"),  # שעה:דקה
        "tables": [],  # כאן נצרף טבלאות להודעת Agent
    }
    st.session_state.chat.append(msg)
    return msg


def say(text: str) -> None:
    """מוסיף הודעת עוזר לצ'אט ומציג אותה במסך."""
    msg = append_chat_message("assistant", text)
    render_chat_message(msg)


# -------------------------------------------------
# סיכום תקופה – מה קרה בשבוע / שבועיים / טווח טבעי
# -------------------------------------------------


def handle_period_question(query: str, today: dt_date, intents: dict[str, Any]) -> None:
    """
    לוגיקה מרוכזת לשאלות כמו:
    'מה קרה בשבועיים האחרונים', 'מה קרה בשבוע האחרון', 'מה קרה ב־X הימים האחרונים'
    """
    rng = intents.get("RESOLVED_RANGE")

    if rng:
        start, end, title = rng
    else:
        # אם אין טווח מפורש – ננחש ימים מהשאלה / INTENT
        days = intents.get("DAYS")
        if not days:
            days = _extract_days_from_query(query, default=14)
            days = _guess_days_hebrew(query, default=days)
        end = today
        start = today - timedelta(days=days - 1)
        title = f"{days} הימים האחרונים"

    # ולידציה מול סימולציה ודאטה
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

    # ----- KPI בסיסיים -----
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

    # ----- מגמת סיכון לאורך הזמן -----
    if not trend.empty:
        add_table(f"מגמות סיכון ({title})", trend)
        first, last = trend.iloc[0]["pct_risk"], trend.iloc[-1]["pct_risk"]
        delta = last - first
        direction = "עלייה" if delta >= 0 else "ירידה"
        say(
            paraphrase_he(
                f"במהלך {title} נראית {direction} בשיעור האוטובוסים בסיכון: "
                f"מ־{first:.1f}% ל־{last:.1f}%."
            )
        )

    # ----- טבלת סיכום לפי אוטובוס ויום (מתחזיות) -----
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
            f"סיכום לפי אוטובוס ויום ({title}) – Top 100 לפי סיכון לשבוע",
            risk_sorted[cols].head(100),
        )

    # ----- מטריקס תקלות בפועל לפי יום ואוטובוס -----
    if not detail.empty:
        render_failures_matrix(detail, title="פירוט תקלות בפועל לפי יום ואוטובוס")

    if trend.empty and not (risk.empty and detail.empty):
        # אין מגמה, אבל יש נתונים – נציין סיכום קצר
        say(f"הצגתי סיכום תקלות וסיכונים עבור {title}.")


# -------------------------------------------------
# לוגיקת המענה (Agent)
# -------------------------------------------------


def answer(query: str):
    today: dt_date = st.session_state.date
    intents = detect_intents(query, today, st.session_state.top_limit)
    log_agent("Detected intents", **intents)
    top_n = intents.get("TOP_N", st.session_state.top_limit)

    # ✅ מקרה מיוחד: "מי בסיכון לתקלות במזגן/בלמים" וכו'
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
            title = (
                f"Top {top_n} אוטובוסים בסיכון גבוה "
                f"({', '.join(fault_cats)}) ב-{days} הימים האחרונים"
            )
            add_table(title, df)
            say("הצגתי את האוטובוסים עם הסיכון הגבוה ביותר לפי סוגי התקלות שביקשת.")
        return

    # מי בסיכון היום (לכל התקלות)
    if intents.get("WHO_AT_RISK_TODAY"):
        with fancy_spinner("מביא את האוטובוסים בסיכון היום..."):
            df = df_at_risk_today(today, top_n)
        if df.empty:
            say(f"לא נמצאו אוטובוסים עם סיכון ≥ 50% בתאריך {today}.")
        else:
            add_table(
                f"מי בסיכון היום (≥50%) — {today}",
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
            say("סיכום יומי הוצג.")
        return

    # כל התקלות בפועל לאוטובוס מסוים
    if intents.get("BUS_ALL_FAILURES") and intents.get("BUS_ID"):
        bus_id = intents["BUS_ID"]
        nice_id = pretty_bus_id(bus_id)

        with fancy_spinner(f"מביא את כל התקלות של {nice_id} בכל התקופה..."):
            df_hist = df_bus_all_failures(bus_id)

        if df_hist.empty:
            say(f"לא מצאתי תקלות מתועדות עבור {nice_id} בכל התקופה.")
        else:
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
            say(f"הצגתי טבלה עם כל התקלות של {nice_id} בכל התקופה.")
        return

    # BUS ספציפי – מצב היום / היסטוריה מהתחזיות
    if intents.get("BUS_ID"):
        bus_id = intents["BUS_ID"]
        nice_id = pretty_bus_id(bus_id)

        # בדיקה אם המשתמש ביקש "כל התקלות" בהקשר של המודל
        want_all_failures_model = bool(
            _re2.search(r"כל\s+התקלות|כל\s+התקולות|all\s+failures", query, _re2.IGNORECASE)
        )

        if want_all_failures_model:
            with fancy_spinner(f"מביא היסטוריה של תחזיות/תקלות עבור {nice_id}..."):
                dfb = df_bus_history(bus_id, limit=200)

            if dfb.empty:
                say(f"לא נמצאו נתוני תחזיות היסטוריים עבור {nice_id}.")
            else:
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
                say("הצגתי היסטוריה של תחזיות/סיכונים לאוטובוס הזה.")
            return

        # ברירת מחדל – מצב היום בלבד
        with fancy_spinner(f"מחשב סיכון עבור {nice_id}..."):
            dfb = df_bus_today(today, bus_id)

        if dfb.empty:
            say(f"לא נמצאו נתונים עבור {nice_id} בתאריך {today}.")
        else:
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
            r = dfb.iloc[0]
            msg = paraphrase_he(
                f"**{nice_id} — {today}**: p7={r.proba_7d:.3f}"
                + (f", p30={r.proba_30d:.3f}" if pd.notnull(r.proba_30d) else "")
                + f". {r.explanation_he}"
            )
            say(msg)
        return

    # החלקים שהוחלפו הכי הרבה בחודש האחרון
    if intents.get("MOST_REPLACED_PARTS"):
        with fancy_spinner("סורק החלפות חלקים בחודש האחרון..."):
            dfp = df_parts_replaced_last_30d(today, 20)
        if dfp.empty:
            say("לא נמצאו החלפות חלקים בחודש האחרון.")
        else:
            add_table(
                f"החלקים שהוחלפו הכי הרבה — 30 יום אחרונים עד {today}",
                dfp,
            )
            top = dfp.iloc[0]
            say(
                paraphrase_he(
                    f'החלק שהוחלף הכי הרבה בחודש האחרון: {top.part_name} (סה"כ {int(top.replaced_count)} החלפות).'
                )
            )
        return

    # אוטובוסים עם הכי הרבה תקלות בפועל
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
            title_str = "אוטובוסים עם הכי הרבה תקלות"
            if ft_list:
                title_str += f" ({', '.join(ft_list)})"
            title_str += f" - {title}"
            add_table(title_str, df)
            top_row = df.iloc[0]
            say(
                paraphrase_he(
                    f"האוטובוס עם הכי הרבה תקלות הוא {top_row.bus_id} "
                    f"עם {int(top_row.failure_count)} תקלות מתועדות בתקופה {title}."
                )
            )
        return

    # טווח טבעי / "מה קרה בשבוע / בשבועיים האחרונים"
    if intents.get("ANY_NATURAL_RANGE") or intents.get("WHAT_HAPPENED_LAST_DAYS"):
        handle_period_question(query, today, intents)
        return

    # Top N היום (סיכון גבוה ביותר)
    if intents.get("TOP_LIST") or intents.get("HIGHEST_RISK_N"):
        n = intents.get("TOP_N", intents.get("TOP_N_TEXT", st.session_state.top_limit))
        with fancy_spinner(f"מחשב Top {n} לסיכון היום..."):
            df = df_at_risk_today(today, n)
        if df.empty:
            say(f"לא נמצאו נתונים ל-{today}.")
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

    # Fallback Agent – שאילתא כללית
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
            add_table(
                shared_state.LAST_AGENT_TITLE or "תוצאה (Agent)",
                df,
            )
            say("פענחתי את הבקשה בעזרת הסוכן והצגתי טבלה מתאימה.")
        return

    say(
        "לא זיהיתי בקשה. נסה: “מי בסיכון היום?”, “BUS 17”, "
        "“מה קרה בשבוע האחרון?”, “Top 10”, "
        "או “לאילו אוטובוסים יש סיכון גבוה לתקלות במזגן ובבלמים?”."
    )


# -------------------------------------------------
# main – נקודת כניסה אחת
# -------------------------------------------------


def main() -> None:
    # אתחול state
    if "chat" not in st.session_state:
        st.session_state.chat: list[dict[str, Any]] = []
    if "date" not in st.session_state:
        st.session_state.date = dt_date(2024, 12, 30)
    if "top_limit" not in st.session_state:
        st.session_state.top_limit = 10

    # Sidebar
    with st.sidebar:
        st.subheader("הגדרות")
        st.session_state.date = st.date_input(
            "📅 תאריך סימולציה:",
            value=st.session_state.date,
            min_value=date(2023, 1, 1),
            max_value=date(2024, 12, 31)
        )
        st.session_state.top_limit = st.number_input(
            "Top N (לרשימות או LIMIT לסוכן):",
            1,
            500,
            st.session_state.top_limit,
            1,
        )
        st.caption(
            "✅ OpenAI key loaded"
            if OPENAI_API_KEY
            else "ℹ️ ללא OpenAI (ניסוח בסיסי בלבד)"
        )
        st.markdown("---")
        if st.button("🧹 נקה טבלאות מוצגות", width="stretch"):
            if "chat" in st.session_state:
                for m in st.session_state.chat:
                    if "tables" in m:
                        m["tables"] = []
            st.success("נוקו הטבלאות מהתצוגה.")

    # כותרת + הסבר
    st.markdown("### 🚌 תחזוקה חכמה — Agent (Modular)")
    st.info(
        "דוגמאות: “מי בסיכון היום?”, “BUS 17 / אוטובוס 9”, "
        "“איזה חלקים הוחלפו הכי הרבה?”, “מה קרה בשבוע האחרון?”, "
        "“לאיזה אוטובוס היו הכי הרבה תקלות במזגן בשבוע האחרון?”."
    )

    # הצגת 10 ההודעות האחרונות
    for m in st.session_state.chat[-10:]:
        render_chat_message(m)

    st.markdown("---")

    # קלט מהמשתמש
    user_msg = st.chat_input(
        "שאלה (עברית/English). אפשר BUS, 'מי בסיכון', 'חלקים הוחלפו', 'Top', 'מה קרה', 'הכי הרבה תקלות'."
    )
    if user_msg:
        msg = append_chat_message("user", user_msg)
        render_chat_message(msg)
        with fancy_spinner("מבצע את הבקשה..."):
            answer(user_msg)



if __name__ == "__main__":
    main()
