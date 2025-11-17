
from __future__ import annotations
import uuid
from datetime import date as dt_date
from typing import Any

import streamlit as st
import pandas as pd

from .config import resolve_openai_key
from .humanize import paraphrase_he
from .intents import detect_intents
from .agent_queries import (
    df_at_risk_today,
    df_bus_today,
    df_parts_replaced_last_30d,
    df_trend_last_days,
    df_bus_most_failures,
    run_fallback_agent,
)
from .failure_mapping import map_failure_types_from_query
from .time_range import parse_natural_range
from .utils_logging import log_agent
from . import shared_state

st.set_page_config(page_title="🚌 תחזוקה חכמה — Agent", page_icon="🚌", layout="wide")

OPENAI_API_KEY = resolve_openai_key()

if "chat" not in st.session_state:
    st.session_state.chat: list[dict[str, Any]] = []
if "tables_store" not in st.session_state:
    st.session_state.tables_store: list[dict[str, Any]] = []
if "date" not in st.session_state:
    st.session_state.date = dt_date(2024, 12, 30)
if "top_limit" not in st.session_state:
    st.session_state.top_limit = 10

with st.sidebar:
    st.subheader("הגדרות")
    st.session_state.date = st.date_input("📅 תאריך סימולציה:", value=st.session_state.date)
    st.session_state.top_limit = st.number_input("Top N (לרשימות):", 1, 200, st.session_state.top_limit, 1)
    st.caption("✅ OpenAI key loaded" if OPENAI_API_KEY else "ℹ️ ללא OpenAI (ניסוח בסיסי בלבד)")
    st.markdown("---")
    if st.button("🧹 נקה טבלאות מוצגות", use_container_width=True):
        st.session_state.tables_store = []
        st.success("נוקו הטבלאות מהתצוגה.")

st.markdown("### 🚌 תחזוקה חכמה — Agent (Modular)")
st.info("דוגמאות: “מי בסיכון היום?”, “BUS 17 / אוטובוס 9”, “איזה חלקים הוחלפו הכי הרבה?”, “מה קרה בשבוע האחרון?”, “לאיזה אוטובוס היו הכי הרבה תקלות במזגן בשבוע האחרון?”.")


class fancy_spinner:
    def __init__(self, msg: str = "מעבד את הבקשה..."):
        self.msg = msg
        self.placeholder = st.empty()

    def __enter__(self):
        self.placeholder.markdown(f"🌀 **{self.msg}**")
        self.spinner = st.spinner(self.msg)
        self.spinner.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.spinner.__exit__(exc_type, exc, tb)
        self.placeholder.empty()


def add_table(title: str, df: pd.DataFrame):
    for item in st.session_state.tables_store:
        if item["title"] == title:
            item["df"] = df.copy()
            return
    st.session_state.tables_store.append({"id": str(uuid.uuid4()), "title": title, "df": df.copy()})


def render_all_tables():
    for item in st.session_state.tables_store:
        st.markdown(f"**{item['title']}**")
        st.dataframe(item["df"], width="stretch", height=320, key=f"df_{item['id']}")
        csv = item["df"].to_csv(index=False).encode("utf-8")
        safe_name = f"{item['title'].replace(' ', '_')}.csv"
        st.download_button("⬇️ הורד CSV", csv, file_name=safe_name, key=f"dl_{item['id']}")


def say(text: str):
    st.session_state.chat.append({"role": "assistant", "text": text})
    st.markdown(text)


def answer(query: str):
    today = st.session_state.date
    intents = detect_intents(query, today, st.session_state.top_limit)
    log_agent("Detected intents", **intents)
    top_n = intents.get("TOP_N", st.session_state.top_limit)

    # מי בסיכון היום
    if intents.get("WHO_AT_RISK_TODAY"):
        with fancy_spinner("מביא את האוטובוסים בסיכון היום..."):
            df = df_at_risk_today(today, top_n)
        if df.empty:
            say(f"לא נמצאו אוטובוסים עם סיכון ≥ 50% בתאריך {today}.")
        else:
            add_table(
                f"מי בסיכון היום (≥50%) — {today}",
                df[["bus_id", "d", "predicted_proba", "predicted_label", "failure_reason", "reason_he", "likely_fault", "where_he", "explanation_he"]],
            )
            say("סיכום יומי הוצג.")
        return

    # BUS ספציפי
    if intents.get("BUS_ID"):
        bus_id = intents["BUS_ID"]
        with fancy_spinner(f"מחשב סיכון עבור {bus_id}..."):
            dfb = df_bus_today(today, bus_id)
        if dfb.empty:
            say(f"לא נמצאו נתונים עבור {bus_id} בתאריך {today}.")
        else:
            add_table(
                f"{bus_id} — פירוט {today}",
                dfb[["bus_id", "d", "proba_7d", "label_7d", "proba_30d", "label_30d", "failure_reason", "reason_he", "likely_fault", "where_he", "explanation_he"]],
            )
            r = dfb.iloc[0]
            msg = paraphrase_he(
                f"**{bus_id} — {today}**: p7={r.proba_7d:.3f}"
                + (f", p30={r.proba_30d:.3f}" if pd.notnull(r.proba_30d) else "")
                + f". {r.explanation_he}"
            )
            say(msg)
        return

    # חלקים
    if intents.get("MOST_REPLACED_PARTS"):
        with fancy_spinner("סורק החלפות חלקים בחודש האחרון..."):
            dfp = df_parts_replaced_last_30d(today, 20)
        if dfp.empty:
            say("לא נמצאו החלפות חלקים בחודש האחרון.")
        else:
            add_table(f"החלקים שהוחלפו הכי הרבה — 30 יום אחרונים עד {today}", dfp)
            top = dfp.iloc[0]
            say(
                paraphrase_he(
                    f"החלק שהוחלף הכי הרבה בחודש האחרון: {top.part_name} (סה"כ {int(top.replaced_count)} החלפות)."
                )
            )
        return

    # אוטובוס עם הכי הרבה תקלות (כלליות או לפי סוג תקלה)
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

    # מגמות בטווח טבעי
    if intents.get("ANY_NATURAL_RANGE") or intents.get("WHAT_HAPPENED_LAST_DAYS"):
        rng = intents.get("RESOLVED_RANGE", None)
        if rng:
            start, end, title = rng
            with fancy_spinner(f"מחשב מגמות סיכון עבור {title}..."):
                trend = df_trend_last_days(start, end)
            if trend.empty:
                say(f"לא נמצאו נתונים בטווח: {title}.")
            else:
                add_table(f"מגמות סיכון ({title})", trend)
                first, last = trend.iloc[0]["pct_risk"], trend.iloc[-1]["pct_risk"]
                delta = last - first
                direction = "עלייה" if delta >= 0 else "ירידה"
                say(paraphrase_he(f"במהלך {title} נראית {direction} בשיעור האוטובוסים בסיכון: מ־{first:.1f}% ל־{last:.1f}%.")
                    )
            return

    # Top N היום
    if intents.get("TOP_LIST") or intents.get("HIGHEST_RISK_N"):
        n = intents.get("TOP_N", intents.get("TOP_N_TEXT", st.session_state.top_limit))
        from .agent_queries import SQL_AT_RISK_TODAY, PRED_SRC  # reuse query
        with fancy_spinner(f"מחשב Top {n} לסיכון היום..."):
            df = df_at_risk_today(today, n)
        if df.empty:
            say(f"לא נמצאו נתונים ל־{today}.")
        else:
            add_table(
                f"Top {n} Highest Risk — {today}",
                df[["bus_id", "d", "predicted_proba", "predicted_label", "failure_reason", "reason_he", "likely_fault", "where_he", "explanation_he"]],
            )
            say("הצגתי את האוטובוסים עם הסיכון הגבוה ביותר היום.")
        return

    # Fallback Agent
    with fancy_spinner("🤖🌀 הסוכן חושב ומרכיב שאילתה..."):
        used = run_fallback_agent(query, today, st.session_state.top_limit, intents.get("DAYS"))
    if used and shared_state.LAST_AGENT_DF is not None:
        add_table(shared_state.LAST_AGENT_TITLE or "תוצאה (Agent)", shared_state.LAST_AGENT_DF)
        say("פענחתי את הבקשה בעזרת הסוכן והצגתי טבלה מתאימה.")
        return

    say("לא זיהיתי בקשה. נסה: “מי בסיכון היום?”, “BUS 17”, “מה קרה בשבוע האחרון?”, “Top 10”, “לאיזה אוטובוס היו הכי הרבה תקלות?”")


for m in st.session_state.chat[-10:]:
    if m["role"] == "user":
        st.markdown(f"🗣️ {m['text']}")
    else:
        st.markdown(m["text"])

st.markdown("---")
user_msg = st.chat_input("שאלה (עברית/English). אפשר BUS, 'מי בסיכון', 'חלקים הוחלפו', 'Top', 'מה קרה', 'הכי הרבה תקלות'.")
if user_msg:
    st.session_state.chat.append({"role": "user", "text": user_msg})
    st.markdown(f"🗣️ {user_msg}")
    with fancy_spinner("מבצע את הבקשה..."):
        answer(user_msg)

st.markdown("---")
render_all_tables()
