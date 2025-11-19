from __future__ import annotations
import uuid
import json
import pandas as pd
from sqlalchemy import text
from .db import engine
from .config import OPENAI_MODEL, resolve_openai_key

OPENAI_API_KEY = resolve_openai_key()


# --- ניהול שיחות ---

def create_conversation(user_id: str, title: str = "New Chat") -> str:
    """יוצר שיחה חדשה עבור משתמש ספציפי"""
    chat_id = str(uuid.uuid4())
    sql = """
        INSERT INTO public.chat_conversations (id, user_id, title)
        VALUES (:id, :uid, :title)
    """
    with engine.begin() as conn:
        conn.execute(text(sql), {"id": chat_id, "uid": user_id, "title": title})
    return chat_id


def list_conversations(user_id: str) -> list[dict]:
    """מחזיר רק את השיחות של המשתמש הנוכחי"""
    sql = """
        SELECT id, title 
        FROM public.chat_conversations 
        WHERE user_id = :uid 
        ORDER BY created_at DESC
    """
    with engine.begin() as conn:
        df = pd.read_sql(text(sql), conn, params={"uid": user_id})
    return df.to_dict(orient="records")


def update_conversation_title(chat_id: str, new_title: str):
    sql = "UPDATE public.chat_conversations SET title = :title WHERE id = :id"
    with engine.begin() as conn:
        conn.execute(text(sql), {"title": new_title, "id": chat_id})


def delete_conversation(chat_id: str):
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM public.chat_conversations WHERE id = :id"), {"id": chat_id})


# --- ניהול הודעות (ללא שינוי מהותי, רק הסדר הטוב) ---

def save_message(chat_id: str, role: str, text_content: str, tables: list = None):
    msg_id = str(uuid.uuid4())
    tables_data = []
    if tables:
        for t in tables:
            # שומרים את ה-DataFrame כ-JSON כדי שנוכל לשחזר אותו
            tables_data.append({
                "id": t["id"],
                "title": t["title"],
                "df_json": t["df"].to_json(orient="split", date_format="iso")
            })

    meta_json = json.dumps({"tables": tables_data}) if tables_data else None

    sql = """
        INSERT INTO public.chat_messages (id, conversation_id, role, content, meta_json)
        VALUES (:id, :cid, :role, :content, :meta)
    """
    with engine.begin() as conn:
        conn.execute(text(sql), {
            "id": msg_id, "cid": chat_id, "role": role,
            "content": text_content, "meta": meta_json
        })


def load_messages(chat_id: str) -> list[dict]:
    sql = """
        SELECT role, content, meta_json, created_at 
        FROM public.chat_messages 
        WHERE conversation_id = :cid 
        ORDER BY created_at ASC
    """
    with engine.begin() as conn:
        df = pd.read_sql(text(sql), conn, params={"cid": chat_id})

    messages = []
    for _, row in df.iterrows():
        tables = []
        if row["meta_json"]:
            try:
                meta = json.loads(row["meta_json"]) if isinstance(row["meta_json"], str) else row["meta_json"]
                for rt in meta.get("tables", []):
                    tables.append({
                        "id": rt["id"],
                        "title": rt["title"],
                        "df": pd.read_json(rt["df_json"], orient="split")
                    })
            except Exception:
                pass

        messages.append({
            "id": str(uuid.uuid4()),
            "role": row["role"],
            "text": row["content"],
            "ts": row["created_at"].strftime("%H:%M"),
            "tables": tables
        })
    return messages


# --- יצירת כותרת אוטומטית ---

def generate_chat_title(first_user_msg: str) -> str:
    if not OPENAI_API_KEY:
        return "New Chat"
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.5,
            messages=[
                {"role": "system",
                 "content": "Summarize user prompt in 3-5 words for a chat title. Keep same language. No quotes."},
                {"role": "user", "content": first_user_msg}
            ],
            max_tokens=20
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return "New Chat"