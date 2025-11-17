
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

SCHEMA_PATH = Path(__file__).with_name("db_schema.json")

with SCHEMA_PATH.open(encoding="utf-8") as f:
    SCHEMA: dict[str, Any] = json.load(f)

SCHEMA_NAME: str = SCHEMA.get("schema", "public")
TABLES_META: dict[str, Any] = SCHEMA.get("tables", {})


def allowed_tables_from_schema() -> set[str]:
    return {f"{SCHEMA_NAME}.{name}" for name in TABLES_META.keys()}


def schema_summary_for_llm() -> str:
    lines: list[str] = []
    for tname, meta in TABLES_META.items():
        cols = meta.get("columns", [])
        cols_str = ", ".join(f"{c.get('name')}:{c.get('type')}" for c in cols)
        lines.append(f"- {tname}({cols_str})")
    return "\n".join(lines)
