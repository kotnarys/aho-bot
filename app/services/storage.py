"""Session and request storage — SQLite for persistence across restarts."""

from __future__ import annotations
import json
import sqlite3
import logging
import os
from typing import Optional
from app.models.schemas import ConversationState

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "aho_bot.db")


def _get_db() -> sqlite3.Connection:
    db_path = os.path.abspath(DB_PATH)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = _get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            employee_email TEXT PRIMARY KEY,
            session_data TEXT NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS requests (
            request_id TEXT PRIMARY KEY,
            employee_email TEXT,
            request_type TEXT,
            request_data TEXT NOT NULL,
            status TEXT DEFAULT 'submitted',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_email TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    conn.close()
    logger.info("SQLite database initialized")


# ── Session operations ─────────────────────────────────

def save_session(state: ConversationState) -> None:
    key = state.employee_email or state.session_id
    data = state.model_dump_json()
    conn = _get_db()
    conn.execute(
        "INSERT OR REPLACE INTO sessions (employee_email, session_data, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)",
        (key, data),
    )
    conn.commit()
    conn.close()


def load_session(session_id: str, employee_email: str | None = None) -> Optional[ConversationState]:
    conn = _get_db()
    key = employee_email or session_id
    row = conn.execute("SELECT session_data FROM sessions WHERE employee_email = ?", (key,)).fetchone()
    conn.close()
    if row:
        return ConversationState.model_validate_json(row["session_data"])
    return None


def delete_session(key: str) -> None:
    conn = _get_db()
    conn.execute("DELETE FROM sessions WHERE employee_email = ?", (key,))
    conn.commit()
    conn.close()


# ── Chat history ───────────────────────────────────────

def save_chat_message(employee_email: str, role: str, content: str) -> None:
    conn = _get_db()
    conn.execute(
        "INSERT INTO chat_history (employee_email, role, content) VALUES (?, ?, ?)",
        (employee_email, role, content),
    )
    conn.commit()
    conn.close()


def load_chat_history(employee_email: str, limit: int = 50) -> list[dict]:
    conn = _get_db()
    rows = conn.execute(
        "SELECT role, content, created_at FROM chat_history WHERE employee_email = ? ORDER BY id DESC LIMIT ?",
        (employee_email, limit),
    ).fetchall()
    conn.close()
    # Reverse to get chronological order
    return [{"role": r["role"], "content": r["content"], "created_at": r["created_at"]} for r in reversed(rows)]


def clear_chat_history(employee_email: str) -> None:
    conn = _get_db()
    conn.execute("DELETE FROM chat_history WHERE employee_email = ?", (employee_email,))
    conn.commit()
    conn.close()


# ── Request (ticket) storage ──────────────────────────

def save_request(request_id: str, data: dict, employee_email: str = "") -> None:
    conn = _get_db()
    conn.execute(
        "INSERT OR REPLACE INTO requests (request_id, employee_email, request_type, request_data, status) VALUES (?, ?, ?, ?, ?)",
        (request_id, employee_email or data.get("employee_email", ""), data.get("type", ""), json.dumps(data, ensure_ascii=False, default=str), data.get("status", "submitted")),
    )
    conn.commit()
    conn.close()


def load_request(request_id: str) -> Optional[dict]:
    conn = _get_db()
    row = conn.execute("SELECT request_data FROM requests WHERE request_id = ?", (request_id,)).fetchone()
    conn.close()
    if row:
        return json.loads(row["request_data"])
    return None


def list_requests(employee_email: str | None = None) -> list[dict]:
    conn = _get_db()
    if employee_email:
        rows = conn.execute("SELECT request_data FROM requests WHERE employee_email = ? ORDER BY created_at DESC", (employee_email,)).fetchall()
    else:
        rows = conn.execute("SELECT request_data FROM requests ORDER BY created_at DESC").fetchall()
    conn.close()
    return [json.loads(r["request_data"]) for r in rows]
