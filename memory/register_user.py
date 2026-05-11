"""
Upserts a Discord user into the users table.
Called from main.py on every message (safe to call repeatedly).
"""
import os
import sqlite3
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "ryo.db")


def register_user(discord_id: str, username: str, display_name: str) -> bool:
    """Returns True if this is a new registration."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute("SELECT discord_id FROM users WHERE discord_id = ?", (discord_id,))
    exists = cur.fetchone() is not None
    if not exists:
        conn.execute(
            "INSERT INTO users (discord_id, username, display_name, registered_at) VALUES (?, ?, ?, ?)",
            (discord_id, username, display_name, datetime.now().isoformat()),
        )
        conn.commit()
    conn.close()
    return not exists
