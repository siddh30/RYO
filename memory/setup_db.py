"""
Creates memory/ryo.db and migrates existing CSV data into it.
Safe to re-run — uses INSERT OR IGNORE so existing rows are skipped.
Also adds discord_id column if upgrading from an older schema.
"""
import csv
import os
import sqlite3

DB_PATH = os.path.join(os.path.dirname(__file__), "ryo.db")
PERMANENT_CSV = os.path.join(os.path.dirname(__file__), "Permanent_Memory.csv")
REMINDERS_CSV = os.path.join(os.path.dirname(__file__), "Reminders.csv")

DDL = """
CREATE TABLE IF NOT EXISTS cost_tracking (
    id                      INTEGER PRIMARY KEY CHECK (id = 1),
    total_cost_usd          REAL    DEFAULT 0.0,
    total_input_tokens      INTEGER DEFAULT 0,
    total_output_tokens     INTEGER DEFAULT 0,
    total_cache_read_tokens INTEGER DEFAULT 0,
    total_messages          INTEGER DEFAULT 0,
    credit_balance_usd      REAL    DEFAULT NULL,
    low_credit_alerted      INTEGER DEFAULT 0,
    last_updated            TEXT
);
INSERT OR IGNORE INTO cost_tracking (id) VALUES (1);

CREATE TABLE IF NOT EXISTS users (
    discord_id   TEXT PRIMARY KEY,
    username     TEXT,
    display_name TEXT,
    registered_at TEXT
);

CREATE TABLE IF NOT EXISTS permanent_memories (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    discord_id     TEXT REFERENCES users(discord_id),
    date_logged    TEXT,
    remember_window TEXT,
    remember_flag  TEXT DEFAULT 'Permanent',
    index_title    TEXT,
    ai_message     TEXT,
    context        TEXT,
    UNIQUE(discord_id, index_title)
);

CREATE TABLE IF NOT EXISTS reminders (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    discord_id            TEXT REFERENCES users(discord_id),
    date_logged           TEXT,
    remember_window       TEXT,
    remember_flag         TEXT DEFAULT 'Not Permanent',
    index_title           TEXT,
    ai_message            TEXT,
    context               TEXT,
    repeat_count          INTEGER DEFAULT 1,
    snooze_interval_mins  INTEGER DEFAULT 30,
    UNIQUE(discord_id, index_title)
);

CREATE TABLE IF NOT EXISTS channel_sessions (
    channel_id   INTEGER PRIMARY KEY,
    session_id   TEXT NOT NULL,
    channel_type TEXT NOT NULL DEFAULT 'general',
    updated_at   TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS channel_history (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_id INTEGER NOT NULL,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    ts         TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_channel_history_channel ON channel_history (channel_id, id DESC);

CREATE TABLE IF NOT EXISTS user_profile (
    discord_id  TEXT NOT NULL,
    key         TEXT NOT NULL,
    value       TEXT NOT NULL,
    updated_at  TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (discord_id, key)
);
"""

INSERT = """
INSERT OR IGNORE INTO {table}
    (discord_id, date_logged, remember_window, remember_flag, index_title, ai_message, context)
VALUES (?, ?, ?, ?, ?, ?, ?)
"""


def add_column_if_missing(conn: sqlite3.Connection, table: str, column: str, col_type: str):
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
        print(f"  Added column {column} to {table}")


def _seed_user_profile():
    """One-time seed of user_profile from existing permanent_memories.
    Uses simple heuristics — the skill will keep it updated going forward."""
    import re, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from memory import profile_db

    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT discord_id, index_title, context FROM permanent_memories WHERE context IS NOT NULL"
    ).fetchall()
    conn.close()

    # Patterns: (profile_key, compiled_regex_to_extract_value)
    extractors = [
        ("preferred_name", re.compile(
            r'(?:call me|goes?\s+by|known\s+as|nickname\s+is|refer(?:red)?\s+(?:to\s+me\s+)?as|'
            r'you\s+can\s+call\s+me|prefer(?:red)?\s+(?:name|to\s+be\s+called)\s+(?:is\s+)?)'
            r'\s*([A-Z][a-z]{1,20})\b', re.IGNORECASE)),
        ("actual_name", re.compile(
            r'(?:full\s+name\s+is|my\s+name\s+is|legal\s+name\s+is)\s+([A-Z][a-zA-Z\s]{2,40}?)(?:[,.]|$)',
            re.IGNORECASE)),
        ("location", re.compile(
            r'(?:live[s]?\s+in|based\s+in|located\s+in|from)\s+([A-Z][a-zA-Z\s,]{2,40}?)(?:[,.]|$)',
            re.IGNORECASE)),
        ("role", re.compile(
            r'(?:I\s+am\s+(?:a\s+|an\s+)?|my\s+(?:role|title|position)\s+is\s+)'
            r'([A-Z][a-zA-Z\s,/]{2,60}?)(?:\s+at\s+|\.|$)',
            re.IGNORECASE)),
    ]

    seeded = 0
    for discord_id, index_title, context in rows:
        for key, pattern in extractors:
            m = pattern.search(context)
            if m:
                value = m.group(1).strip().strip('.,')
                if discord_id:
                    existing = profile_db.get_value(discord_id, key)
                    if not existing:
                        profile_db.set_value(discord_id, key, value)
                        seeded += 1

    if seeded:
        print(f"  Seeded {seeded} user_profile entries from permanent_memories.")


def migrate_csv(conn: sqlite3.Connection, csv_path: str, table: str) -> int:
    if not os.path.exists(csv_path):
        return 0
    count = 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            conn.execute(INSERT.format(table=table), (
                None,  # discord_id = NULL for legacy records
                row.get("date_logged", ""),
                row.get("remember_window", ""),
                row.get("remember_flag", ""),
                row.get("index_title", ""),
                row.get("AImessage", ""),
                row.get("context", ""),
            ))
            count += 1
    return count


def main():
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(DDL)

    # Upgrade existing schema if needed
    add_column_if_missing(conn, "permanent_memories", "discord_id", "TEXT")
    add_column_if_missing(conn, "reminders", "discord_id", "TEXT")
    add_column_if_missing(conn, "reminders", "repeat_count", "INTEGER DEFAULT 1")
    add_column_if_missing(conn, "reminders", "snooze_interval_mins", "INTEGER DEFAULT 30")
    add_column_if_missing(conn, "cost_tracking", "low_credit_alerted", "INTEGER DEFAULT 0")

    p = migrate_csv(conn, PERMANENT_CSV, "permanent_memories")
    r = migrate_csv(conn, REMINDERS_CSV, "reminders")

    conn.commit()
    conn.close()

    # Seed user_profile from existing permanent_memories (safe to re-run)
    _seed_user_profile()

    print(f"Database ready: {DB_PATH}")
    print(f"  Migrated {p} permanent memories, {r} reminders.")


if __name__ == "__main__":
    main()
