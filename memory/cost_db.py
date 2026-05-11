import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "ryo.db")


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def load() -> dict:
    conn = _conn()
    row = conn.execute("SELECT * FROM cost_tracking WHERE id = 1").fetchone()
    conn.close()
    return dict(row) if row else {}


def save(delta: dict):
    """Atomically add delta values to running totals."""
    conn = _conn()
    conn.execute(
        """
        UPDATE cost_tracking SET
            total_cost_usd          = total_cost_usd          + :cost,
            total_input_tokens      = total_input_tokens      + :input_tokens,
            total_output_tokens     = total_output_tokens     + :output_tokens,
            total_cache_read_tokens = total_cache_read_tokens + :cache_read_tokens,
            total_messages          = total_messages          + 1,
            last_updated            = :now
        WHERE id = 1
        """,
        {
            "cost": delta.get("total_cost_usd", 0.0),
            "input_tokens": delta.get("input_tokens", 0),
            "output_tokens": delta.get("output_tokens", 0),
            "cache_read_tokens": delta.get("cache_read_tokens", 0),
            "now": datetime.utcnow().isoformat(timespec="seconds"),
        },
    )
    conn.commit()
    conn.close()


def set_credit_balance(amount_usd: float):
    conn = _conn()
    conn.execute(
        "UPDATE cost_tracking SET credit_balance_usd = ? WHERE id = 1",
        (amount_usd,),
    )
    conn.commit()
    conn.close()
