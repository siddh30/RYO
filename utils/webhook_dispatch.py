import sqlite3
import asyncio
import aiohttp
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'memory', 'ryo.db')

VALID_EVENTS = {"reminder", "low_credit", "new_user", "message"}


def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def add_webhook(event: str, url: str, label: str = "") -> str:
    if event not in VALID_EVENTS:
        return f"Unknown event `{event}`. Valid events: {', '.join(f'`{e}`' for e in sorted(VALID_EVENTS))}"
    conn = _conn()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO webhooks (event, url, label) VALUES (?, ?, ?)",
            (event, url, label),
        )
        conn.commit()
        return f"✅ Webhook added for `{event}`"
    except Exception as e:
        return f"❌ Failed: {e}"
    finally:
        conn.close()


def remove_webhook(event: str) -> str:
    conn = _conn()
    cur = conn.execute("DELETE FROM webhooks WHERE event = ?", (event,))
    conn.commit()
    conn.close()
    return f"✅ Removed webhook(s) for `{event}`" if cur.rowcount else f"No webhook found for `{event}`"


def list_webhooks() -> list[dict]:
    conn = _conn()
    rows = [dict(r) for r in conn.execute("SELECT * FROM webhooks ORDER BY event").fetchall()]
    conn.close()
    return rows


async def dispatch(event: str, payload: dict):
    conn = _conn()
    rows = conn.execute(
        "SELECT url FROM webhooks WHERE event = ?", (event,)
    ).fetchall()
    conn.close()
    if not rows:
        return

    body = {"event": event, "timestamp": datetime.utcnow().isoformat(), **payload}

    async def _post(url: str):
        try:
            async with aiohttp.ClientSession() as session:
                await session.post(url, json=body, timeout=aiohttp.ClientTimeout(total=10))
        except Exception as e:
            print(f"Webhook dispatch error [{event} → {url}]: {e}")

    await asyncio.gather(*[_post(r["url"]) for r in rows])
