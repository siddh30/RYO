"""
Prints permanent memories and reminders for a given user.
Includes legacy records (discord_id IS NULL) visible to all users.
Called by the CEO agent via Bash at the start of each conversation.
"""
import argparse
import os
import sqlite3

DB_PATH = os.path.join(os.path.dirname(__file__), "ryo.db")


def fetch(conn: sqlite3.Connection, table: str, discord_id: str) -> list[dict]:
    conn.row_factory = sqlite3.Row
    cur = conn.execute(
        f"SELECT * FROM {table} WHERE discord_id = ? OR discord_id IS NULL ORDER BY id",
        (discord_id,),
    )
    return [dict(r) for r in cur.fetchall()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-id", required=True, help="Discord user ID")
    args = parser.parse_args()

    if not os.path.exists(DB_PATH):
        print("No memory database found.")
        return

    conn = sqlite3.connect(DB_PATH)
    memories = fetch(conn, "permanent_memories", args.user_id)
    reminders = fetch(conn, "reminders", args.user_id)
    conn.close()

    print("=== PERMANENT MEMORIES ===")
    if memories:
        for m in memories:
            print(f"[{m['index_title']}] {m['context']}")
    else:
        print("(none)")

    print("\n=== REMINDERS ===")
    if reminders:
        for r in reminders:
            print(f"[{r['index_title']}] Until {r['remember_window']}: {r['context']}")
    else:
        print("(none)")


if __name__ == "__main__":
    main()
