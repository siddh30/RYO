import argparse
import os
import sqlite3
import sys
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "memory", "ryo.db")
DB_PATH = os.path.normpath(DB_PATH)


def main():
    parser = argparse.ArgumentParser(description="Store a memory in ryo.db")
    parser.add_argument("--user-id", required=True, dest="user_id")
    parser.add_argument("--flag", required=True, help="'Permanent' or 'Not Permanent'")
    parser.add_argument("--title", required=True)
    parser.add_argument("--context", required=True)
    parser.add_argument("--window", required=True, help="ISO datetime string")
    parser.add_argument("--ai-message", required=True, dest="ai_message")
    parser.add_argument("--repeat-count", type=int, default=1, dest="repeat_count",
                        help="Times to remind: 1=once, N=N times, -1=until stopped")
    parser.add_argument("--snooze-interval", type=int, default=30, dest="snooze_interval",
                        help="Minutes between repeats (used when repeat_count != 1)")
    args = parser.parse_args()

    is_permanent = args.flag.strip().lower() == "permanent"
    table = "permanent_memories" if is_permanent else "reminders"
    full_title = f"Permanent: {args.title}" if is_permanent else f"Not Permanent: {args.title}"

    try:
        datetime.fromisoformat(args.window)
    except ValueError:
        print(f"Error: invalid --window datetime '{args.window}'", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}. Run memory/setup_db.py first.", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    try:
        if is_permanent:
            conn.execute(
                "INSERT OR REPLACE INTO permanent_memories "
                "(discord_id, date_logged, remember_window, remember_flag, index_title, ai_message, context) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (args.user_id, datetime.now().isoformat(), args.window,
                 args.flag.strip(), full_title, args.ai_message, args.context),
            )
        else:
            conn.execute(
                "INSERT OR REPLACE INTO reminders "
                "(discord_id, date_logged, remember_window, remember_flag, index_title, "
                "ai_message, context, repeat_count, snooze_interval_mins) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (args.user_id, datetime.now().isoformat(), args.window,
                 args.flag.strip(), full_title, args.ai_message, args.context,
                 args.repeat_count, args.snooze_interval),
            )
        conn.commit()
    finally:
        conn.close()

    print(args.ai_message)


if __name__ == "__main__":
    main()
