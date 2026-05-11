import argparse
import os
import sqlite3
import sys

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "memory", "ryo.db")
DB_PATH = os.path.normpath(DB_PATH)


def main():
    parser = argparse.ArgumentParser(description="Delete a memory from ryo.db")
    parser.add_argument("--user-id", required=True, dest="user_id", help="Discord user ID")
    parser.add_argument("--title", required=True, help="Exact index_title as stored in the database")
    parser.add_argument("--type", required=True, dest="mem_type", help="'permanent' or 'reminder'")
    args = parser.parse_args()

    table = "permanent_memories" if args.mem_type.strip().lower() == "permanent" else "reminders"

    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}.", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    try:
        # Delete only the row belonging to this user (not legacy/global rows)
        cur = conn.execute(
            f"DELETE FROM {table} WHERE index_title = ? AND discord_id = ?",
            (args.title, args.user_id),
        )
        conn.commit()
        deleted = cur.rowcount
    finally:
        conn.close()

    if deleted == 0:
        print(f"No memory found with index_title '{args.title}' for this user.", file=sys.stderr)
        sys.exit(1)

    print(f"Memory '{args.title}' has been forgotten.")


if __name__ == "__main__":
    main()
