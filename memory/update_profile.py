#!/usr/bin/env python3
"""
CLI to set, append, delete, or list user profile key-value pairs.

Usage:
  # Scalar (replace):
  python memory/update_profile.py --user-id 123 --key preferred_name --value "Sid"

  # Bucket (append — won't duplicate):
  python memory/update_profile.py --user-id 123 --key hobbies --value "marathon running" --append

  # Delete a key:
  python memory/update_profile.py --user-id 123 --key hobbies --delete

  # List all:
  python memory/update_profile.py --user-id 123 --list
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from memory import profile_db


def main():
    parser = argparse.ArgumentParser(description="Manage structured user profile data")
    parser.add_argument("--user-id", required=True, dest="user_id")
    parser.add_argument("--key", help="Profile key (normalized to canonical form)")
    parser.add_argument("--value", help="Value to store or append")
    parser.add_argument("--append", action="store_true",
                        help="Append to bucket key instead of replacing")
    parser.add_argument("--delete", action="store_true", help="Delete this key")
    parser.add_argument("--list", action="store_true",
                        help="Print all profile keys for this user")
    args = parser.parse_args()

    if args.list:
        profile = profile_db.get_all(args.user_id)
        if not profile:
            print(f"No profile data for user {args.user_id}")
        else:
            for k, v in profile.items():
                print(f"{k}: {v}")
        return

    if not args.key:
        print("Error: --key is required unless --list is used", file=sys.stderr)
        sys.exit(1)

    if args.delete:
        deleted = profile_db.delete_value(args.user_id, args.key)
        canonical = profile_db.normalize_key(args.key)
        print(f"✅ Removed: {canonical}" if deleted else f"Key '{canonical}' not found")
        return

    if not args.value:
        print("Error: --value is required unless --delete or --list is used", file=sys.stderr)
        sys.exit(1)

    # Auto-detect: if key is a known bucket, default to append unless explicitly not
    is_bucket = profile_db.is_bucket(args.key)
    use_append = args.append or (is_bucket and not args.delete)

    if use_append:
        canonical = profile_db.append_value(args.user_id, args.key, args.value)
        current = profile_db.get_value(args.user_id, canonical)
        print(f"✅ Profile updated — {canonical}: {current}")
    else:
        canonical = profile_db.set_value(args.user_id, args.key, args.value)
        print(f"✅ Profile updated — {canonical}: {args.value}")


if __name__ == "__main__":
    main()
