#!/usr/bin/env python3
"""
CLI to set or delete a user profile key-value pair.

Usage:
  python memory/update_profile.py --user-id 123 --key preferred_name --value "Sid"
  python memory/update_profile.py --user-id 123 --key preferred_name --delete
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
    parser.add_argument("--key", help="Profile key (will be normalized to canonical form)")
    parser.add_argument("--value", help="Value to store")
    parser.add_argument("--delete", action="store_true", help="Delete this key")
    parser.add_argument("--list", action="store_true", help="Print all profile keys for this user")
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
        if deleted:
            print(f"✅ Removed profile key: {canonical}")
        else:
            print(f"No profile key '{canonical}' found for this user")
        return

    if not args.value:
        print("Error: --value is required unless --delete or --list is used", file=sys.stderr)
        sys.exit(1)

    canonical = profile_db.set_value(args.user_id, args.key, args.value)
    print(f"✅ Profile updated — {canonical}: {args.value}")


if __name__ == "__main__":
    main()
