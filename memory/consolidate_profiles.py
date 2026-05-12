"""
Profile Consolidation Agent

Periodically scans user_profile for semantically overlapping or redundant keys
and merges them using Claude. Runs as a background task or on-demand.

Strategy:
- Scalar keys with similar meaning → keep the more complete value
- Custom keys that overlap with a canonical bucket → absorb into the bucket
- Genuinely distinct keys → leave alone
"""
import asyncio
import json
import os
import sqlite3

import anthropic

DB_PATH = os.path.join(os.path.dirname(__file__), "ryo.db")


def _get_all_users_with_profiles() -> list[str]:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT DISTINCT discord_id FROM user_profile"
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]


async def consolidate_user(discord_id: str, dry_run: bool = False) -> list[str]:
    """
    Analyse and merge semantically similar profile keys for one user.
    Returns a list of human-readable merge descriptions.
    dry_run=True returns what WOULD be merged without writing.
    """
    # Import here to avoid circular imports when used as standalone script
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from memory import profile_db

    profile = profile_db.get_all(discord_id)
    if len(profile) < 2:
        return []

    canonical_desc = "\n".join(
        f"  {k}: {v}" for k, v in profile_db.CANONICAL_KEYS.items()
    )
    profile_text = "\n".join(f"  {k}: {v}" for k, v in profile.items())

    prompt = f"""You are a profile data analyst. A user has these stored profile key-value pairs:

{profile_text}

Canonical bucket/scalar keys available (prefer these as merge targets):
{canonical_desc}

Task: identify keys that are semantically overlapping or redundant and should be merged.

Rules:
1. Only merge keys where the data genuinely belongs together (e.g. "hobby_running" and "sports" both contain physical activity data).
2. Prefer canonical keys as the merge target over custom keys.
3. For bucket merges: combine values as a comma-separated, deduplicated list.
4. For scalar merges: keep the more complete/specific value.
5. Do NOT merge keys that are semantically distinct (e.g. "hobbies" and "interests" can stay separate if they have different content).
6. Do NOT merge if unsure — false merges lose data.

Return ONLY a JSON array (empty array if nothing to merge):
[
  {{
    "target_key": "best_canonical_or_existing_key",
    "absorb_keys": ["key_to_remove_1", "key_to_remove_2"],
    "merged_value": "deduplicated combined value",
    "reason": "one-line explanation"
  }}
]"""

    client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    resp = await client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = resp.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        merges = json.loads(raw)
    except json.JSONDecodeError:
        return [f"Parse error for user {discord_id}: {raw[:100]}"]

    if not merges:
        return []

    results = []
    for m in merges:
        target = profile_db.normalize_key(m["target_key"])
        absorb = [profile_db.normalize_key(k) for k in m.get("absorb_keys", [])]
        value = m["merged_value"]
        reason = m.get("reason", "")

        desc = f"{absorb} → {target}: '{value}' ({reason})"
        results.append(desc)

        if not dry_run:
            profile_db.set_value(discord_id, target, value)
            for old_key in absorb:
                if old_key != target:
                    profile_db.delete_value(discord_id, old_key)

    return results


async def consolidate_all(dry_run: bool = False) -> dict[str, list[str]]:
    """Run consolidation for every user that has profile data."""
    users = _get_all_users_with_profiles()
    report: dict[str, list[str]] = {}
    for discord_id in users:
        merges = await consolidate_user(discord_id, dry_run=dry_run)
        if merges:
            report[discord_id] = merges
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Consolidate user profile keys")
    parser.add_argument("--dry-run", action="store_true", help="Show merges without applying")
    parser.add_argument("--user-id", help="Only consolidate this user")
    args = parser.parse_args()

    async def _main():
        if args.user_id:
            merges = await consolidate_user(args.user_id, dry_run=args.dry_run)
            if merges:
                print(f"{'[DRY RUN] ' if args.dry_run else ''}Merges for {args.user_id}:")
                for m in merges:
                    print(f"  {m}")
            else:
                print("Nothing to merge.")
        else:
            report = await consolidate_all(dry_run=args.dry_run)
            if report:
                for uid, merges in report.items():
                    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}User {uid}:")
                    for m in merges:
                        print(f"  {m}")
            else:
                print("All profiles are clean.")

    asyncio.run(_main())
