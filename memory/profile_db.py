"""
Structured user profile store — key/value pairs per user.

Canonical keys are normalized so different phrasings always land on the same key.
Unknown keys are stored as-is (snake_case), enabling custom per-user fields.
"""
import os
import sqlite3

DB_PATH = os.path.join(os.path.dirname(__file__), "ryo.db")

# Canonical key → human label
CANONICAL_KEYS: dict[str, str] = {
    "preferred_name":  "Preferred name / what to call the user",
    "actual_name":     "Full / legal name",
    "location":        "City or area where the user lives",
    "role":            "Job title and company",
    "email":           "Email address",
    "phone":           "Phone number",
    "age":             "Age or birth year",
    "timezone":        "Timezone",
    "home_airport":    "Home airport",
    "nationality":     "Nationality / citizenship",
    "languages":       "Languages spoken",
}

# Alias → canonical key  (all lowercase, underscores)
_ALIASES: dict[str, str] = {
    # Name variants
    "nickname":          "preferred_name",
    "nick":              "preferred_name",
    "goes_by":           "preferred_name",
    "call_me":           "preferred_name",
    "known_as":          "preferred_name",
    "refer_to_me_as":    "preferred_name",
    "prefer_to_be_called": "preferred_name",
    "preferred_name":    "preferred_name",
    "full_name":         "actual_name",
    "real_name":         "actual_name",
    "legal_name":        "actual_name",
    "name":              "actual_name",
    "first_name":        "preferred_name",
    # Location variants
    "city":              "location",
    "lives_in":          "location",
    "based_in":          "location",
    "from":              "location",
    "home":              "location",
    "address":           "location",
    "lives":             "location",
    # Role variants
    "job":               "role",
    "occupation":        "occupation",
    "position":          "role",
    "title":             "role",
    "works_as":          "role",
    "works_at":          "role",
    "company":           "role",
    "employer":          "role",
    "job_title":         "role",
    # Other
    "tz":                "timezone",
    "airport":           "home_airport",
    "home_airport":      "home_airport",
    "birth_year":        "age",
    "born":              "age",
    "language":          "languages",
}


def normalize_key(raw: str) -> str:
    """Normalize a raw key string to its canonical form (or snake_case if unknown)."""
    clean = raw.strip().lower().replace(" ", "_").replace("-", "_")
    return _ALIASES.get(clean, clean)


def set_value(discord_id: str, key: str, value: str) -> str:
    """Upsert a profile value. Returns the canonical key used."""
    canonical = normalize_key(key)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT OR REPLACE INTO user_profile (discord_id, key, value, updated_at) "
        "VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
        (discord_id, canonical, value),
    )
    conn.commit()
    conn.close()
    return canonical


def get_value(discord_id: str, key: str) -> str | None:
    """Return a single profile value, or None."""
    canonical = normalize_key(key)
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT value FROM user_profile WHERE discord_id = ? AND key = ?",
        (discord_id, canonical),
    ).fetchone()
    conn.close()
    return row[0] if row else None


def get_all(discord_id: str) -> dict[str, str]:
    """Return all profile key-value pairs for a user."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT key, value FROM user_profile WHERE discord_id = ? ORDER BY key",
        (discord_id,),
    ).fetchall()
    conn.close()
    return {r[0]: r[1] for r in rows}


def delete_value(discord_id: str, key: str) -> bool:
    """Delete a profile key. Returns True if a row was deleted."""
    canonical = normalize_key(key)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute(
        "DELETE FROM user_profile WHERE discord_id = ? AND key = ?",
        (discord_id, canonical),
    )
    conn.commit()
    conn.close()
    return cur.rowcount > 0


def format_for_context(discord_id: str) -> str:
    """Return a compact key: value block for injecting into prompts."""
    profile = get_all(discord_id)
    if not profile:
        return ""
    return "\n".join(f"{k}: {v}" for k, v in profile.items())
