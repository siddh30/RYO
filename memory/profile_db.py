"""
Structured user profile store — key/value pairs per user.

Two key types:
  SCALAR keys  — single value, overwritten on update (e.g. preferred_name, location)
  BUCKET keys  — comma-separated list, new items appended (e.g. hobbies, credit_cards)

Canonical keys are normalized so different phrasings always land on the same key.
Unknown keys are stored as snake_case; if they look like a list (sport, hobby, etc.)
the caller should use append_value() so similar facts accumulate in one place.
"""
import os
import sqlite3

DB_PATH = os.path.join(os.path.dirname(__file__), "ryo.db")

# ── Scalar keys ────────────────────────────────────────────────────────────────
# Single value; upsert replaces the previous one.
SCALAR_KEYS: dict[str, str] = {
    "preferred_name":  "Preferred name / what to call the user",
    "actual_name":     "Full / legal name",
    "location":        "City or area where the user lives",
    "role":            "Job title and company",
    "email":           "Email address",
    "phone":           "Phone number",
    "age":             "Age or birth year",
    "timezone":        "Timezone",
    "home_airport":    "Home airport (IATA code or city)",
    "nationality":     "Nationality / citizenship",
    "bio":             "Short personal bio / summary",
}

# ── Bucket keys ────────────────────────────────────────────────────────────────
# Comma-separated accumulator; append_value() adds without duplicating.
BUCKET_KEYS: dict[str, str] = {
    "languages":           "Languages spoken",
    "hobbies":             "Hobbies and leisure activities",
    "sports":              "Sports and physical activities",
    "interests":           "Topics the user is interested in",
    "skills":              "Professional or technical skills",
    "credit_cards":        "Credit / charge cards the user holds",
    "dietary_preferences": "Dietary preferences or restrictions",
    "allergies":           "Allergies or intolerances",
    "pets":                "Pets owned",
    "goals":               "Personal or professional goals",
    "news_interests":      "News topics the user follows",
    "travel_wishlist":     "Destinations the user wants to visit",
}

CANONICAL_KEYS = {**SCALAR_KEYS, **BUCKET_KEYS}

# ── Alias map ──────────────────────────────────────────────────────────────────
_ALIASES: dict[str, str] = {
    # Name variants → scalar
    "nickname":               "preferred_name",
    "nick":                   "preferred_name",
    "goes_by":                "preferred_name",
    "call_me":                "preferred_name",
    "known_as":               "preferred_name",
    "refer_to_me_as":         "preferred_name",
    "prefer_to_be_called":    "preferred_name",
    "first_name":             "preferred_name",
    "full_name":              "actual_name",
    "real_name":              "actual_name",
    "legal_name":             "actual_name",
    "name":                   "actual_name",
    # Location variants → scalar
    "city":                   "location",
    "lives_in":               "location",
    "based_in":               "location",
    "from":                   "location",
    "home":                   "location",
    "address":                "location",
    "lives":                  "location",
    "hometown":               "location",
    # Role variants → scalar
    "job":                    "role",
    "position":               "role",
    "title":                  "role",
    "works_as":               "role",
    "works_at":               "role",
    "company":                "role",
    "employer":               "role",
    "job_title":              "role",
    "occupation":             "role",
    # Other scalar variants
    "tz":                     "timezone",
    "airport":                "home_airport",
    "birth_year":             "age",
    "born":                   "age",
    # Bucket variants
    "language":               "languages",
    "hobby":                  "hobbies",
    "pastime":                "hobbies",
    "sport":                  "sports",
    "exercise":               "sports",
    "workout":                "sports",
    "fitness":                "sports",
    "interest":               "interests",
    "topic":                  "interests",
    "passion":                "interests",
    "skill":                  "skills",
    "expertise":              "skills",
    "card":                   "credit_cards",
    "credit_card":            "credit_cards",
    "amex":                   "credit_cards",
    "visa":                   "credit_cards",
    "mastercard":             "credit_cards",
    "diet":                   "dietary_preferences",
    "dietary":                "dietary_preferences",
    "food_preference":        "dietary_preferences",
    "allergy":                "allergies",
    "intolerance":            "allergies",
    "pet":                    "pets",
    "goal":                   "goals",
    "objective":              "goals",
    "news":                   "news_interests",
    "news_preference":        "news_interests",
    "wishlist":               "travel_wishlist",
    "bucket_list":            "travel_wishlist",
    "want_to_visit":          "travel_wishlist",
}


def normalize_key(raw: str) -> str:
    """Normalize a raw key string to its canonical form (or snake_case if unknown)."""
    clean = raw.strip().lower().replace(" ", "_").replace("-", "_")
    return _ALIASES.get(clean, clean)


def is_bucket(key: str) -> bool:
    """Return True if key is a bucket (accumulator) key."""
    return normalize_key(key) in BUCKET_KEYS


def set_value(discord_id: str, key: str, value: str) -> str:
    """Upsert a scalar profile value. Returns the canonical key used."""
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


def append_value(discord_id: str, key: str, new_item: str) -> str:
    """Append a new item to a bucket key (comma-separated), skipping duplicates.
    Returns the canonical key used."""
    canonical = normalize_key(key)
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT value FROM user_profile WHERE discord_id = ? AND key = ?",
        (discord_id, canonical),
    ).fetchone()

    if row:
        existing = [v.strip() for v in row[0].split(",") if v.strip()]
        if new_item.strip() not in existing:
            merged = ", ".join(existing + [new_item.strip()])
            conn.execute(
                "UPDATE user_profile SET value = ?, updated_at = CURRENT_TIMESTAMP "
                "WHERE discord_id = ? AND key = ?",
                (merged, discord_id, canonical),
            )
    else:
        conn.execute(
            "INSERT INTO user_profile (discord_id, key, value, updated_at) "
            "VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
            (discord_id, canonical, new_item.strip()),
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
