import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone

import anthropic
import discord

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'memory', 'ryo.db')


def _get_all_travel_prefs(guild_id: int | None = None) -> list[dict]:
    """Return travel preferences for all users who have them saved."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT u.display_name, pm.context "
        "FROM permanent_memories pm JOIN users u ON pm.discord_id = u.discord_id "
        "WHERE LOWER(pm.index_title) LIKE '%travel%pref%'"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _store_trip_reminders(discord_id: str, trip_name: str, start_date: datetime):
    """Insert pre-trip reminder rows for 7 days, 2 days, and day before departure."""
    checkpoints = [
        (7,  f"1 week until {trip_name}! Check visas, travel docs, and accommodation."),
        (2,  f"2 days until {trip_name}! Start packing and confirm bookings."),
        (1,  f"Tomorrow is {trip_name}! Finish packing, do online check-in."),
    ]
    conn = sqlite3.connect(DB_PATH)
    for days_before, context in checkpoints:
        window = (start_date - timedelta(days=days_before)).isoformat()
        try:
            conn.execute(
                "INSERT OR IGNORE INTO reminders "
                "(discord_id, date_logged, remember_window, remember_flag, index_title, context, repeat_count, snooze_interval_mins) "
                "VALUES (?, ?, ?, 'Not Permanent', ?, ?, 1, 30)",
                (discord_id, datetime.now().isoformat(), window,
                 f"Pre-trip: {trip_name} (-{days_before}d)", context),
            )
        except Exception:
            pass
    conn.commit()
    conn.close()


async def extract_trip_events(itinerary_text: str, destination: str, start_date: datetime) -> list[dict]:
    """
    Ask Claude to parse the itinerary into structured day events.
    Returns list of {day, title, description, date_iso} dicts.
    """
    client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    prompt = (
        f"Extract the day-by-day events from this travel itinerary for {destination} "
        f"starting {start_date.strftime('%Y-%m-%d')}.\n\n"
        f"Return ONLY a JSON array, no other text. Each element:\n"
        f'{{"day": 1, "title": "short title (max 8 words)", "description": "2-3 sentence summary", "date_iso": "YYYY-MM-DD"}}\n\n'
        f"Itinerary:\n{itinerary_text}"
    )
    resp = await client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.content[0].text.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


async def create_discord_events(
    guild: discord.Guild,
    events: list[dict],
    destination: str,
    start_date: datetime,
) -> list[str]:
    """Create one Discord Scheduled Event per day. Returns list of created event names."""
    created = []
    tz = timezone.utc

    for ev in events:
        try:
            date = datetime.fromisoformat(ev["date_iso"]).replace(tzinfo=tz)
            start = date.replace(hour=9, minute=0)
            end = date.replace(hour=22, minute=0)
            await guild.create_scheduled_event(
                name=f"✈️ Day {ev['day']} — {ev['title']}",
                description=ev.get("description", ""),
                start_time=start,
                end_time=end,
                entity_type=discord.EntityType.external,
                privacy_level=discord.PrivacyLevel.guild_only,
                location=destination,
            )
            created.append(ev["title"])
        except Exception as e:
            print(f"Could not create event for day {ev.get('day')}: {e}")

    return created
