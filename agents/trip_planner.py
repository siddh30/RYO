import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone

import anthropic
import discord

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'memory', 'ryo.db')


def _get_all_travel_prefs(guild_id: int | None = None) -> list[dict]:
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
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def event_title(destination: str, day: int, title: str) -> str:
    return f"{destination}: Day {day} — {title}"


async def get_existing_event_names(guild: discord.Guild) -> dict[str, discord.ScheduledEvent]:
    """Return {event_name: event} for all scheduled events in the guild."""
    try:
        events = await guild.fetch_scheduled_events()
        return {ev.name: ev for ev in events}
    except Exception:
        return {}


async def delete_all_guild_events(guild: discord.Guild) -> int:
    """Delete every scheduled event in the guild. Returns count deleted."""
    events = await guild.fetch_scheduled_events()
    count = 0
    for ev in events:
        try:
            await ev.delete()
            count += 1
        except Exception as e:
            print(f"Could not delete event {ev.name}: {e}")
    return count


async def create_discord_events(
    guild: discord.Guild,
    events: list[dict],
    destination: str,
    start_date: datetime,
    replace_names: set[str] | None = None,
    skip_names: set[str] | None = None,
) -> list[str]:
    """
    Create one Discord Scheduled Event per day.
    replace_names: delete these existing events first, then recreate.
    skip_names: skip these entirely.
    Returns list of created event titles.
    """
    created = []
    tz = timezone.utc
    existing = await get_existing_event_names(guild) if replace_names else {}

    for ev in events:
        name = event_title(destination, ev["day"], ev["title"])

        if skip_names and name in skip_names:
            continue

        if replace_names and name in replace_names and name in existing:
            try:
                await existing[name].delete()
            except Exception:
                pass

        try:
            date = datetime.fromisoformat(ev["date_iso"]).replace(tzinfo=tz)
            start = date.replace(hour=9, minute=0)
            end = date.replace(hour=22, minute=0)
            await guild.create_scheduled_event(
                name=name,
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
