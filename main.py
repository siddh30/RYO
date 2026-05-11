import os
import re
import sqlite3
import sys
from datetime import datetime, timedelta

from dateutil import parser as dateparser


sys.path.insert(0, '.')

from config import Config
conf = Config()

from agents.ceo import run_ceo
from agents.vision import run_vision
from agents.trip_planner import (
    _get_all_travel_prefs, _store_trip_reminders,
    extract_trip_events, create_discord_events,
    get_existing_event_names, delete_all_guild_events, event_title,
)
from memory.register_user import register_user
from memory import cost_db

import discord
from discord.ext import tasks

DISCORD_MAX = 1900
DB_PATH = f"{conf.memory_path}/ryo.db"
BILLING_URL = "https://platform.claude.com/settings/billing"

intents = discord.Intents.default()
intents.message_content = True

_totals: dict = {}
_last_cost: dict = {}
_session_messages: int = 0
_stats_messages: dict[int, list[discord.Message]] = {}  # guild_id -> [embed messages]
_channel_dashboards: dict[int, discord.Message] = {}    # channel_id -> persistent dashboard
_pending_travel_prefs: set[str] = set()  # discord_ids awaiting travel pref questionnaire
_pending_trip_events: dict[int, dict] = {}  # channel_id -> pending duplicate-confirmation state


def _sanitize(text: str) -> str:
    text = re.sub(r'^#{1,6}\s+(.+)$', r'**\1**', text, flags=re.MULTILINE)
    text = re.sub(r'^-{3,}$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)', r'\1 — \2', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _chunk(text: str) -> list[str]:
    if len(text) <= DISCORD_MAX:
        return [text]
    chunks, current = [], []
    length = 0
    for line in text.splitlines(keepends=True):
        if length + len(line) > DISCORD_MAX and current:
            chunks.append("".join(current))
            current, length = [], 0
        current.append(line)
        length += len(line)
    if current:
        chunks.append("".join(current))
    return chunks or [""]


def _bar(ratio: float, width: int = 16) -> str:
    filled = max(0, min(width, round(ratio * width)))
    return "█" * filled + "░" * (width - filled)


STATS_COMMANDS = (
    "`!refresh` — refresh this dashboard\n"
    "🔒 `!setcredits <amount>` — set your credit balance\n"
    "🔒 `!clear` — clear messages in this channel\n"
    "-# 🔒 = owner only"
)


def _embed_credits_dashboard() -> discord.Embed:
    balance = _totals.get("credit_balance_usd")
    spent = _totals.get("total_cost_usd", 0.0)
    total_msgs = _totals.get("total_messages", 0)

    if balance is not None:
        remaining = max(0.0, balance - spent)
        ratio = remaining / balance if balance > 0 else 0.0
        pct = ratio * 100
        color = 0x57F287 if pct > 30 else (0xFEE75C if pct > 10 else 0xED4245)
        bar = _bar(ratio)
        desc = (
            f"{bar}  **{pct:.1f}%**\n\n"
            f"💰 **${remaining:.2f}** remaining  ·  **${spent:.4f}** consumed\n"
            f"📨 **{total_msgs:,}** messages  ·  [Top up]({BILLING_URL})"
        )
    else:
        color = 0x99AAB5
        desc = f"No balance set — use `!setcredits <amount>` to start tracking.\n[Check billing]({BILLING_URL})"

    e = discord.Embed(title="💳  Credits Remaining", description=desc, color=color)
    e.add_field(name="⌨️ Commands", value=STATS_COMMANDS, inline=False)
    last = _totals.get("last_updated", "—")
    e.set_footer(text=f"Last updated {last} UTC")
    e.timestamp = datetime.now()
    return e


def _all_embeds() -> list[discord.Embed]:
    return [_embed_credits_dashboard()]


_DASHBOARD_KEYWORDS = ("Travel Planner", "RYO — Chat", "Available Actions")


def _channel_dashboard_embed(channel_name: str) -> discord.Embed | None:
    """Return persistent actions embed for a channel. Returns None for ryo-stats (handled separately)."""
    if channel_name == "ryo-stats":
        return None

    if channel_name == "ryo-travel":
        e = discord.Embed(
            title="✈️  Travel Planner",
            description="Plan trips, build itineraries, and explore the world.",
            color=0x1DA1F2,
        )
        e.add_field(
            name="💬 Just ask",
            value=(
                "*\"Plan a 5-day trip to Kyoto\"*\n"
                "*\"What should I pack for Bali in July?\"*\n"
                "*\"Best restaurants in Paris under €30?\"*"
            ),
            inline=False,
        )
        e.add_field(
            name="⌨️ Commands",
            value=(
                "`!travel-preferences` — view your travel profile\n"
                "`!travel-preferences update` — update saved preferences\n"
                "`!plan-trip <destination> <start date> <end date>`\n"
                "    ↳ full itinerary + Discord events + pre-trip reminders\n"
                "`!clear-events` — delete all scheduled events in the server\n"
                "🔒 `!clear` — clear all messages"
            ),
            inline=False,
        )
        e.add_field(
            name="💡 Tips",
            value=(
                "📸 Attach an image or PDF and ask a question — Ryo can analyse it\n"
                "↩️ Off-topic messages are automatically answered in **#ryo-general**\n"
                "-# 🔒 = owner only"
            ),
            inline=False,
        )
        e.set_footer(text="Both travellers should run !travel-preferences before planning a trip together")
        return e

    # ryo-general and any other channel
    e = discord.Embed(
        title="🌀  RYO — Chat Naturally",
        description="No commands needed — just talk. Ryo figures out the rest.",
        color=0x57F287,
    )
    e.add_field(
        name="💬 What you can ask",
        value=(
            "📰 *\"What's in the news?\"*\n"
            "🌤️ *\"Weather in Tokyo?\"*\n"
            "🔍 *\"Look up X\"* / *\"Find info on Y\"*\n"
            "🧠 *\"Remember I'm a VP at JPMorgan\"*\n"
            "🗑️ *\"Forget my address\"*\n"
            "⏰ *\"Remind me at 3pm\"* / *\"Remind me 3 times every 10 min\"*"
        ),
        inline=False,
    )
    e.add_field(
        name="⌨️ Commands",
        value=(
            "📸 Attach an image or PDF — Ryo can analyse it\n"
            "🔒 `!clear` — clear all messages\n"
            "-# 🔒 = owner only"
        ),
        inline=False,
    )
    return e


TRAVEL_PREF_QUESTIONNAIRE = """✈️ Let's set up your travel profile! Reply with your answers:

**1.** 💳 Credit cards you carry *(e.g. Amex Gold, Chase Sapphire, Visa Infinite)*
**2.** 🍽️ Favourite cuisines *(e.g. Japanese, Italian, Mexican)*
**3.** 🚗 Can you drive? *(yes / no)*
**4.** 💰 Budget style *(budget / mid-range / luxury)*
**5.** 🏨 Accommodation preference *(hotel / Airbnb / hostel / any)*
**6.** 🎒 Travel style *(adventure / relaxing / cultural / foodie / mix)*
**7.** 🥗 Dietary restrictions *(none / vegetarian / vegan / halal / other)*

Reply with numbered answers and I'll save them to your profile!"""


def _get_channel_session(channel_id: int) -> str | None:
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT session_id FROM channel_sessions WHERE channel_id = ?", (channel_id,)
    ).fetchone()
    conn.close()
    return row[0] if row else None


def _save_channel_session(channel_id: int, session_id: str, channel_type: str = "general"):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT OR REPLACE INTO channel_sessions (channel_id, session_id, channel_type, updated_at)"
        " VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
        (channel_id, session_id, channel_type),
    )
    conn.commit()
    conn.close()


def _clear_channel_session(channel_id: int):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM channel_sessions WHERE channel_id = ?", (channel_id,))
    conn.commit()
    conn.close()


def _parse_trip_args(text: str) -> tuple[str, datetime | None, datetime | None, str]:
    """
    Parse trip command text → (destination, start_date, end_date, extra_context).
    Handles multi-word destinations, natural language dates, 'to/through/until' separators,
    trailing description/notes, and parenthetical content (e.g. hotel addresses).

    Examples:
      'New Orleans 19th May to 24th May'
      'New Orleans for 19th May to 24th May. Staying Virgin Hotel (550 Baronne St)'
      'Tokyo June 1 to June 7 2026 focus on street food and anime'
      'Paris 2026-06-01 2026-06-07, romantic trip, budget €200/day'
    """
    # Pull out parenthetical content (e.g. hotel addresses) — keep as extra context,
    # but don't let complex address tokens confuse the date parser.
    paren_extras = re.findall(r'\([^)]+\)', text)
    text_clean = re.sub(r'\([^)]+\)', '', text).strip()

    halves = re.split(r'\s+(?:to|through|until|[-–→])\s+', text_clean, maxsplit=1, flags=re.IGNORECASE)

    try:
        if len(halves) == 2:
            first, second = halves
            start_dt, start_tokens = dateparser.parse(first, fuzzy_with_tokens=True, dayfirst=False)
            destination_raw = ' '.join(t.strip(',- ') for t in start_tokens if t.strip(',- '))
            # Strip trailing prepositions left over from patterns like "New Orleans for 19th May"
            destination = re.sub(r'\s+\b(?:for|in|at|on)\b\s*$', '', destination_raw, flags=re.IGNORECASE).strip()

            # Separate the end-date from any description that follows
            # e.g. "24th May. staying at Virgin Hotel" or "24th May budget €200"
            # Split on: period/! followed by whitespace, OR two or more spaces
            m = re.search(r'[.!]\s+|\s{2,}', second)
            if m:
                second_date_str = second[:m.start()]
                extra_suffix = second[m.end():].strip('.! ')
            else:
                second_date_str = second
                extra_suffix = ''

            end_dt, end_tokens = dateparser.parse(second_date_str, fuzzy_with_tokens=True, dayfirst=False, default=start_dt)
            extra_from_tokens = ' '.join(t.strip(',- ') for t in end_tokens if t.strip(',- '))
            extra = ' '.join(p for p in [extra_from_tokens, extra_suffix] + paren_extras if p).strip()

            return destination, start_dt, end_dt, extra

        # Fallback: 'Destination YYYY-MM-DD YYYY-MM-DD [extra text]'
        parts = text_clean.split()
        if len(parts) >= 3:
            destination = parts[0]
            start_dt = dateparser.parse(parts[1], dayfirst=False)
            end_dt   = dateparser.parse(parts[2], dayfirst=False)
            extra    = ' '.join(parts[3:])
            return destination, start_dt, end_dt, extra.strip()

    except Exception:
        pass

    return "", None, None, ""


def _get_travel_preferences(discord_id: str) -> str | None:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT context FROM permanent_memories WHERE discord_id = ? AND LOWER(index_title) LIKE '%travel%pref%'",
        (discord_id,),
    ).fetchone()
    conn.close()
    return row["context"] if row else None


def _due_reminders() -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM reminders WHERE discord_id IS NOT NULL"
    ).fetchall()
    conn.close()

    now = datetime.now()
    due = []
    for row in rows:
        r = dict(row)
        try:
            window = datetime.fromisoformat(r["remember_window"].split("+")[0].strip())
            if window <= now:
                due.append(r)
        except (ValueError, AttributeError):
            pass
    return due


def _delete_reminder(row_id: int):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM reminders WHERE id = ?", (row_id,))
    conn.commit()
    conn.close()


def _reschedule_reminder(row_id: int, interval_mins: int, new_count: int):
    next_window = (datetime.now() + timedelta(minutes=interval_mins)).isoformat()
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "UPDATE reminders SET remember_window = ?, repeat_count = ? WHERE id = ?",
        (next_window, new_count, row_id),
    )
    conn.commit()
    conn.close()


class Client(discord.Client):
    async def on_ready(self):
        global _totals
        _totals = cost_db.load()
        print(f"Logged on as {self.user}!")
        self.reminder_loop.start()
        await self._ensure_channel("ryo-travel", "✈️ Travel itineraries and trip planning with RYO.")
        await self._ensure_channel("ryo-general", "💬 General chat with RYO — news, weather, reminders, memory.")
        await self._init_stats_panels()
        await self._init_channel_dashboards()

    async def _init_channel_dashboards(self):
        active_names = {"ryo-general", "ryo-travel"}
        for guild in self.guilds:
            for channel in guild.text_channels:
                if channel.name not in active_names:
                    continue
                await self._restore_channel_dashboard(channel)

    async def _restore_channel_dashboard(self, channel: discord.TextChannel):
        embed = _channel_dashboard_embed(channel.name)
        if embed is None:
            return
        existing = None
        async for msg in channel.history(limit=30, oldest_first=False):
            if msg.author == self.user and msg.embeds:
                title = msg.embeds[0].title or ""
                if any(kw in title for kw in _DASHBOARD_KEYWORDS):
                    existing = msg
                    break
        if existing:
            await existing.edit(embed=embed)
            _channel_dashboards[channel.id] = existing
        else:
            msg = await channel.send(embed=embed)
            _channel_dashboards[channel.id] = msg

    async def _ensure_channel(self, name: str, topic: str):
        for guild in self.guilds:
            if not discord.utils.get(guild.text_channels, name=name):
                try:
                    await guild.create_text_channel(name, topic=topic)
                    print(f"Created #{name} in {guild.name}")
                except discord.Forbidden:
                    print(f"Missing permission to create #{name} in {guild.name}")

    async def _init_stats_panels(self):
        for guild in self.guilds:
            channel = discord.utils.get(guild.text_channels, name="ryo-stats")
            if not channel:
                try:
                    channel = await guild.create_text_channel(
                        "ryo-stats",
                        topic="Live RYO dashboard — updates after every message.",
                    )
                    print(f"Created #ryo-stats in {guild.name}")
                except discord.Forbidden:
                    print(f"Missing permission to create #ryo-stats in {guild.name}")
                    continue

            existing = []
            async for msg in channel.history(limit=10, oldest_first=True):
                if msg.author == self.user and msg.embeds:
                    existing.append(msg)

            embeds = _all_embeds()

            if len(existing) == len(embeds):
                for msg, emb in zip(existing, embeds):
                    await msg.edit(embed=emb)
                _stats_messages[guild.id] = existing
            else:
                await channel.purge(limit=20, check=lambda m: m.author == self.user)
                msgs = [await channel.send(embed=emb) for emb in embeds]
                _stats_messages[guild.id] = msgs

    async def _update_stats_panel(self, guild: discord.Guild):
        if guild.id not in _stats_messages:
            await self._init_stats_panels()
            return
        embeds = _all_embeds()
        for msg, emb in zip(_stats_messages[guild.id], embeds):
            try:
                await msg.edit(embed=emb)
            except discord.NotFound:
                _stats_messages.pop(guild.id, None)
                await self._init_stats_panels()
                return

    @tasks.loop(minutes=1)
    async def reminder_loop(self):
        for reminder in _due_reminders():
            discord_id = reminder["discord_id"]
            context = reminder["context"]
            index_title = reminder.get("index_title", "reminder")
            repeat_count = reminder.get("repeat_count") or 1
            snooze_interval = reminder.get("snooze_interval_mins") or 30
            delivered = False

            if repeat_count == -1:
                footer = f"\n*Repeating every {snooze_interval} min · say \"stop reminding me about this\" to cancel*"
            elif repeat_count > 1:
                footer = f"\n*{repeat_count} reminder(s) left · say \"stop reminding me about this\" to cancel*"
            else:
                footer = ""
            ping_text = f"⏰ **Reminder:** {context}{footer}"

            try:
                user = await self.fetch_user(int(discord_id))
                await user.send(ping_text)
                delivered = True
            except discord.Forbidden:
                for guild in self.guilds:
                    member = guild.get_member(int(discord_id))
                    if not member:
                        continue
                    for channel in guild.text_channels:
                        if channel.permissions_for(guild.me).send_messages:
                            try:
                                await channel.send(f"⏰ {member.mention} **Reminder:** {context}{footer}")
                                delivered = True
                            except Exception:
                                pass
                            break
                    if delivered:
                        break
                if not delivered:
                    print(f"Could not reach {discord_id} — DMs off, no shared channel")
            except Exception as e:
                print(f"Reminder ping error for {discord_id}: {e}")

            if delivered:
                if repeat_count == 1:
                    _delete_reminder(reminder["id"])
                elif repeat_count == -1:
                    _reschedule_reminder(reminder["id"], snooze_interval, -1)
                else:
                    _reschedule_reminder(reminder["id"], snooze_interval, repeat_count - 1)
                print(f"Pinged {discord_id}: {index_title} (repeat_count={repeat_count})")

    async def on_message(self, message):
        global _totals, _last_cost, _session_messages

        if message.author == self.user:
            return

        channel_name = getattr(message.channel, "name", "")

        # !clear — owner only, works in any channel
        if message.content.strip() == "!clear":
            if str(message.author.id) != conf.owner_discord_id:
                await message.channel.send("🚫 Only the bot owner can clear conversations.")
                return
            if not isinstance(message.channel, discord.TextChannel):
                await message.channel.send("⚠️ `!clear` only works in server channels.")
                return
            try:
                await message.delete()
            except discord.Forbidden:
                pass
            deleted = await message.channel.purge(limit=100)
            confirm = await message.channel.send(
                f"🧹 Cleared **{len(deleted)}** message(s).\n"
                f"-# For messages older than 14 days use **Undiscord** — github.com/victornpb/undiscord"
            )
            await confirm.delete(delay=8)
            _clear_channel_session(message.channel.id)
            if channel_name == "ryo-stats" and message.guild:
                await self._init_stats_panels()
            elif isinstance(message.channel, discord.TextChannel):
                _channel_dashboards.pop(message.channel.id, None)
                await self._restore_channel_dashboard(message.channel)
            return

        # ryo-stats: only stats commands work here
        if channel_name == "ryo-stats":
            cmd = message.content.strip()
            if cmd == "!refresh":
                if message.guild:
                    await self._update_stats_panel(message.guild)
            elif cmd.startswith("!setcredits"):
                if str(message.author.id) != conf.owner_discord_id:
                    await message.channel.send("🚫 Only the bot owner can update credits.")
                    return
                parts = cmd.split()
                if len(parts) == 2:
                    try:
                        amount = float(parts[1].lstrip("$"))
                        cost_db.set_credit_balance(amount)
                        _totals = cost_db.load()
                        await message.channel.send(f"✅ Credit balance set to **${amount:.2f}**")
                        if message.guild:
                            await self._update_stats_panel(message.guild)
                    except ValueError:
                        await message.channel.send("Usage: `!setcredits 28.35`")
                else:
                    await message.channel.send("Usage: `!setcredits 28.35`")
            elif not cmd.startswith("!"):
                await message.channel.send("📊 This channel shows cost diagnostics only. Chat with Ryo in **#ryo-general**!")
            return

        discord_id = str(message.author.id)
        display_name = message.author.display_name
        username = message.author.name

        is_new = register_user(discord_id, username, display_name)
        if is_new:
            print(f"New user registered: {display_name} ({discord_id})")

        channel_type = "travel" if channel_name == "ryo-travel" else "general"

        # Image / PDF attachments — route to vision
        image_attachments = [
            (a.url, a.content_type or "image/jpeg")
            for a in message.attachments
            if (a.content_type or "").startswith("image/")
            or a.content_type == "application/pdf"
        ]
        if image_attachments:
            async with message.channel.typing():
                vision_response = await run_vision(
                    user_message=message.content.strip(),
                    attachment_urls=image_attachments,
                    display_name=display_name,
                )
            for chunk in _chunk(_sanitize(vision_response)):
                await message.channel.send(chunk)
            if message.guild:
                await self._update_stats_panel(message.guild)
            return

        # !clear-events — ryo-travel only
        if channel_name == "ryo-travel" and message.content.strip() == "!clear-events":
            if not message.guild:
                return
            count = await delete_all_guild_events(message.guild)
            await message.channel.send(f"🗑️ Deleted **{count}** scheduled event(s).")
            return

        # Pending duplicate-event confirmation
        if channel_name == "ryo-travel" and message.channel.id in _pending_trip_events:
            await self._handle_event_confirm_reply(message)
            return

        # !plan-trip — ryo-travel only
        if channel_name == "ryo-travel" and message.content.strip().startswith("!plan-trip"):
            await self._handle_plan_trip(message, discord_id, display_name)
            return

        # !travel-preferences — ryo-travel only
        if channel_name == "ryo-travel" and message.content.strip() in ("!travel-preferences", "!refresh"):
            prefs = _get_travel_preferences(discord_id)
            if prefs:
                await message.channel.send(
                    f"🗺️ **Your travel profile, {display_name}:**\n\n{prefs}\n\n"
                    f"Want to update? Just type `!travel-preferences update`."
                )
            else:
                _pending_travel_prefs.add(discord_id)
                await message.channel.send(TRAVEL_PREF_QUESTIONNAIRE)
            return

        if channel_name == "ryo-travel" and message.content.strip() == "!travel-preferences update":
            _pending_travel_prefs.add(discord_id)
            await message.channel.send(TRAVEL_PREF_QUESTIONNAIRE)
            return

        # Pending pref reply — save via CEO with travel_preferences_save prompt
        if channel_name == "ryo-travel" and discord_id in _pending_travel_prefs:
            _pending_travel_prefs.discard(discord_id)
            channel_type = "travel_preferences_save"

        channel_id = message.channel.id

        async with message.channel.typing():
            response, cost_info, new_session_id = await run_ceo(
                user_message=message.content.strip(),
                discord_id=discord_id,
                display_name=display_name,
                channel_type=channel_type,
                session_id=_get_channel_session(channel_id),
            )

        if new_session_id:
            _save_channel_session(channel_id, new_session_id, channel_type)

        # Off-topic routing: specialized channel redirects to general
        if response.startswith("<<ROUTE:general>>"):
            redirect_msg = response.replace("<<ROUTE:general>>", "").strip()
            await message.channel.send(_sanitize(redirect_msg))
            if message.guild:
                general = self._get_general_channel(message.guild)
                if general:
                    async with general.typing():
                        gen_response, cost_info, gen_session_id = await run_ceo(
                            user_message=message.content.strip(),
                            discord_id=discord_id,
                            display_name=display_name,
                            channel_type="general",
                            session_id=_get_channel_session(general.id),
                        )
                    if gen_session_id:
                        _save_channel_session(general.id, gen_session_id, "general")
                    for chunk in _chunk(_sanitize(gen_response)):
                        await general.send(f"↩️ *Redirected from #{channel_name}*\n{chunk}")
        else:
            for chunk in _chunk(_sanitize(response)):
                await message.channel.send(chunk)

        if cost_info:
            _last_cost = cost_info
            _session_messages += 1
            cost_db.save(cost_info)
            _totals = cost_db.load()
            await self._check_low_credit()

        if message.guild:
            await self._update_stats_panel(message.guild)

    def _get_general_channel(self, guild: discord.Guild) -> discord.TextChannel | None:
        preferred = discord.utils.get(guild.text_channels, name="ryo-general")
        if preferred and guild.me.permissions_in(preferred).send_messages:
            return preferred
        specialized = {"ryo-stats", "ryo-travel"}
        for ch in guild.text_channels:
            if ch.name not in specialized and guild.me.permissions_in(ch).send_messages:
                return ch
        return None

    async def _handle_plan_trip(self, message: discord.Message, discord_id: str, display_name: str):
        raw = re.sub(r'^!plan-trip\s+', '', message.content.strip(), flags=re.IGNORECASE)
        destination, start_date, end_date, extra_context = _parse_trip_args(raw)

        if not destination or not start_date or not end_date:
            await message.channel.send(
                "Couldn't figure out the destination or dates. Try:\n"
                "`!plan-trip New Orleans 19th May to 24th May`\n"
                "`!plan-trip Tokyo June 1 2026 to June 7 2026`\n"
                "You can add any description after the dates — hotel info, budget, preferences, etc."
            )
            return

        days = (end_date - start_date).days + 1
        status_msg = await message.channel.send(
            f"✈️ Planning your **{days}-day {destination}** trip… this may take a moment!"
        )

        prefs = _get_all_travel_prefs()
        prefs_context = ""
        if prefs:
            prefs_context = "\n\nTravel profiles for this trip:\n" + "\n\n".join(
                f"**{p['display_name']}**: {p['context']}" for p in prefs
            )

        extra_section = f"\n\nAdditional notes: {extra_context}" if extra_context else ""
        trip_prompt = (
            f"Plan a detailed {days}-day itinerary for {destination} "
            f"from {start_date.strftime('%B %d')} to {end_date.strftime('%B %d, %Y')}."
            f"{extra_section}"
            f"{prefs_context}"
        )

        async with message.channel.typing():
            itinerary, cost_info, new_session_id = await run_ceo(
                user_message=trip_prompt,
                discord_id=discord_id,
                display_name=display_name,
                channel_type="travel",
                session_id=_get_channel_session(message.channel.id),
            )
        if new_session_id:
            _save_channel_session(message.channel.id, new_session_id, "travel")

        await status_msg.delete()
        for chunk in _chunk(_sanitize(itinerary)):
            await message.channel.send(chunk)

        if message.guild:
            await message.channel.send("📅 Preparing Discord events…")
            try:
                events = await extract_trip_events(itinerary, destination, start_date)
                existing = await get_existing_event_names(message.guild)
                duplicate_names = {
                    event_title(destination, ev["day"], ev["title"])
                    for ev in events
                    if event_title(destination, ev["day"], ev["title"]) in existing
                }

                if duplicate_names:
                    dup_list = "\n".join(f"• {n}" for n in sorted(duplicate_names))
                    _pending_trip_events[message.channel.id] = {
                        "guild": message.guild,
                        "events": events,
                        "destination": destination,
                        "start_date": start_date,
                        "discord_id": discord_id,
                        "display_name": display_name,
                        "duplicate_names": duplicate_names,
                        "cost_info": cost_info,
                    }
                    await message.channel.send(
                        f"⚠️ **{len(duplicate_names)}** event(s) already exist:\n{dup_list}\n\n"
                        f"Replace them? Reply `yes` to replace all, `no` to skip duplicates."
                    )
                    return

                await self._finish_creating_events(
                    message.channel, message.guild, events, destination,
                    start_date, discord_id, display_name, cost_info,
                )
                return
            except Exception as e:
                print(f"Trip event creation error: {e}")
                await message.channel.send("⚠️ Itinerary ready but couldn't create Discord events.")

    async def _finish_creating_events(
        self, channel, guild, events, destination, start_date,
        discord_id, display_name, cost_info,
        replace_names=None, skip_names=None,
    ):
        created = await create_discord_events(
            guild, events, destination, start_date,
            replace_names=replace_names, skip_names=skip_names,
        )
        if created:
            await channel.send(
                f"✅ Created **{len(created)}** Discord events! Check the Events tab in your server."
            )
        else:
            await channel.send("⚠️ Could not create Discord events — check the bot has **Manage Events** permission.")

        _store_trip_reminders(discord_id, f"{destination} trip", start_date)
        await channel.send(
            f"⏰ Pre-trip reminders set for **{display_name}** — 7 days, 2 days, and 1 day before departure."
        )

        if cost_info:
            global _last_cost, _session_messages, _totals
            _last_cost = cost_info
            _session_messages += 1
            cost_db.save(cost_info)
            _totals = cost_db.load()
            await self._check_low_credit()

        if guild:
            await self._update_stats_panel(guild)

    async def _handle_event_confirm_reply(self, message: discord.Message):
        channel_id = message.channel.id
        pending = _pending_trip_events.pop(channel_id, None)
        if not pending:
            return

        reply = message.content.strip().lower()
        if reply in ("yes", "y", "replace", "replace all", "yes all"):
            await message.channel.send("🔄 Replacing duplicates and creating all events…")
            await self._finish_creating_events(
                message.channel, pending["guild"], pending["events"],
                pending["destination"], pending["start_date"],
                pending["discord_id"], pending["display_name"], pending["cost_info"],
                replace_names=pending["duplicate_names"],
            )
        elif reply in ("no", "n", "skip", "skip all"):
            await message.channel.send("⏭️ Skipping duplicates, creating new events only…")
            await self._finish_creating_events(
                message.channel, pending["guild"], pending["events"],
                pending["destination"], pending["start_date"],
                pending["discord_id"], pending["display_name"], pending["cost_info"],
                skip_names=pending["duplicate_names"],
            )
        else:
            # Put it back and re-ask
            _pending_trip_events[channel_id] = pending
            await message.channel.send("Reply `yes` to replace duplicates or `no` to skip them.")


    async def _check_low_credit(self):
        balance = _totals.get("credit_balance_usd")
        alerted = _totals.get("low_credit_alerted", 0)
        if balance is None or alerted:
            return
        remaining = balance - _totals.get("total_cost_usd", 0.0)
        if remaining < 5.0:
            try:
                user = await self.fetch_user(int(conf.owner_discord_id))
                await user.send(
                    f"⚠️ **Low credit alert!** You have an estimated **${remaining:.2f}** remaining.\n"
                    f"Top up at {BILLING_URL}"
                )
                cost_db.mark_low_credit_alerted()
                _totals["low_credit_alerted"] = 1
            except Exception as e:
                print(f"Low credit alert failed: {e}")


client = Client(intents=intents)
client.run(conf.discord_token)
