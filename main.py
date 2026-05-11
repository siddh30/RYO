import os
import re
import sqlite3
import sys
from datetime import datetime, timedelta

from dateutil import parser as dateparser

import anthropic

sys.path.insert(0, '.')

from config import Config
conf = Config()

from agents.ceo import run_ceo
from agents.vision import run_vision
from agents.trip_planner import (
    _get_all_travel_prefs, _store_trip_reminders,
    extract_trip_events, create_discord_events,
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
_conversation_history: dict[int, list[tuple[str, str]]] = {}  # channel_id -> [(speaker, message), ...]
_channel_input_tokens: dict[int, int] = {}  # channel_id -> latest input_tokens

CONTEXT_WINDOW = 200_000       # claude-sonnet-4-6
COMPRESS_AT    = 0.80          # compress when context hits 80%
COMPRESS_TO    = 0.20          # target summary size ~20% of window


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


def _append_history(channel_id: int, display_name: str, user_msg: str, bot_response: str):
    history = _conversation_history.setdefault(channel_id, [])
    history.append((display_name, user_msg))
    history.append(("Ryo", bot_response))


async def _compress_history(channel_id: int) -> str:
    """Summarise the current conversation history into a compact block."""
    history = _conversation_history.get(channel_id, [])
    if not history:
        return ""
    formatted = "\n".join(f"{name}: {msg}" for name, msg in history)
    client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    resp = await client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=int(CONTEXT_WINDOW * COMPRESS_TO),
        messages=[{
            "role": "user",
            "content": (
                "Compress this Discord conversation into a dense summary that preserves all key facts, "
                "preferences, decisions, and ongoing context Ryo needs to continue seamlessly. "
                "Write in third person, plain prose, no bullet points.\n\n"
                f"{formatted}"
            ),
        }],
    )
    summary = resp.content[0].text
    _conversation_history[channel_id] = [("Summary", summary)]
    _channel_input_tokens[channel_id] = 0
    return summary


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
            _conversation_history.pop(message.channel.id, None)
            _channel_input_tokens.pop(message.channel.id, None)
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
        history = _conversation_history.get(channel_id, [])

        async with message.channel.typing():
            response, cost_info = await run_ceo(
                user_message=message.content.strip(),
                discord_id=discord_id,
                display_name=display_name,
                channel_type=channel_type,
                conversation_history=history,
            )

        # Off-topic routing: specialized channel redirects to general
        if response.startswith("<<ROUTE:general>>"):
            redirect_msg = response.replace("<<ROUTE:general>>", "").strip()
            await message.channel.send(_sanitize(redirect_msg))
            if message.guild:
                general = self._get_general_channel(message.guild)
                if general:
                    gen_history = _conversation_history.get(general.id, [])
                    async with general.typing():
                        gen_response, cost_info = await run_ceo(
                            user_message=message.content.strip(),
                            discord_id=discord_id,
                            display_name=display_name,
                            channel_type="general",
                            conversation_history=gen_history,
                        )
                    for chunk in _chunk(_sanitize(gen_response)):
                        await general.send(f"↩️ *Redirected from #{channel_name}*\n{chunk}")
                    _append_history(general.id, display_name, message.content.strip(), gen_response)
        else:
            for chunk in _chunk(_sanitize(response)):
                await message.channel.send(chunk)
            _append_history(channel_id, display_name, message.content.strip(), response)

        if cost_info:
            _last_cost = cost_info
            _session_messages += 1
            cost_db.save(cost_info)
            _totals = cost_db.load()
            await self._check_low_credit()

            input_tokens = cost_info.get("input_tokens", 0)
            _channel_input_tokens[channel_id] = input_tokens
            if input_tokens > CONTEXT_WINDOW * COMPRESS_AT:
                await self._compress_and_notify(channel_id, message.channel)

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
        parts = message.content.strip().split()
        if len(parts) < 4:
            await message.channel.send(
                "Usage: `!plan-trip <destination> <start date> <end date>`\n"
                "Example: `!plan-trip Tokyo June-1-2026 June-7-2026`"
            )
            return

        destination = parts[1]
        try:
            start_date = dateparser.parse(parts[2], dayfirst=False)
            end_date = dateparser.parse(parts[3], dayfirst=False)
        except Exception:
            await message.channel.send("Couldn't parse those dates. Try: `June 1 2026`, `2026-06-01`, or `01/06/2026`.")
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

        trip_prompt = (
            f"Plan a detailed {days}-day itinerary for {destination} "
            f"from {start_date.strftime('%B %d')} to {end_date.strftime('%B %d, %Y')}."
            f"{prefs_context}"
        )

        async with message.channel.typing():
            itinerary, cost_info = await run_ceo(
                user_message=trip_prompt,
                discord_id=discord_id,
                display_name=display_name,
                channel_type="travel",
            )

        await status_msg.delete()
        for chunk in _chunk(_sanitize(itinerary)):
            await message.channel.send(chunk)

        if message.guild:
            await message.channel.send("📅 Creating Discord events for each day…")
            try:
                events = await extract_trip_events(itinerary, destination, start_date)
                created = await create_discord_events(message.guild, events, destination, start_date)
                if created:
                    await message.channel.send(
                        f"✅ Created **{len(created)}** Discord events! Check the Events tab in your server.\n"
                        f"⏰ Setting pre-trip reminders for you…"
                    )
                else:
                    await message.channel.send("⚠️ Could not create Discord events — check the bot has **Manage Events** permission.")
            except Exception as e:
                print(f"Trip event creation error: {e}")
                await message.channel.send("⚠️ Itinerary ready but couldn't create Discord events.")

        _store_trip_reminders(discord_id, f"{destination} trip", start_date)
        await message.channel.send(
            f"⏰ Pre-trip reminders set for **{display_name}** — 7 days, 2 days, and 1 day before departure."
        )

        if cost_info:
            global _last_cost, _session_messages, _totals
            _last_cost = cost_info
            _session_messages += 1
            cost_db.save(cost_info)
            _totals = cost_db.load()
            await self._check_low_credit()

        if message.guild:
            await self._update_stats_panel(message.guild)

    async def _compress_and_notify(self, channel_id: int, channel):
        await _compress_history(channel_id)
        try:
            user = await self.fetch_user(int(conf.owner_discord_id))
            await user.send(f"🗜️ **Context compressed** in #{getattr(channel, 'name', 'DM')} — conversation was approaching 80% of Sonnet's context window. Summary saved, continuing seamlessly.")
        except Exception:
            pass
        try:
            await channel.send("-# 🗜️ Context compressed — long conversation summarised to stay within limits.")
        except Exception:
            pass

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
