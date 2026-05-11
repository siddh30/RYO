import re
import sqlite3
import sys
from datetime import datetime, timedelta

sys.path.insert(0, '.')

from config import Config
conf = Config()

from agents.ceo import run_ceo
from memory.register_user import register_user
from memory import cost_db

import discord
from discord.ext import tasks

DISCORD_MAX = 1900
DB_PATH = f"{conf.memory_path}/ryo.db"
BILLING_URL = "https://platform.claude.com/settings/billing"

intents = discord.Intents.default()
intents.message_content = True

# Loaded from DB on startup; updated in-memory and persisted after every message
_totals: dict = {}
_last_cost: dict = {}
_session_messages: int = 0
_stats_messages: dict[int, discord.Message] = {}  # guild_id -> live embed message


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


def _build_stats_embed() -> discord.Embed:
    embed = discord.Embed(title="📊 RYO Cost Diagnostics", color=0x5865F2)

    if _last_cost:
        embed.add_field(
            name="Last Message",
            value=(
                f"💰 `${_last_cost['total_cost_usd']:.5f}`\n"
                f"⏱️ `{_last_cost['duration_ms']} ms` · `{_last_cost['num_turns']}` turn(s)\n"
                f"📥 `{_last_cost['input_tokens']:,}` in · `{_last_cost['output_tokens']:,}` out\n"
                f"⚡ Cache read `{_last_cost['cache_read_tokens']:,}` · created `{_last_cost['cache_creation_tokens']:,}`"
            ),
            inline=False,
        )

    all_time_cost = _totals.get("total_cost_usd", 0.0)
    all_time_messages = _totals.get("total_messages", 0)
    all_time_in = _totals.get("total_input_tokens", 0)
    all_time_out = _totals.get("total_output_tokens", 0)
    all_time_cache = _totals.get("total_cache_read_tokens", 0)

    embed.add_field(
        name=f"All-Time Total  ·  {_session_messages} this session",
        value=(
            f"💬 `{all_time_messages}` messages\n"
            f"💰 `${all_time_cost:.5f}` spent\n"
            f"📥 `{all_time_in:,}` in · `{all_time_out:,}` out\n"
            f"⚡ Cache read `{all_time_cache:,}`"
        ),
        inline=False,
    )

    credit_balance = _totals.get("credit_balance_usd")
    if credit_balance is not None:
        remaining = credit_balance - all_time_cost
        bar_total = 20
        filled = max(0, min(bar_total, round((remaining / credit_balance) * bar_total))) if credit_balance > 0 else 0
        bar = "█" * filled + "░" * (bar_total - filled)
        pct = max(0.0, (remaining / credit_balance) * 100) if credit_balance > 0 else 0.0
        embed.add_field(
            name="💳 Credits",
            value=(
                f"`{bar}` {pct:.1f}%\n"
                f"Started `${credit_balance:.2f}` · Spent `${all_time_cost:.5f}`\n"
                f"**Estimated remaining: `${remaining:.2f}`**\n"
                f"[Check billing]({BILLING_URL})"
            ),
            inline=False,
        )
    else:
        embed.add_field(
            name="💳 Credits",
            value=f"Set a balance with `!setcredits <amount>`\n[Check billing]({BILLING_URL})",
            inline=False,
        )

    last_updated = _totals.get("last_updated", "—")
    embed.set_footer(text=f"Last updated {last_updated} UTC · use !setcredits to update balance")
    embed.timestamp = datetime.now()
    return embed


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
        await self._init_stats_panels()

    async def _init_stats_panels(self):
        for guild in self.guilds:
            channel = discord.utils.get(guild.text_channels, name="ryo-stats")
            if not channel:
                continue
            async for msg in channel.history(limit=20):
                if msg.author == self.user and msg.embeds:
                    _stats_messages[guild.id] = msg
                    await msg.edit(embed=_build_stats_embed())
                    break
            else:
                msg = await channel.send(embed=_build_stats_embed())
                _stats_messages[guild.id] = msg

    async def _update_stats_panel(self, guild: discord.Guild):
        if guild.id not in _stats_messages:
            channel = discord.utils.get(guild.text_channels, name="ryo-stats")
            if not channel:
                return
            msg = await channel.send(embed=_build_stats_embed())
            _stats_messages[guild.id] = msg
            return
        try:
            await _stats_messages[guild.id].edit(embed=_build_stats_embed())
        except discord.NotFound:
            _stats_messages.pop(guild.id, None)
            await self._update_stats_panel(guild)

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

        # !setcredits command — owner only
        if message.content.strip().startswith("!setcredits"):
            if str(message.author.id) != conf.owner_discord_id:
                await message.channel.send("🚫 Only the bot owner can update credits.")
                return
            parts = message.content.strip().split()
            if len(parts) == 2:
                try:
                    amount = float(parts[1].lstrip("$"))
                    cost_db.set_credit_balance(amount)
                    _totals = cost_db.load()
                    await message.channel.send(f"✅ Credit balance set to **${amount:.2f}**")
                    if message.guild:
                        await self._update_stats_panel(message.guild)
                    return
                except ValueError:
                    await message.channel.send("Usage: `!setcredits 28.35`")
                    return

        discord_id = str(message.author.id)
        display_name = message.author.display_name
        username = message.author.name

        is_new = register_user(discord_id, username, display_name)
        if is_new:
            print(f"New user registered: {display_name} ({discord_id})")

        async with message.channel.typing():
            response, cost_info = await run_ceo(
                user_message=message.content.strip(),
                discord_id=discord_id,
                display_name=display_name,
            )

        if cost_info:
            _last_cost = cost_info
            _session_messages += 1
            cost_db.save(cost_info)
            _totals = cost_db.load()
            await self._check_low_credit()

        for chunk in _chunk(_sanitize(response)):
            await message.channel.send(chunk)

        if message.guild:
            await self._update_stats_panel(message.guild)

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
