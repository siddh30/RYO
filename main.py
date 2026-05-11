import re
import sqlite3
import sys
from datetime import datetime, timedelta

sys.path.insert(0, '.')

from config import Config
conf = Config()

from agents.ceo import run_ceo
from memory.register_user import register_user

import discord
from discord.ext import tasks

DISCORD_MAX = 1900
DB_PATH = f"{conf.memory_path}/ryo.db"

intents = discord.Intents.default()
intents.message_content = True

_session_stats = {
    "messages": 0,
    "total_cost_usd": 0.0,
    "total_input_tokens": 0,
    "total_output_tokens": 0,
    "total_cache_read_tokens": 0,
    "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}
_last_cost: dict = {}
_stats_messages: dict[int, discord.Message] = {}  # guild_id -> pinned stats message


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

    embed.add_field(
        name="Session Total",
        value=(
            f"💬 `{_session_stats['messages']}` message(s)\n"
            f"💰 `${_session_stats['total_cost_usd']:.5f}`\n"
            f"📥 `{_session_stats['total_input_tokens']:,}` in · `{_session_stats['total_output_tokens']:,}` out\n"
            f"⚡ Cache read `{_session_stats['total_cache_read_tokens']:,}`"
        ),
        inline=False,
    )

    embed.set_footer(text=f"Session started {_session_stats['started_at']} UTC · updates after every message")
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
        print(f"Logged on as {self.user}!")
        self.reminder_loop.start()
        await self._init_stats_panels()

    async def _init_stats_panels(self):
        """Find or create the ryo-stats message in every guild."""
        for guild in self.guilds:
            channel = discord.utils.get(guild.text_channels, name="ryo-stats")
            if not channel:
                continue
            # Try to reuse the most recent bot message in that channel
            async for msg in channel.history(limit=20):
                if msg.author == self.user and msg.embeds:
                    _stats_messages[guild.id] = msg
                    break
            else:
                msg = await channel.send(embed=_build_stats_embed())
                _stats_messages[guild.id] = msg

    async def _update_stats_panel(self, guild: discord.Guild):
        """Edit the stats embed for this guild."""
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
        if message.author == self.user:
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
            global _last_cost
            _last_cost = cost_info
            _session_stats["messages"] += 1
            _session_stats["total_cost_usd"] += cost_info["total_cost_usd"]
            _session_stats["total_input_tokens"] += cost_info["input_tokens"]
            _session_stats["total_output_tokens"] += cost_info["output_tokens"]
            _session_stats["total_cache_read_tokens"] += cost_info["cache_read_tokens"]

        for chunk in _chunk(_sanitize(response)):
            await message.channel.send(chunk)

        if message.guild:
            await self._update_stats_panel(message.guild)


client = Client(intents=intents)
client.run(conf.discord_token)
