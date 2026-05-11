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
from utils.webhook_dispatch import dispatch, add_webhook, remove_webhook, list_webhooks

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
_stats_messages: dict[int, list[discord.Message]] = {}  # guild_id -> [status, credits, last_msg, alltime]


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


def _bar(ratio: float, width: int = 18) -> str:
    filled = max(0, min(width, round(ratio * width)))
    return "█" * filled + "░" * (width - filled)


def _embed_status() -> discord.Embed:
    e = discord.Embed(title="🌀  RYO  —  Work in Balance. Life in Flow.", color=0x57F287)
    e.add_field(name="Status", value="🟢  Online", inline=True)
    e.add_field(name="Session messages", value=f"`{_session_messages}`", inline=True)
    e.add_field(name="Model", value="`claude-sonnet-4-6`", inline=True)
    e.set_footer(text="Updates after every message")
    e.timestamp = datetime.now()
    return e


def _embed_credits() -> discord.Embed:
    balance = _totals.get("credit_balance_usd")
    spent = _totals.get("total_cost_usd", 0.0)

    if balance is not None:
        remaining = max(0.0, balance - spent)
        ratio = remaining / balance if balance > 0 else 0.0
        pct = ratio * 100
        color = 0x57F287 if pct > 30 else (0xFEE75C if pct > 10 else 0xED4245)
        e = discord.Embed(title="💳  Credit Balance", color=color)
        e.add_field(
            name="Remaining",
            value=f"**`${remaining:.2f}`**  /  `${balance:.2f}`",
            inline=False,
        )
        e.add_field(
            name=f"{_bar(ratio)}  {pct:.1f}%",
            value=f"Spent `${spent:.4f}`",
            inline=False,
        )
        e.add_field(name="Top up", value=f"[platform.claude.com]({BILLING_URL})", inline=True)
        e.add_field(name="Update balance", value="`!setcredits <amount>`", inline=True)
    else:
        e = discord.Embed(title="💳  Credit Balance", color=0x99AAB5)
        e.description = f"No balance set.\nUse `!setcredits 28.35` to set it.\n[Check billing]({BILLING_URL})"

    e.timestamp = datetime.now()
    return e


def _embed_last_message() -> discord.Embed:
    e = discord.Embed(title="⚡  Last Message", color=0x5865F2)
    if _last_cost:
        e.add_field(name="Cost", value=f"`${_last_cost['total_cost_usd']:.5f}`", inline=True)
        e.add_field(name="Latency", value=f"`{_last_cost['duration_ms']} ms`", inline=True)
        e.add_field(name="Turns", value=f"`{_last_cost['num_turns']}`", inline=True)
        e.add_field(name="Input tokens", value=f"`{_last_cost['input_tokens']:,}`", inline=True)
        e.add_field(name="Output tokens", value=f"`{_last_cost['output_tokens']:,}`", inline=True)
        e.add_field(name="Cache read", value=f"`{_last_cost['cache_read_tokens']:,}`", inline=True)
    else:
        e.description = "*No messages yet this session.*"
    e.timestamp = datetime.now()
    return e


def _embed_alltime() -> discord.Embed:
    e = discord.Embed(title="📊  All-Time Usage", color=0xEB459E)
    e.add_field(name="Messages", value=f"`{_totals.get('total_messages', 0):,}`", inline=True)
    e.add_field(name="Total cost", value=f"`${_totals.get('total_cost_usd', 0.0):.4f}`", inline=True)
    e.add_field(name="​", value="​", inline=True)
    e.add_field(name="Input tokens", value=f"`{_totals.get('total_input_tokens', 0):,}`", inline=True)
    e.add_field(name="Output tokens", value=f"`{_totals.get('total_output_tokens', 0):,}`", inline=True)
    e.add_field(name="Cache read", value=f"`{_totals.get('total_cache_read_tokens', 0):,}`", inline=True)
    last = _totals.get("last_updated", "—")
    e.set_footer(text=f"Last updated {last} UTC")
    e.timestamp = datetime.now()
    return e


def _all_embeds() -> list[discord.Embed]:
    return [_embed_status(), _embed_credits(), _embed_last_message(), _embed_alltime()]


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
                try:
                    channel = await guild.create_text_channel(
                        "ryo-stats",
                        topic="Live RYO dashboard — updates after every message.",
                    )
                    print(f"Created #ryo-stats in {guild.name}")
                except discord.Forbidden:
                    print(f"Missing permission to create #ryo-stats in {guild.name}")
                    continue

            # Collect existing bot embed messages in order
            existing = []
            async for msg in channel.history(limit=10, oldest_first=True):
                if msg.author == self.user and msg.embeds:
                    existing.append(msg)

            embeds = _all_embeds()

            if len(existing) == len(embeds):
                # Reuse and refresh all panels
                for msg, emb in zip(existing, embeds):
                    await msg.edit(embed=emb)
                _stats_messages[guild.id] = existing
            else:
                # Clear and repost clean dashboard
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
                await dispatch("reminder", {
                    "discord_id": discord_id,
                    "index_title": index_title,
                    "context": context,
                    "repeat_count": repeat_count,
                })

    async def on_message(self, message):
        global _totals, _last_cost, _session_messages

        if message.author == self.user:
            return

        # webhook commands — owner only
        if message.content.strip().startswith("!addwebhook"):
            if str(message.author.id) != conf.owner_discord_id:
                await message.channel.send("🚫 Only the bot owner can manage webhooks.")
                return
            parts = message.content.strip().split()
            if len(parts) >= 3:
                label = " ".join(parts[3:]) if len(parts) > 3 else ""
                await message.channel.send(add_webhook(parts[1], parts[2], label))
            else:
                await message.channel.send("Usage: `!addwebhook <event> <url> [label]`\nEvents: `reminder`, `low_credit`, `new_user`, `message`")
            return

        if message.content.strip().startswith("!removewebhook"):
            if str(message.author.id) != conf.owner_discord_id:
                await message.channel.send("🚫 Only the bot owner can manage webhooks.")
                return
            parts = message.content.strip().split()
            if len(parts) == 2:
                await message.channel.send(remove_webhook(parts[1]))
            else:
                await message.channel.send("Usage: `!removewebhook <event>`")
            return

        if message.content.strip() == "!listwebhooks":
            if str(message.author.id) != conf.owner_discord_id:
                await message.channel.send("🚫 Only the bot owner can manage webhooks.")
                return
            rows = list_webhooks()
            if not rows:
                await message.channel.send("No webhooks configured. Use `!addwebhook <event> <url>`")
            else:
                lines = "\n".join(f"• `{r['event']}` — {r['url']}" + (f" _{r['label']}_" if r['label'] else "") for r in rows)
                await message.channel.send(f"**Configured webhooks:**\n{lines}")
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
            await dispatch("new_user", {"discord_id": discord_id, "display_name": display_name})

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
            await dispatch("message", {
                "discord_id": discord_id,
                "display_name": display_name,
                "cost_usd": cost_info["total_cost_usd"],
                "input_tokens": cost_info["input_tokens"],
                "output_tokens": cost_info["output_tokens"],
                "duration_ms": cost_info["duration_ms"],
            })

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
                await dispatch("low_credit", {"remaining_usd": remaining, "balance_usd": balance})
            except Exception as e:
                print(f"Low credit alert failed: {e}")


client = Client(intents=intents)
client.run(conf.discord_token)
