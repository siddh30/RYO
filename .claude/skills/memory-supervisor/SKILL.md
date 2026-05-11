---
name: memory-supervisor
description: Manage user memory — store, delete, or update memories and reminders. Use when the user asks to remember something, set a reminder, forget something, or update a saved memory.
---

# Memory Supervisor

Manage per-user memory in `memory/ryo.db`. Always use DiscordID from `<CurrentUser>`.

## Classify the request

| User says | Action |
|-----------|--------|
| "Remember X" / "I am Y" / no timeframe | Store → Permanent |
| "Remind me to X by [date]" / has timeframe | Store → Reminder |
| "Forget X" / "Delete X" | Delete |
| "Update X to Y" | Delete old → Store new |

## Before storing — check for duplicates

```bash
python memory/query_memory.py --user-id {DISCORD_ID}
```

If already saved → reply `✅ Already saved, {DisplayName}!` and stop.

## Store — Permanent memory

```bash
python .claude/skills/memory-store/scripts/store_memory.py \
  --user-id "{DISCORD_ID}" \
  --flag "Permanent" \
  --title "short_index_name" \
  --context "exact content" \
  --window "9999-12-31T23:59:59" \
  --ai-message "✅ Got it, {DisplayName}! [one-line confirmation]"
```

## Store — Reminder (ask about repeat preference first)

Before storing a reminder, ask the user:
> "How many times should I remind you — just once, a specific number of times, or keep reminding until you say stop?"

Parse repeat and interval directly from the message if stated. Only ask if missing.

| User says | `--repeat-count` | `--snooze-interval` |
|-----------|-----------------|---------------------|
| "Once" / no preference | `1` | *(omit)* |
| "3 times every 1 minute" | `3` | `1` |
| "twice every hour" | `2` | `60` |
| "every 5 minutes until I say stop" | `-1` | `5` |
| "Until I say stop" with no interval | `-1` | ask user, default `30` |
| "3 times" with no interval | `3` | ask user, default `30` |

`--snooze-interval` is always in **minutes** — convert hours/days if needed (e.g. "every 2 hours" → `120`).

```bash
python .claude/skills/memory-store/scripts/store_memory.py \
  --user-id "{DISCORD_ID}" \
  --flag "Not Permanent" \
  --title "short_index_name" \
  --context "exact content" \
  --window "YYYY-MM-DDTHH:MM:SS" \
  --repeat-count 3 \
  --snooze-interval 30 \
  --ai-message "⏰ Reminder set! [what + when + how many times, one line]"
```

To stop a snoozed/repeating reminder the user can say "stop reminding me about X" → use the Delete flow below.

## Delete

First query memory to get exact `index_title`, then:

```bash
python .claude/skills/memory-delete/scripts/delete_memory.py \
  --user-id "{DISCORD_ID}" \
  --title "Permanent: exact_title" \
  --type "permanent"
```

The script output is the response — print it directly to the user.
