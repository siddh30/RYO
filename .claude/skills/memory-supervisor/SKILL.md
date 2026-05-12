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

## Profile data — always run this AFTER store_memory for personal facts

Whenever the memory is a personal fact about the user, also update the structured profile.
Keys come in two types — pick the right one and use the right command form.

### Key types

**SCALAR** — single value, replaced on update. Use plain `--value`.

| Data | Key |
|---|---|
| What to call the user ("call me X", "refer to me as X", "I am known as X") | `preferred_name` |
| Full / legal name | `actual_name` |
| City / location | `location` |
| Job title + company | `role` |
| Email | `email` |
| Phone | `phone` |
| Age or birth year | `age` |
| Home airport | `home_airport` |
| Timezone | `timezone` |

**BUCKET** — comma-separated accumulator, never overwritten. Always use `--append`.
Bucket keys auto-append even without `--append`, but be explicit.

| Data | Key |
|---|---|
| Languages spoken | `languages` |
| Hobbies / leisure | `hobbies` |
| Sports / physical activities | `sports` |
| General interests / topics | `interests` |
| Professional / technical skills | `skills` |
| Credit or charge cards | `credit_cards` |
| Dietary preferences / restrictions | `dietary_preferences` |
| Allergies / intolerances | `allergies` |
| Pets | `pets` |
| Goals | `goals` |
| News topics followed | `news_interests` |
| Travel wishlist / bucket list | `travel_wishlist` |

### When to invent a custom key

Only invent a new key when the fact **cannot** fit any bucket above.
Name it as a **category** (`snake_case`), not a specific fact — so future similar facts land in the same bucket.

- ❌ `hobby_marathon` — too specific, won't catch "I also cycle"
- ✅ `sports: marathon running` — use the `sports` bucket instead
- ❌ `project_ryo` — reasonable if truly unique
- ✅ `dev_projects` — better if the user might mention more projects

### Commands

```bash
# Scalar — replace
python memory/update_profile.py --user-id "{DISCORD_ID}" --key preferred_name --value "Sid"

# Bucket — append (script also auto-appends for known bucket keys)
python memory/update_profile.py --user-id "{DISCORD_ID}" --key sports --value "marathon running" --append
python memory/update_profile.py --user-id "{DISCORD_ID}" --key credit_cards --value "AMEX Platinum" --append

# Custom category bucket
python memory/update_profile.py --user-id "{DISCORD_ID}" --key dev_projects --value "RYO Discord bot" --append

# List current profile
python memory/update_profile.py --user-id "{DISCORD_ID}" --list
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
