---
name: memory-store
description: Append a new row to the user's memory in ryo.db. Use when directed by the memory-supervisor skill to persist a permanent memory or reminder.
---

# Memory Store

Stores a new memory entry by running the bundled Python script via Bash.

## Usage

```bash
python .claude/skills/memory-store/scripts/store_memory.py \
  --user-id "DISCORD_ID" \
  --flag "Permanent" \
  --title "index_name" \
  --context "content to store" \
  --window "9999-12-31T23:59:59" \
  --ai-message "Confirmation message for user"
```

Always pass the `--user-id` value from the `<CurrentUser>` block in the system prompt.

### Arguments

| Arg | Values | Description |
|-----|--------|-------------|
| `--user-id` | Discord ID string | Scopes the memory to this user |
| `--flag` | `"Permanent"` or `"Not Permanent"` | Determines which table to write to |
| `--title` | string | Short descriptive index name (no prefix — script adds it) |
| `--context` | string | The actual content to remember |
| `--window` | ISO datetime string | `"9999-12-31T23:59:59"` for permanent; specific future datetime for reminders |
| `--ai-message` | string | Friendly confirmation message shown to the user |
