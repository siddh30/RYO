---
name: memory-delete
description: Remove a row from the user's memory in ryo.db. Use when directed by the memory-supervisor skill to forget a permanent memory or reminder.
---

# Memory Delete

Deletes a memory entry by running the bundled Python script via Bash.

## Usage

```bash
python .claude/skills/memory-delete/scripts/delete_memory.py \
  --user-id "DISCORD_ID" \
  --title "Permanent: exact_index_title" \
  --type "permanent"
```

Use `--type "reminder"` for reminders. Always pass the `--user-id` from `<CurrentUser>`.

### Arguments

| Arg | Values | Description |
|-----|--------|-------------|
| `--user-id` | Discord ID string | Scopes the deletion to this user only |
| `--title` | string | The exact `index_title` as stored in the database (including prefix) |
| `--type` | `"permanent"` or `"reminder"` | Determines which table to modify |

Always run `python memory/query_memory.py --user-id {id}` first to get the exact `index_title` before calling this script.
