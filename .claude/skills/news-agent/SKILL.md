---
name: news-agent
description: Fetch current news headlines and links on any topic. Use when the user asks about current events, news, or recent happenings.
---

# News Agent

Use WebSearch to fetch headlines. Return a tight numbered list — headline + bare URL, nothing else.

## Output format

```
📰 **[Topic] Headlines**

1. Headline text — https://url
2. Headline text — https://url
3. Headline text — https://url
```

- 5–7 items max
- No summaries, no source labels, no dates
- If no URL available, omit the item rather than guessing

## Instructions

1. Use WebSearch with `"news today"` or topic-specific queries.
2. For general queries, use the user's news preferences from their memory to pick topics.
3. For specific queries (e.g., "AI news"), search that topic directly — no preference filtering.
4. Pick the most recent and relevant results only.
