import sys
sys.path.append('../')

from claude_agent_sdk import query, ClaudeAgentOptions, ResultMessage

from utils.resource_manager import ResourceManager


CHANNEL_PROMPTS = {
    "travel": "travel_prompt",
    "travel_preferences_save": "travel_preferences_save",
}


async def run_ceo(
    user_message: str,
    discord_id: str,
    display_name: str,
    channel_type: str = "general",
    session_id: str | None = None,
) -> tuple[str, dict, str | None]:
    rm = ResourceManager.get_instance()
    prompt_name = CHANNEL_PROMPTS.get(channel_type, "ceo_prompt")
    base_prompt = rm.prompt_loader(prompt_name)

    # Re-injected fresh each turn so the model always knows the current speaker,
    # even when resuming a session that was started by a different user.
    user_context = (
        f"\n<CurrentUser>\n"
        f"DisplayName: {display_name}\n"
        f"DiscordID: {discord_id}\n"
        f"</CurrentUser>"
    )

    async for message in query(
        prompt=user_message,
        options=ClaudeAgentOptions(
            system_prompt=base_prompt + user_context,
            model="claude-sonnet-4-6",
            allowed_tools=["Read", "Write", "Edit", "Bash", "WebSearch", "WebFetch"],
            permission_mode="dontAsk",
            resume=session_id,
        ),
    ):
        if isinstance(message, ResultMessage):
            usage = message.usage or {}
            cost_info = {
                "total_cost_usd": message.total_cost_usd or 0.0,
                "duration_ms": message.duration_ms,
                "num_turns": message.num_turns,
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "cache_read_tokens": usage.get("cache_read_input_tokens", 0),
                "cache_creation_tokens": usage.get("cache_creation_input_tokens", 0),
            }
            return message.result or "", cost_info, message.session_id

    return "", {}, None
