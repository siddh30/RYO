import sys
sys.path.append('../')

from claude_agent_sdk import query, ClaudeAgentOptions, ResultMessage

from utils.resource_manager import ResourceManager


async def run_ceo(user_message: str, discord_id: str, display_name: str) -> tuple[str, dict]:
    rm = ResourceManager.get_instance()
    base_prompt = rm.prompt_loader("ceo_prompt")

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
            return message.result or "", cost_info

    return "", {}
