import sys
sys.path.append('../')

from claude_agent_sdk import query, ClaudeAgentOptions, ResultMessage

from utils.resource_manager import ResourceManager


async def run_ceo(user_message: str, discord_id: str, display_name: str) -> str:
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
            return message.result or ""

    return ""
