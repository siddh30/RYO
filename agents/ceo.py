import sys
sys.path.append('../')

from claude_agent_sdk import query, ClaudeAgentOptions, ResultMessage, AssistantMessage
from claude_agent_sdk.types import ToolUseBlock

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
    user_memories: str = "",
) -> tuple[str, dict, str | None]:
    rm = ResourceManager.get_instance()
    prompt_name = CHANNEL_PROMPTS.get(channel_type, "ceo_prompt")
    base_prompt = rm.prompt_loader(prompt_name)

    # Re-injected fresh each turn so the model always knows the current speaker.
    # user_memories pre-loads permanent memory so the agent never has to call a tool
    # just to know the user's name or preferences.
    # DisplayName is the resolved preferred name where possible; if Profile says
    # otherwise, always defer to the name stated in Profile.
    user_context = (
        f"\n<CurrentUser>\n"
        f"DisplayName: {display_name}\n"
        f"DiscordID: {discord_id}\n"
        + (f"Profile (use the name stated here when addressing the user):\n{user_memories}\n" if user_memories else "")
        + f"</CurrentUser>"
    )

    tool_calls: list[dict] = []  # telemetry: [{name, summary}]

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
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, ToolUseBlock):
                    tool_calls.append(_summarise_tool(block))

        elif isinstance(message, ResultMessage):
            usage = message.usage or {}
            cost_info = {
                "total_cost_usd": message.total_cost_usd or 0.0,
                "duration_ms": message.duration_ms,
                "num_turns": message.num_turns,
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "cache_read_tokens": usage.get("cache_read_input_tokens", 0),
                "cache_creation_tokens": usage.get("cache_creation_input_tokens", 0),
                "tool_calls": tool_calls,
                "is_error": message.is_error,
            }
            return message.result or "", cost_info, message.session_id

    return "", {}, None


def _summarise_tool(block: ToolUseBlock) -> dict:
    name = block.name
    inp = block.input or {}

    if name == "Skill":
        summary = inp.get("name") or inp.get("skill_name") or "?"
    elif name == "Bash":
        cmd = inp.get("command", "")
        summary = cmd[:80].replace("\n", " ")
    elif name in ("WebSearch", "WebFetch"):
        summary = inp.get("query") or inp.get("url") or ""
        summary = summary[:80]
    elif name == "Read":
        summary = inp.get("file_path", "")[-60:]
    elif name in ("Write", "Edit"):
        summary = inp.get("file_path", "")[-60:]
    else:
        summary = str(inp)[:80]

    return {"name": name, "summary": summary}
