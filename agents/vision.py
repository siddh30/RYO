import base64
import os
import aiohttp
import anthropic

SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
SUPPORTED_DOC_TYPES = {"application/pdf"}

VISION_SYSTEM = """You are Ryo, a sharp assistant on Discord.
The user has sent you one or more files (images or PDFs), possibly with a question.
Analyse the content and answer naturally and concisely.

Discord format rules:
- NO markdown headers — use **bold** for emphasis
- NO horizontal rules
- Keep responses short and punchy
- Use emojis where they naturally fit
"""


async def run_vision(
    user_message: str,
    attachment_urls: list[tuple[str, str]],  # [(url, content_type), ...]
    display_name: str,
) -> str:
    content: list[dict] = []

    async with aiohttp.ClientSession() as session:
        for url, content_type in attachment_urls:
            media_type = content_type.split(";")[0].strip()
            async with session.get(url) as resp:
                file_bytes = await resp.read()
            b64 = base64.standard_b64encode(file_bytes).decode("utf-8")

            if media_type in SUPPORTED_IMAGE_TYPES:
                content.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": b64},
                })
            elif media_type in SUPPORTED_DOC_TYPES:
                content.append({
                    "type": "document",
                    "source": {"type": "base64", "media_type": media_type, "data": b64},
                })

    if not content:
        return "I can analyse images (JPEG, PNG, GIF, WEBP) and PDF files."

    content.append({
        "type": "text",
        "text": user_message or "What does this contain?",
    })

    client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = await client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system=VISION_SYSTEM + f"\nThe user's name is {display_name}.",
        messages=[{"role": "user", "content": content}],
        betas=["pdfs-2024-09-25"],
    )
    return response.content[0].text
