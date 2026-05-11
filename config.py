import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Config():

    base_dir = str(Path(__file__).parent.resolve())
    memory_path = f"{base_dir}/memory"
    prompt_dir_path = f"{base_dir}/prompts"

    discord_token = os.getenv('DISCORD_TOKEN')
