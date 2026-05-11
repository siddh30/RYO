import os
from dotenv import load_dotenv

load_dotenv()


class Config():

    base_dir = '/Users/siddharth/Desktop/Data-Science-Universe/Projects/LLMS/Ryo'
    memory_path = f"{base_dir}/memory"
    prompt_dir_path = f"{base_dir}/prompts"

    discord_token = os.getenv('DISCORD_TOKEN')
