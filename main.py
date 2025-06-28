from config import Config
conf = Config()

##### News Agent ###
from agents.supervisor import ryo
from agents.news_agent import news_agent

##### Connect to Discord #####
import discord

####### Bud
from langchain_core.prompts import PromptTemplate

# Define the intents your bot needs
intents = discord.Intents.default()
intents.message_content = True  # Required for Discord API v2 to read message content


class Client(discord.Client):
    async def on_ready(self):
        print(f'Logged on as {self.user}!')

    async def on_message(self, message):
        # Ensure the bot doesn't respond to its own messages
        if message.author == self.user:
            return
        
        async with message.channel.typing():
            result = ryo.invoke({"messages": [{"role": "user", "content":message.content.strip()}]},
            conf.ryo_configs)
            await message.channel.send(result['messages'][-1].content) 
        


client = Client(intents=intents)
client.run(conf.discord_token)