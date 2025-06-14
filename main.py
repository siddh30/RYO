from config import Config
conf = Config()

##### News Agent ###
from agents.news_agent import news_agent 

##### Connect to Discord #####
import discord

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
        
        result = news_agent.invoke({"messages": [{"role": "user", "content":message.content.strip()}]},
        conf.news_agent_configs)

        await message.channel.send(result['messages'][-1].content) 


client = Client(intents=intents)
client.run(conf.discord_token)

