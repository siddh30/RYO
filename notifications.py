from config import Config
conf = Config()

##### News Agent ###
from agents.news_agent import news_agent 

##### Connect to Discord #####
import discord
import asyncio
from datetime import datetime

intents = discord.Intents.default()
intents.message_content = True

class Client(discord.Client):
    async def on_ready(self):
        print(f'Logged on as {self.user}!')
        # Start our time check in the background
        self.loop.create_task(self.time_checker())
    
    async def time_checker(self):
        while True:
            now = datetime.now()
            # Set your target time here (24-hour format)
            if now.hour == 19 and now.minute == 24:  # 2:00 PM
                # Find first available channel
                for guild in self.guilds:
                    for channel in guild.text_channels:
                        if channel.permissions_for(guild.me).send_messages:
                            await channel.send("🕑 It's 2 PM! This is my scheduled message.")
                            await asyncio.sleep(60)  # Prevent multiple sends
                            break
            await asyncio.sleep(30)  # Check every 30 seconds

    async def on_message(self, message):
        if message.author == self.user:
            return
        
        async with message.channel.typing():
            result = news_agent.invoke(
                {"messages": [{"role": "user", "content": message.content.strip()}]},
                conf.news_agent_configs
            )
            await message.channel.send(result['messages'][-1].content) 

client = Client(intents=intents)
client.run(conf.discord_token)