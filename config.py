import os 
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

######### ENV #########
from dotenv import load_dotenv

load_dotenv()

class Config():

    base_dir = '/Users/siddharth/Desktop/Data-Science-Universe/Projects/LLMS/Ryo'
    memory_path = f"{base_dir}/memory"

    ########## SECRETS #################
    discord_token = os.getenv('DISCORD_TOKEN')


    ############ MODEL ########################
    model = ChatOpenAI(model="gpt-4o")



    ################# NEWS AGENT ################

    ### Name ###
    news_agent_name = 'news_agent'

    ### Tools ###
    news_agent_tools = [TavilySearch(max_results=5,topic="news")]


    ### configurations ###
    news_agent_configs = {"configurable":{"thread_id":1}}

    ryo_configs = {"configurable":{"thread_id":1}}


    #### Agent Prompt ###
    news_agent_prompt = "You are a news reporter, please use all the given tools to provide the latest, most accurate news."


    





