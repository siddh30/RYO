import os 
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_community.utilities import OpenWeatherMapAPIWrapper

######### ENV #########
from dotenv import load_dotenv

load_dotenv()

class Config():

    base_dir = '/Users/siddharth/Desktop/Data-Science-Universe/Projects/LLMS/Ryo'
    memory_path = f"{base_dir}/memory"
    prompt_dir_path = f"{base_dir}/prompts"

    ########## SECRETS #################
    discord_token = os.getenv('DISCORD_TOKEN')


    ############ MODEL ########################
    model = ChatOpenAI(model="gpt-4o")


    ########### Supervisor ################
    ryo_configs = {"configurable":{"thread_id":1}}
    


    ################# NEWS AGENT ################
    news_agent_name = 'news_agent'
    news_agent_tools = [TavilySearch(max_results=5,topic="news")]
    news_agent_configs = {"configurable":{"thread_id":2}}


    ################# SEARCH AGENT #####################
    search_agent_name = 'search_agent'

    weather_tool = OpenWeatherMapAPIWrapper()
    TavilySearch_tool = TavilySearch(max_results=5,topic="general")

    search_agent_tools =  [weather_tool.run, TavilySearch_tool]
    
    search_agent_configs = {"configurable":{"thread_id":3}}


 
 


   





