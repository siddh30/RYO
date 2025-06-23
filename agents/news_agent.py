import sys
sys.path.append('../')

######### React agent ############
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

###### From config #######
from config import Config
conf = Config()


news_agent = create_react_agent(
    model = conf.model,
    tools = conf.news_agent_tools,
    name =  conf.news_agent_name,
    prompt = conf.news_agent_prompt,

) 