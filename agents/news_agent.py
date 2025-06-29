import sys
sys.path.append('../')

######### React agent ############
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

###### From config #######
from config import Config
conf = Config()


######## Resource Manager ########
from utils.resource_manager import ResourceManager
resource_manager = ResourceManager.get_instance()


prompt = resource_manager.prompt_loader("news_agent_prompt")

news_agent = create_react_agent(
    model = conf.model,
    tools = conf.news_agent_tools,
    name =  conf.news_agent_name,
    prompt = prompt,

) 