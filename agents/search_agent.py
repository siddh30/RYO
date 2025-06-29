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


prompt = resource_manager.prompt_loader("search_agent_prompt")

search_agent = create_react_agent(
    model = conf.model,
    tools = conf.search_agent_tools,
    name =  conf.search_agent_name,
    prompt = prompt,

) 