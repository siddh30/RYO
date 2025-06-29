import sys
sys.path.append('../')


import uuid

######### React agent ############
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate

###### From config #######
from config import Config
conf = Config()

####### From Resources #####
from utils.resource_manager import ResourceManager
resource_manager = ResourceManager.get_instance()

import pandas as pd


##### Agent ##############
from langgraph_supervisor import create_supervisor
from langgraph_supervisor.handoff import create_forward_message_tool
from agents.news_agent import news_agent
from agents.search_agent import search_agent
from agents.memory_store_agent import memory_store_agent
from agents.memory_delete_agent import memory_delete_agent



prompt_blueprint = resource_manager.prompt_loader("central_memory_supervisor")

prompt_template = PromptTemplate.from_template(prompt_blueprint)
prompt = prompt_template.invoke({"ToolMessage": '{"tool":"forwarded_response","args":{"from_agent":"news_agent"}}'}).to_string()


forwarding_tool = create_forward_message_tool("forwarded_response")

graph = create_supervisor([memory_store_agent, memory_delete_agent], 
                        model=conf.model, 
                        supervisor_name='central_memory_supervisor',
                        prompt=prompt,
                        output_mode='last_message')

cms = graph.compile(checkpointer=InMemorySaver())
cms.name =  'central_memory_supervisor'


