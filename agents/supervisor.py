import sys
sys.path.append('../')
import pandas as pd
from langgraph_supervisor import create_supervisor
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.prompts import PromptTemplate
from langgraph_supervisor.handoff import create_forward_message_tool

######### Import Agents #########
from agents.news_agent import news_agent
from agents.central_memory_agent import central_memory_agent

###### From config #######
from config import Config
conf = Config()


######## Resource Manager ########
from utils.resource_manager import ResourceManager
resource_manager = ResourceManager.get_instance()



prompt_blueprint = resource_manager.prompt_loader("supervisor_prompt")



try:
    long_term_memory = pd.read_csv(f"{conf.memory_path}/Permanent_Memory.csv")
    long_term_memory_str = long_term_memory.to_json(orient='records', indent=4)

except:
    long_term_memory_str = ""


try:
    reminders = pd.read_csv(f"{conf.memory_path}/Reminders.csv")
    reminders_str = reminders.to_json(orient='records', indent=4)

except:
    reminders_str = ""


prompt_template = PromptTemplate.from_template(prompt_blueprint)
prompt = prompt_template.invoke({"ToolMessage": '{"tool":"forwarded_response","args":{"from_agent":"news_agent"}}',
                            "long_term_memory_string": long_term_memory_str,
                            "reminder_string": reminders_str}).to_string()


forwarding_tool = create_forward_message_tool("forwarded_response")

graph = create_supervisor([news_agent, central_memory_agent], 
                        model=conf.model, 
                        supervisor_name='Ryo',
                        prompt=prompt,
                        tools=[forwarding_tool],
                        output_mode='last_message')

ryo = graph.compile(checkpointer=InMemorySaver())
