import sys
sys.path.append('../')

from langgraph_supervisor import create_supervisor
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.prompts import PromptTemplate

###### From config #######
from config import Config
conf = Config()

import pandas as pd

from agents.news_agent import news_agent
from agents.central_memory_agent import central_memory_agent

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


prompt_template = PromptTemplate.from_template("You are Ryo, health and lifestyle supervisor managing a Central Memory Agent and a News Agent. You will be also provided with Past Interaction asked by the user store permanently and as reminders.\n" 
                                "\n<Agent Instructions>\n"
                                "1. For news related queries use News Agent.\n"
                                "2. For storing memories and reminders, use Central Memory Agent. \n" 
                                "\n<Past Interactions stored as Permanent Memory>\n" 
                                "{long_term_memory_string}\n"
                                "\n<Past Interactions stored as Reminders Memory>\n" 
                                "{reminder_string}")
prompt = prompt_template.invoke({"long_term_memory_string": long_term_memory_str,
                            "reminder_string": reminders_str}).to_string()


graph = create_supervisor([news_agent, central_memory_agent], 
                        model=conf.model, 
                        supervisor_name='Ryo',
                        prompt=prompt
                        
    
    
)
ryo = graph.compile(checkpointer=InMemorySaver())
ryo



