from langgraph.graph import StateGraph, START,END
from typing import TypedDict, Annotated, Literal
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
load_dotenv()
llm = ChatOpenAI()

class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState)-> ChatState:
    messages = state['messages']
    resp = llm.invoke(messages)
    return {'messages':resp}

check_pt = InMemorySaver()

graph = StateGraph(ChatState)
graph.add_node('chat_node', chat_node)
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node',END)

chat = graph.compile(checkpointer=check_pt)
con= {'configurable':{'thread_id':'thread 1'}}
resp = chat.invoke(
    {'messages': [HumanMessage(content='Hi, My name is NP')]},
    config= con,
    
)

print(chat.get_state(config=con).values['messages'])
