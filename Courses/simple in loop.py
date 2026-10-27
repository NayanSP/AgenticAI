import os
from dotenv import load_dotenv
load_dotenv()
from typing import Annotated, TypedDict
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END,START

from langchain_community.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()

result = search.invoke("Magnetic force in the solar system")
print(result)


