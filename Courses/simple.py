from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI

llm = "gpt-4o-mini"

client = ChatOpenAI(model=llm)

resp = client.invoke([
    ("system", "You are a helpful assistant."),
    ("human", "What is magnetic force in the context of the solar system?")
])

print(resp.content)