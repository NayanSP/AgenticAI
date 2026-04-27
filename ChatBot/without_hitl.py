from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from dotenv import load_dotenv
import requests

load_dotenv()
llm = ChatOpenAI()

@tool
def get_stock_price(symbol: str) -> dict:
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    )
    r = requests.get(url)
    return r.json()

@tool
def purchase_Stock(symbol: str, quantity: int)-> dict:
    return {
        'status':'success',
        'message': f'Purchase order placed for {quantity} shares of {symbol}.',
        'symbol' : symbol,
        'quantity': quantity,
    }

tool = [get_stock_price, purchase_Stock]

llm_with_tools = llm.bind_tools(tool)

class ChatState(TypedDict):
    messages:  Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    msg = state['messages']
    resp = llm_with_tools.invoke(msg)
    return {'messages':[resp]}

tool_node = ToolNode(tool)

memory = MemorySaver()

graph = StateGraph(ChatState)
graph.add_node('chat_node',chat_node)
graph.add_node('tools',tool_node)
graph.add_edge(START, 'chat_node')
graph.add_conditional_edges('chat_node',tools_condition)
graph.add_edge('tools', 'chat_node')

chat = graph.compile(checkpointer=memory)

if __name__ == "__main__":
    print("📈 Stock Bot with Tools (get_stock_price, purchase_stock)")
    print("Type 'exit' to quit.\n")

    # thread_id still works with MemorySaver (conversation kept in RAM)
    thread_id = "demo-thread"

    while True:
        user_input = input("You: ")
        if user_input.lower().strip() in {"exit", "quit"}:
            print("Goodbye!")
            break

        # Build initial state for this turn
        state = {"messages": [HumanMessage(content=user_input)]}

        # Run the graph
        result = chat.invoke(
            state,
            config={"configurable": {"thread_id": thread_id}},
        )

        # Get the latest message from the assistant
        messages = result["messages"]
        last_msg = messages[-1]
        print(f"Bot: {last_msg.content}\n")