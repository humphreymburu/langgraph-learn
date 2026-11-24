"""Minimal LangGraph chatbot using a free model via OpenRouter."""

import os
import uuid

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.checkpoint.memory import MemorySaver


load_dotenv()  # reads OPENROUTER_API_KEY from .env

MODEL = "nvidia/nemotron-3.5-super-vl:free"

# OpenRouter speaks the OpenAI API, so ChatOpenAI works with a custom base_url.
llm = ChatOpenAI(
    model=MODEL,
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    extra_body={"reasoning": {"enabled": True}},  # OpenRouter-specific option
)


# A node is just a function: it takes the current state and returns an update.
def chatbot(state: MessagesState) -> dict:
    reply = llm.invoke(state["messages"])
    return {"messages": [reply]}  # MessagesState appends, it doesn't overwrite



# Wire the graph: START -> chatbot -> END
graph_builder = StateGraph(MessagesState)


graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

checkpointer = MemorySaver()

graph = graph_builder.compile(checkpointer=checkpointer)


def main():
    # One thread_id = one conversation. The checkpointer stores its history,
    # so each invoke only needs the new message.
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    while True:
        user_message = input("enter message (or 'quit'): ").strip()
        if user_message.lower() in {"quit", "exit", ""}:
            break
        result = graph.invoke(
            {"messages": [{"role": "user", "content": user_message}]},
            config=config,
        )
        print(f"ai: {result['messages'][-1].content}")


if __name__ == "__main__":
    main()
