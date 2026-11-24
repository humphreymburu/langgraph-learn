"""LangGraph chatbot that shows and preserves the model's reasoning.

ChatOpenAI drops OpenRouter's `reasoning_details`, so this node calls the
plain `openai` client instead and keeps raw message dicts in the graph state.
"""

import operator
import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from openai import OpenAI

load_dotenv()

MODEL = "nvidia/nemotron-3.5-super-vl:free"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)


# Custom state: a list of plain OpenAI-style message dicts.
# `operator.add` is the reducer, so node updates are appended to the list.
class State(TypedDict):
    messages: Annotated[list[dict], operator.add]


def chatbot(state: State) -> dict:
    response = client.chat.completions.create(
        model=MODEL,
        messages=state["messages"],
        extra_body={"reasoning": {"enabled": True}},
    )
    msg = response.choices[0].message
    reasoning_details = getattr(msg, "reasoning_details", None)
    # Pass reasoning_details back unmodified so the model can continue
    # reasoning from where it left off on the next turn.
    return {
        "messages": [
            {
                "role": "assistant",
                "content": msg.content,
                "reasoning_details": reasoning_details,
            }
        ]
    }


builder = StateGraph(State)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)
graph = builder.compile()


def show_last(state: State) -> None:
    last = state["messages"][-1]
    for detail in last.get("reasoning_details") or []:
        text = detail.get("text") or detail.get("summary")
        if text:
            print(f"[reasoning] {text.strip()}\n")
    print(f"[answer] {last['content']}\n")


def main():
    # Turn 1
    state = graph.invoke(
        {"messages": [{"role": "user", "content": "How many r's are in the word 'strawberry'?"}]}
    )
    show_last(state)

    # Turn 2: send the full history (including reasoning_details) back in.
    state["messages"].append({"role": "user", "content": "Are you sure? Think carefully."})
    state = graph.invoke(state)
    show_last(state)


if __name__ == "__main__":
    main()
