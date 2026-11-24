"""Tests for the chatbot graph in main.py.

Most tests swap the real model for a fake one, so they are fast, free and
don't need the network. The last test calls OpenRouter for real and only
runs when you ask for it:  RUN_LIVE=1 uv run pytest
"""

import os
import uuid

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

import main


def new_config():
    """A fresh thread_id per test, so tests don't share conversation history."""
    return {"configurable": {"thread_id": str(uuid.uuid4())}}


@pytest.fixture
def fake_llm(monkeypatch):
    """Replace main.llm with a fake model that returns canned replies in order."""
    fake = GenericFakeChatModel(
        messages=iter([AIMessage(content="There are 3 r's."), AIMessage(content="Yes, 3.")])
    )
    monkeypatch.setattr(main, "llm", fake)
    return fake


def test_graph_appends_ai_reply(fake_llm):
    result = main.graph.invoke(
        {"messages": [{"role": "user", "content": "hi"}]}, config=new_config()
    )

    messages = result["messages"]
    assert [m.type for m in messages] == ["human", "ai"]
    assert messages[1].content == "There are 3 r's."


def test_checkpointer_remembers_same_thread(fake_llm):
    config = new_config()
    main.graph.invoke({"messages": [{"role": "user", "content": "first"}]}, config=config)
    result = main.graph.invoke({"messages": [{"role": "user", "content": "second"}]}, config=config)

    # History from turn 1 is still there: human, ai, human, ai
    assert [m.content for m in result["messages"]] == ["first", "There are 3 r's.", "second", "Yes, 3."]


def test_different_threads_are_separate(fake_llm):
    main.graph.invoke({"messages": [{"role": "user", "content": "first"}]}, config=new_config())
    result = main.graph.invoke({"messages": [{"role": "user", "content": "other"}]}, config=new_config())

    assert [m.content for m in result["messages"]] == ["other", "Yes, 3."]


def test_main_chat_loop(fake_llm, monkeypatch, capsys):
    replies = iter(["How many r's?", "Sure?", "quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(replies))
    main.main()

    out = capsys.readouterr().out
    assert "ai: There are 3 r's." in out
    assert "ai: Yes, 3." in out


@pytest.mark.skipif(not os.getenv("RUN_LIVE"), reason="set RUN_LIVE=1 to call OpenRouter")
def test_live_openrouter():
    result = main.graph.invoke(
        {"messages": [{"role": "user", "content": "Reply with just: OK"}]}, config=new_config()
    )
    assert result["messages"][-1].content.strip()
