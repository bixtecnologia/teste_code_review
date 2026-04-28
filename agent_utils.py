from typing import Any
import inspect
import os

from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI


def create_react_agent_compat(llm: Any, tools: Any, system_message: Any):
    """Create a ReAct agent compatible with multiple LangGraph versions.

    Tries supported kwarg names in this order:
    - messages_modifier
    - state_modifier
    - prompt
    Falls back to calling without modifier if none are supported.
    """
    # Try introspecting the function signature first
    try:
        params = set(inspect.signature(create_react_agent).parameters.keys())
    except Exception:
        params = set()

    if "messages_modifier" in params:
        return create_react_agent(llm, tools, messages_modifier=system_message)
    if "state_modifier" in params:
        return create_react_agent(llm, tools, state_modifier=system_message)
    if "prompt" in params:
        return create_react_agent(llm, tools, prompt=system_message)

    # If signature detection didn't help, brute-force try common kwargs
    for kwargs in (
        {"messages_modifier": system_message},
        {"state_modifier": system_message},
        {"prompt": system_message},
    ):
        try:
            return create_react_agent(llm, tools, **kwargs)
        except TypeError:
            continue

    # Last resort: no modifier
    return create_react_agent(llm, tools)


def create_chat_openai_from_env(default_model: str = "gpt-5") -> ChatOpenAI:
    """Create ChatOpenAI with optional reasoning.effort based on env vars.

    Env:
    - OPENAI_MODEL: model name (default: gpt-5)
    - OPENAI_REASONING_EFFORT: minimal|low|medium|high (default: minimal). Use
      'off'/'none' to disable sending the reasoning parameter.
    """
    model_name = os.getenv("OPENAI_MODEL", default_model)
    effort = os.getenv("OPENAI_REASONING_EFFORT", "minimal").strip().lower()

    kwargs: dict[str, Any] = {"model": model_name}

    # Only attach reasoning param for gpt-5 family unless explicitly disabled
    if (
        model_name.startswith("gpt-5")
        and effort
        and effort not in ("off", "none", "disable")
    ):
        kwargs["reasoning"] = {"effort": effort}

    return ChatOpenAI(**kwargs)
