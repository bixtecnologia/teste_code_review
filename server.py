import os
import json
import time
import threading
from typing import Optional, Iterable, Dict, List, Tuple

from fastapi import FastAPI, HTTPException, Request
from starlette.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_community.utilities.sql_database import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain import hub
from langgraph.prebuilt import create_react_agent
from agent_utils import create_react_agent_compat, create_chat_openai_from_env


# Load environment variables
load_dotenv()


def _build_agent():
    # Database (expects example.db in project root)
    db = SQLDatabase.from_uri("sqlite:///catalog.db")

    # Model (configurable via env OPENAI_MODEL)
    llm = create_chat_openai_from_env(default_model="gpt-5")

    # Tools
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()

    # System prompt
    try:
        prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")
        # Template expects dialect and top_k
        system_message = prompt_template.format(dialect="SQLite", top_k=5)
    except Exception:
        # Fallback system message if Hub is unavailable
        system_message = (
            "You are a helpful SQL agent for SQLite. "
            "Generate correct, efficient SQL for the user question, execute it, and return succinct results."
        )

    # Domain-specific guidance (can be overridden via env SQL_AGENT_HINTS)
    domain_hint = os.getenv(
        "SQL_AGENT_HINTS",
        "When referencing prices, use the table 'product_itens'. For product descriptions, use the table 'products'.",
    )
    if domain_hint:
        system_message = f"{system_message}\n\nDomain guidance: {domain_hint}"

    # Agent
    agent = create_react_agent_compat(llm, tools, system_message)
    return agent


app = FastAPI(title="Quotes SQL Agent API", version="0.1.0")
agent_executor = _build_agent()

# Simple in-memory session store for chat history
SESSION_LOCK = threading.Lock()
SESSION_STORE: Dict[str, List[Tuple[str, str]]] = {}


def _get_history(session_id: Optional[str]) -> List[Tuple[str, str]]:
    if not session_id:
        return []
    with SESSION_LOCK:
        return list(SESSION_STORE.get(session_id, []))


def _append_history(session_id: Optional[str], role: str, content: str) -> None:
    if not session_id:
        return
    with SESSION_LOCK:
        history = SESSION_STORE.setdefault(session_id, [])
        history.append((role, content))


class AskRequest(BaseModel):
    question: str
    session_id: Optional[str] = None


class AskResponse(BaseModel):
    answer: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    question = req.question.strip()
    session_id = (req.session_id or "").strip() or None
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty")

    # Stream and capture the latest assistant message content
    try:
        final_text: Optional[str] = None
        history = _get_history(session_id)
        # Record the user turn in history
        _append_history(session_id, "user", question)
        events = agent_executor.stream(
            {"messages": [*history, ("user", question)]},
            stream_mode="values",
        )
        for event in events:
            msg = event["messages"][-1]
            content = getattr(msg, "content", None)
            if isinstance(content, str):
                final_text = content
            elif isinstance(content, list):
                # Concatenate text parts if content is structured
                parts = []
                for p in content:
                    if isinstance(p, dict) and "text" in p:
                        parts.append(p["text"])
                if parts:
                    final_text = "\n".join(parts)

        if not final_text:
            final_text = "No answer produced."

        # Save assistant turn
        _append_history(session_id, "assistant", final_text.strip())
        return AskResponse(answer=final_text.strip())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")


def _json_sse(event: str, data: dict) -> str:
    return f"event: {event}\n" f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _stream_agent(question: str, session_id: Optional[str]) -> Iterable[bytes]:
    # Initial event
    yield _json_sse("start", {"question": question, "session_id": session_id}).encode("utf-8")

    try:
        # Build conversation context
        history = _get_history(session_id)
        # Record user message immediately
        _append_history(session_id, "user", question)

        # Stream state values and emit the last message each step
        events = agent_executor.stream(
            {"messages": [*history, ("user", question)]},
            stream_mode="values",
        )
        announced_calls = set()
        last_text = None
        for state in events:
            try:
                msgs = state.get("messages", []) if isinstance(state, dict) else []
            except Exception:
                msgs = []
            if not msgs:
                continue
            msg = msgs[-1]
            role = getattr(msg, "type", None) or getattr(msg, "role", None) or "message"
            content = getattr(msg, "content", "")
            text = None
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                parts = []
                for p in content:
                    if isinstance(p, dict) and "text" in p:
                        parts.append(p["text"])
                if parts:
                    text = "\n".join(parts)
            if text is None:
                text = ""

            # Emit tool_call events if the model requested a tool
            try:
                tool_calls = getattr(msg, "tool_calls", None) or []
            except Exception:
                tool_calls = []
            for call in tool_calls:
                try:
                    if isinstance(call, dict):
                        name = call.get("name")
                        args = call.get("args")
                    else:
                        name = getattr(call, "name", None)
                        args = getattr(call, "args", None)
                    key = f"{name}:{json.dumps(args, sort_keys=True, ensure_ascii=False)}"
                except Exception:
                    name, args, key = (getattr(call, "name", "tool"), None, str(call))
                if key not in announced_calls:
                    announced_calls.add(key)
                    yield _json_sse("tool_call", {"name": name, "args": args}).encode("utf-8")

            # Skip echoing user/human/system messages
            if str(role).lower() in ("human", "user", "system"):
                continue

            # Only send deltas to reduce flicker/duplication
            if text != last_text:
                last_text = text
                yield _json_sse("message", {"role": role, "content": text}).encode("utf-8")

        # At end, persist assistant turn with final text
        if last_text is not None:
            _append_history(session_id, "assistant", (last_text or "").strip())

        yield _json_sse("final", {}).encode("utf-8")
    except Exception as e:
        yield _json_sse("error", {"error": str(e)}).encode("utf-8")


@app.get("/ask/stream")
def ask_stream(request: Request, q: Optional[str] = None, question: Optional[str] = None, session_id: Optional[str] = None):
    """Server-Sent Events endpoint for streaming agent progress/messages.

    Use: GET /ask/stream?question=... (EventSource-compatible)
    """
    query = (question or q or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Missing question")

    return StreamingResponse(
        _stream_agent(query, session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
