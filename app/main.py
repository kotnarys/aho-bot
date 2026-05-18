"""FastAPI app — REST + WebSocket with persistent sessions per employee."""

from __future__ import annotations
import logging
import uuid
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from app.models.schemas import ConversationState
from app.agents.graph import process_message
from app.services.storage import (
    init_db, load_session, save_session, load_request, list_requests,
    save_chat_message, load_chat_history, clear_chat_history, delete_session,
)
from app.services.rag import get_knowledge_base

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing database...")
    init_db()
    logger.info("Loading knowledge base...")
    get_knowledge_base()
    logger.info("Ready")
    yield


app = FastAPI(title="АХО Бот API", version="2.0.0", lifespan=lifespan)

WEB_DIR = os.path.join(os.path.dirname(__file__), "..", "web")
if os.path.exists(WEB_DIR):
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str
    employee_email: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    response: str
    step: str
    current_intent: str | None = None
    current_request: dict | None = None
    debug_trace: list[dict] = []


@app.get("/")
async def root():
    index_path = os.path.join(WEB_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())
    email = request.employee_email

    # Load existing session by employee email (persistence across page reloads)
    state = load_session(session_id, employee_email=email)
    if state is None:
        state = ConversationState(session_id=session_id, employee_email=email)
        if email:
            from app.services.employees import get_employee_by_email
            emp = get_employee_by_email(email)
            if emp:
                state.employee_name = emp["name"]
                state.department = emp.get("department")

    bot_response, updated_state, debug_trace = process_message(state, request.message)

    # Save chat messages to history
    if email:
        save_chat_message(email, "user", request.message)
        save_chat_message(email, "assistant", bot_response)

    return ChatResponse(
        session_id=session_id,
        response=bot_response,
        step=updated_state.step,
        current_intent=updated_state.current_intent.value if updated_state.current_intent else None,
        current_request=updated_state.current_request,
        debug_trace=debug_trace,
    )


@app.get("/api/history/{employee_email}")
async def get_history(employee_email: str):
    """Get chat history for an employee — used to restore chat on page reload."""
    messages = load_chat_history(employee_email, limit=50)
    state = load_session("", employee_email=employee_email)
    return {
        "messages": messages,
        "step": state.step if state else "greeting",
        "current_intent": state.current_intent.value if state and state.current_intent else None,
        "current_request": state.current_request if state else None,
    }


@app.post("/api/reset/{employee_email}")
async def reset_session(employee_email: str):
    """Clear session and chat history for an employee."""
    delete_session(employee_email)
    clear_chat_history(employee_email)
    return {"status": "ok", "message": "Session cleared"}


@app.get("/api/requests")
async def get_requests(employee_email: str | None = None):
    return {"requests": list_requests(employee_email)}


@app.get("/api/requests/{request_id}")
async def get_request(request_id: str):
    req = load_request(request_id)
    if req is None:
        return {"error": "Not found"}
    return req


@app.get("/api/health")
async def health():
    return {"status": "healthy", "version": "2.0.0"}
