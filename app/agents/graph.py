"""
AHO Bot v2 — Multi-agent architecture with tool calling.

Architecture:
  Router Agent → [6 Specialist Agents] → Tool Execution → Confirm → Submit

Each specialist agent:
  - Has its own system prompt and domain expertise
  - Collects required fields through dialog
  - Calls mock MCP tools to execute actions
"""

from __future__ import annotations
import json
import logging
import time
import uuid
from typing import TypedDict
from pydantic import ValidationError
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from app.config import get_settings
from app.agents.prompts import (
    ROUTER_PROMPT,
    AGENT_SYSTEM_PROMPTS,
    COLLECT_FIELDS_PROMPT,
    CONFIRMATION_PROMPT,
)
from app.models.schemas import (
    IntentType,
    ConversationState,
    RequestStatus,
    REQUEST_MODEL_MAP,
)
from app.services.rag import get_knowledge_base
from app.services.storage import save_session, save_request
from app.tools.mcp_tools import execute_tool

logger = logging.getLogger(__name__)


# ── LLM ───────────────────────────────────────────────

def _get_llm() -> ChatOpenAI:
    settings = get_settings()
    kwargs = {"model": settings.model_name, "api_key": settings.openai_api_key, "temperature": 0.3}
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    return ChatOpenAI(**kwargs)


def _trace(event_type: str, data: dict, duration_ms: float = 0) -> dict:
    return {"timestamp": time.time(), "type": event_type, "duration_ms": round(duration_ms, 1), "data": data}


def _parse_json(text: str) -> dict | None:
    """Robust JSON extraction from LLM response."""
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
            if text.startswith("json"):
                text = text[4:]
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        # Try to find JSON object in text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    return None


# ── Guardrails: Pydantic validation of LLM output ────

def validate_router_output(data: dict) -> dict:
    """Validate router LLM output against expected schema."""
    valid_agents = [e.value for e in IntentType]
    agent = data.get("agent", "unknown")
    if agent not in valid_agents:
        data["agent"] = "unknown"
    confidence = data.get("confidence", 0.5)
    if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
        data["confidence"] = 0.5
    if "extracted_entities" not in data or not isinstance(data["extracted_entities"], dict):
        data["extracted_entities"] = {}
    return data


def validate_collect_output(data: dict) -> dict:
    """Validate field collection LLM output."""
    if "extracted_fields" not in data or not isinstance(data["extracted_fields"], dict):
        data["extracted_fields"] = {}
    if "all_collected" not in data:
        data["all_collected"] = False
    if "next_question" not in data:
        data["next_question"] = None
    return data


def validate_request_with_pydantic(intent: str, request_data: dict) -> tuple[bool, list[str]]:
    """Validate final request against Pydantic model. Returns (is_valid, errors)."""
    try:
        intent_enum = IntentType(intent)
    except ValueError:
        return False, [f"Unknown intent: {intent}"]
    model_cls = REQUEST_MODEL_MAP.get(intent_enum)
    if not model_cls:
        return False, [f"No model for intent: {intent}"]
    try:
        model_cls.model_validate(request_data)
        return True, []
    except ValidationError as e:
        errors = [f"{err['loc']}: {err['msg']}" for err in e.errors()]
        return False, errors


# ── Graph state ───────────────────────────────────────

class GraphState(TypedDict):
    session_id: str
    user_message: str
    employee_info: dict | None
    current_intent: str | None
    current_request: dict
    step: str
    messages: list[dict]
    bot_response: str
    knowledge_context: str
    missing_fields: list[str]
    requests_history: list[dict]
    debug_trace: list[dict]
    active_agent: str | None
    tool_results: list[dict]


# ── Required fields per agent ─────────────────────────

REQUIRED_FIELDS = {
    IntentType.OFFICE_SUPPLIES: ["items", "delivery_office"],
    IntentType.SIM_CARD: ["target_employee_name", "target_department", "manager_name", "sim_type"],
    IntentType.BUSINESS_TRIP: ["destination_city", "check_in_date", "check_out_date"],
    IntentType.PARKING_PASS: ["car_plate", "car_brand", "pass_start_date", "pass_end_date"],
    IntentType.TAXI: ["pickup_location", "destination", "pickup_time"],
    IntentType.INCIDENT: ["category", "description", "location"],
}

FIELD_LABELS_RU = {
    "items": "товары (название и количество)",
    "delivery_office": "адрес доставки / офис",
    "target_employee_name": "ФИО сотрудника для SIM",
    "target_department": "подразделение сотрудника",
    "manager_name": "ФИО руководителя",
    "sim_type": "тип SIM (физическая / eSIM)",
    "destination_city": "город командировки",
    "check_in_date": "дата заезда",
    "check_out_date": "дата выезда",
    "car_plate": "номер автомобиля",
    "car_brand": "марка и модель авто",
    "pass_start_date": "дата начала пропуска",
    "pass_end_date": "дата окончания пропуска",
    "pickup_location": "откуда забрать",
    "destination": "куда ехать",
    "pickup_time": "время подачи",
    "category": "категория проблемы",
    "description": "описание проблемы",
    "location": "локация (офис, этаж, помещение)",
}

INTENT_LABELS_RU = {
    "office_supplies": "📦 Канцтовары/продукты",
    "sim_card": "📱 SIM-карта",
    "business_trip": "✈️ Командировка",
    "parking_pass": "🅿️ Парковка",
    "taxi": "🚕 Такси",
    "incident": "⚠️ Инцидент",
    "unknown": "❓ Не определено",
}

# Map intent → tool to call on submit
INTENT_TOOL_MAP = {
    "office_supplies": "place_order_komus",
    "sim_card": "request_sim_activation",
    "business_trip": "book_hotel",
    "parking_pass": "issue_parking_pass",
    "taxi": "order_taxi",
    "incident": "create_incident_ticket",
}

# Map intent → tool argument mapping
INTENT_TOOL_ARGS = {
    "office_supplies": lambda r: {"items": r.get("items", []), "delivery_office": r.get("delivery_office", "")},
    "sim_card": lambda r: {"employee_name": r.get("target_employee_name", ""), "department": r.get("target_department", ""), "sim_type": r.get("sim_type", "physical"), "roaming": r.get("international_roaming", False)},
    "business_trip": lambda r: {"city": r.get("destination_city", ""), "check_in": str(r.get("check_in_date", "")), "check_out": str(r.get("check_out_date", "")), "preferences": r.get("hotel_preferences", "")},
    "parking_pass": lambda r: {"car_plate": r.get("car_plate", ""), "car_brand": r.get("car_brand", ""), "start_date": str(r.get("pass_start_date", "")), "end_date": str(r.get("pass_end_date", ""))},
    "taxi": lambda r: {"pickup": r.get("pickup_location", ""), "destination": r.get("destination", ""), "pickup_time": str(r.get("pickup_time", "")), "passengers": r.get("passengers", 1)},
    "incident": lambda r: {"category": r.get("category", "other"), "description": r.get("description", ""), "location": r.get("location", ""), "priority": r.get("priority", "medium")},
}


# ── Node: Router Agent ────────────────────────────────

def router_node(state: GraphState) -> GraphState:
    """Router agent — classifies intent and delegates to specialist."""
    step = state.get("step", "greeting")
    msg = state["user_message"].strip().lower()
    trace = state.get("debug_trace", [])

    # Handle confirmation responses
    if step == "confirm":
        # Confirm → submit
        if any(w in msg for w in ["да", "верно", "подтверждаю", "ок", "отправляй", "yes", "go"]):
            state["step"] = "submit"
            trace.append(_trace("decision", {"node": "router", "action": "user_confirmed → submit", "agent": "router"}))
            state["debug_trace"] = trace
            return state

        # Cancel → reset everything
        if any(w in msg for w in ["отмена", "отменить", "cancel", "стоп", "забудь"]) and not any(w in msg for w in ["изменить", "поменять", "поправ"]):
            state["step"] = "done"
            state["current_intent"] = None
            state["current_request"] = {}
            state["missing_fields"] = []
            state["active_agent"] = None
            state["bot_response"] = "✖️ Заявка отменена. Чем ещё могу помочь?"
            trace.append(_trace("decision", {"node": "router", "action": "user_cancelled → reset"}))
            state["debug_trace"] = trace
            return state

        # Edit request — "нет", "изменить", "поправить", or any message with content
        # Pass to specialist agent to update fields
        state["step"] = "collect"
        trace.append(_trace("decision", {"node": "router", "action": "user wants edit → back to specialist with message"}))
        state["debug_trace"] = trace
        return state

    # Continue collecting if already with a specialist
    if step == "collect" and state.get("current_intent"):
        state["step"] = "collect"
        trace.append(_trace("decision", {"node": "router", "action": f"continue with agent: {state['active_agent'] or state['current_intent']}"}))
        state["debug_trace"] = trace
        return state

    # New request — classify
    state["step"] = "classify"
    trace.append(_trace("node_enter", {"node": "router", "action": "classifying intent", "user_message": state["user_message"]}))
    state["debug_trace"] = trace
    return state


def classify_node(state: GraphState) -> GraphState:
    """Classify intent using router agent."""
    trace = state.get("debug_trace", [])
    llm = _get_llm()
    msg = state["user_message"]
    settings = get_settings()

    prompt = ROUTER_PROMPT.format(message=msg)

    trace.append(_trace("llm_call", {
        "node": "router_agent",
        "purpose": "Intent classification → delegate to specialist",
        "model": settings.model_name,
        "prompt_preview": prompt[:300],
    }))

    t0 = time.time()
    response = llm.invoke([
        SystemMessage(content="Ты — маршрутизатор. Определи агента. Отвечай только JSON."),
        HumanMessage(content=prompt),
    ])
    duration = (time.time() - t0) * 1000

    result = _parse_json(response.content)
    if result is None:
        result = {"agent": "unknown", "confidence": 0.0, "extracted_entities": {}, "reasoning": "JSON parse failed"}

    # Guardrail: validate
    result = validate_router_output(result)

    intent = result["agent"]
    entities = result["extracted_entities"]

    trace.append(_trace("llm_response", {
        "node": "router_agent",
        "raw_response": response.content.strip()[:500],
        "parsed": {
            "agent": intent,
            "agent_label": INTENT_LABELS_RU.get(intent, intent),
            "confidence": result["confidence"],
            "reasoning": result.get("reasoning", ""),
            "extracted_entities": entities,
        },
        "guardrails": "✅ passed validation",
    }, duration))

    # RAG — search by intent + entities + domain keywords, not raw message
    t0 = time.time()
    kb = get_knowledge_base()
    # Build rich RAG query: intent label + entities + domain keywords
    rag_parts = [INTENT_LABELS_RU.get(intent, intent)]
    rag_parts.extend(str(v) for v in entities.values() if v)
    # Add domain-specific keywords for better retrieval
    domain_keywords = {
        "business_trip": "командировка отель проживание лимит суточные трансфер",
        "office_supplies": "канцтовары заказ поставщик Комус Вкусвилл лимит",
        "sim_card": "SIM карта корпоративная МТС тариф роуминг",
        "parking_pass": "парковка пропуск автомобиль",
        "taxi": "такси трансфер корпоративное Яндекс",
        "incident": "инцидент поломка проблема непорядок",
    }
    rag_parts.append(domain_keywords.get(intent, ""))
    rag_query = " ".join(rag_parts).strip()
    knowledge_docs = kb.search(rag_query, n_results=3)
    rag_duration = (time.time() - t0) * 1000
    state["knowledge_context"] = "\n---\n".join(knowledge_docs)

    trace.append(_trace("rag_search", {
        "node": "router_agent",
        "query": rag_query,
        "results_count": len(knowledge_docs),
        "snippets": [doc[:150] + "..." for doc in knowledge_docs],
    }, rag_duration))

    if intent == "unknown":
        state["step"] = "greeting"
        state["bot_response"] = (
            "Я могу помочь с:\n"
            "📦 Заказ канцтоваров и продуктов\n"
            "📱 Оформление корпоративной SIM-карты\n"
            "✈️ Организация командировки\n"
            "🅿️ Пропуск на парковку\n"
            "🚕 Заказ такси\n"
            "⚠️ Сообщить о проблеме (Непорядок!)\n\n"
            "Опиши, что тебе нужно, и я помогу оформить заявку."
        )
        state["debug_trace"] = trace
        return state

    state["current_intent"] = intent
    state["active_agent"] = intent

    # Init request
    req = {"type": intent, **entities}
    emp = state.get("employee_info")
    if emp:
        req["employee_name"] = emp.get("name")
        req["employee_email"] = emp.get("email")
        req["department"] = emp.get("department")
        trace.append(_trace("autofill", {"node": "router_agent", "employee": emp.get("name"), "fields_filled": ["employee_name", "employee_email", "department"]}))

    state["current_request"] = req
    state["step"] = "collect"

    intent_enum = IntentType(intent)
    required = REQUIRED_FIELDS.get(intent_enum, [])
    missing = [f for f in required if not req.get(f)]
    state["missing_fields"] = missing

    trace.append(_trace("agent_delegation", {
        "node": "router_agent",
        "delegated_to": INTENT_LABELS_RU.get(intent, intent),
        "agent_id": intent,
        "required_fields": required,
        "already_filled": [f for f in required if req.get(f)],
        "missing_fields": missing,
    }))

    state["debug_trace"] = trace
    return state


# ── Node: Specialist Agent (collect) ──────────────────

def specialist_node(state: GraphState) -> GraphState:
    """Specialist agent — collects data for its domain."""
    trace = state.get("debug_trace", [])
    llm = _get_llm()
    intent = state.get("current_intent", "")
    request_data = state.get("current_request", {})
    msg = state["user_message"]
    settings = get_settings()

    try:
        intent_enum = IntentType(intent)
    except ValueError:
        state["step"] = "greeting"
        state["bot_response"] = "Произошла ошибка. Давай начнём сначала."
        state["debug_trace"] = trace
        return state

    agent_prompt = AGENT_SYSTEM_PROMPTS.get(intent, "Ты — агент-специалист АХО.")

    required = REQUIRED_FIELDS.get(intent_enum, [])
    missing = [f for f in required if not request_data.get(f)]
    missing_labels = [f"{f} ({FIELD_LABELS_RU.get(f, f)})" for f in missing]

    emp = state.get("employee_info")
    emp_info_str = json.dumps(emp, ensure_ascii=False) if emp else "Не указан"

    # RAG for this specific agent context — use intent keywords + user message
    kb = get_knowledge_base()
    domain_keywords = {
        "business_trip": "командировка отель проживание лимит суточные трансфер бронирование",
        "office_supplies": "канцтовары заказ поставщик Комус Вкусвилл",
        "sim_card": "SIM карта корпоративная МТС тариф роуминг",
        "parking_pass": "парковка пропуск автомобиль",
        "taxi": "такси трансфер корпоративное",
        "incident": "инцидент поломка проблема",
    }
    rag_query = f"{domain_keywords.get(intent, '')} {msg}"
    knowledge_docs = kb.search(rag_query, n_results=3)
    knowledge_context = "\n---\n".join(knowledge_docs)

    prompt = COLLECT_FIELDS_PROMPT.format(
        request_type=intent,
        employee_info=emp_info_str,
        current_data=json.dumps(request_data, ensure_ascii=False, default=str),
        missing_fields=", ".join(missing_labels) if missing_labels else "ВСЕ СОБРАНЫ",
        user_message=msg,
        knowledge_context=knowledge_context,
    )

    trace.append(_trace("llm_call", {
        "node": f"agent:{intent}",
        "purpose": f"Specialist agent [{INTENT_LABELS_RU.get(intent, intent)}] extracting fields",
        "model": settings.model_name,
        "agent_system": agent_prompt[:100],
        "missing_fields": missing_labels,
        "rag_context_snippets": len(knowledge_docs),
    }))

    t0 = time.time()
    response = llm.invoke([
        SystemMessage(content=f"{agent_prompt}\n\nОтвечай только JSON."),
        HumanMessage(content=prompt),
    ])
    duration = (time.time() - t0) * 1000

    result = _parse_json(response.content)
    if result is None:
        result = {"extracted_fields": {}, "next_question": "Не удалось распознать. Повтори, пожалуйста?", "all_collected": False}

    # Guardrail
    result = validate_collect_output(result)

    extracted = result.get("extracted_fields", {})

    trace.append(_trace("llm_response", {
        "node": f"agent:{intent}",
        "raw_response": response.content.strip()[:500],
        "extracted_fields": extracted,
        "next_question": result.get("next_question"),
        "all_collected": result.get("all_collected", False),
        "guardrails": "✅ passed validation",
    }, duration))

    for k, v in extracted.items():
        if v is not None and v != "":
            request_data[k] = v
    state["current_request"] = request_data

    old_missing = list(missing)
    missing = [f for f in required if not request_data.get(f)]
    state["missing_fields"] = missing

    trace.append(_trace("fields_update", {
        "node": f"agent:{intent}",
        "newly_filled": [f for f in old_missing if f not in missing],
        "still_missing": missing,
        "request_snapshot": request_data,
    }))

    if not missing or result.get("all_collected"):
        state["step"] = "confirm"
        trace.append(_trace("decision", {"node": f"agent:{intent}", "action": "all fields → confirmation"}))
        state["debug_trace"] = trace
        state = _generate_confirmation(state)
    else:
        next_q = result.get("next_question")
        if next_q:
            state["bot_response"] = next_q
        else:
            label = FIELD_LABELS_RU.get(missing[0], missing[0])
            state["bot_response"] = f"Укажи, пожалуйста: {label}"
        state["step"] = "collect"
        state["debug_trace"] = trace

    return state


def _generate_confirmation(state: GraphState) -> GraphState:
    trace = state.get("debug_trace", [])
    llm = _get_llm()
    intent = state.get("current_intent", "")
    request_data = state.get("current_request", {})
    settings = get_settings()

    kb = get_knowledge_base()
    knowledge = kb.search(INTENT_LABELS_RU.get(intent, intent), n_results=2)
    knowledge_context = "\n---\n".join(knowledge)

    prompt = CONFIRMATION_PROMPT.format(
        request_type=intent,
        request_data=json.dumps(request_data, ensure_ascii=False, default=str),
        knowledge_context=knowledge_context,
    )

    trace.append(_trace("llm_call", {"node": f"agent:{intent}", "purpose": "Generate confirmation", "model": settings.model_name}))

    t0 = time.time()
    response = llm.invoke([
        SystemMessage(content=AGENT_SYSTEM_PROMPTS.get(intent, "Сформируй сводку.") + "\nСформируй читабельную сводку с эмодзи. Спроси подтверждение."),
        HumanMessage(content=prompt),
    ])
    duration = (time.time() - t0) * 1000

    state["bot_response"] = response.content
    trace.append(_trace("llm_response", {"node": f"agent:{intent}", "summary_preview": response.content[:300]}, duration))
    state["debug_trace"] = trace
    return state


# ── Node: Tool execution + Submit ─────────────────────

def submit_node(state: GraphState) -> GraphState:
    """Execute MCP tool and submit the request."""
    trace = state.get("debug_trace", [])
    request_data = state.get("current_request", {})
    intent = state.get("current_intent", "")

    # Execute tool
    tool_name = INTENT_TOOL_MAP.get(intent)
    tool_args_fn = INTENT_TOOL_ARGS.get(intent)

    tool_result = {}
    if tool_name and tool_args_fn:
        args = tool_args_fn(request_data)
        trace.append(_trace("tool_call", {
            "node": f"agent:{intent}",
            "tool": tool_name,
            "arguments": args,
        }))

        t0 = time.time()
        tool_result = execute_tool(tool_name, args)
        duration = (time.time() - t0) * 1000

        trace.append(_trace("tool_result", {
            "node": f"agent:{intent}",
            "tool": tool_name,
            "result": tool_result,
            "status": tool_result.get("status", "unknown"),
        }, duration))

    # Save
    request_data["status"] = RequestStatus.SUBMITTED.value
    req_id = tool_result.get("order_id") or tool_result.get("ticket_id") or tool_result.get("booking_id") or tool_result.get("pass_id") or str(uuid.uuid4())[:8]
    request_data["id"] = req_id
    request_data["tool_result"] = tool_result

    save_request(req_id, request_data, request_data.get("employee_email", ""))

    history = state.get("requests_history", [])
    history.append(request_data)
    state["requests_history"] = history

    tool_msg = tool_result.get("message", "")
    state["bot_response"] = (
        f"✅ {tool_msg}\n\n"
        f"Статус: **В обработке**\n"
        f"Номер заявки: **{req_id}**\n\n"
        f"Могу помочь ещё с чем-нибудь?"
    )

    trace.append(_trace("submit", {
        "node": f"agent:{intent}",
        "request_id": req_id,
        "tool_used": tool_name,
        "final_json": request_data,
    }))

    state["step"] = "done"
    state["current_intent"] = None
    state["current_request"] = {}
    state["missing_fields"] = []
    state["active_agent"] = None
    state["debug_trace"] = trace
    return state


# ── Routing logic ─────────────────────────────────────

def route_after_router(state: GraphState) -> str:
    step = state.get("step", "greeting")
    return {"classify": "classify", "collect": "specialist", "confirm": "confirm_pass", "submit": "submit"}.get(step, "classify")

def route_after_classify(state: GraphState) -> str:
    return "specialist" if state.get("step") == "collect" else END

# ── Build graph ───────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(GraphState)

    graph.add_node("router", router_node)
    graph.add_node("classify", classify_node)
    graph.add_node("specialist", specialist_node)
    graph.add_node("submit", submit_node)
    graph.add_node("confirm_pass", lambda s: s)

    graph.set_entry_point("router")

    graph.add_conditional_edges("router", route_after_router, {
        "classify": "classify",
        "specialist": "specialist",
        "confirm_pass": "confirm_pass",
        "submit": "submit",
    })
    graph.add_conditional_edges("classify", route_after_classify, {
        "specialist": "specialist",
        END: END,
    })
    graph.add_edge("specialist", END)
    graph.add_edge("submit", END)
    graph.add_edge("confirm_pass", END)

    return graph.compile()


_compiled_graph = None

def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def process_message(session_state: ConversationState, user_message: str) -> tuple[str, ConversationState, list[dict]]:
    graph = get_graph()

    emp_info = None
    if session_state.employee_email:
        from app.services.employees import get_employee_by_email
        emp_info = get_employee_by_email(session_state.employee_email)

    graph_input: GraphState = {
        "session_id": session_state.session_id,
        "user_message": user_message,
        "employee_info": emp_info,
        "current_intent": session_state.current_intent.value if session_state.current_intent else None,
        "current_request": session_state.current_request or {},
        "step": session_state.step,
        "messages": session_state.messages,
        "bot_response": "",
        "knowledge_context": "",
        "missing_fields": [],
        "requests_history": session_state.requests_history,
        "debug_trace": [],
        "active_agent": None,
        "tool_results": [],
    }

    result = graph.invoke(graph_input)

    session_state.step = result.get("step", "greeting")
    session_state.current_intent = IntentType(result["current_intent"]) if result.get("current_intent") else None
    session_state.current_request = result.get("current_request", {})
    session_state.requests_history = result.get("requests_history", [])

    session_state.messages.append({"role": "user", "content": user_message})
    bot_response = result.get("bot_response", "Извини, произошла ошибка. Попробуй ещё раз.")
    session_state.messages.append({"role": "assistant", "content": bot_response})

    save_session(session_state)

    return bot_response, session_state, result.get("debug_trace", [])
