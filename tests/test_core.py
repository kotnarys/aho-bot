"""Unit tests for AHO bot v2 — schemas, storage, tools, guardrails."""

import os
import sys
import pytest
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.models.schemas import (
    IntentType, RequestStatus, ConversationState,
    OfficeSuppliesRequest, SimCardRequest, BusinessTripRequest,
    ParkingPassRequest, TaxiRequest, IncidentReport, OrderItem,
    REQUEST_MODEL_MAP,
)
from app.services.storage import init_db, save_session, load_session, delete_session, save_chat_message, load_chat_history, clear_chat_history
from app.tools.mcp_tools import execute_tool, TOOL_REGISTRY
from app.agents.graph import validate_router_output, validate_collect_output, _parse_json


@pytest.fixture(scope="session", autouse=True)
def setup_db():
    init_db()


# ── Schema tests ──────────────────────────────────────

class TestSchemas:
    def test_intent_types(self):
        assert IntentType.OFFICE_SUPPLIES.value == "office_supplies"
        assert IntentType.TAXI.value == "taxi"
        assert len(IntentType) == 7  # 6 + unknown

    def test_office_supplies_request(self):
        req = OfficeSuppliesRequest(
            items=[OrderItem(name="Ручки", quantity=10)],
            delivery_office="Центральный офис (Москва-Сити)",
        )
        assert req.type == IntentType.OFFICE_SUPPLIES
        assert len(req.items) == 1
        assert req.status == RequestStatus.COLLECTING

    def test_conversation_state(self):
        state = ConversationState(session_id="test-1", employee_email="test@co.ru")
        assert state.step == "greeting"
        assert state.current_intent is None

    def test_request_model_map(self):
        assert len(REQUEST_MODEL_MAP) == 6
        assert REQUEST_MODEL_MAP[IntentType.TAXI] == TaxiRequest


# ── Storage tests ─────────────────────────────────────

class TestStorage:
    def test_session_save_load(self):
        state = ConversationState(session_id="test-s1", employee_email="storage@test.ru")
        state.step = "collect"
        state.current_intent = IntentType.TAXI
        save_session(state)

        loaded = load_session("test-s1", "storage@test.ru")
        assert loaded is not None
        assert loaded.step == "collect"
        assert loaded.current_intent == IntentType.TAXI

        delete_session("storage@test.ru")
        assert load_session("", "storage@test.ru") is None

    def test_chat_history(self):
        email = "chattest@test.ru"
        clear_chat_history(email)
        save_chat_message(email, "user", "Привет")
        save_chat_message(email, "assistant", "Здравствуй!")
        history = load_chat_history(email)
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["content"] == "Здравствуй!"
        clear_chat_history(email)
        assert len(load_chat_history(email)) == 0


# ── Tool tests ────────────────────────────────────────

class TestTools:
    def test_all_tools_registered(self):
        expected = ["place_order_komus", "place_order_vkusvill", "request_sim_activation",
                     "book_hotel", "book_transfer", "issue_parking_pass", "order_taxi", "create_incident_ticket"]
        for t in expected:
            assert t in TOOL_REGISTRY

    def test_execute_all_tools(self):
        cases = [
            ("place_order_komus", {"items": [{"name": "Бумага", "quantity": 5}], "delivery_office": "Офис"}),
            ("place_order_vkusvill", {"items": [{"name": "Кофе", "quantity": 2}], "delivery_office": "Офис"}),
            ("request_sim_activation", {"employee_name": "Иванов", "department": "ИТ", "sim_type": "eSIM"}),
            ("book_hotel", {"city": "Санкт-Петербург", "check_in": "2025-06-10", "check_out": "2025-06-15"}),
            ("book_transfer", {"pickup": "Вокзал", "destination": "Отель", "datetime_str": "2025-06-10 14:00"}),
            ("issue_parking_pass", {"car_plate": "А123БВ777", "car_brand": "Toyota Camry", "start_date": "2025-06-01", "end_date": "2025-12-31"}),
            ("order_taxi", {"pickup": "Москва-Сити", "destination": "Шереметьево", "pickup_time": "18:00"}),
            ("create_incident_ticket", {"category": "equipment", "description": "Принтер не работает", "location": "3 этаж"}),
        ]
        for tool_name, args in cases:
            result = execute_tool(tool_name, args)
            assert result["status"] == "success", f"Tool {tool_name} failed: {result}"

    def test_unknown_tool(self):
        result = execute_tool("fake_tool", {})
        assert result["status"] == "error"


# ── Guardrail tests ───────────────────────────────────

class TestGuardrails:
    def test_validate_router_output_valid(self):
        data = {"agent": "taxi", "confidence": 0.95, "extracted_entities": {"destination": "SVO"}, "reasoning": "ok"}
        result = validate_router_output(data)
        assert result["agent"] == "taxi"

    def test_validate_router_output_invalid_agent(self):
        data = {"agent": "pizza_delivery", "confidence": 0.9}
        result = validate_router_output(data)
        assert result["agent"] == "unknown"

    def test_validate_router_output_bad_confidence(self):
        data = {"agent": "taxi", "confidence": 5.0}
        result = validate_router_output(data)
        assert result["confidence"] == 0.5

    def test_validate_collect_output_empty(self):
        data = {}
        result = validate_collect_output(data)
        assert result["extracted_fields"] == {}
        assert result["all_collected"] == False

    def test_parse_json_clean(self):
        assert _parse_json('{"a": 1}') == {"a": 1}

    def test_parse_json_with_markdown(self):
        assert _parse_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_parse_json_embedded(self):
        assert _parse_json('Here is result: {"a": 1} done') == {"a": 1}

    def test_parse_json_invalid(self):
        assert _parse_json('not json at all') is None
