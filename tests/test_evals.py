"""
Evaluation test suite — 15 dialog scenarios for the AHO bot.
Tests intent classification accuracy and field extraction.

Run: pytest tests/test_evals.py -v
Requires: OPENAI_API_KEY set (uses real LLM calls via OpenRouter)
"""

import os
import sys
import pytest
import json

# Add project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.models.schemas import ConversationState, IntentType
from app.agents.graph import process_message
from app.services.storage import init_db


@pytest.fixture(scope="session", autouse=True)
def setup_db():
    init_db()


def _run_dialog(messages: list[str], employee_email: str = "ivanov@company.ru") -> list[tuple[str, ConversationState, list]]:
    """Run a multi-turn dialog, return list of (response, state, trace) per turn."""
    state = ConversationState(session_id="eval-test", employee_email=employee_email)
    results = []
    for msg in messages:
        response, state, trace = process_message(state, msg)
        results.append((response, state, trace))
    return results


# ── Intent Classification Tests ───────────────────────

class TestIntentClassification:
    """Test that the router agent correctly classifies user intents."""

    @pytest.mark.parametrize("message,expected_intent", [
        ("Закажи ручки и блокнот", IntentType.OFFICE_SUPPLIES),
        ("Нужно заказать кофе в офис", IntentType.OFFICE_SUPPLIES),
        ("Нужна корпоративная сим-карта", IntentType.SIM_CARD),
        ("Оформи eSIM для нового сотрудника", IntentType.SIM_CARD),
        ("Еду в командировку в Питер", IntentType.BUSINESS_TRIP),
        ("Нужно забронировать отель в Казани на 3 ночи", IntentType.BUSINESS_TRIP),
        ("Оформи пропуск на парковку", IntentType.PARKING_PASS),
        ("Нужно место на парковке для моей машины", IntentType.PARKING_PASS),
        ("Такси до Шереметьево к 15:00", IntentType.TAXI),
        ("Закажи такси от офиса до Домодедово", IntentType.TAXI),
        ("Кофемашина не работает", IntentType.INCIDENT),
        ("В туалете на 3 этаже нет бумаги", IntentType.INCIDENT),
        ("Сломался кондиционер в переговорке", IntentType.INCIDENT),
    ])
    def test_intent_classification(self, message, expected_intent):
        state = ConversationState(session_id="eval-intent", employee_email="ivanov@company.ru")
        _, updated_state, trace = process_message(state, message)
        assert updated_state.current_intent == expected_intent, (
            f"Expected {expected_intent.value}, got {updated_state.current_intent}"
        )


# ── Full Dialog Flow Tests ────────────────────────────

class TestDialogFlows:
    """Test complete dialog flows from start to submission."""

    def test_office_supplies_full_flow(self):
        """Full flow: order office supplies → confirm → submit."""
        results = _run_dialog([
            "Закажи 10 ручек и 5 блокнотов",
            "В центральный офис",
            "Да, всё верно",
        ])
        # After last message: should be done with a submitted request
        _, state_final, _ = results[-1]
        assert state_final.step == "done"
        assert len(state_final.requests_history) > 0
        assert state_final.requests_history[-1].get("type") == "office_supplies"

    def test_taxi_with_details(self):
        """Taxi order with all details in one message."""
        results = _run_dialog([
            "Такси от офиса Москва-Сити до Шереметьево к 18:00, 2 пассажира",
        ])
        _, state, _ = results[0]
        assert state.current_intent == IntentType.TAXI

    def test_incident_report(self):
        """Report an incident."""
        results = _run_dialog([
            "Не работает кондиционер в переговорке на 5 этаже",
        ])
        _, state, _ = results[0]
        assert state.current_intent == IntentType.INCIDENT

    def test_business_trip_rag_context(self):
        """Business trip should trigger RAG for hotel limits."""
        results = _run_dialog([
            "Еду в командировку в Питер с 10 по 15 июня",
        ])
        _, state, trace = results[0]
        assert state.current_intent == IntentType.BUSINESS_TRIP
        # Check that RAG was invoked
        rag_traces = [t for t in trace if t["type"] == "rag_search"]
        assert len(rag_traces) > 0, "RAG search should be triggered for business trip"

    def test_sim_card_for_self(self):
        """SIM card request for the employee themselves."""
        results = _run_dialog([
            "Нужна корпоративная SIM для меня, eSIM, без роуминга",
        ])
        _, state, _ = results[0]
        assert state.current_intent == IntentType.SIM_CARD


# ── Guardrail Tests ───────────────────────────────────

class TestGuardrails:
    """Test that guardrails catch invalid data."""

    def test_unknown_intent(self):
        """Out-of-scope request should be handled gracefully."""
        results = _run_dialog([
            "Какая погода завтра в Москве?",
        ])
        _, state, _ = results[0]
        # Should either be unknown or greeting (not crash)
        assert state.step in ("greeting", "collect", "classify")

    def test_empty_message(self):
        """Empty-ish message should not crash."""
        results = _run_dialog(["  "])
        _, state, _ = results[0]
        assert state is not None


# ── Tool Execution Tests (unit) ───────────────────────

class TestToolExecution:
    """Test mock MCP tools directly."""

    def test_komus_order(self):
        from app.tools.mcp_tools import execute_tool
        result = execute_tool("place_order_komus", {
            "items": [{"name": "Ручки", "quantity": 10}],
            "delivery_office": "Центральный офис",
        })
        assert result["status"] == "success"
        assert result["order_id"].startswith("KOM-")

    def test_taxi_order(self):
        from app.tools.mcp_tools import execute_tool
        result = execute_tool("order_taxi", {
            "pickup": "Москва-Сити",
            "destination": "Шереметьево",
            "pickup_time": "18:00",
        })
        assert result["status"] == "success"
        assert result["order_id"].startswith("TXI-")

    def test_incident_ticket(self):
        from app.tools.mcp_tools import execute_tool
        result = execute_tool("create_incident_ticket", {
            "category": "equipment",
            "description": "Кондиционер не работает",
            "location": "5 этаж, переговорка",
            "priority": "high",
        })
        assert result["status"] == "success"
        assert result["ticket_id"].startswith("INC-")

    def test_unknown_tool(self):
        from app.tools.mcp_tools import execute_tool
        result = execute_tool("nonexistent_tool", {})
        assert result["status"] == "error"
