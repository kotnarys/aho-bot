"""
Mock MCP tools for each AHO agent.
Each tool simulates an external service call (booking, ordering, etc.)
and returns a structured result.
"""

import uuid
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def _mock_id() -> str:
    return str(uuid.uuid4())[:8].upper()


# ── Office Supplies Agent Tools ────────────────────────

def tool_place_order_komus(items: list[dict], delivery_office: str, urgent: bool = False) -> dict:
    """Place an order with Komus office supplies."""
    order_id = f"KOM-{_mock_id()}"
    logger.info(f"[MCP:Komus] Order placed: {order_id}")
    return {
        "status": "success",
        "order_id": order_id,
        "supplier": "Комус",
        "items_count": len(items),
        "estimated_delivery": "1-2 рабочих дня",
        "delivery_address": delivery_office,
        "message": f"Заказ {order_id} оформлен в Комус. Доставка в течение 1-2 рабочих дней.",
    }


def tool_place_order_vkusvill(items: list[dict], delivery_office: str) -> dict:
    """Place an order with VkusVill for office kitchen supplies."""
    order_id = f"VKS-{_mock_id()}"
    logger.info(f"[MCP:VkusVill] Order placed: {order_id}")
    return {
        "status": "success",
        "order_id": order_id,
        "supplier": "Вкусвилл",
        "items_count": len(items),
        "estimated_delivery": "ближайшая среда (заказ до вторника 12:00)",
        "delivery_address": delivery_office,
        "message": f"Заказ {order_id} оформлен во Вкусвилл. Доставка в среду.",
    }


# ── SIM Card Agent Tools ──────────────────────────────

def tool_request_sim_activation(
    employee_name: str, department: str, sim_type: str, roaming: bool = False
) -> dict:
    """Request SIM card activation through MTS corporate portal."""
    ticket_id = f"SIM-{_mock_id()}"
    logger.info(f"[MCP:MTS] SIM request: {ticket_id}")
    return {
        "status": "success",
        "ticket_id": ticket_id,
        "operator": "МТС",
        "sim_type": sim_type,
        "roaming_enabled": roaming,
        "estimated_activation": "3-5 рабочих дней",
        "message": f"Заявка {ticket_id} на {sim_type} SIM отправлена в МТС. Ожидайте активации в течение 3-5 дней.",
    }


# ── Business Trip Agent Tools ─────────────────────────

def tool_book_hotel(
    city: str, check_in: str, check_out: str, preferences: str = ""
) -> dict:
    """Book a hotel through Ostrovok.ru corporate account."""
    booking_id = f"HTL-{_mock_id()}"
    logger.info(f"[MCP:Ostrovok] Hotel booked: {booking_id}")
    return {
        "status": "success",
        "booking_id": booking_id,
        "platform": "Ostrovok.ru",
        "city": city,
        "check_in": check_in,
        "check_out": check_out,
        "hotel_name": "Holiday Inn Express" if not preferences else f"Отель по запросу: {preferences}",
        "message": f"Отель забронирован ({booking_id}). {city}, {check_in} — {check_out}.",
    }


def tool_book_transfer(
    pickup: str, destination: str, datetime_str: str, passengers: int = 1
) -> dict:
    """Book an airport/station transfer."""
    transfer_id = f"TRF-{_mock_id()}"
    logger.info(f"[MCP:Transfer] Transfer booked: {transfer_id}")
    return {
        "status": "success",
        "transfer_id": transfer_id,
        "pickup": pickup,
        "destination": destination,
        "datetime": datetime_str,
        "passengers": passengers,
        "message": f"Трансфер {transfer_id} забронирован. {pickup} → {destination}, {datetime_str}.",
    }


# ── Parking Pass Agent Tools ──────────────────────────

def tool_issue_parking_pass(
    car_plate: str, car_brand: str, start_date: str, end_date: str, office: str = ""
) -> dict:
    """Issue a parking pass for the business center."""
    pass_id = f"PRK-{_mock_id()}"
    logger.info(f"[MCP:Parking] Pass issued: {pass_id}")
    return {
        "status": "success",
        "pass_id": pass_id,
        "car_plate": car_plate,
        "car_brand": car_brand,
        "valid_from": start_date,
        "valid_until": end_date,
        "parking_location": "Подземная парковка, уровень P2",
        "message": f"Пропуск {pass_id} оформлен на {car_plate} ({car_brand}). Действует {start_date} — {end_date}.",
    }


# ── Taxi Agent Tools ──────────────────────────────────

def tool_order_taxi(
    pickup: str, destination: str, pickup_time: str, passengers: int = 1
) -> dict:
    """Order a corporate taxi via Yandex Go."""
    order_id = f"TXI-{_mock_id()}"
    logger.info(f"[MCP:YandexGo] Taxi ordered: {order_id}")
    return {
        "status": "success",
        "order_id": order_id,
        "service": "Яндекс Go",
        "tariff": "Комфорт+",
        "pickup": pickup,
        "destination": destination,
        "pickup_time": pickup_time,
        "estimated_cost": "800-1200 ₽",
        "message": f"Такси {order_id} заказано. {pickup} → {destination}, тариф Комфорт+.",
    }


# ── Incident Agent Tools ─────────────────────────────

def tool_create_incident_ticket(
    category: str, description: str, location: str, priority: str = "medium"
) -> dict:
    """Create an incident ticket in the facility management system."""
    ticket_id = f"INC-{_mock_id()}"
    priority_response = {
        "critical": "15 минут",
        "high": "1 час",
        "medium": "4 часа",
        "low": "до конца рабочего дня",
    }
    logger.info(f"[MCP:FacilityMgmt] Incident created: {ticket_id}")
    return {
        "status": "success",
        "ticket_id": ticket_id,
        "category": category,
        "priority": priority,
        "location": location,
        "response_time": priority_response.get(priority, "4 часа"),
        "assigned_to": "Дежурный техник АХО",
        "message": f"Инцидент {ticket_id} зарегистрирован. Приоритет: {priority}. Время реакции: {priority_response.get(priority, '4 часа')}.",
    }


# ── Tool registry ─────────────────────────────────────

TOOL_REGISTRY = {
    "place_order_komus": tool_place_order_komus,
    "place_order_vkusvill": tool_place_order_vkusvill,
    "request_sim_activation": tool_request_sim_activation,
    "book_hotel": tool_book_hotel,
    "book_transfer": tool_book_transfer,
    "issue_parking_pass": tool_issue_parking_pass,
    "order_taxi": tool_order_taxi,
    "create_incident_ticket": tool_create_incident_ticket,
}


def execute_tool(tool_name: str, args: dict) -> dict:
    """Execute a tool by name with given arguments."""
    fn = TOOL_REGISTRY.get(tool_name)
    if fn is None:
        return {"status": "error", "message": f"Unknown tool: {tool_name}"}
    try:
        return fn(**args)
    except Exception as e:
        logger.error(f"Tool execution error [{tool_name}]: {e}")
        return {"status": "error", "message": str(e)}
