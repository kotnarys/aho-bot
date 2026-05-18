"""Pydantic models for all AHO service request types."""

from __future__ import annotations
from datetime import datetime, date
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
import uuid


# ── Enums ──────────────────────────────────────────────

class RequestStatus(str, Enum):
    DRAFT = "draft"
    COLLECTING = "collecting"
    PENDING_CONFIRMATION = "pending_confirmation"
    SUBMITTED = "submitted"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class IntentType(str, Enum):
    OFFICE_SUPPLIES = "office_supplies"
    SIM_CARD = "sim_card"
    BUSINESS_TRIP = "business_trip"
    PARKING_PASS = "parking_pass"
    TAXI = "taxi"
    INCIDENT = "incident"
    UNKNOWN = "unknown"


class IncidentCategory(str, Enum):
    CLEANING = "cleaning"
    EQUIPMENT = "equipment"
    SUPPLIES = "supplies"
    HVAC = "hvac"
    PLUMBING = "plumbing"
    ELECTRICAL = "electrical"
    OTHER = "other"


class IncidentPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SimType(str, Enum):
    PHYSICAL = "physical"
    ESIM = "esim"


class Office(str, Enum):
    HQ = "Центральный офис (Москва-Сити)"
    WAREHOUSE = "Склад (Химки)"
    SERVICE_CENTER = "Сервис-центр (Южнопортовая)"


# ── Item models ────────────────────────────────────────

class OrderItem(BaseModel):
    name: str = Field(description="Название товара")
    quantity: int = Field(default=1, ge=1, description="Количество")
    category: str = Field(default="", description="Категория: канцтовары / продукты / другое")
    url: Optional[str] = Field(default=None, description="Ссылка на товар если указана")


# ── Request schemas ────────────────────────────────────

class BaseRequest(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    type: IntentType
    employee_name: Optional[str] = None
    employee_email: Optional[str] = None
    department: Optional[str] = None
    status: RequestStatus = RequestStatus.COLLECTING
    created_at: datetime = Field(default_factory=datetime.now)
    notes: Optional[str] = None


class OfficeSuppliesRequest(BaseRequest):
    type: IntentType = IntentType.OFFICE_SUPPLIES
    items: list[OrderItem] = Field(default_factory=list)
    supplier: Optional[str] = Field(default=None, description="Комус / Вкусвилл / другой")
    delivery_office: Optional[Office] = None
    urgent: bool = False


class SimCardRequest(BaseRequest):
    type: IntentType = IntentType.SIM_CARD
    target_employee_name: Optional[str] = Field(default=None, description="ФИО сотрудника для SIM")
    target_department: Optional[str] = None
    manager_name: Optional[str] = None
    sim_type: Optional[SimType] = None
    international_roaming: Optional[bool] = None


class BusinessTripRequest(BaseRequest):
    type: IntentType = IntentType.BUSINESS_TRIP
    destination_city: Optional[str] = None
    check_in_date: Optional[date] = None
    check_out_date: Optional[date] = None
    nights: Optional[int] = None
    hotel_preferences: Optional[str] = None
    need_transfer: Optional[bool] = None
    transfer_from: Optional[str] = None
    transfer_to: Optional[str] = None
    transfer_datetime: Optional[datetime] = None


class ParkingPassRequest(BaseRequest):
    type: IntentType = IntentType.PARKING_PASS
    car_plate: Optional[str] = None
    car_brand: Optional[str] = None
    car_model: Optional[str] = None
    pass_start_date: Optional[date] = None
    pass_end_date: Optional[date] = None
    office: Optional[Office] = None


class TaxiRequest(BaseRequest):
    type: IntentType = IntentType.TAXI
    pickup_location: Optional[str] = None
    destination: Optional[str] = None
    pickup_time: Optional[datetime] = None
    passengers: int = Field(default=1, ge=1)
    urgent: bool = False


class IncidentReport(BaseRequest):
    type: IntentType = IntentType.INCIDENT
    category: Optional[IncidentCategory] = None
    priority: Optional[IncidentPriority] = None
    location: Optional[Office] = None
    description: Optional[str] = None
    floor: Optional[str] = None
    room: Optional[str] = None


# ── Intent classification output ───────────────────────

class IntentClassification(BaseModel):
    intent: IntentType
    confidence: float = Field(ge=0, le=1)
    extracted_entities: dict = Field(default_factory=dict)
    reasoning: str = ""


# ── Conversation state ─────────────────────────────────

class ConversationState(BaseModel):
    session_id: str
    employee_name: Optional[str] = None
    employee_email: Optional[str] = None
    department: Optional[str] = None
    current_intent: Optional[IntentType] = None
    current_request: Optional[dict] = None
    messages: list[dict] = Field(default_factory=list)
    pending_field: Optional[str] = None
    step: str = "greeting"  # greeting | classify | collect | confirm | submit
    requests_history: list[dict] = Field(default_factory=list)


# ── Map intent → request model ─────────────────────────

REQUEST_MODEL_MAP = {
    IntentType.OFFICE_SUPPLIES: OfficeSuppliesRequest,
    IntentType.SIM_CARD: SimCardRequest,
    IntentType.BUSINESS_TRIP: BusinessTripRequest,
    IntentType.PARKING_PASS: ParkingPassRequest,
    IntentType.TAXI: TaxiRequest,
    IntentType.INCIDENT: IncidentReport,
}
