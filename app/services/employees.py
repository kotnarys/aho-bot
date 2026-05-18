"""Employee directory service."""

import json
import os
from typing import Optional

EMPLOYEES_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "employees.json")


def _load_employees() -> list[dict]:
    with open(os.path.abspath(EMPLOYEES_PATH), "r", encoding="utf-8") as f:
        return json.load(f)


def find_employee(query: str) -> Optional[dict]:
    """Fuzzy search by name or email."""
    query_lower = query.lower().strip()
    employees = _load_employees()

    # Exact email match
    for emp in employees:
        if emp["email"].lower() == query_lower:
            return emp

    # Partial name match
    matches = []
    for emp in employees:
        name_lower = emp["name"].lower()
        if query_lower in name_lower:
            matches.append(emp)

    if len(matches) == 1:
        return matches[0]
    if matches:
        return matches[0]  # best effort
    return None


def list_employees() -> list[dict]:
    return _load_employees()


def get_employee_by_email(email: str) -> Optional[dict]:
    employees = _load_employees()
    for emp in employees:
        if emp["email"].lower() == email.lower():
            return emp
    return None
