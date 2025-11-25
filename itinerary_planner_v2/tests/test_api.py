from __future__ import annotations

from datetime import datetime
from typing import Any, Dict
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import api
from scripts.planner import Itinerary, ItineraryStep
from scripts.planner_utils import PathSegment, ShortestPathResult


def make_itinerary() -> Itinerary:
    step = ItineraryStep(
        poi_id="poi:A",
        name="Galerie des Glaces",
        arrival_time=datetime(2024, 6, 1, 9, 0),
        departure_time=datetime(2024, 6, 1, 9, 30),
        stay_minutes=30,
    )
    segment = PathSegment(
        from_id="poi:A",
        to_id="poi:B",
        distance_min=12.0,
        is_step_free=True,
        stroller_friendly=True,
        path_type="indoor",
        notes=None,
    )
    travel = ShortestPathResult(
        node_ids=["poi:A", "poi:B"],
        total_minutes=12.0,
        segments=[segment],
    )
    return Itinerary(
        steps=[step],
        travel_minutes=12.0,
        visit_minutes=30,
        idle_minutes=5.0,
        total_minutes=47.0,
        travel_segments=travel,
    )


def test_health_endpoint():
    client = TestClient(api.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_itinerary_success(monkeypatch):
    client = TestClient(api.app)
    monkeypatch.setattr(api, "plan_versailles_itinerary", lambda **kwargs: make_itinerary())

    payload: Dict[str, Any] = {
        "start_time": "2024-06-01T09:00:00",
        "total_duration_minutes": 180,
        "constraints": {
            "interests": ["history"],
            "user_profile": "standard",
            "accessibility": "any",
            "must_include": [],
            "exclude_ids": [],
        },
    }

    response = client.post("/itinerary", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["visit_minutes"] == 30
    assert data["travel_minutes"] == pytest.approx(12.0)
    assert data["idle_minutes"] == pytest.approx(5.0)
    assert len(data["steps"]) == 1
    assert data["travel_segments"]["node_ids"] == ["poi:A", "poi:B"]


def test_itinerary_bad_request(monkeypatch):
    client = TestClient(api.app)

    def raise_value_error(*args, **kwargs):
        raise ValueError("No POIs match the specified constraints.")

    monkeypatch.setattr(api, "plan_versailles_itinerary", raise_value_error)

    payload = {
        "start_time": "2024-06-01T09:00:00",
        "total_duration_minutes": 120,
        "constraints": {
            "interests": ["gardens"],
            "user_profile": "standard",
            "accessibility": "any",
            "must_include": [],
            "exclude_ids": [],
        },
    }

    response = client.post("/itinerary", json=payload)
    assert response.status_code == 400
    assert response.json()["detail"] == "No POIs match the specified constraints."


def test_itinerary_must_include_not_found(monkeypatch):
    client = TestClient(api.app)

    def raise_value_error(*args, **kwargs):
        raise ValueError("Required POI 'versailles:Room:galerie-des-glaces' is unavailable under the current filters.")

    monkeypatch.setattr(api, "plan_versailles_itinerary", raise_value_error)

    payload = {
        "start_time": "2024-06-01T09:00:00",
        "total_duration_minutes": 180,
        "constraints": {
            "interests": ["history"],
            "user_profile": "standard",
            "accessibility": "any",
            "must_include": ["versailles:Room:galerie-des-glaces"],
            "exclude_ids": [],
        },
    }

    response = client.post("/itinerary", json=payload)
    assert response.status_code == 400
    assert response.json()["detail"].startswith("Required POI")


def test_itinerary_invalid_payload():
    client = TestClient(api.app)
    payload = {
        "start_time": "2024-06-01T09:00:00",
        "total_duration_minutes": 180,
        "constraints": {
            "interests": ["history"],
            "user_profile": "invalid_profile",
            "accessibility": "any",
            "must_include": [],
            "exclude_ids": [],
        },
    }

    response = client.post("/itinerary", json=payload)
    assert response.status_code == 422
