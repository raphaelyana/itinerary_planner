from __future__ import annotations

from datetime import datetime, timedelta
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from scripts import planner
from scripts.planner import (
    CandidatePOI,
    ItineraryStep,
    PlannerConstraints,
    select_pois,
    determine_route,
    determine_route_greedy,
    build_itinerary,
)
from scripts.planner_utils import PathSegment, ShortestPathResult


class FakeSession:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeDriver:
    def __init__(self):
        self.closed = False

    def session(self):
        return FakeSession()

    def close(self):
        self.closed = True


def make_candidate(
    poi_id: str,
    name: str,
    visit_minutes: int,
    priority: float = 1.0,
    opening: str = "09:00",
    closing: str = "19:00",
) -> CandidatePOI:
    return CandidatePOI(
        poi_id=poi_id,
        name=name,
        priority_score=priority,
        estimated_visit_minutes=visit_minutes,
        interest_tags=["history"],
        opening_time=opening,
        closing_time=closing,
    )


def fake_route(order):
    segment = PathSegment(
        from_id=order[0],
        to_id=order[-1],
        distance_min=10.0,
        is_step_free=True,
        stroller_friendly=True,
        path_type="indoor",
        notes=None,
    )
    result = ShortestPathResult(node_ids=list(order), total_minutes=10.0, segments=[segment])
    return order, [result], result


def setup_planner_mocks(monkeypatch, candidates, route_order):
    fake_driver = FakeDriver()
    monkeypatch.setattr(planner, "_connect", lambda *args, **kwargs: fake_driver)
    monkeypatch.setattr(planner, "fetch_candidate_pois", lambda *args, **kwargs: candidates)
    monkeypatch.setattr(planner, "determine_route", lambda ids, constraints: fake_route(route_order))
    def build_itinerary(start_time, selected, route, pair_paths):
        time_cursor = start_time
        steps = []
        for poi in selected:
            departure = time_cursor + timedelta(minutes=poi.estimated_visit_minutes)
            steps.append(
                ItineraryStep(
                    poi_id=poi.poi_id,
                    name=poi.name,
                    arrival_time=time_cursor,
                    departure_time=departure,
                    stay_minutes=poi.estimated_visit_minutes,
                )
            )
            time_cursor = departure
        return steps, 0.0

    monkeypatch.setattr(planner, "build_itinerary", build_itinerary)


def test_planner_enforces_must_include(monkeypatch):
    start = datetime(2024, 6, 1, 9, 0)
    candidates = [
        make_candidate("poi:A", "A", 60, priority=2.0),
        make_candidate("poi:B", "B", 30, priority=1.0),
    ]
    setup_planner_mocks(monkeypatch, candidates, ["poi:A", "poi:B"])

    constraints = PlannerConstraints(
        interests=["history"],
        user_profile="standard",
        accessibility="any",
        must_include=["poi:A"],
    )

    itinerary = planner.plan_versailles_itinerary(
        start_time=start,
        total_duration_minutes=180,
        constraints=constraints,
    )

    assert any(step.poi_id == "poi:A" for step in itinerary.steps)
    assert itinerary.travel_minutes == pytest.approx(10.0)


def test_planner_fails_when_must_include_exceeds_duration(monkeypatch):
    start = datetime(2024, 6, 1, 9, 0)
    candidates = [
        make_candidate("poi:A", "A", 180, priority=2.0),
    ]
    setup_planner_mocks(monkeypatch, candidates, ["poi:A"])

    constraints = PlannerConstraints(
        interests=["history"],
        user_profile="standard",
        accessibility="any",
        must_include=["poi:A"],
    )

    with pytest.raises(ValueError):
        planner.plan_versailles_itinerary(
            start_time=start,
            total_duration_minutes=60,
            constraints=constraints,
        )


def test_select_pois_prioritises_must_include():
    candidates = [
        make_candidate("poi:A", "A", 60, priority=2.0),
        make_candidate("poi:B", "B", 30, priority=5.0),
        make_candidate("poi:C", "C", 30, priority=1.0),
    ]
    chosen = select_pois(candidates, total_duration_minutes=90, must_include=["poi:A"])
    ids = [c.poi_id for c in chosen]
    assert ids[0] == "poi:A"
    assert "poi:B" in ids


def test_determine_route_combines_segments(monkeypatch):
    constraints = PlannerConstraints(interests=[], user_profile="standard", accessibility="any")

    def mocked_shortest_path(a, b, **kwargs):
        segment = PathSegment(
            from_id=a,
            to_id=b,
            distance_min=5.0,
            is_step_free=True,
            stroller_friendly=True,
            path_type="indoor",
            notes=None,
        )
        return ShortestPathResult(node_ids=[a, b], total_minutes=5.0, segments=[segment])

    monkeypatch.setattr("scripts.planner.get_shortest_path", mocked_shortest_path)
    order, pair_paths, merged = determine_route(["A", "B", "C"], constraints=constraints)
    assert order == ["A", "B", "C"]
    assert len(pair_paths) == 2
    assert merged.total_minutes == pytest.approx(10.0)
    assert merged.node_ids == ["A", "B", "C"]


def test_build_itinerary_respects_opening_hours():
    start = datetime(2024, 6, 1, 9, 0)
    candidates = [
        make_candidate("poi:A", "A", 30, priority=1.0, opening="09:00", closing="10:00"),
        make_candidate("poi:B", "B", 30, priority=1.0, opening="10:00", closing="11:00"),
    ]
    selected = candidates
    route = ["poi:A", "poi:B"]
    pair_paths = [
        ShortestPathResult(
            node_ids=["poi:A", "poi:B"],
            total_minutes=10,
            segments=[
                PathSegment(
                    from_id="poi:A",
                    to_id="poi:B",
                    distance_min=10,
                    is_step_free=True,
                    stroller_friendly=True,
                    path_type="indoor",
                    notes=None,
                )
            ],
        )
    ]
    steps, idle = build_itinerary(start, selected, route, pair_paths)
    assert steps[0].arrival_time == start
    assert steps[1].arrival_time >= steps[0].departure_time
    assert idle >= 0


def test_determine_route_greedy_handles_large_sets(monkeypatch):
    constraints = PlannerConstraints(interests=[], user_profile="standard", accessibility="any")

    def mocked_shortest_path(a, b, **kwargs):
        # deterministic increasing cost to keep order predictable
        cost = abs(hash((a, b))) % 10 + 1
        segment = PathSegment(
            from_id=a,
            to_id=b,
            distance_min=float(cost),
            is_step_free=True,
            stroller_friendly=True,
            path_type="indoor",
            notes=None,
        )
        return ShortestPathResult(node_ids=[a, b], total_minutes=float(cost), segments=[segment])

    monkeypatch.setattr("scripts.planner.get_shortest_path", mocked_shortest_path)
    pois = [f"poi:{i}" for i in range(10)]
    order, pair_paths, merged = determine_route_greedy(pois, constraints=constraints)
    assert len(order) == len(pois)
    assert len(pair_paths) == len(pois) - 1
    assert merged.total_minutes == pytest.approx(sum(path.total_minutes for path in pair_paths))
