from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmarks.performance import Scenario, generate_mock_candidates, merge_segments, brute_force_route
from scripts.planner import PlannerConstraints
from scripts.planner_utils import ShortestPathResult, PathSegment


def test_generate_mock_candidates_returns_unique_ids():
    poi_ids = [f"poi:{i}" for i in range(5)]
    candidates = generate_mock_candidates(poi_ids, visit_minutes=10)
    assert [c.poi_id for c in candidates] == poi_ids
    assert all(c.estimated_visit_minutes == 10 for c in candidates)


def test_merge_segments_concatenates_paths():
    segments = [
        ShortestPathResult(
            node_ids=["A", "B"],
            total_minutes=5.0,
            segments=[
                PathSegment(
                    from_id="A",
                    to_id="B",
                    distance_min=5.0,
                    is_step_free=True,
                    stroller_friendly=True,
                    path_type="indoor",
                    notes=None,
                )
            ],
        ),
        ShortestPathResult(
            node_ids=["B", "C"],
            total_minutes=6.0,
            segments=[
                PathSegment(
                    from_id="B",
                    to_id="C",
                    distance_min=6.0,
                    is_step_free=True,
                    stroller_friendly=True,
                    path_type="indoor",
                    notes=None,
                )
            ],
        ),
    ]
    merged = merge_segments(segments)
    assert merged.node_ids == ["A", "B", "C"]
    assert merged.total_minutes == pytest.approx(11.0)
    assert len(merged.segments) == 2


def test_brute_force_timeout(monkeypatch):
    constraints = PlannerConstraints(interests=[], user_profile="standard", accessibility="any")

    def slow_solver(a, b, c):
        raise ValueError("no path")

    with pytest.raises(ValueError):
        brute_force_route(["A", "B"], constraints, slow_solver, timeout_sec=0.1)
