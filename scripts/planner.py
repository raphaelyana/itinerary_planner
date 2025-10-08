from __future__ import annotations

import itertools
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, time
from typing import Iterable, List, Optional, Sequence, Tuple

from neo4j import GraphDatabase

from scripts.planner_utils import AccessibilityRequirement, ShortestPathResult, UserProfile, get_shortest_path

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=os.environ.get("PLANNER_LOG_LEVEL", "INFO"))


@dataclass
class PlannerConstraints:
    interests: Sequence[str]
    user_profile: UserProfile
    accessibility: AccessibilityRequirement
    must_include: Sequence[str] = ()
    exclude_ids: Sequence[str] = ()


@dataclass
class CandidatePOI:
    poi_id: str
    name: str
    priority_score: float
    estimated_visit_minutes: int
    interest_tags: List[str]
    opening_time: Optional[str]
    closing_time: Optional[str]


@dataclass
class ItineraryStep:
    poi_id: str
    name: str
    arrival_time: datetime
    departure_time: datetime
    stay_minutes: int


@dataclass
class Itinerary:
    steps: List[ItineraryStep]
    travel_minutes: float
    visit_minutes: int
    idle_minutes: float
    total_minutes: float
    travel_segments: ShortestPathResult


def _connect(uri: Optional[str], user: Optional[str], password: Optional[str]):
    uri = uri or os.getenv("NEO4J_URI", "neo4j://localhost:7687")
    user = user or os.getenv("NEO4J_USERNAME", "neo4j")
    password = password or os.getenv("NEO4J_PASSWORD", "neo4j")
    return GraphDatabase.driver(uri, auth=(user, password))


FILTER_QUERY = """
MATCH (poi:POI)
WHERE poi.opening_time IS NOT NULL
  AND poi.closing_time IS NOT NULL
  AND poi.estimated_visit_minutes IS NOT NULL
  AND poi.id IS NOT NULL
  AND poi.priority_score IS NOT NULL
  AND ($interests IS NULL OR size($interests) = 0 OR any(tag IN poi.interest_tags WHERE tag IN $interests))
  AND ($exclude_ids IS NULL OR NOT poi.id IN $exclude_ids)
  AND (
        $accessibility = 'any'
        OR (
            poi.accessibility_level = 'full'
            AND EXISTS {
                MATCH (poi)-[r:CONNECTS_TO]->(:POI)
                WHERE ($accessibility = 'step_free' AND r.is_step_free = true)
                   OR ($accessibility = 'stroller' AND r.is_step_free = true AND r.stroller_friendly = true)
            }
        )
      )
RETURN poi.id AS id,
       poi.name AS name,
       poi.priority_score AS priority_score,
       poi.estimated_visit_minutes AS estimated_visit_minutes,
       poi.interest_tags AS interest_tags,
       poi.opening_time AS opening_time,
       poi.closing_time AS closing_time
"""


def fetch_candidate_pois(
    session,
    *,
    interests: Sequence[str],
    accessibility: AccessibilityRequirement,
    exclude_ids: Sequence[str],
) -> List[CandidatePOI]:
    result = session.run(
        FILTER_QUERY,
        interests=list(interests),
        exclude_ids=list(exclude_ids),
        accessibility=accessibility,
    )
    return [
        CandidatePOI(
            poi_id=record["id"],
            name=record["name"],
            priority_score=float(record["priority_score"]),
            estimated_visit_minutes=int(record["estimated_visit_minutes"]),
            interest_tags=record["interest_tags"] or [],
            opening_time=record["opening_time"],
            closing_time=record["closing_time"],
        )
        for record in result
    ]


def select_pois(
    candidates: Iterable[CandidatePOI],
    *,
    total_duration_minutes: int,
    min_visit_minutes: int = 10,
    must_include: Sequence[str] = (),
) -> List[CandidatePOI]:
    remaining = total_duration_minutes
    selected: List[CandidatePOI] = []
    already_selected: set[str] = set()
    lookup = {candidate.poi_id: candidate for candidate in candidates}

    for required_id in must_include:
        candidate = lookup.get(required_id)
        if candidate is None:
            continue
        if candidate.poi_id in already_selected:
            continue
        if candidate.estimated_visit_minutes > remaining:
            raise ValueError(
                f"Required POI {candidate.name} cannot fit within the allotted duration. "
                f"Increase total time (needs {candidate.estimated_visit_minutes} minutes)."
            )
        selected.append(candidate)
        already_selected.add(candidate.poi_id)
        remaining -= candidate.estimated_visit_minutes

    for candidate in sorted(candidates, key=lambda c: (-c.priority_score, c.estimated_visit_minutes)):
        if candidate.estimated_visit_minutes < min_visit_minutes:
            continue
        if candidate.poi_id in already_selected:
            continue
        if candidate.estimated_visit_minutes <= remaining:
            selected.append(candidate)
            already_selected.add(candidate.poi_id)
            remaining -= candidate.estimated_visit_minutes
    return selected


def determine_route(
    poi_ids: Sequence[str],
    *,
    constraints: PlannerConstraints,
) -> Tuple[List[str], List[ShortestPathResult], ShortestPathResult]:
    if not poi_ids:
        raise ValueError("At least one POI is required to build a route.")

    if len(poi_ids) == 1:
        zero_path = ShortestPathResult(node_ids=[poi_ids[0]], total_minutes=0.0, segments=[])
        return [poi_ids[0]], [], zero_path

    if len(poi_ids) > 7:
        logger.info("Planner: falling back to greedy solver for %d POIs", len(poi_ids))
        return determine_route_greedy(poi_ids, constraints=constraints)

    best_order: Optional[List[str]] = None
    best_cost = float("inf")
    best_pair_paths: List[ShortestPathResult] = []

    for ordered in itertools.permutations(poi_ids):
        total_cost = 0.0
        pair_paths: List[ShortestPathResult] = []
        valid = True
        for a, b in zip(ordered, ordered[1:]):
            try:
                path = get_shortest_path(
                    a,
                    b,
                    user_profile=constraints.user_profile,
                    accessibility=constraints.accessibility,
                )
            except ValueError:
                valid = False
                break
            total_cost += path.total_minutes
            pair_paths.append(path)
        if not valid:
            continue
        if total_cost < best_cost:
            best_cost = total_cost
            best_order = list(ordered)
            best_pair_paths = pair_paths

    if best_order is None:
        raise ValueError("Could not determine a viable route for the selected POIs.")

    merged = _merge_paths(best_pair_paths)
    logger.info(
        "Planner: permutation solver finished (%d POIs, cost=%.2f)",
        len(poi_ids),
        merged.total_minutes,
    )
    return best_order, best_pair_paths, merged


def _merge_paths(paths: Sequence[ShortestPathResult]) -> ShortestPathResult:
    if not paths:
        return ShortestPathResult(node_ids=[], total_minutes=0.0, segments=[])
    node_ids: List[str] = list(paths[0].node_ids)
    segments: List = list(paths[0].segments)
    total = paths[0].total_minutes

    for path in paths[1:]:
        # Avoid duplicating shared node at join boundary.
        node_ids.extend(path.node_ids[1:])
        segments.extend(path.segments)
        total += path.total_minutes

    return ShortestPathResult(node_ids=node_ids, total_minutes=total, segments=segments)


def determine_route_greedy(
    poi_ids: Sequence[str],
    *,
    constraints: PlannerConstraints,
) -> Tuple[List[str], List[ShortestPathResult], ShortestPathResult]:
    remaining = list(poi_ids)
    if not remaining:
        raise ValueError("At least one POI is required to build a route.")

    ordered: List[str] = [remaining.pop(0)]
    pair_paths: List[ShortestPathResult] = []

    while remaining:
        current = ordered[-1]
        best_candidate = None
        best_path: Optional[ShortestPathResult] = None
        best_cost = float("inf")

        for candidate in remaining:
            try:
                path = get_shortest_path(
                    current,
                    candidate,
                    user_profile=constraints.user_profile,
                    accessibility=constraints.accessibility,
                )
            except ValueError:
                continue

            if path.total_minutes < best_cost:
                best_cost = path.total_minutes
                best_candidate = candidate
                best_path = path

        if best_candidate is None or best_path is None:
            raise ValueError("Unable to build a step-free route for all POIs using greedy heuristic.")

        ordered.append(best_candidate)
        pair_paths.append(best_path)
        remaining.remove(best_candidate)

    merged = _merge_paths(pair_paths)
    logger.info(
        "Planner: greedy solver finished (%d POIs, cost=%.2f)",
        len(ordered),
        merged.total_minutes,
    )
    return ordered, pair_paths, merged


def _parse_time(value: str) -> time:
    return datetime.strptime(value, "%H:%M").time()


def build_itinerary(
    start_time: datetime,
    selected: Sequence[CandidatePOI],
    route: Sequence[str],
    pair_paths: Sequence[ShortestPathResult],
) -> Tuple[List[ItineraryStep], float]:
    steps: List[ItineraryStep] = []
    time_cursor = start_time
    idle_minutes = 0.0
    poi_lookup = {poi.poi_id: poi for poi in selected}

    travel_iter = iter(pair_paths)

    for index, poi_id in enumerate(route):
        poi = poi_lookup.get(poi_id)
        if not poi:
            continue

        if not poi.opening_time or not poi.closing_time:
            raise ValueError(f"POI {poi_id} has missing opening hours.")

        opening_dt = datetime.combine(time_cursor.date(), _parse_time(poi.opening_time))
        closing_dt = datetime.combine(time_cursor.date(), _parse_time(poi.closing_time))

        if time_cursor < opening_dt:
            wait = (opening_dt - time_cursor).total_seconds() / 60
            idle_minutes += wait
            time_cursor = opening_dt

        if time_cursor > closing_dt:
            raise ValueError(f"Arrival at {poi.name} occurs after closing time.")

        stay = poi.estimated_visit_minutes
        departure = time_cursor + timedelta(minutes=stay)

        if departure > closing_dt:
            raise ValueError(f"Visit duration pushes {poi.name} past closing time.")

        steps.append(
            ItineraryStep(
                poi_id=poi.poi_id,
                name=poi.name,
                arrival_time=time_cursor,
                departure_time=departure,
                stay_minutes=stay,
            )
        )

        time_cursor = departure

        if index < len(route) - 1:
            travel_path = next(travel_iter, None)
            if travel_path is None:
                raise ValueError("Travel path segments are misaligned with itinerary order.")
            time_cursor += timedelta(minutes=travel_path.total_minutes)

    return steps, idle_minutes


def plan_versailles_itinerary(
    start_time: datetime,
    total_duration_minutes: int,
    constraints: PlannerConstraints,
    *,
    uri: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
) -> Itinerary:
    driver = _connect(uri, user, password)
    try:
        with driver.session() as session:
            candidates = fetch_candidate_pois(
                session,
                interests=constraints.interests,
                accessibility=constraints.accessibility,
                exclude_ids=constraints.exclude_ids,
            )
    finally:
        driver.close()

    if not candidates:
        logger.info(
            "Planner: no candidates match filters interests=%s accessibility=%s exclude=%s",
            constraints.interests,
            constraints.accessibility,
            constraints.exclude_ids,
        )
        raise ValueError("No POIs match the specified constraints.")

    candidate_lookup = {candidate.poi_id: candidate for candidate in candidates}
    for required in constraints.must_include:
        if required not in candidate_lookup:
            raise ValueError(f"Required POI {required!r} is unavailable under the current filters.")

    selected = select_pois(
        candidates,
        total_duration_minutes=total_duration_minutes,
        must_include=constraints.must_include,
    )
    if not selected:
        raise ValueError("Unable to fit any POIs within the allotted duration.")

    poi_ids = [poi.poi_id for poi in selected]
    logger.info("Planner: selected %d POIs for itinerary", len(poi_ids))
    if constraints.must_include:
        for required in constraints.must_include:
            if required not in poi_ids:
                raise ValueError(f"Required POI {required!r} was not selected.")

    route_order, pair_paths, travel_segments = determine_route(poi_ids, constraints=constraints)
    itinerary_steps, idle_minutes = build_itinerary(start_time, selected, route_order, pair_paths)

    travel_minutes = travel_segments.total_minutes
    visit_minutes = sum(step.stay_minutes for step in itinerary_steps)
    total_minutes = travel_minutes + visit_minutes + idle_minutes

    logger.info(
        "Planner: itinerary built (travel=%.2f, visit=%d, idle=%.2f)",
        travel_minutes,
        visit_minutes,
        idle_minutes,
    )

    return Itinerary(
        steps=itinerary_steps,
        travel_minutes=travel_minutes,
        visit_minutes=visit_minutes,
        idle_minutes=idle_minutes,
        total_minutes=total_minutes,
        travel_segments=travel_segments,
    )
