from __future__ import annotations

import itertools
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from neo4j import GraphDatabase

from scripts.planner_utils import (
    AccessibilityRequirement,
    PathSegment,
    ShortestPathResult,
    UserProfile,
    get_shortest_path,
)
from scripts.path_cache import PathCache

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=os.environ.get("PLANNER_LOG_LEVEL", "INFO"))

DUMMY_START_ID = "__dummy_start__"
DEFAULT_ENTRANCE = "versailles:Garden:cour-dhonneur"
TRIANON_ENTRANCE = "versailles:Trianon:grand-trianon"
CACHE_FILE = os.getenv("PLANNER_DISTANCE_CACHE", "cache/full_path_cache.json")
PATH_CACHE = PathCache(CACHE_FILE, auto_save=False)
_CACHE_DIRTY = False


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


def _compute_pairwise_paths(
    poi_ids: Sequence[str],
    *,
    constraints: PlannerConstraints,
) -> Dict[Tuple[int, int], ShortestPathResult]:
    global _CACHE_DIRTY
    pair_paths: Dict[Tuple[int, int], ShortestPathResult] = {}
    cache_hits = 0
    live_lookups = 0
    profile = constraints.user_profile
    accessibility = constraints.accessibility
    for i, origin in enumerate(poi_ids):
        for j, destination in enumerate(poi_ids):
            if i == j:
                continue
            key = (i, j)
            if key in pair_paths:
                continue

            cached = PATH_CACHE.get(profile, accessibility, origin, destination)
            if cached:
                pair_paths[key] = cached
                cache_hits += 1
                continue

            try:
                path = get_shortest_path(
                    origin,
                    destination,
                    user_profile=constraints.user_profile,
                    accessibility=constraints.accessibility,
                )
            except ValueError as exc:
                raise ValueError(f"No path between {origin!r} and {destination!r}: {exc}") from exc

            pair_paths[key] = path
            PATH_CACHE.store(profile, accessibility, origin, destination, path, persist=False)
            PATH_CACHE.store(profile, accessibility, destination, origin, path, persist=False)
            live_lookups += 1
            _CACHE_DIRTY = True

    if cache_hits and logger.isEnabledFor(logging.DEBUG):
        logger.debug("Planner: distance cache hits=%d, live lookups=%d", cache_hits, live_lookups)

    return pair_paths


def _run_held_karp(
    poi_ids: Sequence[str],
    pair_paths: Dict[Tuple[int, int], ShortestPathResult],
    *,
    start_index: Optional[int] = None,
) -> Tuple[List[int], List[ShortestPathResult], ShortestPathResult]:
    n = len(poi_ids)
    all_mask = (1 << n) - 1
    dp: Dict[Tuple[int, int], Tuple[float, Optional[int]]] = {}

    initial_indices: Iterable[int]
    if start_index is not None:
        initial_indices = (start_index,)
    else:
        initial_indices = range(n)

    for idx in initial_indices:
        dp[(1 << idx, idx)] = (0.0, None)

    for mask in range(1, all_mask + 1):
        for current in range(n):
            if not (mask & (1 << current)):
                continue
            current_state = dp.get((mask, current))
            if current_state is None:
                continue
            current_cost, _ = current_state
            remaining_mask = all_mask ^ mask
            next_candidate = remaining_mask
            while next_candidate:
                next_index = (next_candidate & -next_candidate).bit_length() - 1
                next_candidate &= next_candidate - 1
                key = (current, next_index)
                if key not in pair_paths:
                    continue
                new_mask = mask | (1 << next_index)
                new_cost = current_cost + pair_paths[key].total_minutes
                existing = dp.get((new_mask, next_index))
                if existing is None or new_cost < existing[0]:
                    dp[(new_mask, next_index)] = (new_cost, current)

    best_cost = float("inf")
    best_end: Optional[int] = None
    for idx in range(n):
        state = dp.get((all_mask, idx))
        if state is None:
            continue
        if state[0] < best_cost:
            best_cost = state[0]
            best_end = idx

    if best_end is None:
        raise ValueError("Unable to assemble a route that connects all POIs.")

    order_indices: List[int] = []
    mask = all_mask
    current = best_end
    while current is not None:
        order_indices.append(current)
        state = dp[(mask, current)]
        prev = state[1]
        mask &= ~(1 << current)
        current = prev
    order_indices.reverse()

    pairwise_segments: List[ShortestPathResult] = []
    for a, b in zip(order_indices, order_indices[1:]):
        pairwise_segments.append(pair_paths[(a, b)])

    merged = _merge_paths(pairwise_segments)
    return order_indices, pairwise_segments, merged


def _solve_route_via_held_karp(
    poi_ids: Sequence[str],
    *,
    constraints: PlannerConstraints,
) -> Tuple[List[str], List[ShortestPathResult], ShortestPathResult]:
    base_ids = list(dict.fromkeys(poi_ids))
    if not base_ids:
        raise ValueError("At least one POI is required to build a route.")
    if len(base_ids) == 1:
        zero_path = ShortestPathResult(node_ids=[base_ids[0]], total_minutes=0.0, segments=[])
        return [base_ids[0]], [], zero_path

    includes_trianon = any(poi.startswith("versailles:Trianon") for poi in base_ids)
    start_candidates: List[str] = [DEFAULT_ENTRANCE]
    if includes_trianon:
        start_candidates.append(TRIANON_ENTRANCE)

    candidate_results: List[Tuple[float, List[str], List[ShortestPathResult]]] = []

    for start_id in start_candidates:
        scenario_ids = list(base_ids)
        if start_id not in scenario_ids:
            scenario_ids.append(start_id)

        try:
            base_pair_paths = _compute_pairwise_paths(scenario_ids, constraints=constraints)
        except ValueError:
            continue

        pair_paths: Dict[Tuple[int, int], ShortestPathResult] = {}
        for (i, j), path in base_pair_paths.items():
            pair_paths[(i + 1, j + 1)] = path

        augmented_ids = [DUMMY_START_ID] + scenario_ids
        start_idx = scenario_ids.index(start_id) + 1
        pair_paths[(0, start_idx)] = ShortestPathResult(
            node_ids=[DUMMY_START_ID, start_id], total_minutes=0.0, segments=[]
        )

        try:
            order_indices, pair_segments, merged = _run_held_karp(
                augmented_ids, pair_paths, start_index=0
            )
        except ValueError:
            continue

        # remove dummy index and associated zero-cost segment
        filtered_indices = [idx for idx in order_indices if idx != 0]
        filtered_segments = [
            segment
            for segment, (a, b) in zip(
                pair_segments, zip(order_indices, order_indices[1:])
            )
            if a != 0 and b != 0
        ]

        if not filtered_indices:
            continue

        merged_filtered = _merge_paths(filtered_segments)
        route_order = [augmented_ids[idx] for idx in filtered_indices]
        candidate_results.append((merged_filtered.total_minutes, route_order, filtered_segments))

    if candidate_results:
        best_cost, best_order, best_segments = min(
            candidate_results, key=lambda item: item[0]
        )
        logger.info(
            "Planner: Held-Karp solver finished with fixed start (%d POIs, cost=%.2f)",
            len(best_order),
            best_cost,
        )
        merged = _merge_paths(best_segments)
        return best_order, best_segments, merged

    # Fallback: no scenario succeeded; solve on original set without dummy
    pair_paths = _compute_pairwise_paths(base_ids, constraints=constraints)
    order_indices, pair_segments, merged = _run_held_karp(base_ids, pair_paths)
    logger.info(
        "Planner: Held-Karp solver finished (%d POIs, cost=%.2f)",
        len(base_ids),
        merged.total_minutes,
    )
    route_order = [base_ids[idx] for idx in order_indices]
    return route_order, pair_segments, merged


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
        return _solve_route_via_held_karp(poi_ids, constraints=constraints)

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
            logger.error(
                "Planner: greedy solver stuck at %s with remaining=%s (accessibility=%s)",
                current,
                remaining,
                constraints.accessibility,
            )
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
    logger.info("Planner: greedy route order %s", ordered)
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
            if index < len(route) - 1:
                travel_path = next(travel_iter, None)
                if travel_path is None:
                    raise ValueError("Travel path segments are misaligned with itinerary order.")
                time_cursor += timedelta(minutes=travel_path.total_minutes)
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

    global _CACHE_DIRTY
    if _CACHE_DIRTY:
        PATH_CACHE.save()
        _CACHE_DIRTY = False

    return Itinerary(
        steps=itinerary_steps,
        travel_minutes=travel_minutes,
        visit_minutes=visit_minutes,
        idle_minutes=idle_minutes,
        total_minutes=total_minutes,
        travel_segments=travel_segments,
    )
