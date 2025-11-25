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
    normalize_neo4j_uri,
)
from scripts.path_cache import PathCache
from configs.network_config import (
    get_entrance_for_pois,
    get_exit_for_pois,
    DEFAULT_ENTRANCE,
    DEFAULT_EXIT,
    get_all_entrances,
)

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=os.environ.get("PLANNER_LOG_LEVEL", "INFO"))

DUMMY_START_ID = "__dummy_start__"
CACHE_FILE = os.getenv("PLANNER_DISTANCE_CACHE", "cache/full_path_cache.json")
PATH_CACHE = PathCache(CACHE_FILE, auto_save=False)
_CACHE_DIRTY = False
LUNCH_BREAK_ID = "__lunch_break__"
GREEDY_THRESHOLD = 10
PERM_THRESHOLD = 7
MIN_TRAVEL_PER_HOP = 5.0

# Zone interface nodes for cross-zone transitions
ZONE_INTERFACES = {
    ('Castle', 'Garden'): [
        ('versailles:Castle:cour-dhonneur', 'versailles:Garden:acces-jardins-cour-des-princes'),
        ('versailles:Castle:pavillon-dufour-sortie', 'versailles:Garden:pdv-parterre-du-midi-orangerie'),
        ('versailles:Room:vestibule-de-marbre', 'versailles:Garden:acces-jardins-cour-des-princes'),
    ],
    ('Garden', 'Park'): [
        ('versailles:Garden:entree-parc-bassin-apollon', 'versailles:Park:grand-canal'),
    ],
    ('Garden', 'Trianon'): [
        ('versailles:Garden:entree-parc-bassin-apollon', 'versailles:Trianon:entree-petit-trianon'),
    ],
    ('Park', 'Trianon'): [
        ('versailles:Park:etoile-royale', 'versailles:Trianon:entree-grand-trianon'),
        ('versailles:Park:grand-canal', 'versailles:Trianon:entree-grand-trianon'),
        ('versailles:Park:grand-canal', 'versailles:Trianon:entree-petit-trianon'),
    ],
}
PROFILE_TRAVEL_RATIO = {
    "standard": 0.4,
    "family": 0.35,
    "elder": 0.3,
}
PROFILE_TRAVEL_MIN = {
    "standard": 60.0,
    "family": 50.0,
    "elder": 40.0,
}
PROFILE_TRAVEL_MAX = {
    "standard": 150.0,
    "family": 110.0,
    "elder": 90.0,
}


def _parse_time(value: str) -> time:
    return datetime.strptime(value, "%H:%M").time()


@dataclass
class BudgetConstraint:
    """Budget information for ticket pricing."""
    total_budget: float  # Total budget in euros
    num_adults: int = 1
    num_children_under_18: int = 0
    num_youth_18_25_eu: int = 0
    all_eu_residents: bool = True
    has_reduced_rate_cards: int = 0


@dataclass
class PlannerConstraints:
    interests: Sequence[str]
    user_profile: UserProfile
    accessibility: AccessibilityRequirement
    must_include: Sequence[str] = ()
    exclude_ids: Sequence[str] = ()
    lunch_break: Optional[bool] = None
    hard_time_limits: bool = False
    start_poi: Optional[str] = None  # Optional: specific entrance to use
    finish_poi: Optional[str] = None  # Optional: specific exit to use
    budget: Optional[BudgetConstraint] = None  # Optional: budget constraints


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
    uri = normalize_neo4j_uri(uri or os.getenv("NEO4J_URI"))
    user = user or os.getenv("NEO4J_USERNAME", "neo4j")
    password = password or os.getenv("NEO4J_PASSWORD", "neo4j")
    # Configure connection pool for AuraDB stability
    # Note: neo4j+s:// URI already includes encryption
    return GraphDatabase.driver(
        uri,
        auth=(user, password),
        max_connection_pool_size=50,
        connection_timeout=30.0,
        max_connection_lifetime=3600,
        connection_acquisition_timeout=60.0,
    )


FILTER_QUERY = """
MATCH (poi:POI)
WHERE poi.estimated_visit_minutes IS NOT NULL
  AND poi.id IS NOT NULL
  AND poi.priority_score IS NOT NULL
  AND ($interests IS NULL OR size($interests) = 0 OR any(tag IN poi.interest_tags WHERE tag IN $interests))
  AND (
        $accessibility = 'any'
        OR poi.accessibility_level = 'full'
      )
RETURN poi.id AS id,
       poi.name AS name,
       poi.priority_score AS priority_score,
       poi.estimated_visit_minutes AS estimated_visit_minutes,
       poi.interest_tags AS interest_tags,
       null AS opening_time,
       null AS closing_time
"""


def fetch_candidate_pois(
    session,
    *,
    interests: Sequence[str],
    accessibility: AccessibilityRequirement,
    exclude_ids: Sequence[str],
    accessible_zones: Optional[Set[str]] = None,
) -> List[CandidatePOI]:
    """
    Fetch candidate POIs with optional zone filtering based on budget.

    Args:
        session: Neo4j session
        interests: Interest tags to filter by
        accessibility: Accessibility requirements
        exclude_ids: POI IDs to exclude
        accessible_zones: If provided, only return POIs from these zones

    Returns:
        List of candidate POIs
    """
    excluded = set(exclude_ids)
    raw_result = session.run(
        FILTER_QUERY,
        interests=list(interests),
        accessibility=accessibility,
    )
    candidates = [
        CandidatePOI(
            poi_id=record["id"],
            name=record["name"],
            priority_score=float(record["priority_score"]),
            estimated_visit_minutes=int(record["estimated_visit_minutes"]),
            interest_tags=record["interest_tags"] or [],
            opening_time=record["opening_time"],
            closing_time=record["closing_time"],
        )
        for record in raw_result
    ]

    # Filter by exclusions
    candidates = [c for c in candidates if c.poi_id not in excluded]

    # Filter by accessible zones (budget-based)
    if accessible_zones:
        def poi_in_accessible_zone(poi_id: str) -> bool:
            """Check if POI is in an accessible zone."""
            for zone in accessible_zones:
                if zone in poi_id:
                    return True
            return False

        candidates = [c for c in candidates if poi_in_accessible_zone(c.poi_id)]

    return candidates


def _auto_max_poi_cap(constraints: PlannerConstraints, total_duration_minutes: int) -> Optional[int]:
    """Limit POI count so routing stays tractable."""
    if constraints.must_include:
        return None
    # Conservative cap keeps pairwise path computation reasonable.
    return 8


def _should_schedule_lunch(start_time: datetime, total_duration_minutes: int, preference: Optional[bool]) -> bool:
    if preference is not None:
        return preference
    day_end = start_time + timedelta(minutes=total_duration_minutes)
    window_start = datetime.combine(start_time.date(), time(11, 0))
    window_end = datetime.combine(start_time.date(), time(14, 0))
    overlap_start = max(start_time, window_start)
    overlap_end = min(day_end, window_end)
    overlap_minutes = max(0.0, (overlap_end - overlap_start).total_seconds() / 60.0)
    return overlap_minutes >= 120.0


def _opening_minutes(poi: CandidatePOI) -> int:
    if not poi.opening_time:
        return 0
    t = _parse_time(poi.opening_time)
    return t.hour * 60 + t.minute


def _order_pois_by_opening(
    selected: Sequence[CandidatePOI],
    *,
    start_time: datetime,
    late_threshold_minutes: int = 120,
) -> List[CandidatePOI]:
    start_minutes = start_time.hour * 60 + start_time.minute

    def sort_key(poi: CandidatePOI) -> Tuple[int, int, float]:
        opening = _opening_minutes(poi)
        is_late = 1 if opening - start_minutes > late_threshold_minutes else 0
        return (is_late, opening or start_minutes, -poi.priority_score)

    return sorted(selected, key=sort_key)


def _travel_budget_minutes(profile: UserProfile, total_duration_minutes: int) -> float:
    ratio = PROFILE_TRAVEL_RATIO.get(profile, 0.4)
    min_budget = PROFILE_TRAVEL_MIN.get(profile, 60.0)
    max_budget = PROFILE_TRAVEL_MAX.get(profile, float("inf"))
    budget = max(min_budget, total_duration_minutes * ratio)
    return min(budget, max_budget)


def _minimum_possible_minutes(selected: Sequence[CandidatePOI]) -> float:
    visit_sum = sum(poi.estimated_visit_minutes for poi in selected)
    travel_lb = max(0, (len(selected) - 1) * MIN_TRAVEL_PER_HOP)
    return visit_sum + travel_lb


def _has_only_required(selected: Sequence[CandidatePOI], constraints: PlannerConstraints) -> bool:
    required = set(constraints.must_include)
    if not required:
        return False
    return all(poi.poi_id in required for poi in selected)


def _time_bounds_ok(
    steps: Sequence[ItineraryStep],
    *,
    start_time: datetime,
    total_duration_minutes: int,
    poi_lookup: Dict[str, CandidatePOI],
    enforce_hard: bool,
) -> bool:
    if not steps:
        return True
    allowed_end = start_time + timedelta(minutes=total_duration_minutes)
    if steps[-1].departure_time > allowed_end:
        logger.debug(
            "Planner: total duration exceeded (end=%s, allowed=%s)",
            steps[-1].departure_time,
            allowed_end,
        )
        return False
    if not enforce_hard:
        return True

    for step in steps:
        if step.poi_id == LUNCH_BREAK_ID:
            continue
        poi = poi_lookup.get(step.poi_id)
        if poi and poi.poi_id.startswith("versailles:Room"):
            if poi.closing_time:
                closing_dt = datetime.combine(step.arrival_time.date(), _parse_time(poi.closing_time))
                limit_dt = closing_dt - timedelta(minutes=45)
                if step.arrival_time > limit_dt:
                    logger.debug(
                        "Planner: castle entry too late for %s (arrival=%s, limit=%s)",
                        step.name,
                        step.arrival_time,
                        limit_dt,
                    )
                    return False
            break
    return True


def _prune_latest_optional(
    selected: Sequence[CandidatePOI],
    constraints: PlannerConstraints,
) -> Optional[List[CandidatePOI]]:
    optional = [poi for poi in selected if poi.poi_id not in constraints.must_include]
    if not optional:
        return None
    optional.sort(key=lambda poi: (_opening_minutes(poi), poi.priority_score))
    remove = optional[-1]
    return [poi for poi in selected if poi.poi_id != remove.poi_id]


def _prune_low_priority(
    selected: Sequence[CandidatePOI],
    constraints: PlannerConstraints,
) -> Optional[Tuple[List[CandidatePOI], List[str]]]:
    optional = [poi for poi in selected if poi.poi_id not in constraints.must_include]
    if not optional:
        return None
    optional.sort(key=lambda poi: (poi.priority_score, poi.estimated_visit_minutes))
    remove = optional[0]
    remaining = [poi for poi in selected if poi.poi_id != remove.poi_id]
    return remaining, [remove.poi_id]


def select_pois(
    candidates: Iterable[CandidatePOI],
    *,
    total_duration_minutes: int,
    min_visit_minutes: int = 10,
    must_include: Sequence[str] = (),
    max_pois: Optional[int] = None,
    start_time: Optional[datetime] = None,
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

    # Higher priority first, break ties by better score-per-minute, then shorter visits.
    def _effective_priority(candidate: CandidatePOI) -> float:
        base = candidate.priority_score
        if start_time and candidate.opening_time:
            opening_dt = datetime.combine(
                start_time.date(), _parse_time(candidate.opening_time)
            )
            if opening_dt > start_time:
                delay_minutes = (opening_dt - start_time).total_seconds() / 60.0
                penalty = min(delay_minutes / 30.0, base)
                base -= penalty
        # Morning bias: prioritize castle highlights when user did not opt out.
        if start_time and start_time.hour < 12:
            if ("must_see" in candidate.interest_tags or "history" in candidate.interest_tags) and (
                candidate.poi_id.startswith("versailles:Room") or candidate.poi_id.startswith("versailles:Castle")
            ):
                base += 1.0
        return max(base, 0.0)

    def _sort_key(candidate: CandidatePOI) -> Tuple[float, float, int]:
        duration = max(candidate.estimated_visit_minutes, 1)
        effective_priority = _effective_priority(candidate)
        score_per_min = effective_priority / duration
        return (-effective_priority, -score_per_min, candidate.estimated_visit_minutes)

    for candidate in sorted(candidates, key=_sort_key):
        if candidate.estimated_visit_minutes < min_visit_minutes:
            continue
        if candidate.poi_id in already_selected:
            continue
        if max_pois is not None and len(selected) >= max_pois:
            break
        if candidate.estimated_visit_minutes <= remaining:
            selected.append(candidate)
            already_selected.add(candidate.poi_id)
            remaining -= candidate.estimated_visit_minutes
    return selected


# Canonical Castle tour order (based on actual connections in data)
_CASTLE_TOUR_ORDER = [
    "versailles:Room:galeries-de-lhistoire",
    "versailles:Room:salles-des-croisades",
    "versailles:Room:salles-louis-xiv",
    "versailles:Room:chapelle-royale",
    "versailles:Room:salon-dhercule",
    "versailles:Room:salon-de-labondance",
    "versailles:Room:salon-de-venus",
    "versailles:Room:salon-de-diane",
    "versailles:Room:salon-de-mars",
    "versailles:Room:salon-de-mercure",
    "versailles:Room:salon-dapollon",
    "versailles:Room:salon-de-la-guerre",
    "versailles:Room:galerie-des-glaces",
    "versailles:Room:salon-de-la-paix",
    "versailles:Room:chambre-de-la-reine",
    "versailles:Room:salon-des-nobles",
    "versailles:Room:antichambre-du-grand-couvert-reine",
    "versailles:Room:salle-des-gardes-de-la-reine",
    "versailles:Room:cabinet-du-conseil",
    "versailles:Room:chambre-du-roi",
    "versailles:Room:salle-du-sacre",
    "versailles:Room:appartements-de-madame-de-maintenon",
    "versailles:Room:galerie-des-batailles",
    "versailles:Room:salle-1792",
    "versailles:Room:galerie-basse",
]

_CASTLE_TOUR_INDEX = {poi_id: idx for idx, poi_id in enumerate(_CASTLE_TOUR_ORDER)}


def _reorder_castle_pois_if_needed(
    poi_ids: List[str],
    constraints: PlannerConstraints,
) -> List[str]:
    """
    Reorder Castle POIs to follow the directed tour path.

    The Castle has a one-way tour structure. This function detects Castle POIs
    and reorders them according to the canonical tour sequence.
    """
    # Identify which POIs are Castle rooms
    castle_pois = [pid for pid in poi_ids if pid.startswith("versailles:Room:")]

    if len(castle_pois) <= 1:
        return poi_ids  # Nothing to reorder

    logger.debug("Planner: detected %d Castle POIs, reordering by tour sequence", len(castle_pois))

    # Sort Castle POIs by their position in the canonical tour
    sorted_castle = sorted(
        castle_pois,
        key=lambda pid: _CASTLE_TOUR_INDEX.get(pid, 9999)  # Unknown POIs go to end
    )

    logger.info("Planner: reordered %d Castle POIs to follow tour path", len(sorted_castle))

    # Reconstruct full POI list: keep non-Castle POIs in place, replace Castle POIs with sorted order
    result = []
    castle_iter = iter(sorted_castle)
    for pid in poi_ids:
        if pid in castle_pois:
            result.append(next(castle_iter))
        else:
            result.append(pid)

    return result


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
                logger.warning(
                    "Planner: no path from %s to %s (profile=%s, accessibility=%s)",
                    origin,
                    destination,
                    constraints.user_profile,
                    constraints.accessibility,
                )
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

    # Determine entrance to use
    if constraints.start_poi:
        # Use explicitly specified entrance
        start_candidates: List[str] = [constraints.start_poi]
    else:
        # Auto-detect best entrance based on POI zones
        best_entrance = get_entrance_for_pois(base_ids)
        start_candidates: List[str] = [best_entrance]

        # Also try default entrance if different
        if best_entrance != DEFAULT_ENTRANCE:
            start_candidates.append(DEFAULT_ENTRANCE)

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
    poi_ids = list(poi_ids)
    if not poi_ids:
        raise ValueError("At least one POI is required to build a route.")

    if len(poi_ids) == 1:
        zero_path = ShortestPathResult(node_ids=[poi_ids[0]], total_minutes=0.0, segments=[])
        return [poi_ids[0]], [], zero_path

    # Reorder Castle POIs to follow directed tour path
    poi_ids = _reorder_castle_pois_if_needed(poi_ids, constraints)

    # If all POIs are museum interior rooms (Castle or Trianon), use the pre-ordered sequence directly
    # (Museum interiors have directed one-way tours, so TSP/Held-Karp doesn't apply)
    all_museum_rooms = all(
        pid.startswith("versailles:Room:") or
        (pid.startswith("versailles:Trianon:") and any(
            room_name in pid for room_name in [
                "boudoir-imperatrice", "salon-des-glaces", "chambre-de-limperatrice",
                "salon-de-la-chapelle", "salon-des-seigneurs", "salon-rond",
                "salon-de-famille-de-lempereur", "chambre-reine-des-belges",
                "salon-de-musique", "salon-famille-louis-philippe", "salon-malachite",
                "cabinet-topographique", "salon-frais", "galerie-des-cotelle",
                "salon-des-jardins", "premier-salon", "salle-des-gardes-petit-trianon",
                "salon-du-billard-petit-trianon", "piece-dargenterie-petit-trianon",
                "salle-glaces-mouvantes-petit-trianon", "les-deux-fruiteries",
                "escalier-dhonneur-petit-trianon", "antichambre-petit-trianon",
                "grande-salle-a-manger-petit-trianon", "petite-salle-a-manger-petit-trianon",
                "salon-de-compagnie-petit-trianon", "chambre-a-coucher-de-la-reine-petit-trianon",
                "boudoir-des-glaces-mouvantes-petit-trianon", "cabinet-de-toilette-de-la-reine-petit-trianon",
            ]
        ))
        for pid in poi_ids
    )
    if all_museum_rooms and len(poi_ids) > 1:
        logger.info("Planner: all %d POIs are museum interior rooms, attempting pre-ordered tour sequence", len(poi_ids))
        # Try to build sequential path through the ordered rooms
        try:
            pair_paths: List[ShortestPathResult] = []
            for i in range(len(poi_ids) - 1):
                path = get_shortest_path(
                    poi_ids[i],
                    poi_ids[i + 1],
                    user_profile=constraints.user_profile,
                    accessibility=constraints.accessibility,
                )
                pair_paths.append(path)
            merged = _merge_paths(pair_paths)
            logger.info("Planner: sequential museum tour successful")
            return poi_ids, pair_paths, merged
        except ValueError as exc:
            # Sequential path failed (likely due to branch disconnection)
            # Fall back to greedy routing which can handle disconnected subsets
            logger.warning(
                "Planner: sequential museum path failed (%s), falling back to greedy routing",
                str(exc)
            )
            return determine_route_greedy(poi_ids, constraints=constraints)

    n = len(poi_ids)
    # Route solver selection:
    # - > GREEDY_THRESHOLD: use greedy heuristic (fast for large sets)
    # - > PERM_THRESHOLD: use Held-Karp DP
    # - otherwise: exact permutation
    if n > GREEDY_THRESHOLD:
        logger.info("Planner: using greedy solver (%d POIs)", n)
        return determine_route_greedy(poi_ids, constraints=constraints)
    if n > PERM_THRESHOLD:
        logger.info("Planner: using Held-Karp solver (%d POIs)", n)
        return _solve_route_via_held_karp(poi_ids, constraints=constraints)

    try:
        raw_pair_paths = _compute_pairwise_paths(poi_ids, constraints=constraints)
    except ValueError as exc:
        raise ValueError("Unable to pre-compute pairwise paths for permutation solver.") from exc

    pair_lookup: Dict[Tuple[str, str], ShortestPathResult] = {
        (poi_ids[i], poi_ids[j]): path for (i, j), path in raw_pair_paths.items()
    }

    best_order: Optional[List[str]] = None
    best_cost = float("inf")
    best_pair_paths: List[ShortestPathResult] = []

    for ordered in itertools.permutations(poi_ids):
        total_cost = 0.0
        pair_segments: List[ShortestPathResult] = []
        valid = True
        for a, b in zip(ordered, ordered[1:]):
            path = pair_lookup.get((a, b))
            if path is None:
                valid = False
                break
            total_cost += path.total_minutes
            pair_segments.append(path)
        if not valid:
            continue
        if total_cost < best_cost:
            best_cost = total_cost
            best_order = list(ordered)
            best_pair_paths = pair_segments

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
    # Reorder Castle POIs to follow directed tour path
    poi_ids = _reorder_castle_pois_if_needed(list(poi_ids), constraints)

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


def build_itinerary(
    start_time: datetime,
    selected: Sequence[CandidatePOI],
    route: Sequence[str],
    pair_paths: Sequence[ShortestPathResult],
    lunch_break_minutes: int = 0,
) -> Tuple[List[ItineraryStep], float]:
    steps: List[ItineraryStep] = []
    time_cursor = start_time
    idle_minutes = 0.0
    poi_lookup = {poi.poi_id: poi for poi in selected}

    travel_iter = iter(pair_paths)
    lunch_inserted = lunch_break_minutes <= 0
    lunch_duration = timedelta(minutes=lunch_break_minutes)
    lunch_window_start = datetime.combine(start_time.date(), time(11, 0))
    lunch_window_end = datetime.combine(start_time.date(), time(14, 0))

    def maybe_insert_lunch() -> None:
        nonlocal time_cursor, idle_minutes, lunch_inserted
        if lunch_inserted or lunch_break_minutes <= 0:
            return
        if lunch_window_start <= time_cursor <= lunch_window_end:
            logger.debug(
                "Planner: inserting lunch break at %s for %d minutes",
                time_cursor,
                lunch_break_minutes,
            )
            steps.append(
                ItineraryStep(
                    poi_id=LUNCH_BREAK_ID,
                    name="Lunch Break",
                    arrival_time=time_cursor,
                    departure_time=time_cursor + lunch_duration,
                    stay_minutes=lunch_break_minutes,
                )
            )
            time_cursor += lunch_duration
            idle_minutes += lunch_break_minutes
            lunch_inserted = True

    for index, poi_id in enumerate(route):
        poi = poi_lookup.get(poi_id)
        if not poi:
            if index < len(route) - 1:
                travel_path = next(travel_iter, None)
                if travel_path is None:
                    raise ValueError("Travel path segments are misaligned with itinerary order.")
                time_cursor += timedelta(minutes=travel_path.total_minutes)
            continue

        maybe_insert_lunch()

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

    if lunch_break_minutes > 0 and not lunch_inserted:
        if time_cursor < lunch_window_start:
            time_cursor = lunch_window_start
        if time_cursor <= lunch_window_end:
            logger.debug(
                "Planner: late lunch break insertion at %s for %d minutes",
                time_cursor,
                lunch_break_minutes,
            )
            steps.append(
                ItineraryStep(
                    poi_id=LUNCH_BREAK_ID,
                    name="Lunch Break",
                    arrival_time=time_cursor,
                    departure_time=time_cursor + lunch_duration,
                    stay_minutes=lunch_break_minutes,
                )
            )
            idle_minutes += lunch_break_minutes

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
    # Budget validation (if budget constraints provided)
    accessible_zones = None
    if constraints.budget:
        from configs.pricing_config import (
            create_visitor_group,
            get_accessible_zones,
            validate_budget_for_interests,
        )

        # Create visitor group from budget constraint
        visitors = create_visitor_group(
            num_adults=constraints.budget.num_adults,
            num_children_under_18=constraints.budget.num_children_under_18,
            num_youth_18_25_eu=constraints.budget.num_youth_18_25_eu,
            all_eu_residents=constraints.budget.all_eu_residents,
            has_reduced_rate_cards=constraints.budget.has_reduced_rate_cards,
        )

        # Determine accessible zones based on budget
        accessible_zones = get_accessible_zones(
            visitors,
            constraints.budget.total_budget,
            start_time
        )

        logger.info(
            "Planner: budget constraint active (€%.2f). Accessible zones: %s",
            constraints.budget.total_budget,
            accessible_zones
        )

    driver = _connect(uri, user, password)
    try:
        with driver.session() as session:
            candidates = fetch_candidate_pois(
                session,
                interests=constraints.interests,
                accessibility=constraints.accessibility,
                exclude_ids=constraints.exclude_ids,
                accessible_zones=accessible_zones,
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

    max_pois = _auto_max_poi_cap(constraints, total_duration_minutes)
    if max_pois is None:
        selected = select_pois(
            candidates,
            total_duration_minutes=total_duration_minutes,
            must_include=constraints.must_include,
            start_time=start_time,
        )
    else:
        cap = max_pois
        selected: List[CandidatePOI] = []
        while cap >= 1:
            selected = select_pois(
                candidates,
                total_duration_minutes=total_duration_minutes,
                must_include=constraints.must_include,
                max_pois=cap,
                start_time=start_time,
            )
            if selected:
                break
            cap -= 1
        if not selected:
            selected = select_pois(
                candidates,
                total_duration_minutes=total_duration_minutes,
                must_include=constraints.must_include,
                start_time=start_time,
            )
    if not selected:
        raise ValueError("Unable to fit any POIs within the allotted duration.")

    poi_ids = [poi.poi_id for poi in selected]
    logger.info("Planner: selected %d POIs for itinerary", len(poi_ids))
    if constraints.must_include:
        for required in constraints.must_include:
            if required not in poi_ids:
                raise ValueError(f"Required POI {required!r} was not selected.")

    selected_current = list(selected)
    while True:
        ordered_selected = _order_pois_by_opening(selected_current, start_time=start_time)
        if constraints.hard_time_limits:
            min_possible = _minimum_possible_minutes(ordered_selected)
            if min_possible > total_duration_minutes:
                logger.debug(
                    "Planner: minimum possible duration %.1f exceeds limit %d",
                    min_possible,
                    total_duration_minutes,
                )
                pruned = _prune_latest_optional(ordered_selected, constraints)
                if pruned:
                    removed_ids = set(poi.poi_id for poi in ordered_selected) - set(poi.poi_id for poi in pruned)
                    logger.debug(
                        "Planner: pruning late-opening POI(s) before routing: %s",
                        ", ".join(sorted(removed_ids)),
                    )
                    selected_current = pruned
                    continue
                last_error = "Even minimal schedule exceeds requested time window with required POIs."
                logger.warning("Planner: %s", last_error)
                raise ValueError(last_error)

        max_attempts = len(ordered_selected)
        last_error: Optional[str] = None
        for _ in range(max_attempts):
            poi_ids = [poi.poi_id for poi in ordered_selected]
            route_order, pair_paths, travel_segments = determine_route(poi_ids, constraints=constraints)
            lunch_flag = _should_schedule_lunch(start_time, total_duration_minutes, constraints.lunch_break)
            lunch_minutes = 45 if lunch_flag else 0
            itinerary_steps, idle_minutes = build_itinerary(
                start_time,
                ordered_selected,
                route_order,
                pair_paths,
                lunch_break_minutes=lunch_minutes,
            )
            if not constraints.hard_time_limits:
                break
            poi_lookup = {poi.poi_id: poi for poi in ordered_selected}
            if _time_bounds_ok(
                itinerary_steps,
                start_time=start_time,
                total_duration_minutes=total_duration_minutes,
                poi_lookup=poi_lookup,
                enforce_hard=True,
            ):
                break
            logger.debug(
                "Planner: hard limits violated (after %d POIs). Travel=%.1f Visit=%d Idle=%.1f",
                len(ordered_selected),
                travel_segments.total_minutes,
                sum(poi.estimated_visit_minutes for poi in ordered_selected),
                idle_minutes,
            )
            pruned = _prune_latest_optional(ordered_selected, constraints)
            if not pruned:
                last_error = "Unable to satisfy hard time limits with required POIs."
                break
            removed = set(poi.poi_id for poi in ordered_selected) - set(poi.poi_id for poi in pruned)
            logger.debug("Planner: pruning late-opening POI(s): %s", ", ".join(sorted(removed)))
            ordered_selected = pruned
        else:
            last_error = "Unable to compute itinerary within requested time window."

        if last_error:
            logger.warning("Planner: %s", last_error)
            raise ValueError(last_error)

        travel_budget = _travel_budget_minutes(constraints.user_profile, total_duration_minutes)
        if travel_segments.total_minutes <= travel_budget or not selected_current:
            selected_current = ordered_selected
            break

        if _has_only_required(ordered_selected, constraints) and travel_segments.total_minutes > travel_budget:
            logger.warning(
                "Planner: travel %.1f exceeds budget %.1f but only required POIs remain.",
                travel_segments.total_minutes,
                travel_budget,
            )
            selected_current = ordered_selected
            break

        prune_result = _prune_low_priority(ordered_selected, constraints)
        if not prune_result:
            logger.warning(
                "Planner: travel %.1f exceeds budget %.1f but only required POIs remain.",
                travel_segments.total_minutes,
                travel_budget,
            )
            selected_current = ordered_selected
            break

        ordered_selected, removed_ids = prune_result
        logger.debug(
            "Planner: pruning POI(s) to reduce travel burden: %s",
            ", ".join(sorted(removed_ids)),
        )
        selected_current = ordered_selected
        continue

    selected = selected_current

    travel_minutes = travel_segments.total_minutes
    visit_minutes = sum(
        step.stay_minutes for step in itinerary_steps if step.poi_id != LUNCH_BREAK_ID
    )
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
