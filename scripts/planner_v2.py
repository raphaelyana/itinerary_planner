"""
Optimized itinerary planner using predefined paths for Castle and Trianon interiors.

New Architecture:
- Castle interior: Predefined routes (no TSP), selected by time budget
- Trianon interior: Predefined paths with accessibility variants
- Gardens + Trianon gardens: TSP (greedy nearest-neighbor)
- Smart time allocation across zones
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Sequence, Set

from neo4j import GraphDatabase, Session

from scripts.planner_utils import (
    AccessibilityRequirement,
    UserProfile,
    get_shortest_path,
    ShortestPathResult,
    normalize_neo4j_uri,
)
from scripts.zone_allocation import (
    detect_requested_zones,
    allocate_time_budget,
    ZoneAllocation,
)
from scripts.castle_routes import get_viable_castle_routes
from scripts.trianon_paths import (
    GRAND_TRIANON_PATH,
    get_petit_trianon_path,
    get_trianon_tsp_start,
    get_trianon_exclusions,
)
from scripts.path_validator import (
    select_valid_castle_route,
    select_valid_petit_trianon_path,
    PathValidationError,
)

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=os.environ.get("PLANNER_LOG_LEVEL", "INFO"))


@dataclass
class PlannerConstraints:
    """User constraints for itinerary planning."""
    interests: Sequence[str]
    user_profile: UserProfile
    accessibility: AccessibilityRequirement
    must_include: Sequence[str] = ()
    exclude_ids: Sequence[str] = ()
    start_poi: Optional[str] = None
    finish_poi: Optional[str] = None


@dataclass
class ItineraryStep:
    """Single step in an itinerary."""
    poi_id: str
    name: str
    arrival_time: datetime
    departure_time: datetime
    stay_minutes: int


@dataclass
class Itinerary:
    """Complete itinerary with timing."""
    steps: List[ItineraryStep]
    travel_minutes: float
    visit_minutes: int
    total_minutes: float
    travel_segments: ShortestPathResult


def _connect(uri: Optional[str], user: Optional[str], password: Optional[str]):
    """Create Neo4j driver connection."""
    uri = normalize_neo4j_uri(uri or os.getenv("NEO4J_URI"))
    user = user or os.getenv("NEO4J_USERNAME", "neo4j")
    password = password or os.getenv("NEO4J_PASSWORD", "neo4j")

    return GraphDatabase.driver(
        uri,
        auth=(user, password),
        max_connection_pool_size=50,
        connection_timeout=30.0,
        max_connection_lifetime=3600,
        connection_acquisition_timeout=60.0,
    )


def fetch_garden_pois(
    session: Session,
    interests: Sequence[str],
    accessibility: AccessibilityRequirement,
    exclude_ids: Sequence[str],
    zone_filter: str,
) -> List[dict]:
    """
    Fetch POIs for gardens/parks where TSP will be applied.

    Parameters
    ----------
    session : Session
        Neo4j session
    interests : Sequence[str]
        Interest tags
    accessibility : AccessibilityRequirement
        Accessibility requirement
    exclude_ids : Sequence[str]
        POI IDs to exclude
    zone_filter : str
        Zone to filter (e.g., "Garden", "Park", "Trianon")

    Returns
    -------
    List[dict]
        POI records with id, name, estimated_visit_minutes
    """
    query = """
    MATCH (poi:POI)
    WHERE poi.estimated_visit_minutes IS NOT NULL
      AND poi.id IS NOT NULL
      AND poi.id CONTAINS $zone_filter
      AND ($interests IS NULL OR size($interests) = 0
           OR any(tag IN poi.interest_tags WHERE tag IN $interests))
      AND ($accessibility = 'any' OR poi.accessibility_level = 'full')
      AND NOT poi.id IN $exclude_ids
    RETURN poi.id AS id,
           poi.name AS name,
           poi.estimated_visit_minutes AS estimated_visit_minutes,
           poi.priority_score AS priority_score
    ORDER BY poi.priority_score DESC
    """

    result = session.run(
        query,
        zone_filter=zone_filter,
        interests=list(interests),
        accessibility=accessibility,
        exclude_ids=list(exclude_ids),
    )

    return [dict(record) for record in result]


def greedy_tsp_on_pois(
    session: Session,
    poi_ids: List[str],
    start_poi: str,
    accessibility: AccessibilityRequirement,
    time_budget_minutes: int,
) -> List[str]:
    """
    Apply greedy nearest-neighbor TSP to select and order POIs.

    Parameters
    ----------
    session : Session
        Neo4j session
    poi_ids : List[str]
        Candidate POI IDs
    start_poi : str
        Starting POI for TSP
    accessibility : AccessibilityRequirement
        Accessibility requirement
    time_budget_minutes : int
        Available time budget for this zone

    Returns
    -------
    List[str]
        Ordered POI IDs (greedy path)
    """
    if not poi_ids:
        return []

    # Simple greedy algorithm: pick nearest unvisited POI
    ordered_path = [start_poi]
    unvisited = set(poi_ids)
    current_time = 0

    # Get POI visit durations
    poi_durations = {}
    query = "MATCH (p:POI {id: $poi_id}) RETURN p.estimated_visit_minutes AS duration"
    for poi_id in poi_ids:
        result = session.run(query, poi_id=poi_id)
        record = result.single()
        if record:
            poi_durations[poi_id] = record["duration"] or 10

    current = start_poi

    while unvisited and current_time < time_budget_minutes:
        # Find nearest unvisited POI
        best_poi = None
        best_distance = float('inf')

        for candidate in unvisited:
            path_result = get_shortest_path(
                session, current, candidate, accessibility, "standard"
            )
            if path_result and path_result.total_minutes < best_distance:
                best_distance = path_result.total_minutes
                best_poi = candidate

        if best_poi is None:
            # No reachable POIs remain
            break

        # Check if we have time to visit
        visit_time = poi_durations.get(best_poi, 10)
        if current_time + best_distance + visit_time > time_budget_minutes:
            break

        # Visit this POI
        ordered_path.append(best_poi)
        unvisited.remove(best_poi)
        current = best_poi
        current_time += best_distance + visit_time

    return ordered_path


def create_itinerary_v2(
    *,
    start_time: datetime,
    total_duration_minutes: int,
    constraints: PlannerConstraints,
    uri: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
) -> Itinerary:
    """
    Create optimized itinerary using new architecture.

    Parameters
    ----------
    start_time : datetime
        Visit start time
    total_duration_minutes : int
        Total available time
    constraints : PlannerConstraints
        User constraints
    uri : Optional[str]
        Neo4j URI
    user : Optional[str]
        Neo4j username
    password : Optional[str]
        Neo4j password

    Returns
    -------
    Itinerary
        Planned itinerary

    Raises
    ------
    PathValidationError
        If no valid Castle path exists
    """
    driver = _connect(uri, user, password)

    try:
        with driver.session() as session:
            # Step 1: Detect requested zones
            requested_zones = detect_requested_zones(
                list(constraints.interests),
                list(constraints.must_include),
            )

            logger.info(f"Detected zones: {requested_zones}")

            # Step 2: Allocate time budget across zones
            allocation = allocate_time_budget(
                total_duration_minutes, requested_zones
            )

            logger.info(f"Time allocation: {allocation}")

            # Step 3: Build path for each zone
            full_path = []

            # Castle interior (predefined path)
            if "castle" in requested_zones:
                castle_candidates = get_viable_castle_routes(allocation.castle_minutes)
                try:
                    castle_path = select_valid_castle_route(
                        session,
                        castle_candidates,
                        constraints.accessibility,
                        list(constraints.exclude_ids),
                    )
                    full_path.extend(castle_path)
                    logger.info(f"Selected Castle route with {len(castle_path)} POIs")
                except PathValidationError as e:
                    logger.error(f"Castle path validation failed: {e}")
                    raise

            # Gardens (TSP)
            if "gardens" in requested_zones:
                garden_pois = fetch_garden_pois(
                    session,
                    constraints.interests,
                    constraints.accessibility,
                    constraints.exclude_ids,
                    ":Garden:",
                )

                if garden_pois:
                    # Start from last Castle POI or default entrance
                    garden_start = full_path[-1] if full_path else "versailles:Garden:acces-jardins-cour-des-princes"

                    garden_path = greedy_tsp_on_pois(
                        session,
                        [p["id"] for p in garden_pois[:20]],  # Limit to top 20
                        garden_start,
                        constraints.accessibility,
                        allocation.gardens_minutes,
                    )

                    full_path.extend(garden_path[1:])  # Exclude start (already in path)
                    logger.info(f"Added {len(garden_path)-1} Garden POIs")

            # Trianon interior (predefined path) + Trianon gardens (TSP)
            if "trianon" in requested_zones:
                # Grand Trianon interior
                grand_trianon_path = GRAND_TRIANON_PATH.copy()
                full_path.extend(grand_trianon_path)

                # Petit Trianon interior
                try:
                    petit_trianon_path = select_valid_petit_trianon_path(
                        session, constraints.accessibility
                    )
                    full_path.extend(petit_trianon_path)
                    logger.info(f"Added Petit Trianon path ({len(petit_trianon_path)} POIs)")
                except PathValidationError as e:
                    logger.warning(f"Petit Trianon path unavailable: {e}")

                # Trianon gardens (TSP)
                trianon_garden_pois = fetch_garden_pois(
                    session,
                    constraints.interests,
                    constraints.accessibility,
                    list(constraints.exclude_ids) + get_trianon_exclusions(grand_trianon_path),
                    ":Trianon:",
                )

                if trianon_garden_pois:
                    tsp_start = get_trianon_tsp_start(grand_trianon_path)
                    trianon_garden_path = greedy_tsp_on_pois(
                        session,
                        [p["id"] for p in trianon_garden_pois[:15]],
                        tsp_start,
                        constraints.accessibility,
                        allocation.trianon_minutes // 2,  # Reserve half for gardens
                    )

                    full_path.extend(trianon_garden_path[1:])
                    logger.info(f"Added {len(trianon_garden_path)-1} Trianon garden POIs")

            # Step 4: Compute full path with timing
            if not full_path:
                raise ValueError("No POIs selected for itinerary")

            # Compute paths between consecutive POIs
            all_segments = []
            total_travel_time = 0.0

            for i in range(len(full_path) - 1):
                from_poi = full_path[i]
                to_poi = full_path[i + 1]

                segment_result = get_shortest_path(
                    from_poi,
                    to_poi,
                    accessibility=constraints.accessibility,
                    user_profile=constraints.user_profile,
                    uri=uri,
                    user=user,
                    password=password,
                )

                if segment_result:
                    all_segments.extend(segment_result.segments)
                    total_travel_time += segment_result.total_minutes

            # Build composite travel result
            from scripts.planner_utils import ShortestPathResult
            travel_segments = ShortestPathResult(
                node_ids=full_path,
                total_minutes=total_travel_time,
                segments=all_segments,
            )

            if not travel_segments:
                raise ValueError("Could not compute path between selected POIs")

            # Build itinerary steps with timing
            steps = []
            current_time = start_time
            total_visit = 0

            for poi_id in full_path:
                # Get POI details
                query = """
                MATCH (p:POI {id: $poi_id})
                RETURN p.name AS name, p.estimated_visit_minutes AS visit_minutes
                """
                result = session.run(query, poi_id=poi_id)
                record = result.single()

                if not record:
                    continue

                visit_minutes = record["visit_minutes"] or 10

                steps.append(ItineraryStep(
                    poi_id=poi_id,
                    name=record["name"],
                    arrival_time=current_time,
                    departure_time=current_time + timedelta(minutes=visit_minutes),
                    stay_minutes=visit_minutes,
                ))

                current_time += timedelta(minutes=visit_minutes)
                total_visit += visit_minutes

            return Itinerary(
                steps=steps,
                travel_minutes=travel_segments.total_minutes,
                visit_minutes=total_visit,
                total_minutes=travel_segments.total_minutes + total_visit,
                travel_segments=travel_segments,
            )

    finally:
        driver.close()
