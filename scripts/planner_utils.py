from __future__ import annotations

import atexit
import os
import time
from dataclasses import dataclass
from threading import Lock
from typing import Dict, List, Literal, Optional, Tuple

from neo4j import GraphDatabase, Driver
from neo4j.exceptions import SessionExpired, ServiceUnavailable

UserProfile = Literal["standard", "family", "elder"]
AccessibilityRequirement = Literal["any", "step_free", "stroller"]

PROFILE_WEIGHT_PROPERTY: Dict[UserProfile, str] = {
    "standard": "base_walk_min",
    "family": "family_walk_min",
    "elder": "elder_walk_min",
}

_DRIVER_CACHE: Dict[Tuple[str, str, str], Driver] = {}
_DRIVER_LOCK = Lock()


def normalize_neo4j_uri(raw_uri: Optional[str]) -> str:
    """
    Ensure the Neo4j URI has a scheme. Render's service hostport env var is scheme-less.
    """
    if not raw_uri:
        return "neo4j://localhost:7687"
    if "://" not in raw_uri:
        return f"neo4j://{raw_uri}"
    return raw_uri


def _get_driver(uri: str, user: str, password: str) -> Driver:
    key = (uri, user, password)
    with _DRIVER_LOCK:
        driver = _DRIVER_CACHE.get(key)
        if driver is None:
            # Configure connection pool for AuraDB stability
            # Note: neo4j+s:// URI already includes encryption, don't set encrypted=True
            driver = GraphDatabase.driver(
                uri,
                auth=(user, password),
                max_connection_pool_size=50,               # Increase pool size for concurrent queries
                connection_timeout=30.0,                   # 30 second connection timeout
                max_connection_lifetime=3600,              # Rotate connections every hour
                connection_acquisition_timeout=60.0,       # Wait up to 60s for available connection
            )
            _DRIVER_CACHE[key] = driver
        return driver


def _close_cached_drivers() -> None:
    with _DRIVER_LOCK:
        for driver in _DRIVER_CACHE.values():
            try:
                driver.close()
            except Exception:
                pass
        _DRIVER_CACHE.clear()


atexit.register(_close_cached_drivers)


@dataclass
class PathSegment:
    """Describes one step in the returned path."""

    from_id: str
    to_id: str
    distance_min: float
    is_step_free: bool
    stroller_friendly: bool
    path_type: Optional[str]
    notes: Optional[str]


@dataclass
class ShortestPathResult:
    """Container for the end-to-end path payload."""

    node_ids: List[str]
    total_minutes: float
    segments: List[PathSegment]


def _validate_profile(profile: UserProfile) -> str:
    if profile not in PROFILE_WEIGHT_PROPERTY:
        supported = ", ".join(PROFILE_WEIGHT_PROPERTY)
        raise ValueError(f"Unsupported user profile {profile!r}. Expected one of: {supported}.")
    return PROFILE_WEIGHT_PROPERTY[profile]


def _build_filters(requirement: AccessibilityRequirement) -> AccessibilityRequirement:
    accepted = {"any", "step_free", "stroller"}
    if requirement not in accepted:
        supported = ", ".join(sorted(accepted))
        raise ValueError(f"Unknown accessibility requirement: {requirement!r}. Expected one of: {supported}.")
    return requirement


def get_shortest_path(
    start_id: str,
    end_id: str,
    *,
    user_profile: UserProfile = "standard",
    accessibility: AccessibilityRequirement = "any",
    uri: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    default_edge_weight: float = 5.0,
) -> ShortestPathResult:
    """
    Compute the optimal route between two POIs using APOC path expansion.

    Parameters
    ----------
    start_id, end_id:
        Node identifiers (matching the ``id`` column in ``pois.csv``).
    user_profile:
        Chooses which relationship weight to prioritise: ``standard`` (base),
        ``family``, or ``elder``.
    accessibility:
        ``any`` for no filter, ``step_free`` for step-free routes, ``stroller`` for
        routes that are both step-free and marked stroller-friendly.
    uri, user, password:
        Optional overrides for the Neo4j connection; defaults to the environment
        variables ``NEO4J_URI``, ``NEO4J_USERNAME``, ``NEO4J_PASSWORD``.
    default_edge_weight:
        Fallback value when a relationship is missing the selected weight property.

    Returns
    -------
    ShortestPathResult
        The ordered set of node IDs, total travel minutes, and per-segment metadata.
    """

    weight_property = _validate_profile(user_profile)
    accessibility = _build_filters(accessibility)

    if start_id == end_id:
        return ShortestPathResult(node_ids=[start_id], total_minutes=0.0, segments=[])

    uri = normalize_neo4j_uri(uri or os.getenv("NEO4J_URI"))
    user = user or os.getenv("NEO4J_USERNAME", "neo4j")
    password = password or os.getenv("NEO4J_PASSWORD", "neo4j")

    driver = _get_driver(uri, user, password)

    # Retry logic for handling transient AuraDB connection issues
    max_retries = 3
    base_delay = 1.0

    for attempt in range(max_retries):
        try:
            with driver.session() as session:
                filters: List[str] = []
                if accessibility == "step_free":
                    filters.append("all(rel IN relationships(path) WHERE rel.is_step_free = true)")
                elif accessibility == "stroller":
                    filters.append(
                        "all(rel IN relationships(path) WHERE rel.is_step_free = true AND rel.stroller_friendly = true)"
                    )

                filters_text = ""
                if filters:
                    filters_text = "AND " + " AND ".join(filters)

                record = session.run(
                    f"""
                        MATCH (start:POI {{id: $start_id}})
                        WITH start
                        MATCH (end:POI {{id: $end_id}})
                        CALL apoc.path.expandConfig(start, {{
                            relationshipFilter: "CONNECTS_TO>",
                            minLevel: 1,
                            maxLevel: 30,
                            uniqueness: "RELATIONSHIP_PATH",
                            endNodes: [end],
                            terminatorNodes: [end]
                        }})
                        YIELD path
                        WHERE last(nodes(path)) = end
                          {filters_text}
                        WITH path,
                             reduce(cost = 0.0, rel IN relationships(path) |
                                   cost + coalesce(rel[$weight_property], $default_weight)) AS total_cost
                        ORDER BY total_cost ASC, length(path) ASC
                        LIMIT 1
                        RETURN [node IN nodes(path) | node.id] AS node_ids,
                               [rel IN relationships(path) |
                                    {{
                                        from_id: startNode(rel).id,
                                        to_id: endNode(rel).id,
                                        distance: coalesce(rel[$weight_property], $default_weight),
                                        is_step_free: rel.is_step_free,
                                        stroller_friendly: rel.stroller_friendly,
                                        path_type: rel.path_type,
                                        notes: rel.notes
                                    }}
                               ] AS segments,
                               total_cost AS total_minutes
                    """,
                    start_id=start_id,
                    end_id=end_id,
                    weight_property=weight_property,
                    default_weight=float(default_edge_weight),
                ).single()

                # Check if we got a result
                if record is None:
                    start_exists = session.run(
                        "MATCH (n:POI {id: $id}) RETURN count(n) AS count",
                        id=start_id,
                    ).single()["count"] > 0
                    end_exists = session.run(
                        "MATCH (n:POI {id: $id}) RETURN count(n) AS count",
                        id=end_id,
                    ).single()["count"] > 0

                    if not start_exists:
                        raise ValueError(f"Start node {start_id!r} is unknown.")
                    if not end_exists:
                        raise ValueError(f"End node {end_id!r} is unknown.")
                    raise ValueError(f"No path found between {start_id!r} and {end_id!r}.")

                # Success - break out of retry loop
                break

        except (SessionExpired, ServiceUnavailable) as e:
            if attempt < max_retries - 1:
                # Exponential backoff: 1s, 2s, 4s
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
                continue
            else:
                # Final attempt failed, re-raise the exception
                raise

    # Process the successful result
    node_ids: List[str] = record["node_ids"]
    total_minutes = float(record["total_minutes"])
    segments_data = record["segments"] or []

    segments = [
        PathSegment(
            from_id=segment["from_id"],
            to_id=segment["to_id"],
            distance_min=float(segment["distance"]),
            is_step_free=bool(segment["is_step_free"]),
            stroller_friendly=bool(segment["stroller_friendly"]),
            path_type=str(segment["path_type"]) if segment["path_type"] is not None else None,
            notes=str(segment["notes"]) if segment["notes"] is not None else None,
        )
        for segment in segments_data
    ]

    return ShortestPathResult(node_ids=node_ids, total_minutes=total_minutes, segments=segments)
