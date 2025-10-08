from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

from neo4j import GraphDatabase

UserProfile = Literal["standard", "family", "elder"]
AccessibilityRequirement = Literal["any", "step_free", "stroller"]

PROFILE_WEIGHT_PROPERTY: Dict[UserProfile, str] = {
    "standard": "base_walk_min",
    "family": "family_walk_min",
    "elder": "elder_walk_min",
}


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


def _build_filters(requirement: AccessibilityRequirement) -> Tuple[bool, bool]:
    if requirement == "any":
        return False, False
    if requirement == "step_free":
        return True, False
    if requirement == "stroller":
        return True, True
    raise ValueError(f"Unknown accessibility requirement: {requirement!r}")


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
    Compute the optimal route between two POIs using APOC's Dijkstra implementation.

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
        variables ``NEO4J_URI``, ``NEO4J_USERNAME``, ``NEO4J_PASSWORD`` or the Neo4j defaults.
    default_edge_weight:
        Fallback value when a relationship is missing the selected weight property.

    Returns
    -------
    ShortestPathResult
        The ordered set of node IDs, total travel minutes, and per-segment metadata.
    """

    weight_property = _validate_profile(user_profile)
    require_step_free, require_stroller = _build_filters(accessibility)

    uri = uri or os.getenv("NEO4J_URI", "neo4j://localhost:7687")
    user = user or os.getenv("NEO4J_USERNAME", "neo4j")
    password = password or os.getenv("NEO4J_PASSWORD", "neo4j")

    cypher = """
    MATCH (start:POI {id: $start_id}), (end:POI {id: $end_id})
    CALL apoc.algo.dijkstraWithDefaultWeight(
        start,
        end,
        $relationship_filter,
        $weight_property,
        $default_weight
    )
    YIELD path, weight
    WHERE ($require_step_free = false OR all(rel IN relationships(path) WHERE rel.is_step_free = true))
      AND ($require_stroller = false OR all(rel IN relationships(path) WHERE rel.stroller_friendly = true))
    RETURN [node IN nodes(path) | node.id] AS node_ids,
           [rel IN relationships(path) |
                {from: startNode(rel).id,
                 to: endNode(rel).id,
                 distance: coalesce(rel[$weight_property], $default_weight),
                 is_step_free: rel.is_step_free,
                 stroller_friendly: rel.stroller_friendly,
                 path_type: rel.path_type,
                 notes: rel.notes}
           ] AS segments,
           weight AS total
    LIMIT 1
    """

    params = {
        "start_id": start_id,
        "end_id": end_id,
        "relationship_filter": "CONNECTS_TO>",
        "weight_property": weight_property,
        "default_weight": float(default_edge_weight),
        "require_step_free": require_step_free,
        "require_stroller": require_stroller,
    }

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session() as session:
            record = session.run(cypher, params).single()
            if record is None:
                raise ValueError(f"No path found between {start_id!r} and {end_id!r}.")

            node_ids: List[str] = record["node_ids"]
            segments_raw: List[Dict[str, object]] = record["segments"]
            segments = [
                PathSegment(
                    from_id=segment["from"],
                    to_id=segment["to"],
                    distance_min=float(segment["distance"]),
                    is_step_free=bool(segment["is_step_free"]),
                    stroller_friendly=bool(segment["stroller_friendly"]),
                    path_type=str(segment["path_type"]) if segment["path_type"] is not None else None,
                    notes=str(segment["notes"]) if segment["notes"] is not None else None,
                )
                for segment in segments_raw
            ]

            return ShortestPathResult(
                node_ids=node_ids,
                total_minutes=float(record["total"]),
                segments=segments,
            )
    finally:
        driver.close()
