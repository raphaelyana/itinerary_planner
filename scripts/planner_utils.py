from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

from neo4j import GraphDatabase
from neo4j.exceptions import ClientError

UserProfile = Literal["standard", "family", "elder"]
AccessibilityRequirement = Literal["any", "step_free", "stroller"]

PROFILE_WEIGHT_PROPERTY: Dict[UserProfile, str] = {
    "standard": "base_walk_min",
    "family": "family_walk_min",
    "elder": "elder_walk_min",
}

GDS_NODE_PROJECTION = "MATCH (n:POI) RETURN id(n) AS id"
GDS_RELATIONSHIP_QUERY_TEMPLATE = """
MATCH (a:POI)-[r:CONNECTS_TO]->(b:POI)
{where_clause}
RETURN id(a) AS source,
       id(b) AS target,
       r.base_walk_min AS base_walk_min,
       r.family_walk_min AS family_walk_min,
       r.elder_walk_min AS elder_walk_min,
       r.is_step_free AS is_step_free,
       r.stroller_friendly AS stroller_friendly,
       r.path_type AS path_type,
       r.notes AS notes
"""

GRAPH_CONFIGS: Dict[AccessibilityRequirement, Dict[str, str]] = {
    "any": {"name": "itinerary_graph_any", "where_clause": ""},
    "step_free": {"name": "itinerary_graph_step_free", "where_clause": "WHERE r.is_step_free = true"},
    "stroller": {
        "name": "itinerary_graph_stroller",
        "where_clause": "WHERE r.is_step_free = true AND r.stroller_friendly = true",
    },
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


def _build_filters(requirement: AccessibilityRequirement) -> AccessibilityRequirement:
    if requirement not in GRAPH_CONFIGS:
        supported = ", ".join(GRAPH_CONFIGS)
        raise ValueError(f"Unknown accessibility requirement: {requirement!r}. Expected one of: {supported}.")
    return requirement


def _ensure_gds_graph(session, accessibility: AccessibilityRequirement) -> str:
    config = GRAPH_CONFIGS[accessibility]
    graph_name = config["name"]
    exists_record = session.run("CALL gds.graph.exists($name) YIELD exists", name=graph_name).single()
    if exists_record is not None and exists_record["exists"]:
        return graph_name

    relationship_query = GDS_RELATIONSHIP_QUERY_TEMPLATE.format(where_clause=config["where_clause"])

    try:
        session.run(
            """
            CALL gds.graph.project.cypher($graph_name, $node_query, $relationship_query)
            """,
            graph_name=graph_name,
            node_query=GDS_NODE_PROJECTION,
            relationship_query=relationship_query,
        ).consume()
    except ClientError as exc:  # pragma: no cover - defensive against concurrent creation
        if "already exists" not in str(exc):
            raise

    return graph_name


def _collect_segments(session, node_ids: List[str], weight_property: str, default_weight: float) -> List[PathSegment]:
    if len(node_ids) < 2:
        return []

    pairs = [
        {"from": node_ids[idx], "to": node_ids[idx + 1], "order": idx}
        for idx in range(len(node_ids) - 1)
    ]
    result = session.run(
        """
        UNWIND $pairs AS pair
        MATCH (src:POI {id: pair.from})-[rel:CONNECTS_TO]->(dst:POI {id: pair.to})
        RETURN pair.order AS order,
               pair.from AS from_id,
               pair.to AS to_id,
               coalesce(rel[$weight_property], $default_weight) AS distance,
               rel.is_step_free AS is_step_free,
               rel.stroller_friendly AS stroller_friendly,
               rel.path_type AS path_type,
               rel.notes AS notes
        ORDER BY order
        """,
        pairs=pairs,
        weight_property=weight_property,
        default_weight=default_weight,
    )

    segments: List[PathSegment] = []
    for record in result:
        segments.append(
            PathSegment(
                from_id=record["from_id"],
                to_id=record["to_id"],
                distance_min=float(record["distance"]),
                is_step_free=bool(record["is_step_free"]),
                stroller_friendly=bool(record["stroller_friendly"]),
                path_type=str(record["path_type"]) if record["path_type"] is not None else None,
                notes=str(record["notes"]) if record["notes"] is not None else None,
            )
        )
    return segments


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
    Compute the optimal route between two POIs using Neo4j GDS Dijkstra.

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

    uri = uri or os.getenv("NEO4J_URI", "neo4j://localhost:7687")
    user = user or os.getenv("NEO4J_USERNAME", "neo4j")
    password = password or os.getenv("NEO4J_PASSWORD", "neo4j")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session() as session:
            graph_name = _ensure_gds_graph(session, accessibility)

            record = session.run(
                """
                MATCH (start:POI {id: $start_id}), (end:POI {id: $end_id})
                CALL gds.shortestPath.dijkstra.stream($graph_name, {
                    sourceNode: id(start),
                    targetNode: id(end),
                    relationshipWeightProperty: $weight_property
                })
                YIELD nodeIds, totalCost
                RETURN [node IN gds.util.asNodes(nodeIds) | node.id] AS node_ids,
                       totalCost AS total_minutes
                """,
                start_id=start_id,
                end_id=end_id,
                graph_name=graph_name,
                weight_property=weight_property,
            ).single()

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

            node_ids: List[str] = record["node_ids"]
            total_minutes = float(record["total_minutes"])
            segments = _collect_segments(session, node_ids, weight_property, float(default_edge_weight))

            return ShortestPathResult(node_ids=node_ids, total_minutes=total_minutes, segments=segments)
    finally:
        driver.close()
