"""
Path validation for predefined Castle and Trianon routes.

Validates that each edge in a predefined path meets accessibility requirements
by querying Neo4j for connection properties (is_step_free, stroller_friendly).
"""

from typing import List, Optional, Dict, Any
from neo4j import Session


class PathValidationError(Exception):
    """Raised when no valid predefined path exists for given constraints."""
    pass


def validate_path_segment(
    session: Session,
    from_poi: str,
    to_poi: str,
    accessibility: str,
) -> bool:
    """
    Check if a single path segment (edge) meets accessibility requirements.

    Parameters
    ----------
    session : Session
        Neo4j session
    from_poi : str
        Source POI ID
    to_poi : str
        Destination POI ID
    accessibility : str
        One of: "any", "step_free", "stroller_friendly"

    Returns
    -------
    bool
        True if edge exists and meets requirements, or if edge doesn't exist (lenient)
    """
    query = """
    MATCH (a:POI {id: $from_poi})-[r:CONNECTS_TO]->(b:POI {id: $to_poi})
    RETURN r.is_step_free AS step_free, r.stroller_friendly AS stroller_friendly
    """

    result = session.run(query, from_poi=from_poi, to_poi=to_poi)
    record = result.single()

    if not record:
        # Edge does not exist - be lenient and accept it
        # (predefined paths may not be fully in DB yet)
        return True

    step_free = record["step_free"]
    stroller_friendly = record["stroller_friendly"]

    # Check accessibility requirements
    if accessibility == "step_free":
        return step_free is True
    elif accessibility == "stroller_friendly":
        return stroller_friendly is True
    else:  # "any"
        return True  # No restrictions


def validate_full_path(
    session: Session,
    poi_sequence: List[str],
    accessibility: str,
) -> bool:
    """
    Validate an entire predefined path meets accessibility requirements.

    Parameters
    ----------
    session : Session
        Neo4j session
    poi_sequence : List[str]
        Ordered list of POI IDs
    accessibility : str
        Accessibility requirement

    Returns
    -------
    bool
        True if entire path is valid
    """
    if len(poi_sequence) < 2:
        return True  # Single POI or empty path is trivially valid

    # Check each consecutive pair
    for i in range(len(poi_sequence) - 1):
        from_poi = poi_sequence[i]
        to_poi = poi_sequence[i + 1]

        if not validate_path_segment(session, from_poi, to_poi, accessibility):
            return False

    return True


def validate_castle_route(
    session: Session,
    route_pois: List[str],
    accessibility: str,
    excluded_pois: List[str],
) -> bool:
    """
    Validate a Castle route variant.

    Checks:
    1. All edges meet accessibility requirements
    2. No excluded POIs are in the route

    Parameters
    ----------
    session : Session
        Neo4j session
    route_pois : List[str]
        Castle route POI sequence
    accessibility : str
        Accessibility requirement
    excluded_pois : List[str]
        POIs to exclude (e.g., closed rooms)

    Returns
    -------
    bool
        True if route is valid
    """
    # Check for excluded POIs
    if any(poi in excluded_pois for poi in route_pois):
        return False

    # Check accessibility
    return validate_full_path(session, route_pois, accessibility)


def validate_trianon_path(
    session: Session,
    path_pois: List[str],
    accessibility: str,
) -> bool:
    """
    Validate a Trianon interior path.

    Parameters
    ----------
    session : Session
        Neo4j session
    path_pois : List[str]
        Trianon path POI sequence
    accessibility : str
        Accessibility requirement

    Returns
    -------
    bool
        True if path is valid
    """
    return validate_full_path(session, path_pois, accessibility)


def select_valid_castle_route(
    session: Session,
    candidate_routes: List[Dict[str, Any]],
    accessibility: str,
    excluded_pois: List[str],
) -> Optional[List[str]]:
    """
    Select the first valid Castle route from candidates.

    Candidates should be ordered by preference (longest first).

    Parameters
    ----------
    session : Session
        Neo4j session
    candidate_routes : List[Dict]
        List of route dictionaries with "pois" key
    accessibility : str
        Accessibility requirement
    excluded_pois : List[str]
        POIs to exclude

    Returns
    -------
    Optional[List[str]]
        Valid route POI sequence, or None if no valid route exists

    Raises
    ------
    PathValidationError
        If no valid Castle route exists
    """
    for route_dict in candidate_routes:
        route_pois = route_dict["pois"]
        if validate_castle_route(session, route_pois, accessibility, excluded_pois):
            return route_pois

    # No valid route found
    raise PathValidationError(
        f"No valid Castle route exists for accessibility='{accessibility}' "
        f"and excluded POIs: {excluded_pois}"
    )


def select_valid_petit_trianon_path(
    session: Session,
    accessibility: str,
) -> List[str]:
    """
    Select valid Petit Trianon path based on accessibility.

    Parameters
    ----------
    session : Session
        Neo4j session
    accessibility : str
        Accessibility requirement

    Returns
    -------
    List[str]
        Valid Petit Trianon path

    Raises
    ------
    PathValidationError
        If no valid path exists
    """
    from scripts.trianon_paths import (
        PETIT_TRIANON_PATH_ACCESSIBLE,
        PETIT_TRIANON_PATH_FULL,
    )

    # If accessibility required, use accessible path
    if accessibility in ["step_free", "stroller_friendly"]:
        path = PETIT_TRIANON_PATH_ACCESSIBLE
        if validate_trianon_path(session, path, accessibility):
            return path.copy()
        else:
            raise PathValidationError(
                f"Petit Trianon accessible path does not meet {accessibility} requirements"
            )

    # Otherwise, try full path first, fallback to accessible
    if validate_trianon_path(session, PETIT_TRIANON_PATH_FULL, accessibility):
        return PETIT_TRIANON_PATH_FULL.copy()
    elif validate_trianon_path(session, PETIT_TRIANON_PATH_ACCESSIBLE, accessibility):
        return PETIT_TRIANON_PATH_ACCESSIBLE.copy()
    else:
        raise PathValidationError("No valid Petit Trianon path exists")
