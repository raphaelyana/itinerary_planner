"""
Predefined route variants for Castle and Trianon interiors.

Each route is a curated sequence of POIs representing a coherent visit path.
Routes are selected based on available time budget and visitor interests.

Note: Gardens and Trianon gardens use TSP (greedy nearest-neighbor).
Only Castle interior and Trianon interior have predefined paths.
"""

from typing import List, Dict, Literal
from scripts.castle_paths import CASTLE_TRUNK

RouteVariant = Literal["essential", "standard", "extended", "comprehensive"]
TrianonVariant = Literal["quick", "standard", "complete"]

# Castle route variants with estimated total duration (minutes)
# Using real paths from castle_paths.py CASTLE_TRUNK
CASTLE_ROUTES: Dict[RouteVariant, Dict[str, any]] = {
    "essential": {
        "duration_min": 45,  # Minimum time needed
        "pois": [
            "versailles:Castle:cour-dhonneur",
            "versailles:Castle:acces-grands-appartements",
            "versailles:Room:salon-dhercule",
            "versailles:Room:salon-dapollon",
            "versailles:Room:galerie-des-glaces",
            "versailles:Room:chambre-de-la-reine",
        ],
        "description": "Core highlights only - minimal visit",
    },
    "standard": {
        "duration_min": 60,
        "pois": [
            "versailles:Castle:cour-dhonneur",
            "versailles:Castle:acces-grands-appartements",
            "versailles:Room:galeries-de-lhistoire",
            "versailles:Room:salles-louis-xiv",
            "versailles:Room:chapelle-royale",  # Detour
            "versailles:Room:salon-dhercule",
            "versailles:Room:salon-de-venus",
            "versailles:Room:salon-de-mars",
            "versailles:Room:salon-dapollon",
            "versailles:Room:salon-de-la-guerre",
            "versailles:Room:galerie-des-glaces",
            "versailles:Room:salon-de-la-paix",
            "versailles:Room:chambre-de-la-reine",
        ],
        "description": "Standard royal apartments tour with Chapelle",
    },
    "extended": {
        "duration_min": 90,
        "pois": CASTLE_TRUNK[:],  # Full trunk path
        "description": "Extended tour - complete trunk sequence",
    },
    "comprehensive": {
        "duration_min": 120,
        "pois": CASTLE_TRUNK + [
            "versailles:Room:galerie-des-batailles",
            "versailles:Castle:pavillon-dufour-sortie",
        ],
        "description": "Comprehensive tour with Galerie des Batailles",
    },
}


def get_viable_castle_routes(available_minutes: int) -> List[Dict[str, any]]:
    """
    Get Castle route variants that fit within time budget, ordered by duration (longest first).

    Parameters
    ----------
    available_minutes : int
        Time budget allocated for the Castle visit

    Returns
    -------
    List[Dict]
        List of route dictionaries ordered by preference
    """
    # Filter routes that fit within the time budget
    viable_routes = [
        route
        for route in CASTLE_ROUTES.values()
        if route["duration_min"] <= available_minutes
    ]

    if not viable_routes:
        # Time too short, return essential route anyway
        viable_routes = [CASTLE_ROUTES["essential"]]

    # Sort by duration (longest first) to maximize time use
    viable_routes.sort(key=lambda r: r["duration_min"], reverse=True)

    return viable_routes


def select_castle_route(
    available_minutes: int,
    interests: List[str],
) -> List[str]:
    """
    Select the best Castle route variant based on available time and interests.

    Parameters
    ----------
    available_minutes : int
        Time budget allocated for the Castle visit
    interests : List[str]
        User interests (e.g., ["art", "history"])

    Returns
    -------
    List[str]
        Ordered list of POI IDs for the selected route
    """
    # Filter routes that fit within the time budget
    viable_routes = {
        variant: route
        for variant, route in CASTLE_ROUTES.items()
        if route["duration_min"] <= available_minutes
    }

    if not viable_routes:
        # Time too short, return essential route anyway
        return CASTLE_ROUTES["essential"]["pois"]

    # Select the longest route that fits (maximize use of allocated time)
    best_variant = max(
        viable_routes.keys(),
        key=lambda v: CASTLE_ROUTES[v]["duration_min"],
    )

    selected_pois = CASTLE_ROUTES[best_variant]["pois"].copy()

    # Interest-based filtering (optional future enhancement)
    # For now, just return the selected route as-is
    # TODO: Filter POIs based on interest_tags if needed

    return selected_pois


def estimate_castle_duration(route_pois: List[str]) -> int:
    """
    Estimate total duration for a Castle route (visit + walk time).

    This is a rough estimate based on number of POIs.
    Actual duration will be computed by the planner with real walk times.

    Parameters
    ----------
    route_pois : List[str]
        List of POI IDs in the route

    Returns
    -------
    int
        Estimated duration in minutes
    """
    # Rough heuristic: ~10 min per POI on average (visit + walk)
    return len(route_pois) * 10


# Trianon interior route variants
TRIANON_ROUTES: Dict[TrianonVariant, Dict[str, any]] = {
    "quick": {
        "duration_min": 30,
        "pois": [
            # Grand Trianon essentials only
            "versailles:Trianon:grand-trianon-peristyle",
            "versailles:Trianon:grand-trianon-galerie",
        ],
        "description": "Quick Grand Trianon visit",
    },
    "standard": {
        "duration_min": 45,
        "pois": [
            "versailles:Trianon:grand-trianon-peristyle",
            "versailles:Trianon:grand-trianon-galerie",
            "versailles:Trianon:petit-trianon",
        ],
        "description": "Grand + Petit Trianon",
    },
    "complete": {
        "duration_min": 60,
        "pois": [
            "versailles:Trianon:grand-trianon-peristyle",
            "versailles:Trianon:grand-trianon-galerie",
            "versailles:Trianon:petit-trianon",
            "versailles:Trianon:theatre-de-la-reine",
        ],
        "description": "Complete Trianon interior tour",
    },
}


def select_trianon_route(
    available_minutes: int,
    interests: List[str],
) -> List[str]:
    """
    Select the best Trianon interior route based on available time.

    Parameters
    ----------
    available_minutes : int
        Time budget for Trianon interior
    interests : List[str]
        User interests

    Returns
    -------
    List[str]
        Ordered list of POI IDs for Trianon interior
    """
    viable_routes = {
        variant: route
        for variant, route in TRIANON_ROUTES.items()
        if route["duration_min"] <= available_minutes
    }

    if not viable_routes:
        return TRIANON_ROUTES["quick"]["pois"]

    best_variant = max(
        viable_routes.keys(),
        key=lambda v: TRIANON_ROUTES[v]["duration_min"],
    )

    return TRIANON_ROUTES[best_variant]["pois"].copy()
