"""
Network configuration for Versailles itinerary planner.
Defines valid start points (entrances) and finish points (exits).
"""

# Valid entrance points (start of itinerary)
ENTRANCE_POINTS = {
    # Castle entrances
    "versailles:Castle:cour-dhonneur": {
        "name": "Main Palace Entrance (Cour d'Honneur)",
        "zone": "Castle",
        "default": True,
    },

    # Garden entrances
    "versailles:Garden:acces-jardins-cour-des-princes": {
        "name": "Gardens Entrance (Cour des Princes)",
        "zone": "Garden",
    },
    "versailles:Garden:entree-parc-bassin-apollon": {
        "name": "Park Entrance (Basin of Apollo)",
        "zone": "Garden",
    },
    "versailles:Garden:grille-de-neptune": {
        "name": "Neptune Gate",
        "zone": "Garden",
    },

    # Trianon entrances
    "versailles:Trianon:entree-grand-trianon": {
        "name": "Grand Trianon Entrance",
        "zone": "Trianon",
    },
    "versailles:Trianon:entree-petit-trianon": {
        "name": "Petit Trianon Entrance",
        "zone": "Trianon",
    },
}

# Valid exit points (end of itinerary)
EXIT_POINTS = {
    # Castle exits
    "versailles:Castle:cour-dhonneur-sortie": {
        "name": "Palace Exit (Cour d'Honneur)",
        "zone": "Castle",
        "default": True,
    },

    # Garden exits
    "versailles:Garden:entree-parc-bassin-apollon": {
        "name": "Basin of Apollo Exit",
        "zone": "Garden",
    },
    "versailles:Garden:grille-de-neptune": {
        "name": "Neptune Gate Exit",
        "zone": "Garden",
    },

    # Trianon exits
    "versailles:Trianon:entree-grand-trianon": {
        "name": "Grand Trianon Exit",
        "zone": "Trianon",
    },
    "versailles:Trianon:entree-petit-trianon": {
        "name": "Petit Trianon Exit",
        "zone": "Trianon",
    },
}

# Default entrance/exit based on zone
DEFAULT_ENTRANCE = "versailles:Castle:cour-dhonneur"
DEFAULT_EXIT = "versailles:Castle:cour-dhonneur-sortie"

# Zone-specific defaults
ZONE_ENTRANCES = {
    "Castle": "versailles:Castle:cour-dhonneur",
    "Garden": "versailles:Garden:acces-jardins-cour-des-princes",
    "Park": "versailles:Garden:entree-parc-bassin-apollon",
    "Trianon": "versailles:Trianon:entree-grand-trianon",
}

ZONE_EXITS = {
    "Castle": "versailles:Castle:cour-dhonneur-sortie",
    "Garden": "versailles:Garden:grille-de-neptune",
    "Park": "versailles:Garden:entree-parc-bassin-apollon",
    "Trianon": "versailles:Trianon:entree-grand-trianon",
}


def get_entrance_for_pois(poi_ids: list[str]) -> str:
    """
    Determine best entrance based on POI zones.

    Args:
        poi_ids: List of POI IDs to visit

    Returns:
        Best entrance POI ID
    """
    # Count POIs by zone
    zone_counts = {}
    for poi_id in poi_ids:
        if "Trianon" in poi_id:
            zone_counts["Trianon"] = zone_counts.get("Trianon", 0) + 1
        elif "Garden" in poi_id or "Park" in poi_id:
            zone_counts["Garden"] = zone_counts.get("Garden", 0) + 1
        elif "Castle" in poi_id:
            zone_counts["Castle"] = zone_counts.get("Castle", 0) + 1

    # Return entrance for most common zone
    if not zone_counts:
        return DEFAULT_ENTRANCE

    most_common_zone = max(zone_counts, key=zone_counts.get)
    return ZONE_ENTRANCES.get(most_common_zone, DEFAULT_ENTRANCE)


def get_exit_for_pois(poi_ids: list[str]) -> str:
    """
    Determine best exit based on POI zones.

    Args:
        poi_ids: List of POI IDs to visit

    Returns:
        Best exit POI ID
    """
    # Count POIs by zone
    zone_counts = {}
    for poi_id in poi_ids:
        if "Trianon" in poi_id:
            zone_counts["Trianon"] = zone_counts.get("Trianon", 0) + 1
        elif "Garden" in poi_id or "Park" in poi_id:
            zone_counts["Garden"] = zone_counts.get("Garden", 0) + 1
        elif "Castle" in poi_id:
            zone_counts["Castle"] = zone_counts.get("Castle", 0) + 1

    # Return exit for most common zone
    if not zone_counts:
        return DEFAULT_EXIT

    most_common_zone = max(zone_counts, key=zone_counts.get)
    return ZONE_EXITS.get(most_common_zone, DEFAULT_EXIT)


def validate_entrance(poi_id: str) -> bool:
    """Check if POI ID is a valid entrance."""
    return poi_id in ENTRANCE_POINTS


def validate_exit(poi_id: str) -> bool:
    """Check if POI ID is a valid exit."""
    return poi_id in EXIT_POINTS


def get_all_entrances() -> list[str]:
    """Get list of all valid entrance POI IDs."""
    return list(ENTRANCE_POINTS.keys())


def get_all_exits() -> list[str]:
    """Get list of all valid exit POI IDs."""
    return list(EXIT_POINTS.keys())
