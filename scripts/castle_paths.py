"""
Pre-computed valid paths through Versailles Castle interior.

The Castle has a complex structure with multiple valid tour routes.
This module stores pre-computed paths to avoid expensive graph traversal
during itinerary planning.
"""

from typing import Dict, List, Tuple, Optional

# Complete Castle tour sequences
# Each sequence represents a valid walking path through the Castle

# Common trunk: All tours start with this sequence
# This is the main linear path through the Castle
CASTLE_TRUNK = [
    "versailles:Castle:cour-dhonneur",
    "versailles:Castle:acces-grands-appartements",
    "versailles:Room:galeries-de-lhistoire",
    "versailles:Room:salles-louis-xiv",
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
    "versailles:Room:salle-du-sacre",
]

# Optional detour paths (bidirectional branches off main trunk)
# These rooms can be visited as detours from specific points in the trunk

# Detour 1: Salles des Croisades (off galeries-de-lhistoire)
PATH_DETOUR_CROISADES = [
    "versailles:Room:galeries-de-lhistoire",
    "versailles:Room:salles-des-croisades",
]

# Detour 2: Chapelle Royale (off salles-louis-xiv)
PATH_DETOUR_CHAPELLE = [
    "versailles:Room:salles-louis-xiv",
    "versailles:Room:chapelle-royale",
    "versailles:Room:salon-dhercule",
]

# Detour 3: Chambre du Roi and Cabinet du Conseil (off galerie-des-glaces)
PATH_DETOUR_CHAMBRE_ROI = [
    "versailles:Room:galerie-des-glaces",
    "versailles:Room:chambre-du-roi",
    "versailles:Room:cabinet-du-conseil",
]

# Detour 4: Madame de Maintenon (off salle-du-sacre)
PATH_DETOUR_MAINTENON = [
    "versailles:Room:salle-du-sacre",
    "versailles:Room:appartements-de-madame-de-maintenon",
]

# Path 1: Galerie des Batailles exit
PATH_GALERIE_BATAILLES = CASTLE_TRUNK + [
    "versailles:Room:galerie-des-batailles",
    "versailles:Castle:pavillon-dufour-sortie",
]

# Path 2: Simple 1792 exit
PATH_SALLE_1792_SIMPLE = CASTLE_TRUNK + [
    "versailles:Room:salle-1792",
    "versailles:Castle:pavillon-dufour-sortie",
]

# Path 3: Historical rooms sequence (Napoleonic galleries)
PATH_HISTORICAL_ROOMS = CASTLE_TRUNK + [
    "versailles:Room:salle-1792",
    "versailles:Room:salle-1796-1797",
    "versailles:Room:salle-1797-1798",
    "versailles:Room:salle-1798-1799",
    "versailles:Room:salle-1802-1804",
    "versailles:Room:salle-1804",
    "versailles:Room:salle-1805-allemagne",
    "versailles:Room:salle-1805-autriche",
    "versailles:Room:salle-1805-prusse-pologne",
    "versailles:Room:salle-1805-1807-prusse",
    "versailles:Room:salle-1807-espagne",
    "versailles:Room:salle-1807-1808",
    "versailles:Room:salle-1809-1811",
    "versailles:Room:salle-marengo",
    "versailles:Room:galerie-basse",
    "versailles:Room:vestibule-de-marbre",
]

# Path 4a: Nieces apartments
PATH_APPARTEMENTS_NIECES = CASTLE_TRUNK + [
    "versailles:Room:salle-1792",
    "versailles:Castle:pavillon-dufour-sortie",
    "versailles:Castle:cour-royale",
    "versailles:Castle:entree-appartements-nieces",
    "versailles:Room:premiere-antichambre",
    "versailles:Room:deuxieme-antichambre",
    "versailles:Room:grand-cabinet-victoire",
    "versailles:Room:chambre-victoire",
    "versailles:Room:cabinet-interieur-victoire",
    "versailles:Room:cabinet-interieur-adelaide",
    "versailles:Room:chambre-adelaide",
    "versailles:Room:grand-cabinet-adelaide",
    "versailles:Room:salle-des-hoquetons",
    "versailles:Castle:pavillon-dufour-sortie",
]

# Path 4b: Dauphine apartments
PATH_APPARTEMENTS_DAUPHINE = CASTLE_TRUNK + [
    "versailles:Room:salle-1792",
    "versailles:Castle:pavillon-dufour-sortie",
    "versailles:Castle:cour-royale",
    "versailles:Castle:entree-appartements-dauphine",
    "versailles:Room:cabinet-interieur-dauphine",
    "versailles:Room:chambre-dauphine",
    "versailles:Room:grand-cabinet-dauphine",
    "versailles:Room:deuxieme-antichambre-dauphine",
    "versailles:Room:premiere-antichambre-dauphine",
    "versailles:Castle:entree-appartements-dauphine",
    "versailles:Castle:cour-royale",
    "versailles:Castle:pavillon-dufour-sortie",
]

# Create variants with detours inserted into the trunk
# This allows pathfinding from any point in trunk to any point in detours

def _create_path_variants_with_detours():
    """Generate all path variants by inserting detours at correct positions."""
    variants = [
        # Base paths
        PATH_GALERIE_BATAILLES,
        PATH_SALLE_1792_SIMPLE,
        PATH_HISTORICAL_ROOMS,
        PATH_APPARTEMENTS_NIECES,
        PATH_APPARTEMENTS_DAUPHINE,
    ]

    # For each detour, find its branch point in CASTLE_TRUNK and create variants
    # Detour 1: Croisades branches from galeries-de-lhistoire (index 2)
    galeries_idx = CASTLE_TRUNK.index("versailles:Room:galeries-de-lhistoire")
    path_with_croisades = (
        CASTLE_TRUNK[:galeries_idx+1] +
        ["versailles:Room:salles-des-croisades"] +
        CASTLE_TRUNK[galeries_idx+1:]
    )
    variants.append(path_with_croisades)

    # Detour 2: Chapelle branches from salles-louis-xiv (index 3)
    louis_xiv_idx = CASTLE_TRUNK.index("versailles:Room:salles-louis-xiv")
    salon_hercule_idx = CASTLE_TRUNK.index("versailles:Room:salon-dhercule")
    path_with_chapelle = (
        CASTLE_TRUNK[:louis_xiv_idx+1] +
        ["versailles:Room:chapelle-royale"] +
        CASTLE_TRUNK[salon_hercule_idx:]
    )
    variants.append(path_with_chapelle)

    # Detour 3: Chambre/Cabinet branches from galerie-des-glaces (index 12)
    glaces_idx = CASTLE_TRUNK.index("versailles:Room:galerie-des-glaces")
    path_with_chambre = (
        CASTLE_TRUNK[:glaces_idx+1] +
        ["versailles:Room:chambre-du-roi", "versailles:Room:cabinet-du-conseil"] +
        CASTLE_TRUNK[glaces_idx+1:]
    )
    variants.append(path_with_chambre)

    # Detour 4: Maintenon branches from salle-du-sacre (index 18)
    sacre_idx = CASTLE_TRUNK.index("versailles:Room:salle-du-sacre")
    path_with_maintenon = (
        CASTLE_TRUNK[:sacre_idx+1] +
        ["versailles:Room:appartements-de-madame-de-maintenon"] +
        CASTLE_TRUNK[sacre_idx+1:]
    )
    variants.append(path_with_maintenon)

    # Also add standalone detour paths for direct detour-to-detour routing
    variants.extend([
        PATH_DETOUR_CROISADES,
        PATH_DETOUR_CHAPELLE,
        PATH_DETOUR_CHAMBRE_ROI,
        PATH_DETOUR_MAINTENON,
    ])

    return variants

# All main paths including detours
ALL_CASTLE_PATHS = _create_path_variants_with_detours()


def _build_path_lookup() -> Dict[Tuple[str, str], List[str]]:
    """
    Build a lookup table for all valid Castle interior paths.

    Returns a dictionary mapping (start_id, end_id) to the sequential path.
    """
    lookup: Dict[Tuple[str, str], List[str]] = {}

    for main_path in ALL_CASTLE_PATHS:
        # For each main path, create lookups for all subsequences
        for i in range(len(main_path)):
            for j in range(i + 1, len(main_path)):
                start_id = main_path[i]
                end_id = main_path[j]
                path_segment = main_path[i:j+1]

                # Only store if we don't already have this pair
                # (prefer shortest path if multiple routes exist)
                key = (start_id, end_id)
                if key not in lookup or len(path_segment) < len(lookup[key]):
                    lookup[key] = path_segment

    return lookup


# Global lookup table
_CASTLE_PATH_LOOKUP = _build_path_lookup()


def get_castle_path(start_id: str, end_id: str) -> Optional[List[str]]:
    """
    Get pre-computed path between two Castle interior POIs.

    Parameters
    ----------
    start_id : str
        Starting POI ID (must be Castle or Room zone)
    end_id : str
        Ending POI ID (must be Castle or Room zone)

    Returns
    -------
    Optional[List[str]]
        Ordered list of POI IDs from start to end, or None if no path exists
    """
    return _CASTLE_PATH_LOOKUP.get((start_id, end_id))


def is_castle_interior(poi_id: str) -> bool:
    """Check if a POI is part of Castle interior (Room or Castle entrance/exit)."""
    return ":Room:" in poi_id or ":Castle:" in poi_id


def get_all_castle_pois() -> List[str]:
    """Get all Castle POIs covered by pre-computed paths."""
    all_pois = set()
    for path in ALL_CASTLE_PATHS:
        all_pois.update(path)
    return sorted(all_pois)
