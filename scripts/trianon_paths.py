"""
Predefined paths for Grand Trianon and Petit Trianon interiors.

These are linear, curated paths (not TSP).
After completing the interior path, TSP is applied to garden POIs.
"""

from typing import List, Optional

# Grand Trianon interior path (single route)
# Path as specified by user: entree -> boudoir -> salon des glaces -> chambre ->
# salon chapelle -> salon seigneurs -> salon rond -> salon famille empereur ->
# chambre reine belges -> salon musique -> salon famille louis philippe ->
# salon malachite -> cabinet topographique -> salon frais -> galerie cotelle ->
# salon jardins -> premier salon (TSP starts here, NEVER return to salon-des-jardins)
GRAND_TRIANON_PATH: List[str] = [
    "versailles:Trianon:entree-grand-trianon",
    "versailles:Trianon:boudoir-imperatrice-Marie-Louise",
    "versailles:Trianon:salon-des-glaces",
    "versailles:Trianon:chambre-de-limperatrice",
    "versailles:Trianon:salon-de-la-chapelle",
    "versailles:Trianon:salon-des-seigneurs",
    "versailles:Trianon:salon-rond",
    "versailles:Trianon:salon-de-famille-de-lempereur",
    "versailles:Trianon:chambre-reine-des-belges",
    "versailles:Trianon:salon-de-musique",
    "versailles:Trianon:salon-famille-louis-philippe",
    "versailles:Trianon:salon-malachite",
    "versailles:Trianon:cabinet-topographique",
    "versailles:Trianon:salon-frais",
    "versailles:Trianon:galerie-des-cotelle",
    "versailles:Trianon:salon-des-jardins",  # DO NOT return to this POI after TSP starts
    "versailles:Trianon:premier-salon",       # TSP starts from here
]

# Petit Trianon - Accessible path (step-free, no stairs)
PETIT_TRIANON_PATH_ACCESSIBLE: List[str] = [
    "versailles:Trianon:entree-petit-trianon",
    "versailles:Trianon:salle-des-gardes-petit-trianon",
    "versailles:Trianon:salon-du-billard-petit-trianon",
    "versailles:Trianon:piece-dargenterie-petit-trianon",
    "versailles:Trianon:salle-glaces-mouvantes-petit-trianon",
    "versailles:Trianon:les-deux-fruiteries",
    "versailles:Trianon:escalier-dhonneur-petit-trianon",  # View only, do not climb
    # TSP starts from here (back to entrance area for gardens access)
]

# Petit Trianon - Full path (includes upper floor via stairs)
PETIT_TRIANON_PATH_FULL: List[str] = [
    "versailles:Trianon:entree-petit-trianon",
    "versailles:Trianon:salle-des-gardes-petit-trianon",
    "versailles:Trianon:salon-du-billard-petit-trianon",
    "versailles:Trianon:piece-dargenterie-petit-trianon",
    "versailles:Trianon:salle-glaces-mouvantes-petit-trianon",
    "versailles:Trianon:les-deux-fruiteries",
    "versailles:Trianon:escalier-dhonneur-petit-trianon",
    "versailles:Trianon:antichambre-petit-trianon",
    "versailles:Trianon:grande-salle-a-manger-petit-trianon",
    "versailles:Trianon:petite-salle-a-manger-petit-trianon",
    "versailles:Trianon:salon-de-compagnie-petit-trianon",
    "versailles:Trianon:chambre-a-coucher-de-la-reine-petit-trianon",
    "versailles:Trianon:boudoir-des-glaces-mouvantes-petit-trianon",
    "versailles:Trianon:cabinet-de-toilette-de-la-reine-petit-trianon",
    "versailles:Trianon:entree-petit-trianon",  # Returns to entrance
    # TSP starts from here
]


def get_petit_trianon_path(accessibility_required: bool) -> List[str]:
    """
    Get Petit Trianon interior path based on accessibility requirements.

    Parameters
    ----------
    accessibility_required : bool
        True if step-free/stroller access is required

    Returns
    -------
    List[str]
        Ordered POI IDs for Petit Trianon interior
    """
    if accessibility_required:
        return PETIT_TRIANON_PATH_ACCESSIBLE.copy()
    else:
        return PETIT_TRIANON_PATH_FULL.copy()


def get_grand_trianon_path() -> List[str]:
    """
    Get Grand Trianon interior path.

    Returns
    -------
    List[str]
        Ordered POI IDs for Grand Trianon interior
    """
    return GRAND_TRIANON_PATH.copy()


def get_trianon_tsp_start(path: List[str]) -> str:
    """
    Get the POI ID where TSP should start after completing interior path.

    Parameters
    ----------
    path : List[str]
        The completed interior path

    Returns
    -------
    str
        POI ID to start TSP from
    """
    return path[-1]  # Last POI of interior path


def get_trianon_exclusions(path: List[str]) -> List[str]:
    """
    Get POI IDs that should be excluded from TSP after interior visit.

    For Grand Trianon: exclude "salon-des-jardins" (must not return there).
    For Petit Trianon: no exclusions.

    Parameters
    ----------
    path : List[str]
        The completed interior path

    Returns
    -------
    List[str]
        POI IDs to exclude from TSP
    """
    # Grand Trianon: exclude salon-des-jardins
    if "versailles:Trianon:salon-des-jardins" in path:
        return ["versailles:Trianon:salon-des-jardins"]

    return []
