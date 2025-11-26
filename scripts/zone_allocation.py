"""
Smart time allocation across Versailles zones (Castle, Gardens, Trianon).

Automatically distributes visit duration based on requested zones
to ensure balanced experience.
"""

from dataclasses import dataclass
from typing import List, Set


@dataclass
class ZoneAllocation:
    """Time budget allocation across zones."""

    castle_minutes: int
    gardens_minutes: int
    trianon_minutes: int
    total_minutes: int

    def __repr__(self) -> str:
        return (
            f"ZoneAllocation(Castle={self.castle_minutes}min, "
            f"Gardens={self.gardens_minutes}min, Trianon={self.trianon_minutes}min)"
        )


def detect_requested_zones(
    interests: List[str],
    must_include: List[str],
) -> Set[str]:
    """
    Detect which zones the visitor is interested in.

    Parameters
    ----------
    interests : List[str]
        Interest tags (e.g., ["gardens", "architecture"])
    must_include : List[str]
        Explicit POI IDs user wants to visit

    Returns
    -------
    Set[str]
        Set of zones: {"castle", "gardens", "trianon"}
    """
    zones = set()

    # Detect from interests
    interest_keywords = {
        "castle": ["art", "history", "architecture", "king", "royal"],
        "gardens": ["gardens", "nature", "fountains", "outdoors", "landscape"],
        "trianon": ["trianon", "marie-antoinette", "queen", "hamlet"],
    }

    for zone, keywords in interest_keywords.items():
        if any(kw in interests for kw in keywords):
            zones.add(zone)

    # Detect from must_include POIs
    for poi_id in must_include:
        if ":Room:" in poi_id or ":Castle:" in poi_id:
            zones.add("castle")
        elif ":Garden:" in poi_id or ":Park:" in poi_id:
            zones.add("gardens")
        elif ":Trianon:" in poi_id:
            zones.add("trianon")

    # Default: if nothing detected, assume Castle + Gardens
    if not zones:
        zones = {"castle", "gardens"}

    return zones


def allocate_time_budget(
    total_minutes: int,
    requested_zones: Set[str],
) -> ZoneAllocation:
    """
    Distribute total visit time across requested zones.

    Strategy:
    - Castle + Gardens: 50/50 split
    - Castle + Gardens + Trianon: 40/40/20 split (Trianon smaller)
    - Single zone: 100% allocated

    Parameters
    ----------
    total_minutes : int
        Total available visit duration
    requested_zones : Set[str]
        Zones to visit (e.g., {"castle", "gardens"})

    Returns
    -------
    ZoneAllocation
        Time budget per zone
    """
    # Reserve 10% for transit between zones
    usable_minutes = int(total_minutes * 0.9)

    castle_min = 0
    gardens_min = 0
    trianon_min = 0

    if requested_zones == {"castle"}:
        castle_min = usable_minutes

    elif requested_zones == {"gardens"}:
        gardens_min = usable_minutes

    elif requested_zones == {"trianon"}:
        trianon_min = usable_minutes

    elif requested_zones == {"castle", "gardens"}:
        # 50/50 split
        castle_min = usable_minutes // 2
        gardens_min = usable_minutes - castle_min

    elif requested_zones == {"castle", "trianon"}:
        # 60/40 split (Castle priority)
        castle_min = int(usable_minutes * 0.6)
        trianon_min = usable_minutes - castle_min

    elif requested_zones == {"gardens", "trianon"}:
        # 60/40 split (Gardens priority)
        gardens_min = int(usable_minutes * 0.6)
        trianon_min = usable_minutes - gardens_min

    elif requested_zones == {"castle", "gardens", "trianon"}:
        # 40/40/20 split (Trianon smaller)
        castle_min = int(usable_minutes * 0.4)
        gardens_min = int(usable_minutes * 0.4)
        trianon_min = usable_minutes - castle_min - gardens_min

    return ZoneAllocation(
        castle_minutes=castle_min,
        gardens_minutes=gardens_min,
        trianon_minutes=trianon_min,
        total_minutes=total_minutes,
    )
