"""
Smart time allocation across Versailles zones (Castle, Gardens, Trianon).

Automatically distributes visit duration based on requested zones
to ensure balanced experience.
"""

from dataclasses import dataclass
from datetime import datetime, time
from typing import List, Set, Optional


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
    # Note: "art" and "history" trigger BOTH castle AND gardens
    # (gardens have sculptures, fountains, and historical significance)
    interest_keywords = {
        "castle": ["art", "history", "architecture", "king", "royal"],
        "gardens": ["gardens", "nature", "fountains", "outdoors", "landscape", "art", "history"],
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
    zone_complexity: Optional[dict] = None,
) -> ZoneAllocation:
    """
    Distribute total visit time across requested zones adaptively.

    Strategy:
    - Allocate proportionally based on zone complexity (POI count/density)
    - If no complexity data: use sensible defaults
    - Ensure minimum time per zone (30 min minimum)

    Parameters
    ----------
    total_minutes : int
        Total available visit duration
    requested_zones : Set[str]
        Zones to visit (e.g., {"castle", "gardens"})
    zone_complexity : Optional[dict]
        Complexity scores per zone: {"castle": 100, "gardens": 80, "trianon": 60}
        Higher = more POIs/content/time needed

    Returns
    -------
    ZoneAllocation
        Time budget per zone
    """
    # Reserve 10% for transit between zones
    usable_minutes = int(total_minutes * 0.9)

    # Default complexity if not provided
    if zone_complexity is None:
        zone_complexity = {
            "castle": 100,  # Dense interior, many rooms
            "gardens": 80,   # Large but sparser
            "trianon": 70,   # Interior + gardens, medium
        }

    castle_min = 0
    gardens_min = 0
    trianon_min = 0

    # Single zone: allocate all time
    if len(requested_zones) == 1:
        zone = list(requested_zones)[0]
        if zone == "castle":
            castle_min = usable_minutes
        elif zone == "gardens":
            gardens_min = usable_minutes
        elif zone == "trianon":
            trianon_min = usable_minutes

    else:
        # Multi-zone: proportional allocation based on complexity
        total_complexity = sum(zone_complexity.get(z, 50) for z in requested_zones)

        for zone in requested_zones:
            complexity = zone_complexity.get(zone, 50)
            proportion = complexity / total_complexity
            allocated = int(usable_minutes * proportion)

            # Enforce minimum 30 minutes per zone
            allocated = max(allocated, 30)

            if zone == "castle":
                castle_min = allocated
            elif zone == "gardens":
                gardens_min = allocated
            elif zone == "trianon":
                trianon_min = allocated

        # Adjust if we exceeded budget due to minimums
        total_allocated = castle_min + gardens_min + trianon_min
        if total_allocated > usable_minutes:
            # Scale down proportionally
            scale = usable_minutes / total_allocated
            castle_min = int(castle_min * scale)
            gardens_min = int(gardens_min * scale)
            trianon_min = usable_minutes - castle_min - gardens_min

    return ZoneAllocation(
        castle_minutes=castle_min,
        gardens_minutes=gardens_min,
        trianon_minutes=trianon_min,
        total_minutes=total_minutes,
    )


def determine_zone_order(
    requested_zones: Set[str],
    start_time: Optional[datetime] = None,
) -> List[str]:
    """
    Determine optimal visit order for zones based on time of day.

    Strategy:
    - Morning (before 12:00): Castle → Gardens → Trianon (default order)
    - Afternoon (12:00-16:00): Trianon → Gardens → Castle (Trianon closes earlier)
    - Late afternoon (after 16:00): Castle → Gardens (skip Trianon if closing soon)

    Parameters
    ----------
    requested_zones : Set[str]
        Zones to visit
    start_time : Optional[datetime]
        Visit start time (if None, use default order)

    Returns
    -------
    List[str]
        Ordered list of zones to visit
    """
    if start_time is None:
        # Default order: Castle → Gardens → Trianon
        return _default_zone_order(requested_zones)

    hour = start_time.hour

    # Morning visit (before 12:00)
    if hour < 12:
        return _default_zone_order(requested_zones)

    # Afternoon visit (12:00-16:00): prioritize Trianon
    elif 12 <= hour < 16:
        if "trianon" in requested_zones:
            # Start with Trianon (closes earlier, farther from entrance)
            order = ["trianon"]
            if "gardens" in requested_zones:
                order.append("gardens")
            if "castle" in requested_zones:
                order.append("castle")
            return order
        else:
            return _default_zone_order(requested_zones)

    # Late afternoon (after 16:00): skip Trianon if included
    else:
        # Too late for Trianon, focus on Castle + Gardens
        zones_copy = requested_zones.copy()
        zones_copy.discard("trianon")  # Remove Trianon
        return _default_zone_order(zones_copy)


def _default_zone_order(zones: Set[str]) -> List[str]:
    """Return default visit order: Castle → Gardens → Trianon."""
    order = []
    if "castle" in zones:
        order.append("castle")
    if "gardens" in zones:
        order.append("gardens")
    if "trianon" in zones:
        order.append("trianon")
    return order
