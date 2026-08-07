"""
Visitor scenarios: the problem instance an itinerary is planned for.

A scenario is everything the planner is told about one visitor. It is separate
from the environment so that solvers, the RL policy and the API all consume the
same description of a request.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

from versailles.graph import ACCESSIBILITIES, PROFILES, ZONES, Accessibility, UserProfile


@dataclass
class Scenario:
    """
    One planning request.

    Parameters
    ----------
    start_time : datetime
        When the visit begins. Drives opening-hours feasibility.
    duration_minutes : int
        Total time available, travel included.
    interests : Sequence[str]
        Interest tags the visitor cares about, scoring POI utility.
    profile : UserProfile
        Walking speed: ``base``, ``family`` or ``elder``.
    accessibility : Accessibility
        ``any``, ``step_free`` or ``stroller``. Prunes both edges and POIs.
    avoid_zones : Sequence[str]
        Zones the visitor refuses ("I hate Trianon"). Hard-masked: the agent
        cannot enter them at all.
    must_include : Sequence[str]
        POI ids the itinerary must contain.
    exclude_ids : Sequence[str]
        POI ids the itinerary must not contain.
    start_poi, finish_poi : Optional[str]
        Fixed endpoints. Default to the main entrance when unset.
    """

    start_time: datetime
    duration_minutes: int
    interests: Sequence[str] = ()
    profile: UserProfile = "base"
    accessibility: Accessibility = "any"
    avoid_zones: Sequence[str] = ()
    must_include: Sequence[str] = ()
    exclude_ids: Sequence[str] = ()
    start_poi: Optional[str] = None
    finish_poi: Optional[str] = None
    name: str = "unnamed"

    def __post_init__(self) -> None:
        if self.profile not in PROFILES:
            raise ValueError(f"profile must be one of {PROFILES}, got {self.profile!r}")
        if self.accessibility not in ACCESSIBILITIES:
            raise ValueError(
                f"accessibility must be one of {ACCESSIBILITIES}, "
                f"got {self.accessibility!r}"
            )
        unknown = set(self.avoid_zones) - set(ZONES)
        if unknown:
            raise ValueError(f"unknown zone(s) in avoid_zones: {sorted(unknown)}")
        if self.duration_minutes <= 0:
            raise ValueError("duration_minutes must be positive")

    @property
    def end_time(self) -> datetime:
        return self.start_time + timedelta(minutes=self.duration_minutes)

    @property
    def allowed_zones(self) -> Optional[List[str]]:
        """Zones the visitor will accept, or None when unrestricted."""
        if not self.avoid_zones:
            return None
        return [z for z in ZONES if z not in set(self.avoid_zones)]

    def describe(self) -> str:
        parts = [
            f"{self.duration_minutes}min from {self.start_time:%a %H:%M}",
            f"profile={self.profile}",
            f"access={self.accessibility}",
        ]
        if self.interests:
            parts.append("interests=" + ",".join(self.interests))
        if self.avoid_zones:
            parts.append("avoiding=" + ",".join(self.avoid_zones))
        if self.must_include:
            parts.append(f"must_include={len(self.must_include)}")
        return " | ".join(parts)


# ----------------------------------------------------------------------
# Canonical evaluation suite
# ----------------------------------------------------------------------

_TUE_9AM = datetime(2026, 7, 28, 9, 0)  # a Tuesday in high season
_TUE_NOON = datetime(2026, 7, 28, 12, 0)


def standard_suite() -> List[Scenario]:
    """
    The fixed benchmark set. Every solver is scored on exactly these, so
    numbers stay comparable across runs and across algorithms.
    """
    return [
        Scenario(
            name="half_day_classic",
            start_time=_TUE_9AM,
            duration_minutes=240,
            interests=["history", "must_see", "art"],
        ),
        Scenario(
            name="short_highlights",
            start_time=_TUE_9AM,
            duration_minutes=120,
            interests=["must_see"],
        ),
        Scenario(
            name="garden_lover",
            start_time=_TUE_9AM,
            duration_minutes=180,
            interests=["garden", "fountains", "scenic"],
        ),
        Scenario(
            name="hates_trianon",
            start_time=_TUE_9AM,
            duration_minutes=240,
            interests=["history", "art"],
            avoid_zones=["Trianon"],
        ),
        Scenario(
            name="hates_castle",
            start_time=_TUE_9AM,
            duration_minutes=240,
            interests=["scenic", "garden"],
            avoid_zones=["Castle"],
        ),
        Scenario(
            name="afternoon_trianon",
            start_time=_TUE_NOON,
            duration_minutes=180,
            interests=["marie_antoinette", "lifestyle"],
        ),
        Scenario(
            name="stroller_family",
            start_time=_TUE_9AM,
            duration_minutes=210,
            interests=["scenic", "must_see"],
            profile="family",
            accessibility="stroller",
        ),
        Scenario(
            name="elder_step_free",
            start_time=_TUE_9AM,
            duration_minutes=150,
            interests=["history", "art"],
            profile="elder",
            accessibility="step_free",
        ),
        Scenario(
            name="full_day",
            start_time=_TUE_9AM,
            duration_minutes=480,
            interests=["history", "art", "garden", "must_see"],
        ),
        Scenario(
            name="marie_antoinette",
            start_time=_TUE_9AM,
            duration_minutes=240,
            interests=["marie_antoinette", "lifestyle", "garden"],
        ),
        Scenario(
            name="rainy_day",
            start_time=_TUE_9AM,
            duration_minutes=180,
            interests=["rain_safe", "art", "history"],
        ),
        Scenario(
            name="napoleon",
            start_time=_TUE_9AM,
            duration_minutes=150,
            interests=["napoleon_i", "art_militaire", "history"],
        ),
    ]


def training_distribution(rng, n: int = 1) -> List[Scenario]:
    """
    Sample random scenarios for RL training.

    Randomising duration, interests, profile, accessibility and zone avoidance
    is what forces the policy to generalise instead of memorising one route.
    """
    import numpy as np

    tag_pool = [
        "history", "art", "garden", "scenic", "must_see", "rain_safe",
        "fountains", "marie_antoinette", "louis_xiv", "architecture",
        "lifestyle", "napoleon_i", "photo_spot", "bosquet", "art_militaire",
    ]
    scenarios = []
    for _ in range(n):
        hour = int(rng.integers(9, 15))
        n_tags = int(rng.integers(1, 5))
        interests = list(rng.choice(tag_pool, size=n_tags, replace=False))

        avoid: List[str] = []
        roll = rng.random()
        if roll < 0.12:
            avoid = ["Trianon"]
        elif roll < 0.20:
            avoid = ["Castle"]
        elif roll < 0.24:
            avoid = ["Trianon", "Park"]

        scenarios.append(
            Scenario(
                name="train",
                start_time=_TUE_9AM.replace(hour=hour),
                duration_minutes=int(rng.integers(90, 480)),
                interests=interests,
                profile=str(rng.choice(PROFILES, p=[0.6, 0.25, 0.15])),
                accessibility=str(rng.choice(ACCESSIBILITIES, p=[0.75, 0.15, 0.10])),
                avoid_zones=avoid,
            )
        )
    return scenarios
