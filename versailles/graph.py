"""
In-memory graph layer for the Versailles estate.

The CSVs under ``data/`` are the source of truth. This module loads them into a
NetworkX graph once, then precomputes all-pairs shortest-path travel times as
dense numpy matrices. Everything downstream (environment, solvers, API) reads
those matrices, so no query ever touches a database.

The estate has ~161 POIs, so a full 161x161 float matrix is ~200 KB. Computing
one takes milliseconds. There is no reason to keep a graph server in the loop.

Travel time depends on two things, so we build one matrix per combination:
  - user profile: base / family / elder (different walking speeds in the CSV)
  - accessibility: any / step_free / stroller (drops non-conforming edges)
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

UserProfile = str  # "base" | "family" | "elder"
Accessibility = str  # "any" | "step_free" | "stroller"

PROFILES: Tuple[UserProfile, ...] = ("base", "family", "elder")
ACCESSIBILITIES: Tuple[Accessibility, ...] = ("any", "step_free", "stroller")

_WALK_COLUMN = {
    "base": "base_walk_min",
    "family": "family_walk_min",
    "elder": "elder_walk_min",
}

# Zones as they appear in the `zone` column of pois.csv.
ZONES: Tuple[str, ...] = ("Castle", "Gardens", "Trianon", "Park")

_DAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")

# Versailles high season runs April 1 - October 31.
_HIGH_SEASON_MONTHS = frozenset(range(4, 11))

# POIs reference this ruleset but opening_hours.csv has no rows for it.
# Falling back to PALACE_DEFAULT keeps those 18 POIs schedulable; see
# `validate()` which reports the gap rather than hiding it.
_RULESET_FALLBACK = {"PALACE_SPECIAL": "PALACE_DEFAULT"}


def _parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _parse_float(value: str, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _parse_int(value: str, default: int = 0) -> int:
    return int(round(_parse_float(value, default)))


@dataclass(frozen=True)
class POI:
    """A single point of interest."""

    id: str
    name: str
    zone: str
    category: str
    interest_tags: Tuple[str, ...]
    visit_minutes: int
    accessibility_level: str
    priority_score: float
    opening_ruleset_id: str
    wing: str = ""
    garden_type: str = ""
    facility_type: str = ""
    entrance_type: str = ""

    @property
    def is_step_free(self) -> bool:
        return self.accessibility_level == "full"

    @property
    def is_transit(self) -> bool:
        """
        True for gates, entrances and exits: nodes you walk through rather than
        visit. They carry no visit time and earn no reward, but remain in the
        graph because routes must pass through them.
        """
        return self.visit_minutes <= 0


@dataclass(frozen=True)
class Connection:
    """A walkable link between two POIs."""

    from_id: str
    to_id: str
    walk_minutes: Dict[str, float]
    is_step_free: bool
    stroller_friendly: bool
    path_type: str
    notes: str = ""


@dataclass
class OpeningRule:
    """Opening window for one ruleset on one day in one season."""

    ruleset_id: str
    day: str
    season: str
    open_local: time
    close_local: time


@dataclass
class ValidationReport:
    """Result of checking the CSVs for structural problems."""

    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def __str__(self) -> str:
        lines = []
        for e in self.errors:
            lines.append(f"ERROR   {e}")
        for w in self.warnings:
            lines.append(f"WARNING {w}")
        if not lines:
            lines.append("Graph data is clean.")
        return "\n".join(lines)


def _parse_time(value: str) -> time:
    hour, minute = value.strip().split(":")
    return time(int(hour), int(minute))


def _season_for(day: date) -> str:
    return "High" if day.month in _HIGH_SEASON_MONTHS else "Low"


class VersaillesGraph:
    """
    The estate as an in-memory graph plus precomputed travel-time matrices.

    Attributes
    ----------
    pois : Dict[str, POI]
        Every POI keyed by id.
    order : List[str]
        Canonical POI ordering. Row/column ``i`` of every matrix is ``order[i]``.
    index : Dict[str, int]
        Inverse of ``order``.
    """

    def __init__(
        self,
        pois: Dict[str, POI],
        connections: List[Connection],
        opening_rules: List[OpeningRule],
    ) -> None:
        self.pois = pois
        self.connections = connections
        self.opening_rules = opening_rules

        self.order: List[str] = sorted(pois)
        self.index: Dict[str, int] = {pid: i for i, pid in enumerate(self.order)}
        self.n = len(self.order)

        self._rules_by_key: Dict[Tuple[str, str, str], OpeningRule] = {
            (r.ruleset_id, r.day, r.season): r for r in opening_rules
        }

        self._graphs: Dict[Accessibility, nx.Graph] = {}
        self._matrices: Dict[Tuple[UserProfile, Accessibility], np.ndarray] = {}

        # Vectorised POI attributes, aligned with `self.order`.
        self.visit_minutes = np.array(
            [pois[p].visit_minutes for p in self.order], dtype=np.float32
        )
        self.priority = np.array(
            [pois[p].priority_score for p in self.order], dtype=np.float32
        )
        self.zone_of = np.array([pois[p].zone for p in self.order], dtype=object)
        # Gates and entrances are traversable but not collectable.
        self.is_transit = np.array(
            [pois[p].is_transit for p in self.order], dtype=bool
        )
        self.is_visitable = ~self.is_transit

        self.all_tags: List[str] = sorted(
            {t for p in pois.values() for t in p.interest_tags}
        )
        self._tag_index = {t: i for i, t in enumerate(self.all_tags)}
        self.tag_matrix = np.zeros((self.n, len(self.all_tags)), dtype=np.float32)
        for pid, poi in pois.items():
            row = self.index[pid]
            for tag in poi.interest_tags:
                self.tag_matrix[row, self._tag_index[tag]] = 1.0

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_csv(cls, data_dir: Path | str = DATA_DIR) -> "VersaillesGraph":
        data_dir = Path(data_dir)
        pois = _load_pois(data_dir / "main_data" / "pois.csv")
        connections = _load_connections(data_dir / "main_data" / "connections.csv")
        rules = _load_opening_hours(data_dir / "main_data" / "opening_hours.csv")
        return cls(pois, connections, rules)

    # ------------------------------------------------------------------
    # Graph views
    # ------------------------------------------------------------------

    def graph(self, accessibility: Accessibility = "any") -> nx.Graph:
        """NetworkX view of the estate under an accessibility constraint."""
        if accessibility not in ACCESSIBILITIES:
            raise ValueError(f"unknown accessibility {accessibility!r}")
        if accessibility in self._graphs:
            return self._graphs[accessibility]

        g = nx.Graph()
        for pid, poi in self.pois.items():
            g.add_node(pid, **{"zone": poi.zone, "name": poi.name})

        for conn in self.connections:
            if conn.from_id not in self.pois or conn.to_id not in self.pois:
                continue
            if accessibility == "step_free" and not conn.is_step_free:
                continue
            if accessibility == "stroller" and not conn.stroller_friendly:
                continue
            # Keep the cheaper edge if the CSV lists a pair twice.
            existing = g.get_edge_data(conn.from_id, conn.to_id)
            if existing and existing["base"] <= conn.walk_minutes["base"]:
                continue
            g.add_edge(
                conn.from_id,
                conn.to_id,
                path_type=conn.path_type,
                **conn.walk_minutes,
            )

        self._graphs[accessibility] = g
        return g

    def travel_matrix(
        self,
        profile: UserProfile = "base",
        accessibility: Accessibility = "any",
    ) -> np.ndarray:
        """
        All-pairs shortest walking time in minutes.

        Unreachable pairs are ``np.inf``. The result is cached per
        (profile, accessibility) pair.
        """
        key = (profile, accessibility)
        if key in self._matrices:
            return self._matrices[key]
        if profile not in PROFILES:
            raise ValueError(f"unknown profile {profile!r}")

        g = self.graph(accessibility)
        matrix = np.full((self.n, self.n), np.inf, dtype=np.float32)
        np.fill_diagonal(matrix, 0.0)

        for source, dists in nx.all_pairs_dijkstra_path_length(g, weight=profile):
            i = self.index[source]
            for target, dist in dists.items():
                matrix[i, self.index[target]] = dist

        self._matrices[key] = matrix
        return matrix

    def shortest_path(
        self,
        from_id: str,
        to_id: str,
        profile: UserProfile = "base",
        accessibility: Accessibility = "any",
    ) -> Optional[List[str]]:
        """Node-by-node walking route, or None when no route exists."""
        g = self.graph(accessibility)
        if from_id not in g or to_id not in g:
            return None
        try:
            return nx.shortest_path(g, from_id, to_id, weight=profile)
        except nx.NetworkXNoPath:
            return None

    # ------------------------------------------------------------------
    # Opening hours
    # ------------------------------------------------------------------

    def opening_window(
        self, poi_id: str, when: date
    ) -> Optional[Tuple[time, time]]:
        """
        Opening window for a POI on a given date, or None when closed.

        Falls back to a default ruleset for POIs whose ruleset has no rows in
        opening_hours.csv (see ``_RULESET_FALLBACK``).
        """
        poi = self.pois[poi_id]
        ruleset = poi.opening_ruleset_id or ""
        day_name = _DAY_NAMES[when.weekday()]
        season = _season_for(when)

        for candidate in (ruleset, _RULESET_FALLBACK.get(ruleset)):
            if not candidate:
                continue
            rule = self._rules_by_key.get((candidate, day_name, season))
            if rule:
                return rule.open_local, rule.close_local
        return None

    def is_open_at(self, poi_id: str, when: datetime) -> bool:
        window = self.opening_window(poi_id, when.date())
        if window is None:
            return False
        opens, closes = window
        return opens <= when.time() <= closes

    def open_mask(self, when: datetime) -> np.ndarray:
        """Boolean mask over ``self.order``: True where the POI is open."""
        return np.array(
            [self.is_open_at(pid, when) for pid in self.order], dtype=bool
        )

    # ------------------------------------------------------------------
    # Utility scoring
    # ------------------------------------------------------------------

    def interest_utility(self, interests: Sequence[str]) -> np.ndarray:
        """
        Per-POI utility given a visitor's declared interests.

        Combines the curated ``priority_score`` with how many of the visitor's
        interest tags the POI matches. A POI with no matching tag keeps a small
        floor so that a highly-rated landmark is not scored at zero just because
        the visitor did not tick its category.
        """
        base = self.priority.copy()
        if not interests:
            return base

        wanted = np.zeros(len(self.all_tags), dtype=np.float32)
        for tag in interests:
            if tag in self._tag_index:
                wanted[self._tag_index[tag]] = 1.0

        matches = self.tag_matrix @ wanted
        return base * (0.25 + matches)

    def zone_mask(self, allowed_zones: Optional[Iterable[str]]) -> np.ndarray:
        """Boolean mask selecting POIs in the allowed zones (None = all)."""
        if allowed_zones is None:
            return np.ones(self.n, dtype=bool)
        allowed = set(allowed_zones)
        return np.array([z in allowed for z in self.zone_of], dtype=bool)

    def accessible_mask(self, accessibility: Accessibility) -> np.ndarray:
        """Boolean mask of POIs a visitor with this requirement can enter."""
        if accessibility == "any":
            return np.ones(self.n, dtype=bool)
        return np.array(
            [self.pois[p].accessibility_level == "full" for p in self.order],
            dtype=bool,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> ValidationReport:
        """Check the loaded data for problems that would break planning."""
        report = ValidationReport()

        known = set(self.pois)
        for conn in self.connections:
            for endpoint in (conn.from_id, conn.to_id):
                if endpoint not in known:
                    report.errors.append(
                        f"connection {conn.from_id} -> {conn.to_id} references "
                        f"unknown POI {endpoint!r}"
                    )

        g = self.graph("any")
        isolated = [p for p in self.order if g.degree(p) == 0]
        if isolated:
            report.errors.append(
                f"{len(isolated)} POI(s) have no connections: {isolated[:5]}"
            )

        components = list(nx.connected_components(g))
        if len(components) > 1:
            sizes = sorted((len(c) for c in components), reverse=True)
            report.errors.append(
                f"graph splits into {len(components)} disconnected components "
                f"(sizes {sizes}); routes cannot cross between them"
            )

        referenced = {p.opening_ruleset_id for p in self.pois.values() if p.opening_ruleset_id}
        defined = {r.ruleset_id for r in self.opening_rules}
        for missing in sorted(referenced - defined):
            count = sum(
                1 for p in self.pois.values() if p.opening_ruleset_id == missing
            )
            fallback = _RULESET_FALLBACK.get(missing)
            detail = f"falling back to {fallback}" if fallback else "POIs treated as closed"
            report.warnings.append(
                f"ruleset {missing!r} used by {count} POI(s) has no rows in "
                f"opening_hours.csv; {detail}"
            )

        for poi in self.pois.values():
            if poi.visit_minutes < 0:
                report.warnings.append(
                    f"POI {poi.id} has negative visit_minutes={poi.visit_minutes}"
                )

        for accessibility in ("step_free", "stroller"):
            sub = self.graph(accessibility)
            parts = [c for c in nx.connected_components(sub) if len(c) > 1]
            if len(parts) > 1:
                report.warnings.append(
                    f"under accessibility={accessibility!r} the graph splits into "
                    f"{len(parts)} reachable clusters; some POIs are unroutable"
                )

        return report

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"VersaillesGraph(pois={self.n}, connections={len(self.connections)}, "
            f"tags={len(self.all_tags)})"
        )


# ----------------------------------------------------------------------
# CSV loaders
# ----------------------------------------------------------------------


def _load_pois(path: Path) -> Dict[str, POI]:
    pois: Dict[str, POI] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            pid = (row.get("id") or "").strip()
            if not pid:
                continue
            tags = tuple(
                t.strip()
                for t in (row.get("interest_tags") or "").split(";")
                if t.strip()
            )
            pois[pid] = POI(
                id=pid,
                name=(row.get("name") or pid).strip(),
                zone=(row.get("zone") or "").strip(),
                category=(row.get("category") or "").strip(),
                interest_tags=tags,
                visit_minutes=_parse_int(row.get("estimated_visit_minutes"), 10),
                accessibility_level=(row.get("accessibility_level") or "").strip(),
                priority_score=_parse_float(row.get("priority_score"), 0.0),
                opening_ruleset_id=(row.get("opening_ruleset_id") or "").strip(),
                wing=(row.get("wing") or "").strip(),
                garden_type=(row.get("garden_type") or "").strip(),
                facility_type=(row.get("facility_type") or "").strip(),
                entrance_type=(row.get("entrance_type") or "").strip(),
            )
    return pois


def _load_connections(path: Path) -> List[Connection]:
    connections: List[Connection] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            from_id = (row.get("from_id") or "").strip()
            to_id = (row.get("to_id") or "").strip()
            if not from_id or not to_id:
                continue
            walk = {
                profile: _parse_float(row.get(column), 1.0)
                for profile, column in _WALK_COLUMN.items()
            }
            connections.append(
                Connection(
                    from_id=from_id,
                    to_id=to_id,
                    walk_minutes=walk,
                    is_step_free=_parse_bool(row.get("is_step_free")),
                    stroller_friendly=_parse_bool(row.get("stroller_friendly")),
                    path_type=(row.get("path_type") or "").strip(),
                    notes=(row.get("notes") or "").strip(),
                )
            )
    return connections


def _load_opening_hours(path: Path) -> List[OpeningRule]:
    rules: List[OpeningRule] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            ruleset = (row.get("ruleset_id") or "").strip()
            if not ruleset:
                continue
            rules.append(
                OpeningRule(
                    ruleset_id=ruleset,
                    day=(row.get("day") or "").strip(),
                    season=(row.get("season") or "").strip(),
                    open_local=_parse_time(row["open_local"]),
                    close_local=_parse_time(row["close_local"]),
                )
            )
    return rules


_CACHED: Optional[VersaillesGraph] = None


def load_graph(data_dir: Path | str = DATA_DIR, refresh: bool = False) -> VersaillesGraph:
    """Load the estate graph, reusing the process-wide instance by default."""
    global _CACHED
    if _CACHED is None or refresh:
        _CACHED = VersaillesGraph.from_csv(data_dir)
        logger.info("loaded %r", _CACHED)
    return _CACHED
