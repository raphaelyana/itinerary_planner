"""
Gymnasium environment for the Versailles Orienteering Problem with Time Windows.

This is *not* a TSP. A TSP orders a fixed set of stops; here the hard part is
choosing which stops are worth the time at all. The agent starts at an entrance
with a time budget, repeatedly picks the next POI to visit, and stops when it
chooses to or when nothing feasible remains. It must be able to reach the exit
before the budget runs out.

Constraints are enforced as **hard action masks**, never as penalties. The agent
physically cannot emit an itinerary that walks through a closed door, uses a
staircase in a wheelchair, or enters a zone the visitor refused. A masked
formulation means every rollout, including a randomly initialised one, is a
route a real visitor could follow.

Reward per step:
    utility(poi) - travel_penalty * travel_minutes - wait_penalty * wait_minutes

where utility blends the curated ``priority_score`` with interest-tag matches.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from versailles.graph import ZONES, VersaillesGraph, load_graph
from versailles.scenario import Scenario, training_distribution

if TYPE_CHECKING:  # avoids a circular import at runtime
    from versailles.utility import UtilityModel

# Default entrance/exit when a scenario does not pin them.
DEFAULT_START_POI = "versailles:Castle:cour-dhonneur"

# Feature layout for each POI row in the observation.
N_ZONE_FEATURES = len(ZONES)
N_POI_FEATURES = 7 + N_ZONE_FEATURES


@dataclass
class StepRecord:
    """One visited stop, with the timing actually achieved."""

    poi_id: str
    name: str
    arrival_minute: float
    wait_minutes: float
    visit_minutes: float
    departure_minute: float
    travel_minutes: float
    utility: float


@dataclass
class EpisodeResult:
    """Summary of a completed episode, used by solvers and evaluation alike."""

    scenario: Scenario
    steps: List[StepRecord]
    total_utility: float
    total_travel: float
    total_wait: float
    total_visit: float
    elapsed_minutes: float
    feasible: bool = True
    violations: List[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.violations is None:
            self.violations = []

    @property
    def poi_ids(self) -> List[str]:
        return [s.poi_id for s in self.steps]

    @property
    def n_visited(self) -> int:
        return len(self.steps)

    def summary(self) -> str:
        return (
            f"{self.n_visited} POIs | utility {self.total_utility:.2f} | "
            f"travel {self.total_travel:.0f}min | wait {self.total_wait:.0f}min | "
            f"elapsed {self.elapsed_minutes:.0f}/{self.scenario.duration_minutes}min"
        )


class VersaillesEnv(gym.Env):
    """
    OPTW environment over the Versailles estate.

    Actions
    -------
    ``0 .. n-1``  travel to that POI and visit it
    ``n``         stop and walk to the exit

    Observations
    ------------
    Dict with:
      ``pois``   (n, N_POI_FEATURES) per-POI features
      ``global`` (4,) budget and progress signals
      ``mask``   (n+1,) bool, True where the action is legal
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        graph: Optional[VersaillesGraph] = None,
        scenario: Optional[Scenario] = None,
        *,
        utility_model: Optional["UtilityModel"] = None,
        travel_penalty: Optional[float] = None,
        wait_penalty: Optional[float] = None,
        max_wait_minutes: float = 45.0,
        randomize: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.graph = graph if graph is not None else load_graph()
        self.n = self.graph.n
        self._fixed_scenario = scenario
        self.randomize = randomize

        # The utility model owns both the per-POI value and the minutes-to-value
        # exchange rates, since tuning one without the other is meaningless.
        from versailles.utility import DEFAULT_UTILITY

        self.utility_model = utility_model or DEFAULT_UTILITY
        self.travel_penalty = (
            travel_penalty
            if travel_penalty is not None
            else self.utility_model.travel_weight
        )
        self.wait_penalty = (
            wait_penalty
            if wait_penalty is not None
            else self.utility_model.wait_weight
        )
        self.max_wait_minutes = max_wait_minutes

        self._rng = np.random.default_rng(seed)

        self.action_space = spaces.Discrete(self.n + 1)
        self.observation_space = spaces.Dict(
            {
                "pois": spaces.Box(
                    low=-1.0, high=np.inf, shape=(self.n, N_POI_FEATURES), dtype=np.float32
                ),
                "global": spaces.Box(low=-1.0, high=np.inf, shape=(4,), dtype=np.float32),
                "mask": spaces.Box(low=0, high=1, shape=(self.n + 1,), dtype=np.int8),
            }
        )

        self._zone_onehot = np.zeros((self.n, N_ZONE_FEATURES), dtype=np.float32)
        for i, zone in enumerate(self.graph.zone_of):
            if zone in ZONES:
                self._zone_onehot[i, ZONES.index(zone)] = 1.0

        self.scenario: Scenario
        self.reset(seed=seed)

    # ------------------------------------------------------------------
    # Episode setup
    # ------------------------------------------------------------------

    def _resolve_scenario(self) -> Scenario:
        if self.randomize:
            return training_distribution(self._rng, 1)[0]
        if self._fixed_scenario is None:
            raise ValueError("VersaillesEnv needs either scenario= or randomize=True")
        return self._fixed_scenario

    def _prepare(self, scenario: Scenario) -> None:
        """Precompute everything that is constant for this scenario."""
        g = self.graph
        self.scenario = scenario

        self.distance = g.travel_matrix(scenario.profile, scenario.accessibility)
        self.utility = self.utility_model.values(g, scenario.interests)

        # Opening windows in minutes from the scenario start.
        day = scenario.start_time.date()
        start_min = scenario.start_time.hour * 60 + scenario.start_time.minute
        opens = np.full(self.n, np.inf, dtype=np.float32)
        closes = np.full(self.n, -np.inf, dtype=np.float32)
        for i, pid in enumerate(g.order):
            window = g.opening_window(pid, day)
            if window is None:
                continue
            o, c = window
            opens[i] = o.hour * 60 + o.minute - start_min
            closes[i] = c.hour * 60 + c.minute - start_min
        self.opens_at = opens
        self.closes_at = closes

        # Static eligibility: everything not dependent on the current position.
        eligible = g.is_visitable.copy()
        eligible &= g.zone_mask(scenario.allowed_zones)
        eligible &= g.accessible_mask(scenario.accessibility)
        if scenario.exclude_ids:
            for pid in scenario.exclude_ids:
                if pid in g.index:
                    eligible[g.index[pid]] = False
        eligible &= np.isfinite(opens)  # no known hours -> treated as closed
        self.eligible = eligible

        start_id = scenario.start_poi or DEFAULT_START_POI
        finish_id = scenario.finish_poi or start_id
        if start_id not in g.index:
            raise ValueError(f"unknown start_poi {start_id!r}")
        if finish_id not in g.index:
            raise ValueError(f"unknown finish_poi {finish_id!r}")
        self.start_idx = g.index[start_id]
        self.finish_idx = g.index[finish_id]

        self.budget = float(scenario.duration_minutes)
        self._max_utility = float(self.utility[eligible].sum()) or 1.0

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        scenario = (options or {}).get("scenario") or self._resolve_scenario()
        self._prepare(scenario)

        self.visited = np.zeros(self.n, dtype=bool)
        self.current = self.start_idx
        self.clock = 0.0
        self.steps: List[StepRecord] = []
        self.total_travel = 0.0
        self.total_wait = 0.0
        self.total_visit = 0.0
        self.total_utility = 0.0
        self.done = False

        return self._observe(), {"scenario": scenario.name}

    def step(
        self, action: int
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        if self.done:
            raise RuntimeError("step() called on a finished episode; call reset()")

        mask = self.action_mask()
        if not mask[action]:
            # Should be unreachable: callers must sample from the mask. Ending
            # the episode with a penalty is safer than silently corrupting the
            # itinerary, and makes the bug visible in training curves.
            self.done = True
            return self._observe(), -1.0, True, False, {"invalid_action": True}

        if action == self.n:  # stop
            self.done = True
            return self._observe(), 0.0, True, False, {"stopped": True}

        travel = float(self.distance[self.current, action])
        arrival = self.clock + travel
        wait = max(0.0, float(self.opens_at[action]) - arrival)
        start_visit = arrival + wait
        visit = float(self.graph.visit_minutes[action])
        departure = start_visit + visit

        poi_id = self.graph.order[action]
        utility = float(self.utility[action])

        self.steps.append(
            StepRecord(
                poi_id=poi_id,
                name=self.graph.pois[poi_id].name,
                arrival_minute=arrival,
                wait_minutes=wait,
                visit_minutes=visit,
                departure_minute=departure,
                travel_minutes=travel,
                utility=utility,
            )
        )

        self.visited[action] = True
        self.current = action
        self.clock = departure
        self.total_travel += travel
        self.total_wait += wait
        self.total_visit += visit
        self.total_utility += utility

        reward = (
            utility
            - self.travel_penalty * travel
            - self.wait_penalty * wait
        )

        next_mask = self.action_mask()
        # Only the stop action remains -> nothing more can be reached.
        if not next_mask[: self.n].any():
            self.done = True

        return self._observe(), reward, self.done, False, {"poi": poi_id}

    # ------------------------------------------------------------------
    # Masking
    # ------------------------------------------------------------------

    def action_mask(self) -> np.ndarray:
        """
        Boolean mask over actions. Stopping is always legal; a POI is legal only
        if the visitor can walk there, get in, finish the visit before closing,
        and still reach the exit within the remaining budget.
        """
        mask = np.zeros(self.n + 1, dtype=bool)
        mask[self.n] = True  # stopping is always allowed

        if self.done:
            return mask

        candidates = self.eligible & ~self.visited
        if not candidates.any():
            return mask

        travel = self.distance[self.current]
        arrival = self.clock + travel
        wait = np.maximum(0.0, self.opens_at - arrival)
        departure = arrival + wait + self.graph.visit_minutes

        feasible = candidates
        feasible &= np.isfinite(travel)              # reachable on this graph
        feasible &= wait <= self.max_wait_minutes    # no absurd loitering
        feasible &= departure <= self.closes_at      # visit ends before closing
        feasible &= (
            departure + self.distance[:, self.finish_idx] <= self.budget
        )                                            # can still reach the exit

        mask[: self.n] = feasible
        return mask

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _observe(self) -> Dict[str, np.ndarray]:
        g = self.graph
        mask = self.action_mask()
        remaining = max(0.0, self.budget - self.clock)

        travel = self.distance[self.current].copy()
        travel[~np.isfinite(travel)] = -1.0

        features = np.zeros((self.n, N_POI_FEATURES), dtype=np.float32)
        features[:, 0] = self.utility / max(self.utility.max(), 1e-6)
        features[:, 1] = g.visit_minutes / 60.0
        features[:, 2] = travel / 60.0
        features[:, 3] = self.visited.astype(np.float32)
        features[:, 4] = mask[: self.n].astype(np.float32)
        features[:, 5] = np.clip(
            (self.opens_at - self.clock) / 60.0, -2.0, 2.0
        )
        features[:, 6] = np.clip(
            (self.closes_at - self.clock) / 60.0, -2.0, 2.0
        )
        features[:, 7:] = self._zone_onehot

        global_features = np.array(
            [
                remaining / max(self.budget, 1.0),
                self.clock / max(self.budget, 1.0),
                self.visited.sum() / max(self.n, 1),
                self.total_utility / self._max_utility,
            ],
            dtype=np.float32,
        )

        return {
            "pois": features,
            "global": global_features,
            "mask": mask.astype(np.int8),
        }

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def result(self) -> EpisodeResult:
        """Package the finished episode, including the walk back to the exit."""
        return_leg = float(self.distance[self.current, self.finish_idx])
        if not np.isfinite(return_leg):
            return_leg = 0.0
        elapsed = self.clock + return_leg

        violations: List[str] = []
        if elapsed > self.budget + 1e-6:
            violations.append(
                f"itinerary takes {elapsed:.1f}min, budget is {self.budget:.0f}min"
            )

        return EpisodeResult(
            scenario=self.scenario,
            steps=list(self.steps),
            total_utility=self.total_utility,
            total_travel=self.total_travel + return_leg,
            total_wait=self.total_wait,
            total_visit=self.total_visit,
            elapsed_minutes=elapsed,
            feasible=not violations,
            violations=violations,
        )

    def render(self) -> str:  # pragma: no cover - display helper
        lines = [f"Scenario: {self.scenario.describe()}"]
        base = self.scenario.start_time
        for i, s in enumerate(self.steps, 1):
            arrive = base + timedelta(minutes=s.arrival_minute + s.wait_minutes)
            depart = base + timedelta(minutes=s.departure_minute)
            wait = f" (waited {s.wait_minutes:.0f}min)" if s.wait_minutes > 0 else ""
            lines.append(
                f"{i:2d}. {arrive:%H:%M}-{depart:%H:%M} {s.name}{wait}"
            )
        lines.append(self.result().summary())
        return "\n".join(lines)
