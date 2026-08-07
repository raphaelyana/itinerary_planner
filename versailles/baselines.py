"""Non-learned planners used as comparison baselines."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from versailles.agent import Agent
from versailles.env import EpisodeResult, VersaillesEnv
from versailles.scenario import Scenario

logger = logging.getLogger(__name__)

_TIME_SCALE = 10
_BOOL_TRUE = 3


class NearestAgent(Agent):
    """Greedy nearest-neighbour. Minimises walking, ignores value."""

    name = "nearest"

    def act(self, obs: Dict[str, np.ndarray], env: VersaillesEnv) -> int:
        mask = obs["mask"].astype(bool)
        visits = np.flatnonzero(mask[: env.n])
        if visits.size == 0:
            return env.n
        travel = env.distance[env.current][visits]
        return int(visits[np.argmin(travel)])


class GreedyDensityAgent(Agent):
    """Greedy on utility per minute spent."""

    name = "greedy_density"

    def __init__(self, epsilon: float = 0.0, seed: int = 0) -> None:
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

    def act(self, obs: Dict[str, np.ndarray], env: VersaillesEnv) -> int:
        mask = obs["mask"].astype(bool)
        visits = np.flatnonzero(mask[: env.n])
        if visits.size == 0:
            return env.n

        if self.epsilon and self.rng.random() < self.epsilon:
            return int(self.rng.choice(visits))

        travel = env.distance[env.current][visits]
        arrival = env.clock + travel
        wait = np.maximum(0.0, env.opens_at[visits] - arrival)
        visit = env.graph.visit_minutes[visits]

        cost = travel + wait + visit
        cost = np.maximum(cost, 1e-3)
        score = env.utility[visits] / cost
        return int(visits[np.argmax(score)])


class ORToolsAgent(Agent):
    """Reference solver: single-vehicle routing with optional nodes."""

    name = "ortools"

    def __init__(
        self,
        time_limit_seconds: int = 15,
        candidate_limit: int = 120,
        warm_start: bool = True,
        log: bool = False,
    ) -> None:
        self.time_limit_seconds = time_limit_seconds
        self.candidate_limit = candidate_limit
        self.warm_start = warm_start
        self.log = log
        self._plan: List[int] = []
        self._cursor = 0

    @staticmethod
    def _greedy_pois(env: VersaillesEnv) -> List[int]:
        """The greedy itinerary as POI indices, used to seed the search."""
        cache_key = id(env.scenario)
        cached = getattr(env, "_greedy_seed_cache", None)
        if cached and cached[0] == cache_key:
            return cached[1]

        scratch = VersaillesEnv(
            graph=env.graph,
            scenario=env.scenario,
            utility_model=env.utility_model,
            travel_penalty=env.travel_penalty,
            wait_penalty=env.wait_penalty,
            max_wait_minutes=env.max_wait_minutes,
        )
        try:
            result = GreedyDensityAgent().solve(env.scenario, env=scratch)
            pois = [env.graph.index[p] for p in result.poi_ids]
        except Exception as exc:
            logger.debug("greedy warm start failed: %s", exc)
            pois = []

        env._greedy_seed_cache = (cache_key, pois)  # type: ignore[attr-defined]
        return pois

    def _greedy_route(self, env: VersaillesEnv, nodes: List[int]) -> List[int]:
        """The warm-start route expressed as model node indices."""
        position = {poi: i for i, poi in enumerate(nodes)}
        return [
            position[poi] for poi in self._greedy_pois(env) if poi in position
        ]

    def act(self, obs: Dict[str, np.ndarray], env: VersaillesEnv) -> int:
        while self._cursor < len(self._plan):
            action = self._plan[self._cursor]
            self._cursor += 1
            if obs["mask"][action]:
                return action
            logger.debug("dropping infeasible planned stop %s", env.graph.order[action])
        return env.n

    def solve(
        self,
        scenario: Scenario,
        env: Optional[VersaillesEnv] = None,
        **env_kwargs: Any,
    ) -> EpisodeResult:
        env = env or VersaillesEnv(scenario=scenario, **env_kwargs)
        env.reset(options={"scenario": scenario})
        self._plan = self._build_plan(env)
        self._cursor = 0
        return super().solve(scenario, env=env)

    # ------------------------------------------------------------------

    def build_model(self, env: VersaillesEnv):
        """
        Construct the routing model without solving it.

        Returns ``(routing, manager, nodes, params)``, or None when there is
        nothing to plan.
        """
        return self._build(env)

    def _build_plan(self, env: VersaillesEnv) -> List[int]:
        built = self._build(env)
        if built is None:
            return []
        routing, manager, nodes, params = built

        solution = None
        if self.warm_start:
            seed = self._greedy_route(env, nodes)
            if seed:
                assignment = routing.ReadAssignmentFromRoutes([seed], True)
                if assignment is not None:
                    solution = routing.SolveFromAssignmentWithParameters(
                        assignment, params
                    )
                else:
                    logger.debug("warm-start route rejected by the model")

        if solution is None:
            solution = routing.SolveWithParameters(params)
        if solution is None:
            logger.warning("OR-Tools found no solution for %s", env.scenario.name)
            return []

        depot = 0
        end = len(nodes) - 1 if env.finish_idx != env.start_idx else 0

        plan: List[int] = []
        index = routing.Start(0)
        while not routing.IsEnd(index):
            node_i = manager.IndexToNode(index)
            if node_i != depot and node_i != end:
                plan.append(nodes[node_i])
            index = solution.Value(routing.NextVar(index))
        return plan

    def _build(self, env: VersaillesEnv):
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2

        reachable = env.eligible & np.isfinite(env.distance[env.start_idx])
        reachable &= np.isfinite(env.distance[:, env.finish_idx])
        candidates = np.flatnonzero(reachable)
        if candidates.size == 0:
            return None

        seed_pois = self._greedy_pois(env) if self.warm_start else []

        if candidates.size > self.candidate_limit:
            travel = env.distance[env.start_idx][candidates]
            score = env.utility[candidates] / np.maximum(
                travel + env.graph.visit_minutes[candidates], 1e-3
            )
            kept = candidates[np.argsort(-score)[: self.candidate_limit]]
            candidates = np.array(
                sorted(set(kept.tolist()) | set(seed_pois)), dtype=int
            )

        nodes = [env.start_idx] + [int(c) for c in candidates]
        if env.finish_idx != env.start_idx:
            nodes.append(env.finish_idx)
        n_nodes = len(nodes)
        depot = 0
        distinct_finish = env.finish_idx != env.start_idx
        end = n_nodes - 1 if distinct_finish else 0

        if distinct_finish:
            manager = pywrapcp.RoutingIndexManager(n_nodes, 1, [depot], [end])
        else:
            manager = pywrapcp.RoutingIndexManager(n_nodes, 1, depot)
        routing = pywrapcp.RoutingModel(manager)

        distance = env.distance
        visit_minutes = env.graph.visit_minutes
        big = 10 ** 7

        def travel_time(from_index: int, to_index: int) -> int:
            i = nodes[manager.IndexToNode(from_index)]
            j = nodes[manager.IndexToNode(to_index)]
            d = distance[i, j]
            if not np.isfinite(d):
                return big
            service = visit_minutes[j] if manager.IndexToNode(to_index) != end else 0.0
            return int(round((d + service) * _TIME_SCALE))

        transit_cb = routing.RegisterTransitCallback(travel_time)

        def arc_cost(from_index: int, to_index: int) -> int:
            i = nodes[manager.IndexToNode(from_index)]
            j = nodes[manager.IndexToNode(to_index)]
            d = distance[i, j]
            if not np.isfinite(d):
                return big
            return int(round(d * env.travel_penalty * _TIME_SCALE * 100))

        routing.SetArcCostEvaluatorOfAllVehicles(
            routing.RegisterTransitCallback(arc_cost)
        )

        horizon = int(round(env.budget * _TIME_SCALE))
        max_slack = int(round(env.max_wait_minutes * _TIME_SCALE))
        routing.AddDimension(transit_cb, max_slack, horizon, True, "Time")
        time_dim = routing.GetDimensionOrDie("Time")

        for node_i in range(1, len(nodes) - (1 if end != depot else 0)):
            poi = nodes[node_i]
            visit = float(visit_minutes[poi])
            earliest_departure = max(0.0, float(env.opens_at[poi])) + visit
            latest_departure = min(float(env.closes_at[poi]), env.budget)
            if latest_departure < earliest_departure:
                continue
            index = manager.NodeToIndex(node_i)
            time_dim.CumulVar(index).SetRange(
                int(round(earliest_departure * _TIME_SCALE)),
                int(round(latest_departure * _TIME_SCALE)),
            )

        utility_scale = 1000.0
        for node_i in range(1, len(nodes) - (1 if end != depot else 0)):
            penalty = int(round(float(env.utility[nodes[node_i]]) * utility_scale))
            routing.AddDisjunction([manager.NodeToIndex(node_i)], max(penalty, 1))

        params = pywrapcp.DefaultRoutingSearchParameters()
        params.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        params.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )

        ls = params.local_search_operators
        for operator in (
            "use_relocate_and_make_active",
            "use_exchange_and_make_active",
            "use_exchange_path_start_ends_and_make_active",
            "use_extended_swap_active",
            "use_node_pair_swap_active",
            "use_inactive_lns",
        ):
            setattr(ls, operator, _BOOL_TRUE)

        params.time_limit.FromSeconds(self.time_limit_seconds)
        params.log_search = self.log

        return routing, manager, nodes, params


def all_baselines(seed: int = 0, ortools_seconds: int = 10) -> List[Agent]:
    """The standard comparison set used by ``versailles.evaluate``."""
    from versailles.agent import RandomAgent

    return [
        RandomAgent(seed=seed),
        NearestAgent(),
        GreedyDensityAgent(),
        ORToolsAgent(time_limit_seconds=ortools_seconds),
    ]
