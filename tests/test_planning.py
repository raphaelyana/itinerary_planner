"""
Tests for the invariants the planner must never violate.

These are deliberately about *guarantees* rather than specific numbers: that no
agent can produce an itinerary a real visitor could not follow, that hard
constraints are hard, and that the solver hierarchy holds. Numbers move when the
utility model is tuned; these properties must not.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from versailles.agent import RandomAgent
from versailles.baselines import GreedyDensityAgent, NearestAgent, ORToolsAgent
from versailles.env import VersaillesEnv
from versailles.graph import load_graph
from versailles.scenario import Scenario, standard_suite, training_distribution


@pytest.fixture(scope="module")
def graph():
    return load_graph()


@pytest.fixture(scope="module")
def suite():
    return {s.name: s for s in standard_suite()}


# ----------------------------------------------------------------------
# Graph data
# ----------------------------------------------------------------------


def test_graph_data_is_valid(graph):
    report = graph.validate()
    assert report.ok, f"graph validation failed:\n{report}"


def test_graph_is_fully_connected(graph):
    matrix = graph.travel_matrix("base", "any")
    assert np.isfinite(matrix).all(), "some POI pairs are unreachable"


def test_every_connection_endpoint_exists(graph):
    known = set(graph.pois)
    for conn in graph.connections:
        assert conn.from_id in known, f"unknown POI {conn.from_id}"
        assert conn.to_id in known, f"unknown POI {conn.to_id}"


def test_transit_nodes_have_no_visit_time(graph):
    for pid in graph.order:
        poi = graph.pois[pid]
        assert poi.is_transit == (poi.visit_minutes <= 0)


def test_trianon_opens_at_noon(graph):
    window = graph.opening_window(
        "versailles:Trianon:entree-grand-trianon", datetime(2026, 7, 28).date()
    )
    assert window is not None
    assert window[0].hour == 12, "Trianon must not be schedulable before noon"


# ----------------------------------------------------------------------
# Environment guarantees
# ----------------------------------------------------------------------


def test_random_rollouts_are_always_feasible(graph, suite):
    """Masking, not penalties: even random play must stay inside the rules."""
    rng = np.random.default_rng(0)
    for scenario in suite.values():
        env = VersaillesEnv(graph=graph, scenario=scenario, seed=0)
        obs, _ = env.reset()
        while True:
            legal = np.flatnonzero(obs["mask"].astype(bool))
            obs, _, done, _, _ = env.step(int(rng.choice(legal)))
            if done:
                break
        result = env.result()
        assert result.feasible, f"{scenario.name}: {result.violations}"
        assert result.elapsed_minutes <= scenario.duration_minutes + 1e-6


def test_stop_action_is_always_legal(graph, suite):
    env = VersaillesEnv(graph=graph, scenario=suite["short_highlights"])
    obs, _ = env.reset()
    assert obs["mask"][env.n] == 1


def test_illegal_action_is_rejected(graph, suite):
    env = VersaillesEnv(graph=graph, scenario=suite["hates_trianon"])
    obs, _ = env.reset()
    trianon = [
        i for i, pid in enumerate(graph.order) if graph.pois[pid].zone == "Trianon"
    ]
    assert obs["mask"][trianon[0]] == 0
    _, reward, done, _, info = env.step(trianon[0])
    assert done and info.get("invalid_action") and reward < 0


def test_transit_nodes_are_never_visited(graph, suite):
    result = GreedyDensityAgent().solve(suite["full_day"])
    for pid in result.poi_ids:
        assert not graph.pois[pid].is_transit, f"{pid} is a transit node"


def test_visited_pois_are_never_repeated(graph, suite):
    result = GreedyDensityAgent().solve(suite["full_day"])
    assert len(result.poi_ids) == len(set(result.poi_ids))


# ----------------------------------------------------------------------
# Hard constraints
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "scenario_name,forbidden_zone",
    [("hates_trianon", "Trianon"), ("hates_castle", "Castle")],
)
def test_zone_avoidance_is_absolute(graph, suite, scenario_name, forbidden_zone):
    scenario = suite[scenario_name]
    for agent in (RandomAgent(seed=3), NearestAgent(), GreedyDensityAgent()):
        result = agent.solve(scenario)
        zones = {graph.pois[p].zone for p in result.poi_ids}
        assert forbidden_zone not in zones, f"{agent.name} entered {forbidden_zone}"


@pytest.mark.parametrize("scenario_name", ["elder_step_free", "stroller_family"])
def test_accessibility_is_absolute(graph, suite, scenario_name):
    scenario = suite[scenario_name]
    for agent in (RandomAgent(seed=5), GreedyDensityAgent()):
        result = agent.solve(scenario)
        for pid in result.poi_ids:
            assert (
                graph.pois[pid].accessibility_level == "full"
            ), f"{agent.name} routed through inaccessible {pid}"


def test_opening_hours_are_respected(graph, suite):
    """No visit may start before opening or end after closing."""
    for scenario in suite.values():
        result = GreedyDensityAgent().solve(scenario)
        for step in result.steps:
            window = graph.opening_window(step.poi_id, scenario.start_time.date())
            assert window is not None, f"{step.poi_id} scheduled with no opening hours"
            start_min = scenario.start_time.hour * 60 + scenario.start_time.minute
            opens = window[0].hour * 60 + window[0].minute - start_min
            closes = window[1].hour * 60 + window[1].minute - start_min
            begin = step.arrival_minute + step.wait_minutes
            assert begin >= opens - 1e-6, f"{step.poi_id} entered before opening"
            assert step.departure_minute <= closes + 1e-6, f"{step.poi_id} left after closing"


def test_excluded_pois_are_never_visited(graph, suite):
    base = suite["half_day_classic"]
    banned = "versailles:Room:galerie-des-glaces"
    scenario = Scenario(
        name="excluded",
        start_time=base.start_time,
        duration_minutes=base.duration_minutes,
        interests=base.interests,
        exclude_ids=[banned],
    )
    result = GreedyDensityAgent().solve(scenario)
    assert banned not in result.poi_ids


def test_randomized_training_scenarios_stay_feasible(graph):
    """The training distribution must not emit unsolvable instances."""
    rng = np.random.default_rng(11)
    agent = GreedyDensityAgent()
    for scenario in training_distribution(rng, 25):
        result = agent.solve(scenario)
        assert result.feasible, f"{scenario.describe()}: {result.violations}"


# ----------------------------------------------------------------------
# Solver quality
# ----------------------------------------------------------------------


def test_greedy_beats_random_and_nearest(graph, suite):
    scenario = suite["half_day_classic"]
    random_score = RandomAgent(seed=1).solve(scenario).total_utility
    nearest_score = NearestAgent().solve(scenario).total_utility
    greedy_score = GreedyDensityAgent().solve(scenario).total_utility
    assert greedy_score > nearest_score > random_score


@pytest.mark.slow
def test_ortools_is_at_least_as_good_as_greedy(graph, suite):
    """
    The reference solver is warm-started from greedy, so it can only improve.
    A regression here means the warm start broke - which previously let a
    constructive heuristic beat the "near-optimal" solver.
    """
    for name in ("half_day_classic", "elder_step_free", "marie_antoinette"):
        scenario = suite[name]
        greedy = GreedyDensityAgent().solve(scenario).total_utility
        ortools = ORToolsAgent(time_limit_seconds=10).solve(scenario).total_utility
        assert ortools >= greedy - 1e-6, f"{name}: ortools {ortools} < greedy {greedy}"


# ----------------------------------------------------------------------
# Utility model
# ----------------------------------------------------------------------


def test_interests_change_what_is_selected(graph, suite):
    base = suite["half_day_classic"]
    gardens = Scenario(
        name="g",
        start_time=base.start_time,
        duration_minutes=base.duration_minutes,
        interests=["garden", "fountains"],
    )
    history = Scenario(
        name="h",
        start_time=base.start_time,
        duration_minutes=base.duration_minutes,
        interests=["history", "art"],
    )
    a = set(GreedyDensityAgent().solve(gardens).poi_ids)
    b = set(GreedyDensityAgent().solve(history).poi_ids)
    assert a != b, "interests have no effect on selection"


def test_utility_model_is_swappable(graph, suite):
    from versailles.utility import LinearUtility

    scenario = suite["garden_lover"]
    strict = LinearUtility(unmatched_floor=0.0)
    env = VersaillesEnv(graph=graph, scenario=scenario, utility_model=strict)
    result = GreedyDensityAgent().solve(scenario, env=env)
    for pid in result.poi_ids:
        tags = set(graph.pois[pid].interest_tags)
        assert tags & set(scenario.interests), (
            f"{pid} has no matching tag but was chosen with zero floor"
        )
