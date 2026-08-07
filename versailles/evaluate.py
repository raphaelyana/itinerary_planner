"""
Scoring harness: run agents over the benchmark suite and compare them.

Every agent is driven through the same environment on the same scenarios with
the same reward, so the resulting table is a fair comparison rather than a set
of numbers that happen to sit next to each other.

Usage
-----
    python -m versailles.evaluate                    # all baselines
    python -m versailles.evaluate --scenario full_day
    python -m versailles.evaluate --json results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from versailles.agent import Agent
from versailles.env import EpisodeResult, VersaillesEnv
from versailles.graph import VersaillesGraph, load_graph
from versailles.scenario import Scenario, standard_suite

logger = logging.getLogger(__name__)


@dataclass
class ScenarioScore:
    """One agent's result on one scenario."""

    agent: str
    scenario: str
    utility: float
    n_pois: int
    travel_minutes: float
    wait_minutes: float
    visit_minutes: float
    elapsed_minutes: float
    budget_minutes: int
    feasible: bool
    solve_seconds: float
    violations: List[str]
    poi_ids: List[str]


@dataclass
class Comparison:
    """Full result set across agents and scenarios."""

    scores: List[ScenarioScore]

    def by_agent(self) -> Dict[str, List[ScenarioScore]]:
        grouped: Dict[str, List[ScenarioScore]] = {}
        for score in self.scores:
            grouped.setdefault(score.agent, []).append(score)
        return grouped

    def mean_utility(self) -> Dict[str, float]:
        return {
            agent: float(np.mean([s.utility for s in scores]))
            for agent, scores in self.by_agent().items()
        }

    def normalized(self, reference: str = "ortools") -> Dict[str, float]:
        """
        Mean utility as a fraction of a reference agent's, per scenario.

        This is the number that answers "how good is my policy really?" - the
        raw mean is dominated by long scenarios, while the per-scenario ratio
        weights every visitor request equally.
        """
        grouped = self.by_agent()
        if reference not in grouped:
            return {}
        ref = {s.scenario: s.utility for s in grouped[reference]}
        out = {}
        for agent, scores in grouped.items():
            ratios = [
                s.utility / ref[s.scenario]
                for s in scores
                if ref.get(s.scenario, 0.0) > 0
            ]
            out[agent] = float(np.mean(ratios)) if ratios else 0.0
        return out

    def table(self, reference: str = "ortools") -> str:
        grouped = self.by_agent()
        means = self.mean_utility()
        ratios = self.normalized(reference)

        rows = []
        header = (
            f"{'agent':<18}{'utility':>10}{'vs ref':>9}{'POIs':>7}"
            f"{'travel':>9}{'wait':>7}{'infeas':>8}{'sec/plan':>10}"
        )
        rows.append(header)
        rows.append("-" * len(header))

        for agent in sorted(means, key=lambda a: -means[a]):
            scores = grouped[agent]
            infeasible = sum(1 for s in scores if not s.feasible)
            rows.append(
                f"{agent:<18}"
                f"{means[agent]:>10.2f}"
                f"{ratios.get(agent, 0.0) * 100:>8.0f}%"
                f"{np.mean([s.n_pois for s in scores]):>7.1f}"
                f"{np.mean([s.travel_minutes for s in scores]):>9.0f}"
                f"{np.mean([s.wait_minutes for s in scores]):>7.0f}"
                f"{infeasible:>8d}"
                f"{np.mean([s.solve_seconds for s in scores]):>10.3f}"
            )
        return "\n".join(rows)

    def per_scenario_table(self) -> str:
        grouped = self.by_agent()
        agents = sorted(grouped)
        scenarios = [s.scenario for s in grouped[agents[0]]]

        width = max(len(s) for s in scenarios) + 2
        rows = ["".ljust(width) + "".join(f"{a:>16}" for a in agents)]
        rows.append("-" * (width + 16 * len(agents)))

        lookup = {(s.agent, s.scenario): s for s in self.scores}
        for scenario in scenarios:
            line = scenario.ljust(width)
            best = max(
                (lookup[(a, scenario)].utility for a in agents if (a, scenario) in lookup),
                default=0.0,
            )
            for agent in agents:
                score = lookup.get((agent, scenario))
                if score is None:
                    line += f"{'-':>16}"
                    continue
                marker = "*" if score.utility >= best - 1e-9 else " "
                flag = "!" if not score.feasible else marker
                line += f"{score.utility:>15.2f}{flag}"
            rows.append(line)
        rows.append("")
        rows.append("* best on that scenario   ! infeasible itinerary")
        return "\n".join(rows)

    def to_json(self, path: Path | str) -> None:
        payload = {"scores": [asdict(s) for s in self.scores]}
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def score_agent(
    agent: Agent,
    scenarios: Sequence[Scenario],
    graph: Optional[VersaillesGraph] = None,
    **env_kwargs,
) -> List[ScenarioScore]:
    """Run one agent across scenarios and collect its scores."""
    graph = graph or load_graph()
    scores: List[ScenarioScore] = []

    for scenario in scenarios:
        env = VersaillesEnv(graph=graph, scenario=scenario, **env_kwargs)
        started = time.perf_counter()
        try:
            result = agent.solve(scenario, env=env)
        except Exception as exc:  # a broken agent should not kill the sweep
            logger.error("%s failed on %s: %s", agent.name, scenario.name, exc)
            scores.append(
                ScenarioScore(
                    agent=agent.name,
                    scenario=scenario.name,
                    utility=0.0,
                    n_pois=0,
                    travel_minutes=0.0,
                    wait_minutes=0.0,
                    visit_minutes=0.0,
                    elapsed_minutes=0.0,
                    budget_minutes=scenario.duration_minutes,
                    feasible=False,
                    solve_seconds=time.perf_counter() - started,
                    violations=[f"{type(exc).__name__}: {exc}"],
                    poi_ids=[],
                )
            )
            continue
        elapsed = time.perf_counter() - started

        scores.append(
            ScenarioScore(
                agent=agent.name,
                scenario=scenario.name,
                utility=result.total_utility,
                n_pois=result.n_visited,
                travel_minutes=result.total_travel,
                wait_minutes=result.total_wait,
                visit_minutes=result.total_visit,
                elapsed_minutes=result.elapsed_minutes,
                budget_minutes=scenario.duration_minutes,
                feasible=result.feasible,
                solve_seconds=elapsed,
                violations=list(result.violations),
                poi_ids=result.poi_ids,
            )
        )
    return scores


def compare(
    agents: Iterable[Agent],
    scenarios: Optional[Sequence[Scenario]] = None,
    graph: Optional[VersaillesGraph] = None,
    **env_kwargs,
) -> Comparison:
    """Score every agent on every scenario."""
    scenarios = list(scenarios if scenarios is not None else standard_suite())
    graph = graph or load_graph()

    scores: List[ScenarioScore] = []
    for agent in agents:
        logger.info("scoring %s", agent.name)
        scores.extend(score_agent(agent, scenarios, graph=graph, **env_kwargs))
    return Comparison(scores=scores)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Compare Versailles planners.")
    parser.add_argument("--scenario", help="run a single named scenario")
    parser.add_argument("--json", help="write full results to this path")
    parser.add_argument(
        "--ortools-seconds", type=int, default=10, help="solver time limit per scenario"
    )
    parser.add_argument("--per-scenario", action="store_true", help="show the detail table")
    parser.add_argument("--checkpoint", help="also score a trained policy from this path")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    from versailles.baselines import all_baselines

    scenarios = standard_suite()
    if args.scenario:
        scenarios = [s for s in scenarios if s.name == args.scenario]
        if not scenarios:
            parser.error(f"no scenario named {args.scenario!r}")

    agents: List[Agent] = list(all_baselines(ortools_seconds=args.ortools_seconds))

    if args.checkpoint:
        # Your trained policy joins the comparison here; see versailles/train.py.
        from versailles.train import load_policy_agent

        agents.append(load_policy_agent(args.checkpoint))

    graph = load_graph()
    report = graph.validate()
    if not report.ok:
        print("Graph data has errors; results may be meaningless:")
        print(report)
        print()

    result = compare(agents, scenarios, graph=graph)

    print(f"Scenarios: {len(scenarios)}   Agents: {len(agents)}")
    print()
    print(result.table())
    if args.per_scenario:
        print()
        print(result.per_scenario_table())

    if args.json:
        result.to_json(args.json)
        print(f"\nWrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
