from __future__ import annotations

import itertools
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Optional

import click
import sys
import random

from neo4j.exceptions import ServiceUnavailable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.planner_utils import ShortestPathResult, PathSegment
from scripts.planner import (
    CandidatePOI,
    PlannerConstraints,
    determine_route,
    determine_route_greedy,
    select_pois,
)


@dataclass
class Scenario:
    name: str
    poi_ids: List[str]
    visit_minutes: int


DEFAULT_SCENARIOS: List[Scenario] = [
    Scenario(
        name="palace_enfilade",
        poi_ids=[
            "versailles:Room:salon-dhercule",
            "versailles:Room:salon-de-labondance",
            "versailles:Room:salon-de-venus",
            "versailles:Room:salon-de-diane",
            "versailles:Room:salon-de-mars",
            "versailles:Room:salon-de-mercure",
            "versailles:Room:salon-dapollon",
        ],
        visit_minutes=8,
    ),
    Scenario(
        name="palace_plus_courtyards",
        poi_ids=[
            "versailles:Garden:cour-dhonneur",
            "versailles:Garden:cour-royale",
            "versailles:Room:galeries-de-lhistoire",
            "versailles:Room:chapelle-royale",
            "versailles:Room:galerie-des-glaces",
            "versailles:Room:grand-appartement-de-la-reine",
            "versailles:Room:cabinet-du-conseil",
            "versailles:Garden:acces-principal-jardins-parterres-deau",
        ],
        visit_minutes=10,
    ),
    Scenario(
        name="extended_mixed",
        poi_ids=[
            "versailles:Garden:cour-dhonneur",
            "versailles:Room:galeries-de-lhistoire",
            "versailles:Room:galerie-des-glaces",
            "versailles:Room:grand-appartement-de-la-reine",
            "versailles:Room:appartements-du-dauphin-et-de-la-dauphine",
            "versailles:Garden:acces-principal-jardins-parterres-deau",
            "poi:mock:trianon1",
            "poi:mock:trianon2",
            "poi:mock:park1",
        ],
        visit_minutes=12,
    ),
    Scenario(
        name="trianon_loop",
        poi_ids=[
            "poi:mock:grand-trianon",
            "poi:mock:pavillon-francaise",
            "poi:mock:gardens-grand-trianon",
            "poi:mock:petit-trianon",
            "poi:mock:queen_hamlet",
            "poi:mock:temple_amour",
        ],
        visit_minutes=9,
    ),
]


def generate_mock_candidates(poi_ids: Sequence[str], visit_minutes: int) -> List[CandidatePOI]:
    candidates: List[CandidatePOI] = []
    for poi_id in poi_ids:
        candidates.append(
            CandidatePOI(
                poi_id=poi_id,
                name=poi_id.split(":")[-1],
                priority_score=random.uniform(1.0, 2.0),
                estimated_visit_minutes=visit_minutes,
                interest_tags=["bench"],
                opening_time="09:00",
                closing_time="18:00",
            )
        )
    return candidates


def make_mock_shortest_path() -> callable:
    def solver(a: str, b: str, user_profile: str, accessibility: str, **_: object) -> ShortestPathResult:
        base = float(abs(hash((a, b))) % 12 + 5)
        segment = PathSegment(
            from_id=a,
            to_id=b,
            distance_min=base,
            is_step_free=True,
            stroller_friendly=True,
            path_type="indoor",
            notes="mock",
        )
        return ShortestPathResult(node_ids=[a, b], total_minutes=base, segments=[segment])

    return solver


def brute_force_route(poi_ids: Sequence[str], constraints: PlannerConstraints, solver, timeout_sec: float = 30.0) -> ShortestPathResult:
    best_order = None
    best_cost = float("inf")
    best_path = None
    start_time = time.perf_counter()
    for order in itertools.permutations(poi_ids):
        if time.perf_counter() - start_time > timeout_sec:
            raise TimeoutError("Brute-force search timed out.")
        total = 0.0
        segments: List[ShortestPathResult] = []
        valid = True
        for a, b in zip(order, order[1:]):
            try:
                path = solver(a, b, constraints)
            except ValueError:
                valid = False
                break
            total += path.total_minutes
            segments.append(path)
        if not valid:
            continue
        if total < best_cost:
            best_cost = total
            best_order = list(order)
            best_path = merge_segments(segments)
    if best_order is None or best_path is None:
        raise ValueError("Brute-force could not find a route.")
    return best_path


def merge_segments(segments: Iterable[ShortestPathResult]) -> ShortestPathResult:
    segments = list(segments)
    if not segments:
        return ShortestPathResult(node_ids=[], total_minutes=0.0, segments=[])
    node_ids = list(segments[0].node_ids)
    total = segments[0].total_minutes
    segment_list = list(segments[0].segments)
    for segment in segments[1:]:
        node_ids.extend(segment.node_ids[1:])
        total += segment.total_minutes
        segment_list.extend(segment.segments)
    return ShortestPathResult(node_ids=node_ids, total_minutes=total, segments=segment_list)


@click.command()
@click.option("--max-pois", default=9, help="Maximum number of POIs to test.")
@click.option("--trials", default=5, help="Number of random trials per scenario.")
@click.option("--output", type=click.Path(writable=True), default="benchmarks.json")
@click.option("--solver", type=click.Choice(["default", "greedy"], case_sensitive=False), default="default")
@click.option("--offline", is_flag=True, help="Use synthetic distances instead of querying Neo4j.")
@click.option("--seed", default=42, show_default=True, help="Random seed for candidate generation.")
def run_benchmarks(max_pois: int, trials: int, output: str, solver: str, offline: bool, seed: int) -> None:
    import scripts.planner_utils as planner_utils
    import scripts.planner as planner_module

    constraints = PlannerConstraints(interests=[], user_profile="standard", accessibility="any")
    results = {"runs": []}
    original_solver_utils = planner_utils.get_shortest_path
    original_solver_planner = planner_module.get_shortest_path

    mock_solver: Optional[callable] = make_mock_shortest_path() if offline else None

    def apply_solver(custom_solver: callable) -> None:
        planner_utils.get_shortest_path = custom_solver  # type: ignore
        planner_module.get_shortest_path = custom_solver  # type: ignore

    if mock_solver:
        apply_solver(mock_solver)

    random.seed(seed)

    try:
        for scenario in DEFAULT_SCENARIOS:
            for trial in range(trials):
                random.seed(seed + trial)
                candidates = generate_mock_candidates(scenario.poi_ids, scenario.visit_minutes)
                selected = select_pois(
                    candidates,
                    total_duration_minutes=scenario.visit_minutes * len(candidates),
                    must_include=[],
                )
                poi_ids = [c.poi_id for c in selected]

                if not poi_ids or len(poi_ids) > max_pois:
                    continue

                attempt_offline = offline
                for attempt in range(2):
                    try:
                        start = time.perf_counter()
                        if solver.lower() == "greedy":
                            order, pair_paths, merged = determine_route_greedy(poi_ids, constraints=constraints)
                        else:
                            order, pair_paths, merged = determine_route(poi_ids, constraints=constraints)
                        greedy_duration = time.perf_counter() - start
                        break
                    except ServiceUnavailable:
                        if attempt == 0:
                            click.echo("Neo4j unavailable, switching to offline mock solver.")
                            mock_solver = make_mock_shortest_path()
                            apply_solver(mock_solver)
                            attempt_offline = True
                            continue
                        raise
                else:
                    continue

                try:
                    brute_start = time.perf_counter()
                    brute_path = brute_force_route(
                        poi_ids,
                        constraints,
                        lambda a, b, c: planner_utils.get_shortest_path(
                            a, b, user_profile=c.user_profile, accessibility=c.accessibility
                        ),
                    )
                    brute_duration = time.perf_counter() - brute_start
                    optimal_cost = brute_path.total_minutes
                except (ValueError, TimeoutError):
                    brute_duration = None
                    optimal_cost = None

                results["runs"].append(
                    {
                        "scenario": scenario.name,
                        "trial": trial,
                        "poi_count": len(poi_ids),
                        "solver": solver.lower(),
                        "offline": attempt_offline,
                        "greedy_cost": merged.total_minutes,
                        "greedy_time_sec": greedy_duration,
                        "brute_cost": optimal_cost,
                        "brute_time_sec": brute_duration,
                    }
                )
    finally:
        planner_utils.get_shortest_path = original_solver_utils  # type: ignore
        planner_module.get_shortest_path = original_solver_planner  # type: ignore

    Path(output).write_text(json.dumps(results, indent=2))
    click.echo(f"Saved benchmarks to {output}")


if __name__ == "__main__":
    run_benchmarks()
