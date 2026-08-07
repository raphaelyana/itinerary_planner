"""
Drawing the estate and the routes chosen on it.

The CSVs carry no coordinates, so positions come from a Kamada-Kawai layout
seeded with the walking-time matrix. Because the embedding is driven by real
travel times, the result is close to a map: POIs a two-minute walk apart land
near each other. Good enough to see *why* an itinerary is bad, which is the
point - a table tells you a route scored 31, a picture tells you it crossed the
gardens three times.

    python -m versailles.viz --map                      # the estate
    python -m versailles.viz --route full_day           # a solved itinerary
    python -m versailles.viz --compare full_day         # every agent, side by side
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # headless by default; no display needed
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from versailles.env import EpisodeResult
from versailles.graph import ZONES, VersaillesGraph, load_graph

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("benchmarks/figures")

ZONE_COLORS: Dict[str, str] = {
    "Castle": "#b8860b",
    "Gardens": "#2e8b57",
    "Trianon": "#8b3a62",
    "Park": "#4682b4",
}
EDGE_COLOR = "#d8d5cc"
TRANSIT_COLOR = "#bbbbbb"

_layout_cache: Dict[int, Dict[str, np.ndarray]] = {}


def compute_layout(
    graph: VersaillesGraph,
    profile: str = "base",
    seed: int = 7,
    method: str = "spring",
) -> Dict[str, np.ndarray]:
    """
    Position every POI for drawing.

    The estate graph is chain-like - average degree 2.6, diameter 29 hops, with
    most POIs having exactly two neighbours - because the CSV encodes the order
    visitors walk through rooms. Kamada-Kawai, which tries to match Euclidean to
    path distance, folds such a graph into a featureless ring. A force-directed
    layout seeded by walking times keeps the chain legible and lets the zones
    separate, so ``spring`` is the default.

    Pass ``method="kamada_kawai"`` for the distance-faithful version when you
    care about relative travel times more than readability.
    """
    key = hash((id(graph), profile, seed, method))
    if key in _layout_cache:
        return _layout_cache[key]

    g = graph.graph("any")

    if method == "kamada_kawai":
        matrix = graph.travel_matrix(profile, "any")
        finite = matrix[np.isfinite(matrix)]
        fallback = float(finite.max()) * 2 if finite.size else 1.0
        dist = {
            source: {
                target: (
                    float(matrix[i, j]) if np.isfinite(matrix[i, j]) else fallback
                )
                for j, target in enumerate(graph.order)
            }
            for i, source in enumerate(graph.order)
        }
        pos = nx.kamada_kawai_layout(g, dist=dist)
    else:
        # Short walks pull POIs together, so weight is the inverse of minutes.
        weighted = nx.Graph()
        weighted.add_nodes_from(g.nodes())
        for u, v, data in g.edges(data=True):
            minutes = max(float(data.get(profile, 1.0)), 0.25)
            weighted.add_edge(u, v, weight=1.0 / minutes)
        pos = nx.spring_layout(
            weighted, weight="weight", seed=seed, iterations=400, k=0.35
        )

    _layout_cache[key] = pos
    return pos


def _draw_base(
    ax,
    graph: VersaillesGraph,
    pos: Dict[str, np.ndarray],
    *,
    highlight: Optional[Sequence[str]] = None,
    node_size: float = 26.0,
    show_edges: bool = True,
) -> None:
    g = graph.graph("any")

    if show_edges:
        segments = [(pos[u], pos[v]) for u, v in g.edges()]
        for (x0, y0), (x1, y1) in segments:
            ax.plot([x0, x1], [y0, y1], color=EDGE_COLOR, linewidth=0.6, zorder=1)

    highlight_set = set(highlight or ())
    for zone in ZONES:
        ids = [
            p
            for p in graph.order
            if graph.pois[p].zone == zone and p not in highlight_set
        ]
        if not ids:
            continue
        xs = [pos[p][0] for p in ids]
        ys = [pos[p][1] for p in ids]
        sizes = [
            node_size if not graph.pois[p].is_transit else node_size * 0.35
            for p in ids
        ]
        ax.scatter(
            xs,
            ys,
            s=sizes,
            c=ZONE_COLORS.get(zone, "#777777"),
            label=zone,
            alpha=0.55,
            linewidths=0,
            zorder=2,
        )

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_map(
    graph: Optional[VersaillesGraph] = None,
    output: Path | str = OUTPUT_DIR / "estate_map.png",
    annotate_top: int = 12,
) -> Path:
    """Draw the whole estate, labelling the highest-priority POIs."""
    graph = graph or load_graph()
    pos = compute_layout(graph)

    fig, ax = plt.subplots(figsize=(13, 10))
    _draw_base(ax, graph, pos, node_size=34)

    if annotate_top:
        top = np.argsort(-graph.priority)[:annotate_top]
        for i in top:
            pid = graph.order[i]
            ax.annotate(
                graph.pois[pid].name,
                pos[pid],
                fontsize=7,
                alpha=0.85,
                xytext=(4, 4),
                textcoords="offset points",
            )

    # `connections` counts CSV rows, which include reverse duplicates; the
    # graph's edge count is the number of distinct walkable links.
    ax.set_title(
        f"Versailles estate - {graph.n} POIs, "
        f"{graph.graph('any').number_of_edges()} walkable links\n"
        "layout derived from walking times",
        fontsize=12,
    )
    ax.legend(loc="upper right", frameon=False, fontsize=9, markerscale=2)
    return _save(fig, output)


def plot_route(
    result: EpisodeResult,
    graph: Optional[VersaillesGraph] = None,
    output: Optional[Path | str] = None,
    title: Optional[str] = None,
    ax=None,
) -> Optional[Path]:
    """
    Draw one itinerary over the estate.

    Stops are numbered in visit order and joined by the actual walking path
    between them, so detours and backtracking are visible rather than implied.
    """
    graph = graph or load_graph()
    pos = compute_layout(graph, result.scenario.profile)

    own_figure = ax is None
    if own_figure:
        fig, ax = plt.subplots(figsize=(12, 9))
    else:
        fig = ax.get_figure()

    visited = result.poi_ids
    _draw_base(ax, graph, pos, highlight=visited, node_size=20)

    # Walk the real graph path between consecutive stops so the drawn line
    # follows corridors rather than cutting through walls.
    start = result.scenario.start_poi or visited[0] if visited else None
    sequence = ([start] if start else []) + list(visited)
    cmap = plt.get_cmap("viridis")

    for i in range(len(sequence) - 1):
        path = graph.shortest_path(
            sequence[i],
            sequence[i + 1],
            profile=result.scenario.profile,
            accessibility=result.scenario.accessibility,
        )
        if not path:
            continue
        xs = [pos[p][0] for p in path]
        ys = [pos[p][1] for p in path]
        ax.plot(
            xs,
            ys,
            color=cmap(i / max(len(sequence) - 1, 1)),
            linewidth=2.0,
            alpha=0.85,
            zorder=3,
            solid_capstyle="round",
        )

    for order, pid in enumerate(visited, 1):
        x, y = pos[pid]
        ax.scatter([x], [y], s=110, c="white", edgecolors="black", linewidths=1.0, zorder=4)
        ax.text(x, y, str(order), fontsize=6, ha="center", va="center", zorder=5)

    ax.set_title(
        title
        or f"{result.scenario.name}  -  {result.summary()}",
        fontsize=10,
    )

    if own_figure:
        return _save(fig, output or OUTPUT_DIR / f"route_{result.scenario.name}.png")
    return None


def plot_comparison(
    results: Sequence[Tuple[str, EpisodeResult]],
    graph: Optional[VersaillesGraph] = None,
    output: Optional[Path | str] = None,
) -> Path:
    """Draw several agents' itineraries for one scenario side by side."""
    graph = graph or load_graph()
    n = len(results)
    cols = min(n, 2)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(11 * cols, 8 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (name, result) in zip(axes, results):
        plot_route(
            result,
            graph=graph,
            ax=ax,
            title=f"{name}  -  {result.summary()}",
        )
    for ax in axes[len(results):]:
        ax.axis("off")

    scenario = results[0][1].scenario
    fig.suptitle(f"{scenario.name}: {scenario.describe()}", fontsize=13)
    return _save(
        fig, output or OUTPUT_DIR / f"compare_{scenario.name}.png"
    )


def plot_training_curve(
    history: Dict[str, Sequence[float]],
    output: Path | str = OUTPUT_DIR / "training_curve.png",
    reference: Optional[Dict[str, float]] = None,
) -> Path:
    """
    Plot training metrics against baseline reference lines.

    Call this from your training loop with whatever you record; horizontal
    reference lines for the baselines make it obvious whether the policy is
    actually learning something useful or merely improving on itself.
    """
    keys = [k for k, v in history.items() if len(v)]
    if not keys:
        raise ValueError("history is empty")

    fig, axes = plt.subplots(len(keys), 1, figsize=(9, 3 * len(keys)), sharex=True)
    axes = np.atleast_1d(axes)

    for ax, key in zip(axes, keys):
        values = np.asarray(history[key], dtype=float)
        ax.plot(values, linewidth=1.2, label=key)
        if values.size > 20:
            window = max(values.size // 20, 2)
            smooth = np.convolve(values, np.ones(window) / window, mode="valid")
            ax.plot(
                np.arange(len(smooth)) + window - 1,
                smooth,
                linewidth=2.0,
                alpha=0.8,
                label=f"{key} (smoothed)",
            )
        if reference and key in reference:
            ax.axhline(
                reference[key],
                color="crimson",
                linestyle="--",
                linewidth=1.0,
                label="baseline",
            )
        ax.set_ylabel(key)
        ax.legend(fontsize=8, frameon=False)
        ax.grid(alpha=0.2)

    axes[-1].set_xlabel("update")
    fig.suptitle("Training progress")
    return _save(fig, output)


def _save(fig, output: Path | str) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", output)
    return output


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Visualize the estate and routes.")
    parser.add_argument("--map", action="store_true", help="draw the estate map")
    parser.add_argument("--route", metavar="SCENARIO", help="draw one agent's route")
    parser.add_argument("--compare", metavar="SCENARIO", help="draw all agents' routes")
    parser.add_argument("--agent", default="ortools", help="agent for --route")
    parser.add_argument("--output", help="output path")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    graph = load_graph()

    if not any([args.map, args.route, args.compare]):
        args.map = True

    from versailles.baselines import all_baselines
    from versailles.scenario import standard_suite

    suite = {s.name: s for s in standard_suite()}

    if args.map:
        print(plot_map(graph, args.output or OUTPUT_DIR / "estate_map.png"))

    if args.route:
        if args.route not in suite:
            parser.error(f"unknown scenario {args.route!r}; try one of {sorted(suite)}")
        agents = {a.name: a for a in all_baselines()}
        if args.agent not in agents:
            parser.error(f"unknown agent {args.agent!r}; try one of {sorted(agents)}")
        result = agents[args.agent].solve(suite[args.route])
        print(plot_route(result, graph, args.output))

    if args.compare:
        if args.compare not in suite:
            parser.error(f"unknown scenario {args.compare!r}")
        scenario = suite[args.compare]
        results = [(a.name, a.solve(scenario)) for a in all_baselines()]
        print(plot_comparison(results, graph, args.output))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
