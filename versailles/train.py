"""
Training harness: everything around the RL algorithm, but not the algorithm.

What is provided here:
  - reproducible seeding
  - rollout collection against randomised scenarios
  - metric logging and training curves
  - checkpoint save/load
  - periodic evaluation against the baselines, so a training curve means
    something in absolute terms rather than only relative to itself

What is left to you, marked clearly below:
  - the update rule (advantage estimation, losses, optimiser step)

Run it:
    python -m versailles.train --updates 500 --run-name ppo-v1
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from versailles.env import VersaillesEnv
from versailles.graph import VersaillesGraph, load_graph
from versailles.scenario import Scenario, standard_suite, training_distribution

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = Path("checkpoints")


@dataclass
class TrainConfig:
    """Everything that defines a run. Saved next to the checkpoint."""

    run_name: str = "run"
    seed: int = 0
    updates: int = 500
    episodes_per_update: int = 32
    eval_every: int = 25
    eval_scenarios: int = 12
    device: str = "cpu"

    # Algorithm hyperparameters. Add whatever your method needs; they are
    # serialised with the checkpoint so a result can be reproduced later.
    learning_rate: float = 3e-4
    gamma: float = 0.99
    extra: Dict[str, float] = field(default_factory=dict)


@dataclass
class Rollout:
    """One episode's trajectory, ready for whatever update rule you write."""

    observations: List[Dict[str, np.ndarray]]
    actions: List[int]
    rewards: List[float]
    masks: List[np.ndarray]
    scenario: Scenario
    total_reward: float
    total_utility: float
    n_visited: int


class MetricLog:
    """Append-only metric store that can write curves and a JSON record."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.history: Dict[str, List[float]] = {}

    def record(self, **metrics: float) -> None:
        for key, value in metrics.items():
            self.history.setdefault(key, []).append(float(value))

    def save(self) -> None:
        (self.run_dir / "metrics.json").write_text(
            json.dumps(self.history, indent=2), encoding="utf-8"
        )

    def plot(self, reference: Optional[Dict[str, float]] = None) -> Optional[Path]:
        from versailles.viz import plot_training_curve

        if not any(self.history.values()):
            return None
        return plot_training_curve(
            self.history, self.run_dir / "training_curve.png", reference
        )


def collect_rollout(
    env: VersaillesEnv,
    agent,
    scenario: Optional[Scenario] = None,
) -> Rollout:
    """
    Run one episode and record everything an update rule might need.

    Works with any object exposing ``act(obs, env) -> int``, so you can collect
    rollouts from a baseline agent too - useful for behaviour cloning, or just
    to sanity-check the pipeline before the network exists.
    """
    obs, _ = env.reset(options={"scenario": scenario} if scenario else None)

    observations: List[Dict[str, np.ndarray]] = []
    actions: List[int] = []
    rewards: List[float] = []
    masks: List[np.ndarray] = []

    while True:
        observations.append({k: v.copy() for k, v in obs.items()})
        masks.append(obs["mask"].astype(bool).copy())

        action = agent.act(obs, env)
        obs, reward, done, _, _ = env.step(action)

        actions.append(int(action))
        rewards.append(float(reward))
        if done:
            break

    result = env.result()
    return Rollout(
        observations=observations,
        actions=actions,
        rewards=rewards,
        masks=masks,
        scenario=env.scenario,
        total_reward=float(sum(rewards)),
        total_utility=result.total_utility,
        n_visited=result.n_visited,
    )


def baseline_reference(
    graph: Optional[VersaillesGraph] = None,
    scenarios: Optional[Sequence[Scenario]] = None,
    ortools_seconds: int = 5,
) -> Dict[str, float]:
    """
    Mean baseline utility on the benchmark suite.

    Drawn as reference lines on the training curve. Without them a rising curve
    only shows the policy beating its former self, which is not the question.
    """
    from versailles.baselines import GreedyDensityAgent, ORToolsAgent
    from versailles.evaluate import compare

    graph = graph or load_graph()
    scenarios = scenarios or standard_suite()
    agents = [GreedyDensityAgent(), ORToolsAgent(time_limit_seconds=ortools_seconds)]
    result = compare(agents, scenarios, graph=graph)
    return result.mean_utility()


def evaluate_policy(
    agent,
    graph: Optional[VersaillesGraph] = None,
    scenarios: Optional[Sequence[Scenario]] = None,
) -> Dict[str, float]:
    """Score the current policy on the fixed benchmark suite."""
    from versailles.evaluate import score_agent

    graph = graph or load_graph()
    scenarios = list(scenarios or standard_suite())
    scores = score_agent(agent, scenarios, graph=graph)
    return {
        "eval_utility": float(np.mean([s.utility for s in scores])),
        "eval_pois": float(np.mean([s.n_pois for s in scores])),
        "eval_infeasible": float(sum(1 for s in scores if not s.feasible)),
    }


def save_checkpoint(path: Path, network, config: TrainConfig, update: int) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": network.state_dict(),
            "config": asdict(config),
            "update": update,
        },
        path,
    )
    logger.info("saved checkpoint %s", path)


def load_policy_agent(path: Path | str, device: str = "cpu", deterministic: bool = True):
    """
    Rebuild a trained policy from a checkpoint.

    Used by ``python -m versailles.evaluate --checkpoint ...`` so a trained
    policy can be dropped into the comparison table alongside the baselines.
    """
    import torch

    from versailles.policy import PolicyAgent, PolicyNetwork

    payload = torch.load(path, map_location=device, weights_only=False)
    graph = load_graph()

    network = PolicyNetwork(n_pois=graph.n)
    network.load_state_dict(payload["state_dict"])
    return PolicyAgent(network, deterministic=deterministic, device=device, name="policy")


def train(config: TrainConfig) -> None:
    """Main training loop. The algorithm goes where marked."""
    import torch

    from versailles.policy import PolicyAgent, PolicyNetwork

    rng = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)

    graph = load_graph()
    report = graph.validate()
    if not report.ok:
        raise SystemExit(f"graph data has errors, refusing to train:\n{report}")

    run_dir = CHECKPOINT_DIR / config.run_name
    log = MetricLog(run_dir)
    (run_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2), encoding="utf-8"
    )

    env = VersaillesEnv(graph=graph, randomize=True, seed=config.seed)
    network = PolicyNetwork(n_pois=graph.n).to(config.device)

    # ==================================================================
    # ===================  YOUR CODE STARTS HERE  ======================
    # ==================================================================
    #
    # optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate)
    #
    # ==================================================================

    optimizer = None  # <- replace

    reference = baseline_reference(graph)
    logger.info("baseline reference: %s", reference)

    agent = PolicyAgent(network, deterministic=False, device=config.device)
    started = time.time()

    for update in range(1, config.updates + 1):
        # ---- collect experience -------------------------------------
        scenarios = training_distribution(rng, config.episodes_per_update)
        rollouts = [collect_rollout(env, agent, sc) for sc in scenarios]

        log.record(
            train_reward=float(np.mean([r.total_reward for r in rollouts])),
            train_utility=float(np.mean([r.total_utility for r in rollouts])),
            train_pois=float(np.mean([r.n_visited for r in rollouts])),
        )

        # ==============================================================
        # ==================  YOUR UPDATE RULE HERE  ===================
        # ==============================================================
        #
        # `rollouts` holds, per episode:
        #     observations : list of dicts with "pois", "global", "mask"
        #     actions      : list of ints
        #     rewards      : list of floats
        #     masks        : list of bool arrays
        #
        # Steps a policy-gradient method would take:
        #   1. returns / advantages from rewards (and values, if you have a
        #      critic). Episodes have different lengths, so pad or concatenate.
        #   2. re-run the network over the stored observations to get fresh
        #      log-probabilities and values.
        #   3. compute your loss - policy term, value term, entropy bonus.
        #   4. backward, clip gradients, optimiser step.
        #
        # Use `masked_categorical(logits, mask)` from versailles.policy so the
        # log-probabilities match the distribution that was sampled from.
        #
        # Record whatever you want on the curve, e.g.:
        #   log.record(policy_loss=..., value_loss=..., entropy=...)
        # ==============================================================

        # ---- periodic evaluation ------------------------------------
        if update % config.eval_every == 0 or update == config.updates:
            greedy_agent = PolicyAgent(network, deterministic=True, device=config.device)
            metrics = evaluate_policy(greedy_agent, graph)
            log.record(**metrics)
            log.save()

            elapsed = time.time() - started
            logger.info(
                "update %d/%d  train_reward %.2f  eval_utility %.2f  (%.0fs)",
                update,
                config.updates,
                log.history["train_reward"][-1],
                metrics["eval_utility"],
                elapsed,
            )
            save_checkpoint(run_dir / "latest.pt", network, config, update)

    log.save()
    curve = log.plot(reference={"eval_utility": reference.get("ortools", 0.0)})
    if curve:
        logger.info("training curve: %s", curve)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train a Versailles planning policy.")
    parser.add_argument("--run-name", default="run")
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--episodes-per-update", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cpu", help="cpu, mps or cuda")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="verify the harness end to end using a baseline agent, no network",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.smoke_test:
        return _smoke_test(args.seed)

    train(
        TrainConfig(
            run_name=args.run_name,
            updates=args.updates,
            episodes_per_update=args.episodes_per_update,
            eval_every=args.eval_every,
            seed=args.seed,
            learning_rate=args.lr,
            device=args.device,
        )
    )
    return 0


def _smoke_test(seed: int) -> int:
    """
    Exercise the whole harness with a baseline agent instead of a network.

    Confirms that rollout collection, metric logging and evaluation all work
    before any RL code exists, so a later failure is unambiguously in the
    algorithm rather than the plumbing.
    """
    from versailles.baselines import GreedyDensityAgent

    graph = load_graph()
    rng = np.random.default_rng(seed)
    env = VersaillesEnv(graph=graph, randomize=True, seed=seed)
    agent = GreedyDensityAgent()

    log = MetricLog(CHECKPOINT_DIR / "smoke-test")
    for update in range(3):
        rollouts = [
            collect_rollout(env, agent, sc) for sc in training_distribution(rng, 8)
        ]
        log.record(
            train_reward=float(np.mean([r.total_reward for r in rollouts])),
            train_utility=float(np.mean([r.total_utility for r in rollouts])),
            train_pois=float(np.mean([r.n_visited for r in rollouts])),
        )
        print(
            f"update {update + 1}: reward "
            f"{log.history['train_reward'][-1]:.2f}, "
            f"{log.history['train_pois'][-1]:.1f} POIs/episode"
        )

    metrics = evaluate_policy(agent, graph)
    log.record(**metrics)
    log.save()
    print("eval:", metrics)
    curve = log.plot()
    print("curve:", curve)
    print("\nHarness OK. Implement PolicyNetwork in versailles/policy.py next.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
