"""
The agent interface every planner implements.

Anything that can choose actions - a random policy, a greedy heuristic, an
OR-Tools solution replayed step by step, or your trained RL policy - implements
``Agent``. Evaluation, visualization and the API all consume this interface, so
a new algorithm becomes comparable to every existing one the moment it satisfies
these two methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np

from versailles.env import EpisodeResult, VersaillesEnv
from versailles.scenario import Scenario


class Agent(ABC):
    """Chooses one action per step given an observation and its legal mask."""

    name: str = "agent"

    @abstractmethod
    def act(self, obs: Dict[str, np.ndarray], env: VersaillesEnv) -> int:
        """
        Return the index of the chosen action.

        The returned action **must** be legal under ``obs["mask"]``. The
        environment rejects illegal actions rather than silently accepting them.
        """

    def reset(self, scenario: Scenario) -> None:
        """Hook called once per episode, before the first ``act``."""

    def solve(
        self,
        scenario: Scenario,
        env: Optional[VersaillesEnv] = None,
        **env_kwargs: Any,
    ) -> EpisodeResult:
        """
        Run one full episode and return the resulting itinerary.

        Every agent is scored by driving the same environment, so constraint
        handling, timing and reward are identical across algorithms by
        construction rather than by convention.
        """
        env = env or VersaillesEnv(scenario=scenario, **env_kwargs)
        obs, _ = env.reset(options={"scenario": scenario})
        self.reset(scenario)

        while True:
            action = self.act(obs, env)
            if not obs["mask"][action]:
                raise ValueError(
                    f"{self.name} chose illegal action {action} "
                    f"({env.graph.order[action] if action < env.n else 'STOP'})"
                )
            obs, _, done, _, _ = env.step(action)
            if done:
                break

        return env.result()


class RandomAgent(Agent):
    """Uniform choice among legal actions. The floor every method must clear."""

    name = "random"

    def __init__(self, seed: int = 0, stop_probability: float = 0.0) -> None:
        self.rng = np.random.default_rng(seed)
        self.stop_probability = stop_probability

    def act(self, obs: Dict[str, np.ndarray], env: VersaillesEnv) -> int:
        mask = obs["mask"].astype(bool)
        legal = np.flatnonzero(mask)
        # Prefer to keep visiting: stopping early is almost never optimal, so
        # the floor baseline should not be crippled by random early exits.
        visits = legal[legal < env.n]
        if visits.size and self.rng.random() >= self.stop_probability:
            return int(self.rng.choice(visits))
        return int(self.rng.choice(legal))
