"""
How much a visitor is assumed to value each POI.

This module is deliberately small and deliberately separate. The reward the
agent maximises is only as good as the utility function behind it - a perfect
solver optimising the wrong objective still produces bad itineraries, and the
evaluation table cannot detect that, because every agent is scored against the
same assumption.

``LinearUtility`` holds the current hand-picked constants in one place, with
every parameter named and exposed. Anything implementing ``UtilityModel`` can
replace it, including a version whose parameters are learned rather than
guessed - see ``LearnableUtility`` for the intended shape.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Dict, Optional, Sequence

import numpy as np

from versailles.graph import VersaillesGraph


class UtilityModel(ABC):
    """Maps (graph, visitor interests) to a per-POI value vector."""

    name: str = "utility"

    @abstractmethod
    def values(
        self, graph: VersaillesGraph, interests: Sequence[str]
    ) -> np.ndarray:
        """Return a float array aligned with ``graph.order``."""

    # Cost weights live alongside utility because they are on the same scale:
    # the reward is `utility - travel_weight * minutes`, so tuning one without
    # the other is meaningless.
    travel_weight: float = 0.02
    wait_weight: float = 0.01


@dataclass
class LinearUtility(UtilityModel):
    """
    The default: curated priority scaled by interest-tag overlap.

        value = priority_score * (unmatched_floor + tag_weight * n_matched_tags)

    Parameters
    ----------
    unmatched_floor : float
        Fraction of its priority a POI keeps when it matches none of the
        visitor's interests. Stops a famous landmark scoring zero just because
        the visitor did not tick its category.
    tag_weight : float
        Value added per matched interest tag.
    saturating : bool
        When True, matched tags pass through ``sqrt`` so the tenth matching tag
        adds less than the first. Closer to how people actually judge appeal.
    travel_weight, wait_weight : float
        Minutes-to-utility exchange rates used by the environment's reward.
    """

    unmatched_floor: float = 0.25
    tag_weight: float = 1.0
    saturating: bool = False
    travel_weight: float = 0.02
    wait_weight: float = 0.01
    name: str = "linear"

    def values(
        self, graph: VersaillesGraph, interests: Sequence[str]
    ) -> np.ndarray:
        base = graph.priority.astype(np.float32)
        if not interests:
            return base

        wanted = np.zeros(len(graph.all_tags), dtype=np.float32)
        for tag in interests:
            idx = graph.all_tags.index(tag) if tag in graph.all_tags else None
            if idx is not None:
                wanted[idx] = 1.0

        matches = graph.tag_matrix @ wanted
        if self.saturating:
            matches = np.sqrt(matches)
        return base * (self.unmatched_floor + self.tag_weight * matches)

    def with_params(self, **kwargs) -> "LinearUtility":
        """Return a copy with some parameters overridden."""
        return replace(self, **kwargs)

    def to_dict(self) -> Dict[str, float]:
        return {
            "unmatched_floor": self.unmatched_floor,
            "tag_weight": self.tag_weight,
            "saturating": float(self.saturating),
            "travel_weight": self.travel_weight,
            "wait_weight": self.wait_weight,
        }


@dataclass
class PerTagUtility(UtilityModel):
    """
    Per-tag weights instead of one shared ``tag_weight``.

    This is the version worth *learning*: a 40-dimensional weight vector, one
    entry per interest tag, plus the two cost weights. Fit it so the planner
    reproduces itineraries you already consider good - your curated Castle and
    Trianon routes are exactly that kind of supervision - and the objective
    stops being a guess.

    Initialised to behave identically to ``LinearUtility`` so that swapping it
    in changes nothing until the weights are actually trained.
    """

    tag_weights: Optional[np.ndarray] = None
    unmatched_floor: float = 0.25
    travel_weight: float = 0.02
    wait_weight: float = 0.01
    name: str = "per_tag"

    def ensure_initialized(self, graph: VersaillesGraph) -> np.ndarray:
        if self.tag_weights is None:
            self.tag_weights = np.ones(len(graph.all_tags), dtype=np.float32)
        return self.tag_weights

    def values(
        self, graph: VersaillesGraph, interests: Sequence[str]
    ) -> np.ndarray:
        weights = self.ensure_initialized(graph)
        base = graph.priority.astype(np.float32)
        if not interests:
            return base

        wanted = np.zeros(len(graph.all_tags), dtype=np.float32)
        for tag in interests:
            if tag in graph.all_tags:
                wanted[graph.all_tags.index(tag)] = 1.0

        matches = graph.tag_matrix @ (wanted * weights)
        return base * (self.unmatched_floor + matches)

    # ------------------------------------------------------------------
    # ==================================================================
    # YOUR CODE: fitting the weights
    # ==================================================================
    #
    # `fit_to_reference` is where inverse optimal control would go: adjust
    # `tag_weights`, `unmatched_floor` and the cost weights so that a solver
    # using this utility reproduces itineraries you already judge good.
    #
    # A workable recipe, if you want one:
    #   1. Collect reference itineraries (scenario -> ordered POI ids).
    #   2. For candidate parameters, run GreedyDensityAgent or ORToolsAgent.
    #   3. Score agreement with the reference (Jaccard over POI sets, or
    #      Kendall tau over the order).
    #   4. Optimise with any derivative-free method - CMA-ES, or scipy's
    #      Nelder-Mead. The parameter vector is ~42-dimensional and each
    #      evaluation is a fast solve, so this is tractable on a laptop.
    #
    # Left unimplemented on purpose: this is a modelling decision, not
    # plumbing, and the choice of agreement metric determines what you learn.
    # ------------------------------------------------------------------

    def fit_to_reference(
        self,
        graph: VersaillesGraph,
        references: Dict[str, Sequence[str]],
        **kwargs,
    ) -> "PerTagUtility":
        raise NotImplementedError(
            "fit_to_reference is yours to implement; see the notes above."
        )


DEFAULT_UTILITY = LinearUtility()
