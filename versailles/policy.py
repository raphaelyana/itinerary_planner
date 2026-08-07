"""
Where your RL policy goes.

The plumbing around the algorithm is done: observations are already flat float
arrays, the action mask is already boolean, and ``PolicyAgent`` already plugs
whatever you build into the evaluation harness and the route plots.

What is deliberately *not* written: the network, the loss, and the update rule.
Those are the parts worth writing yourself.

Two things about this environment that matter for the architecture:

1. **The action space is the node set.** 161 actions, of which only a handful
   are legal at any step. Score every POI and mask, rather than emitting a
   fixed-size categorical over something else.

2. **Masking must happen in logit space.** Set illegal actions to ``-inf``
   *before* the softmax, never zero the probabilities afterwards - otherwise the
   log-probabilities used by the policy gradient do not match the distribution
   you actually sampled from, and the gradients are quietly wrong. This is the
   single most common bug in masked policy-gradient implementations.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:  # torch is optional until you start training
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    nn = object  # type: ignore

from versailles.agent import Agent
from versailles.env import N_POI_FEATURES, VersaillesEnv


def masked_categorical(logits, mask):
    """
    Build a categorical distribution over legal actions only.

    Provided because getting this wrong is subtle and silent: illegal actions
    must be driven to ``-inf`` *before* the softmax so that sampling and
    ``log_prob`` agree. Everything else in the algorithm is yours.

    Parameters
    ----------
    logits : torch.Tensor
        Shape ``(batch, n_actions)``.
    mask : torch.Tensor
        Bool tensor of the same shape; True where the action is legal.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("torch is required; pip install torch")
    masked = logits.masked_fill(~mask.bool(), float("-inf"))
    return torch.distributions.Categorical(logits=masked)


def flatten_observation(obs: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Flatten an observation dict into a single float vector.

    Layout: ``[pois.ravel(), global]``, length ``n * N_POI_FEATURES + 4``.
    Use it if you want a plain MLP; ignore it if your network consumes the
    ``(n, N_POI_FEATURES)`` matrix directly, which is the better structure since
    it lets one shared encoder score every POI.
    """
    return np.concatenate([obs["pois"].ravel(), obs["global"]]).astype(np.float32)


# ======================================================================
# ===================  YOUR CODE STARTS HERE  ==========================
# ======================================================================
#
# Suggested shape for a policy network, given the observation format:
#
#   pois:   (batch, n, N_POI_FEATURES)   per-POI features
#   global: (batch, 4)                   budget and progress
#   mask:   (batch, n + 1)               legal actions
#
#   encoder:  Linear(N_POI_FEATURES -> h) applied to every POI
#             (optionally a few attention layers over the POI dimension so a
#             POI's score can depend on what else is still available)
#   context:  Linear(4 -> h) from the global features
#   actor:    score each POI from (poi_embedding, context) -> (batch, n)
#             plus one extra logit for the STOP action -> (batch, n + 1)
#   critic:   pooled POI embeddings + context -> (batch, 1)
#
# Then mask with `masked_categorical(logits, mask)` above.
#
# ======================================================================


class PolicyNetwork(nn.Module if TORCH_AVAILABLE else object):  # type: ignore[misc]
    """
    Your policy network.

    Must expose:
        forward(pois, global_features, mask) -> (distribution, value)

    where ``distribution`` comes from ``masked_categorical`` and ``value`` has
    shape ``(batch,)``.
    """

    def __init__(self, n_pois: int, n_features: int = N_POI_FEATURES, hidden: int = 128):
        if not TORCH_AVAILABLE:
            raise ImportError("torch is required; pip install torch")
        super().__init__()
        self.n_pois = n_pois
        self.n_features = n_features
        self.hidden = hidden

        # ----- YOUR LAYERS HERE -----
        raise NotImplementedError(
            "Define your policy network in versailles/policy.py"
        )

    def forward(self, pois, global_features, mask):
        # ----- YOUR FORWARD PASS HERE -----
        raise NotImplementedError


class PolicyAgent(Agent):
    """
    Adapter that lets a trained network act in the environment.

    Once ``PolicyNetwork`` works, this class needs no changes: it already makes
    your policy comparable to every baseline through ``versailles.evaluate`` and
    drawable through ``versailles.viz``.
    """

    name = "policy"

    def __init__(
        self,
        network: "PolicyNetwork",
        deterministic: bool = True,
        device: str = "cpu",
        name: Optional[str] = None,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("torch is required; pip install torch")
        self.network = network.to(device)
        self.network.eval()
        self.deterministic = deterministic
        self.device = device
        if name:
            self.name = name

    def act(self, obs: Dict[str, np.ndarray], env: VersaillesEnv) -> int:
        with torch.no_grad():
            pois = torch.as_tensor(obs["pois"], device=self.device).unsqueeze(0)
            glob = torch.as_tensor(obs["global"], device=self.device).unsqueeze(0)
            mask = torch.as_tensor(
                obs["mask"].astype(bool), device=self.device
            ).unsqueeze(0)

            dist, _ = self.network(pois, glob, mask)
            action = (
                torch.argmax(dist.logits, dim=-1)
                if self.deterministic
                else dist.sample()
            )
        return int(action.item())
