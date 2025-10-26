"""
STDP eligibility trace implementations.

STDP eligibilities are exponentially decaying traces of pre/post spikes:
    x_pre(t) = α_pre · x_pre(t-1) + spike_pre(t)
    x_post(t) = α_post · x_post(t-1) + spike_post(t)

where α = exp(-dt/τ).

For Sparse connections, we maintain per-synapse traces.
For Dense connections, we maintain per-neuron traces (more efficient).
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from ...sparse_connections import StaticSparse
    from ...dense_connections import StaticDense
    import torch

from ..base import EligibilityBase
import torch
import math


class EligibilitySTDPSparse(EligibilityBase):
    """STDP eligibility traces for sparse connections.

    Maintains per-synapse exponential traces of pre and post spikes.

    Parameters
    ----------
    tau_pre : float
        Time constant for presynaptic trace decay (in seconds).
    tau_post : float
        Time constant for postsynaptic trace decay (in seconds).
    dt : float
        Simulation timestep (in seconds). Default: 1e-3 (1ms).

    Attributes
    ----------
    x_pre : torch.Tensor
        Presynaptic traces, shape (num_synapses,)
    x_post : torch.Tensor
        Postsynaptic traces, shape (num_synapses,)
    alpha_pre : torch.Tensor
        Decay factor for presynaptic traces
    alpha_post : torch.Tensor
        Decay factor for postsynaptic traces
    """

    def __init__(self, tau_pre: float = 20e-3, tau_post: float = 20e-3, dt: float = 1e-3):
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.dt = dt

        # State variables (allocated in bind())
        self.x_pre = None
        self.x_post = None
        self.alpha_pre = None
        self.alpha_post = None

    def bind(self, conn: StaticSparse) -> None:
        """Allocate trace buffers for this sparse connection."""
        num_synapses = conn.size
        device = conn.device

        # Allocate traces
        self.x_pre = torch.zeros(num_synapses, dtype=torch.float32, device=device)
        self.x_post = torch.zeros(num_synapses, dtype=torch.float32, device=device)

        # Compute decay factors
        self.alpha_pre = torch.tensor(
            math.exp(-self.dt / self.tau_pre),
            dtype=torch.float32,
            device=device
        )
        self.alpha_post = torch.tensor(
            math.exp(-self.dt / self.tau_post),
            dtype=torch.float32,
            device=device
        )

    def step(self, conn: StaticSparse) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute eligibility traces for current timestep.

        Returns
        -------
        tuple of (x_pre, x_post)
            Both tensors of shape (num_synapses,)
        """
        # 1. Decay traces
        self.x_pre *= self.alpha_pre
        self.x_post *= self.alpha_post

        # 2. Get current spikes (per-synapse)
        pre_spikes = conn.pre.get_spikes_at(conn.delay, conn.idx_pre)
        post_spikes = conn.pos.get_spikes_at(1, conn.idx_pos)

        # 3. Add spike contributions
        self.x_pre += pre_spikes.to(torch.float32)
        self.x_post += post_spikes.to(torch.float32)

        return self.x_pre, self.x_post


class EligibilitySTDPDense(EligibilityBase):
    """STDP eligibility traces for dense connections.

    Maintains per-neuron exponential traces (more efficient than per-synapse
    for dense connectivity).

    Parameters
    ----------
    tau_pre : float
        Time constant for presynaptic trace decay (in seconds).
    tau_post : float
        Time constant for postsynaptic trace decay (in seconds).
    dt : float
        Simulation timestep (in seconds). Default: 1e-3 (1ms).

    Attributes
    ----------
    x_pre : torch.Tensor
        Presynaptic traces, shape (num_pre_neurons,)
    x_post : torch.Tensor
        Postsynaptic traces, shape (num_post_neurons,)
    alpha_pre : torch.Tensor
        Decay factor for presynaptic traces
    alpha_post : torch.Tensor
        Decay factor for postsynaptic traces
    """

    def __init__(self, tau_pre: float = 20e-3, tau_post: float = 20e-3, dt: float = 1e-3):
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.dt = dt

        # State variables (allocated in bind())
        self.x_pre = None
        self.x_post = None
        self.alpha_pre = None
        self.alpha_post = None

    def bind(self, conn: StaticDense) -> None:
        """Allocate trace buffers for this dense connection."""
        num_pre = conn.pre.size
        num_post = conn.pos.size
        device = conn.device

        # Allocate traces (per-neuron, not per-synapse)
        self.x_pre = torch.zeros(num_pre, dtype=torch.float32, device=device)
        self.x_post = torch.zeros(num_post, dtype=torch.float32, device=device)

        # Compute decay factors
        self.alpha_pre = torch.tensor(
            math.exp(-self.dt / self.tau_pre),
            dtype=torch.float32,
            device=device
        )
        self.alpha_post = torch.tensor(
            math.exp(-self.dt / self.tau_post),
            dtype=torch.float32,
            device=device
        )

    def step(self, conn: StaticDense) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute eligibility traces for current timestep.

        Returns
        -------
        tuple of (x_pre, x_post)
            x_pre: shape (num_pre_neurons,)
            x_post: shape (num_post_neurons,)
        """
        # 1. Decay traces
        self.x_pre *= self.alpha_pre
        self.x_post *= self.alpha_post

        # 2. Get current spikes (all neurons in pre/post groups)
        from ... import globals
        t = globals.simulator.local_circuit.t
        t_idx = (t - conn.delay) % conn.pre.delay_max
        pre_spikes = conn.pre._spike_buffer[:, t_idx].float().squeeze(-1)
        post_spikes = conn.pos.get_spikes().float()

        # 3. Add spike contributions
        self.x_pre += pre_spikes
        self.x_post += post_spikes

        return self.x_pre, self.x_post
