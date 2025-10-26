"""
Vogels (iSTDP) eligibility trace implementations.

Vogels inhibitory STDP only requires presynaptic traces:
    x_pre(t) = α_pre · x_pre(t-1) + spike_pre(t)

where α = exp(-dt/τ).

The learning signal handles the homeostatic component (L' = z_post - ρ₀).

For Sparse connections, we maintain per-synapse traces.
For Dense connections, we maintain per-neuron traces (more efficient).
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Tuple
import torch
import math

if TYPE_CHECKING:
    from ...sparse_connections import StaticSparse
    from ...dense_connections import StaticDense

from ..base import EligibilityBase


class EligibilityPreSparse(EligibilityBase):
    """Presynaptic-only eligibility traces for sparse connections (Vogels iSTDP).

    Maintains per-synapse exponential traces of presynaptic spikes only.

    Parameters
    ----------
    tau_pre : float
        Time constant for presynaptic trace decay (in seconds).
    dt : float
        Simulation timestep (in seconds). Default: 1e-3 (1ms).

    Attributes
    ----------
    x_pre : torch.Tensor
        Presynaptic traces, shape (num_synapses,)
    alpha_pre : torch.Tensor
        Decay factor for presynaptic traces
    """

    def __init__(self, tau_pre: float = 20e-3, dt: float = 1e-3):
        self.tau_pre = tau_pre
        self.dt = dt

        # State variables (allocated in bind())
        self.x_pre = None
        self.alpha_pre = None

    def bind(self, conn: StaticSparse) -> None:
        """Allocate trace buffers for this sparse connection."""
        num_synapses = conn.size
        device = conn.device

        # Allocate traces
        self.x_pre = torch.zeros(num_synapses, dtype=torch.float32, device=device)

        # Compute decay factor
        self.alpha_pre = torch.tensor(
            math.exp(-self.dt / self.tau_pre),
            dtype=torch.float32,
            device=device
        )

    def step(self, conn: StaticSparse) -> torch.Tensor:
        """Compute eligibility traces for current timestep.

        Returns
        -------
        torch.Tensor
            Presynaptic traces, shape (num_synapses,)
        """
        # 1. Decay traces
        self.x_pre *= self.alpha_pre

        # 2. Get current presynaptic spikes (per-synapse)
        pre_spikes = conn.pre.get_spikes_at(conn.delay, conn.idx_pre)

        # 3. Add spike contributions
        self.x_pre += pre_spikes.to(torch.float32)

        return self.x_pre


class EligibilityPreDense(EligibilityBase):
    """Presynaptic-only eligibility traces for dense connections (Vogels iSTDP).

    Maintains per-neuron exponential traces (more efficient than per-synapse
    for dense connectivity).

    Parameters
    ----------
    tau_pre : float
        Time constant for presynaptic trace decay (in seconds).
    dt : float
        Simulation timestep (in seconds). Default: 1e-3 (1ms).

    Attributes
    ----------
    x_pre : torch.Tensor
        Presynaptic traces, shape (num_pre_neurons,)
    alpha_pre : torch.Tensor
        Decay factor for presynaptic traces
    """

    def __init__(self, tau_pre: float = 20e-3, dt: float = 1e-3):
        self.tau_pre = tau_pre
        self.dt = dt

        # State variables (allocated in bind())
        self.x_pre = None
        self.alpha_pre = None

    def bind(self, conn: StaticDense) -> None:
        """Allocate trace buffers for this dense connection."""
        num_pre = conn.pre.size
        device = conn.device

        # Allocate traces (per-neuron, not per-synapse)
        self.x_pre = torch.zeros(num_pre, dtype=torch.float32, device=device)

        # Compute decay factor
        self.alpha_pre = torch.tensor(
            math.exp(-self.dt / self.tau_pre),
            dtype=torch.float32,
            device=device
        )

    def step(self, conn: StaticDense) -> torch.Tensor:
        """Compute eligibility traces for current timestep.

        Returns
        -------
        torch.Tensor
            Presynaptic traces, shape (num_pre_neurons,)
        """
        # 1. Decay traces
        self.x_pre *= self.alpha_pre

        # 2. Get current presynaptic spikes (all neurons in pre group)
        from ... import globals
        t = globals.simulator.local_circuit.t
        t_idx = (t - conn.delay) % conn.pre.delay_max
        pre_spikes = conn.pre._spike_buffer[:, t_idx].float().squeeze(-1)

        # 3. Add spike contributions
        self.x_pre += pre_spikes

        return self.x_pre
