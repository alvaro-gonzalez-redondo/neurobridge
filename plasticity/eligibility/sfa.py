"""
SFA (Slow Feature Analysis) eligibility trace implementations.

SFA uses difference-of-exponentials (DoE) traces:
    e(t) = x_fast(t) - x_slow(t)

where:
    x_fast(t) = α_fast · x_fast(t-1) + spike(t)
    x_slow(t) = α_slow · x_slow(t-1) + spike(t)

with α = exp(-dt/τ).

The learning signal for SFA is typically a high-pass filtered version
of the postsynaptic output (see signals/hpf.py).

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


class EligibilityDoESparse(EligibilityBase):
    """Difference-of-exponentials eligibility traces for sparse connections (SFA).

    Maintains per-synapse exponential traces with two different time constants.
    The eligibility trace is the difference between fast and slow components.

    Parameters
    ----------
    tau_fast : float
        Time constant for fast trace decay (in seconds). Default: 10e-3 (10ms)
    tau_slow : float
        Time constant for slow trace decay (in seconds). Default: 100e-3 (100ms)
    dt : float
        Simulation timestep (in seconds). Default: 1e-3 (1ms)

    Attributes
    ----------
    x_fast : torch.Tensor
        Fast exponential traces, shape (num_synapses,)
    x_slow : torch.Tensor
        Slow exponential traces, shape (num_synapses,)
    alpha_fast : torch.Tensor
        Decay factor for fast traces
    alpha_slow : torch.Tensor
        Decay factor for slow traces
    """

    def __init__(self, tau_fast: float = 10e-3, tau_slow: float = 100e-3, dt: float = 1e-3):
        self.tau_fast = tau_fast
        self.tau_slow = tau_slow
        self.dt = dt

        # State variables (allocated in bind())
        self.x_fast = None
        self.x_slow = None
        self.alpha_fast = None
        self.alpha_slow = None

    def bind(self, conn: StaticSparse) -> None:
        """Allocate trace buffers for this sparse connection."""
        num_synapses = conn.size
        device = conn.device

        # Allocate traces
        self.x_fast = torch.zeros(num_synapses, dtype=torch.float32, device=device)
        self.x_slow = torch.zeros(num_synapses, dtype=torch.float32, device=device)

        # Compute decay factors
        self.alpha_fast = torch.tensor(
            math.exp(-self.dt / self.tau_fast),
            dtype=torch.float32,
            device=device
        )
        self.alpha_slow = torch.tensor(
            math.exp(-self.dt / self.tau_slow),
            dtype=torch.float32,
            device=device
        )

    def step(self, conn: StaticSparse) -> torch.Tensor:
        """Compute eligibility traces for current timestep.

        Returns
        -------
        torch.Tensor
            Difference-of-exponentials eligibility, shape (num_synapses,)
        """
        # 1. Decay traces
        self.x_fast *= self.alpha_fast
        self.x_slow *= self.alpha_slow

        # 2. Get current presynaptic spikes (per-synapse)
        pre_spikes = conn.pre.get_spikes_at(conn.delay, conn.idx_pre)

        # 3. Add spike contributions
        spike_contrib = pre_spikes.to(torch.float32)
        self.x_fast += spike_contrib
        self.x_slow += spike_contrib

        # 4. Return difference
        return self.x_fast - self.x_slow


class EligibilityDoEDense(EligibilityBase):
    """Difference-of-exponentials eligibility traces for dense connections (SFA).

    Maintains per-neuron exponential traces (more efficient than per-synapse
    for dense connectivity). The eligibility trace is the difference between
    fast and slow components.

    Parameters
    ----------
    tau_fast : float
        Time constant for fast trace decay (in seconds). Default: 10e-3 (10ms)
    tau_slow : float
        Time constant for slow trace decay (in seconds). Default: 100e-3 (100ms)
    dt : float
        Simulation timestep (in seconds). Default: 1e-3 (1ms)

    Attributes
    ----------
    x_fast : torch.Tensor
        Fast exponential traces, shape (num_pre_neurons,)
    x_slow : torch.Tensor
        Slow exponential traces, shape (num_pre_neurons,)
    alpha_fast : torch.Tensor
        Decay factor for fast traces
    alpha_slow : torch.Tensor
        Decay factor for slow traces
    """

    def __init__(self, tau_fast: float = 10e-3, tau_slow: float = 100e-3, dt: float = 1e-3):
        self.tau_fast = tau_fast
        self.tau_slow = tau_slow
        self.dt = dt

        # State variables (allocated in bind())
        self.x_fast = None
        self.x_slow = None
        self.alpha_fast = None
        self.alpha_slow = None

    def bind(self, conn: StaticDense) -> None:
        """Allocate trace buffers for this dense connection."""
        num_pre = conn.pre.size
        device = conn.device

        # Allocate traces (per-neuron, not per-synapse)
        self.x_fast = torch.zeros(num_pre, dtype=torch.float32, device=device)
        self.x_slow = torch.zeros(num_pre, dtype=torch.float32, device=device)

        # Compute decay factors
        self.alpha_fast = torch.tensor(
            math.exp(-self.dt / self.tau_fast),
            dtype=torch.float32,
            device=device
        )
        self.alpha_slow = torch.tensor(
            math.exp(-self.dt / self.tau_slow),
            dtype=torch.float32,
            device=device
        )

    def step(self, conn: StaticDense) -> torch.Tensor:
        """Compute eligibility traces for current timestep.

        Returns
        -------
        torch.Tensor
            Difference-of-exponentials eligibility, shape (num_pre_neurons,)
        """
        # 1. Decay traces
        self.x_fast *= self.alpha_fast
        self.x_slow *= self.alpha_slow

        # 2. Get current presynaptic spikes (all neurons in pre group)
        from ... import globals
        t = globals.simulator.local_circuit.t
        t_idx = (t - conn.delay) % conn.pre.delay_max
        pre_spikes = conn.pre._spike_buffer[:, t_idx].float().squeeze(-1)

        # 3. Add spike contributions
        self.x_fast += pre_spikes
        self.x_slow += pre_spikes

        # 4. Return difference
        return self.x_fast - self.x_slow
