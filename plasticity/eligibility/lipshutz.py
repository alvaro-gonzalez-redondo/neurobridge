"""
Lipshutz (2020) eligibility trace implementations adapted for spiking neurons.

Original Lipshutz et al. (2020) SFA uses:
    ΔW = 2η(ȳ_i x̄_j - y_i x_j)

This can be rewritten as an eligibility trace:
    e_ij = (ȳ_post - y_post) × (x̄_pre - x_pre)

where:
    - x = presynaptic activity (spikes for spiking neurons)
    - x̄ = slow trace of presynaptic activity
    - y = postsynaptic activity (voltage for continuous signal)
    - ȳ = slow trace of postsynaptic activity

This implements a mexican-hat STDP kernel directly in the eligibility:
    - Potentiation when both signals are slow (ȳ > y, x̄ > x)
    - Depression when signals are fast (y > ȳ, x > x̄)

The key advantage for spiking neurons: Using membrane voltage V instead of
firing rate y provides a continuous signal that captures all temporal dynamics.

Reference:
---------
Lipshutz, D., et al. (2020). "A biologically plausible neural network for
slow feature analysis." NeurIPS.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import torch
import math

if TYPE_CHECKING:
    from ...sparse_connections import StaticSparse
    from ...dense_connections import StaticDense

from ..base import EligibilityBase


class EligibilityLipshutzVoltageSparse(EligibilityBase):
    """Lipshutz (2020) eligibility for sparse connections using voltage.

    Maintains slow exponential traces for presynaptic spikes. The eligibility
    is the temporal difference:

        e_i(t) = x̄_i(t) - x_i(t)

    Combined with the postsynaptic temporal difference from SignalLipshutzVoltage,
    this implements the mexican-hat STDP kernel from Lipshutz et al. (2020):

        Δw_ij = η · (x̄_i - x_i) × (V̄_j - V_j)

    Parameters
    ----------
    tau_slow_pre : float
        Time constant for slow presynaptic trace (in seconds).
        Default: 2500e-3 (2.5s) - captures very slow features
    dt : float
        Simulation timestep (in seconds). Default: 1e-3 (1ms)

    Attributes
    ----------
    x_bar : torch.Tensor
        Slow presynaptic traces, shape (num_synapses,)
    alpha_slow_pre : torch.Tensor
        Decay factor for presynaptic traces
    """

    def __init__(
        self,
        tau_slow_pre: float = 2500e-3,
        dt: float = 1e-3
    ):
        self.tau_slow_pre = tau_slow_pre
        self.dt = dt

        # State variables (allocated in bind())
        self.x_bar = None
        self.alpha_slow_pre = None

    def bind(self, conn: StaticSparse) -> None:
        """Allocate trace buffers for this sparse connection."""
        num_synapses = conn.size
        device = conn.device

        # Allocate presynaptic traces
        self.x_bar = torch.zeros(num_synapses, dtype=torch.float32, device=device)

        # Compute decay factor
        self.alpha_slow_pre = torch.tensor(
            math.exp(-self.dt / self.tau_slow_pre),
            dtype=torch.float32,
            device=device
        )

    def step(self, conn: StaticSparse) -> torch.Tensor:
        """Compute Lipshutz eligibility traces for current timestep.

        Returns
        -------
        torch.Tensor
            Presynaptic temporal difference, shape (num_synapses,)

        Note
        ----
        The postsynaptic component (V̄ - V) is computed by SignalLipshutzVoltage.
        The product e_ij = (x̄_i - x_i) × (V̄_j - V_j) is computed by
        the update rule (UpdateOjaSparse).
        """
        # === PRESYNAPTIC: Slow trace of spikes ===
        # 1. Decay slow trace
        self.x_bar *= self.alpha_slow_pre

        # 2. Get current presynaptic spikes (per-synapse)
        x_pre = conn.pre.get_spikes_at(conn.delay, conn.idx_pre).float()

        # 3. Add spike contributions to slow trace
        self.x_bar += x_pre

        # 4. Compute temporal difference (slow - fast)
        # This is the presynaptic eligibility component
        e_pre = self.x_bar - x_pre

        return e_pre


class EligibilityLipshutzVoltageDense(EligibilityBase):
    """Lipshutz (2020) eligibility for dense connections using voltage.

    Maintains slow exponential traces for presynaptic spikes. The eligibility
    is the temporal difference:

        e_i(t) = x̄_i(t) - x_i(t)

    Combined with the postsynaptic temporal difference from SignalLipshutzVoltage,
    this implements the mexican-hat STDP kernel from Lipshutz et al. (2020):

        Δw_ij = η · (x̄_i - x_i) × (V̄_j - V_j)

    For dense connections, we maintain per-neuron traces (more efficient than
    per-synapse). The outer product is computed by UpdateOjaDense.

    Parameters
    ----------
    tau_slow_pre : float
        Time constant for slow presynaptic trace (in seconds).
        Default: 2500e-3 (2.5s) - captures very slow features
    dt : float
        Simulation timestep (in seconds). Default: 1e-3 (1ms)

    Attributes
    ----------
    x_bar : torch.Tensor
        Slow presynaptic traces, shape (num_pre_neurons,)
    alpha_slow_pre : torch.Tensor
        Decay factor for presynaptic traces
    """

    def __init__(
        self,
        tau_slow_pre: float = 2500e-3,
        dt: float = 1e-3
    ):
        self.tau_slow_pre = tau_slow_pre
        self.dt = dt

        # State variables (allocated in bind())
        self.x_bar = None
        self.alpha_slow_pre = None

    def bind(self, conn: StaticDense) -> None:
        """Allocate trace buffers for this dense connection."""
        num_pre = conn.pre.size
        device = conn.device

        # Allocate presynaptic traces (per-neuron, not per-synapse)
        self.x_bar = torch.zeros(num_pre, dtype=torch.float32, device=device)

        # Compute decay factor
        self.alpha_slow_pre = torch.tensor(
            math.exp(-self.dt / self.tau_slow_pre),
            dtype=torch.float32,
            device=device
        )

    def step(self, conn: StaticDense) -> torch.Tensor:
        """Compute Lipshutz eligibility traces for current timestep.

        Returns
        -------
        torch.Tensor
            Presynaptic temporal difference, shape (num_pre_neurons,)

        Note
        ----
        The postsynaptic component (V̄ - V) is computed by SignalLipshutzVoltage.
        The outer product e_ij = (x̄_i - x_i) × (V̄_j - V_j) is computed by
        the update rule (UpdateOjaDense).
        """
        # === PRESYNAPTIC: Slow trace of spikes ===
        # 1. Decay slow trace
        self.x_bar *= self.alpha_slow_pre

        # 2. Get current presynaptic spikes (all neurons in pre group)
        from ... import globals
        t = globals.simulator.local_circuit.t
        t_idx = (t - conn.delay) % conn.pre.delay_max
        x_pre = conn.pre._spike_buffer[:, t_idx].float().squeeze(-1)

        # 3. Add spike contributions to slow trace
        self.x_bar += x_pre * (1-self.alpha_slow_pre)

        # 4. Compute temporal difference (slow - fast)
        # This is the presynaptic eligibility component
        e_pre = self.x_bar - x_pre  # shape: (num_pre,)

        return e_pre
