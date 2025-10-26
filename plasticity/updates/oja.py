"""
Oja's learning rule update policy.

Implements Oja's normalized Hebbian learning rule:
    Δw_ij = η · e_ij · L'_j - β · w_ij · ȳ_j²

where:
    e_ij = eligibility trace (typically DoE for SFA)
    L'_j = learning signal (typically HPF for SFA, can be modulated)
    ȳ_j = filtered postsynaptic activity (spike rate estimate)
    η = learning rate (Hebbian term)
    β = normalization coefficient

The normalization term uses the filtered postsynaptic spike rate ȳ_j, NOT the
learning signal L'_j. This is important because:
- L' can be modulated by reward, neuromodulators, etc.
- Normalization should depend on the actual neuron activity
- This maintains competition between synapses based on output activity

Reference:
    Oja, E. (1982). Simplified neuron model as a principal component analyzer.
    Journal of mathematical biology, 15(3), 267-273.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Dict, Any
import torch
import math

if TYPE_CHECKING:
    from ...sparse_connections import StaticSparse
    from ...dense_connections import StaticDense

from ..base import UpdatePolicyBase


class UpdateOjaSparse(UpdatePolicyBase):
    """Oja's learning rule for sparse connections.

    Applies normalized Hebbian learning:
        Δw = η · e · L' - β · w · ȳ²

    The normalization term uses filtered postsynaptic spike rate ȳ,
    not the learning signal L'. This keeps weights bounded without hard clamping
    and maintains proper competition based on actual neuron activity.

    Parameters
    ----------
    eta : float
        Hebbian learning rate. Default: 1e-4
    beta : float
        Normalization coefficient. Default: 1e-4
    tau_trace : float
        Time constant for postsynaptic activity filter (in seconds).
        Default: 20e-3 (20ms) - typical for spike rate estimation
    dt : float
        Simulation timestep (in seconds). Default: 1e-3 (1ms)
    w_min : float
        Minimum weight value (clamp). Default: 0.0
    w_max : float
        Maximum weight value (clamp). Default: 1.0

    Attributes
    ----------
    eta : float
        Hebbian learning rate
    beta : float
        Normalization coefficient
    tau_trace : float
        Time constant for activity filter
    dt : float
        Simulation timestep
    w_min : float
        Minimum weight bound
    w_max : float
        Maximum weight bound
    y_bar : torch.Tensor or None
        Filtered postsynaptic activity (spike rate estimate)
    alpha_trace : torch.Tensor or None
        Decay factor for activity filter
    """

    def __init__(
        self,
        eta: float = 1e-4,
        beta: float = 1e-4,
        tau_trace: float = 20e-3,
        dt: float = 1e-3,
        w_min: float = 0.0,
        w_max: float = 1.0
    ):
        self.eta = eta
        self.beta = beta
        self.tau_trace = tau_trace
        self.dt = dt
        self.w_min = w_min
        self.w_max = w_max

        # State variables (allocated in bind())
        self.y_bar = None
        self.alpha_trace = None

    def bind(self, conn: StaticSparse) -> None:
        """Initialize filtered activity state."""
        num_post = conn.pos.size
        device = conn.device

        # Allocate filtered postsynaptic activity trace
        self.y_bar = torch.zeros(num_post, dtype=torch.float32, device=device)

        # Compute decay factor
        self.alpha_trace = torch.tensor(
            math.exp(-self.dt / self.tau_trace),
            dtype=torch.float32,
            device=device
        )

    def apply(
        self,
        conn: StaticSparse,
        eligibility: torch.Tensor,
        learning_signal: torch.Tensor,
        modulators: Optional[Dict[str, Any]] = None
    ) -> None:
        """Apply Oja's learning rule weight update.

        Parameters
        ----------
        conn : StaticSparse
            The connection to update
        eligibility : torch.Tensor
            Eligibility traces e, shape (num_synapses,)
        learning_signal : torch.Tensor
            Learning signal L' per postsynaptic neuron, shape (num_post_neurons,)
        modulators : dict, optional
            Additional modulation signals (not used in basic Oja rule)
        """
        # Update filtered postsynaptic activity (spike rate estimate)
        y_post = conn.pos.get_spikes().float()  # shape: (num_post,)
        self.y_bar = self.alpha_trace * self.y_bar + (1 - self.alpha_trace) * y_post

        # Get learning signal for each synapse based on postsynaptic neuron
        L_prime_per_synapse = learning_signal[conn.idx_pos]

        # Hebbian term: η · e · L'
        hebbian = self.eta * eligibility * L_prime_per_synapse

        # Normalization term: -β · w · ȳ²
        # Uses filtered activity, NOT learning signal
        y_bar_per_synapse = self.y_bar[conn.idx_pos]
        normalization = -self.beta * conn.weight * (y_bar_per_synapse * y_bar_per_synapse)

        # Total update
        dw = hebbian + normalization

        # Update weights
        conn.weight += dw

        # Clamp weights to valid range
        conn.weight.clamp_(self.w_min, self.w_max)


class UpdateOjaDense(UpdatePolicyBase):
    """Oja's learning rule for dense connections.

    Applies normalized Hebbian learning:
        Δw = η · e · L' - β · w · ȳ²

    The normalization term uses filtered postsynaptic spike rate ȳ,
    not the learning signal L'. For dense connections, eligibility is
    per-neuron (not per-synapse), so we use outer product for the Hebbian term.

    Parameters
    ----------
    eta : float
        Hebbian learning rate. Default: 1e-4
    beta : float
        Normalization coefficient. Default: 1e-4
    tau_trace : float
        Time constant for postsynaptic activity filter (in seconds).
        Default: 20e-3 (20ms) - typical for spike rate estimation
    dt : float
        Simulation timestep (in seconds). Default: 1e-3 (1ms)
    w_min : float
        Minimum weight value (clamp). Default: 0.0
    w_max : float
        Maximum weight value (clamp). Default: 1.0

    Attributes
    ----------
    eta : float
        Hebbian learning rate
    beta : float
        Normalization coefficient
    tau_trace : float
        Time constant for activity filter
    dt : float
        Simulation timestep
    w_min : float
        Minimum weight bound
    w_max : float
        Maximum weight bound
    y_bar : torch.Tensor or None
        Filtered postsynaptic activity (spike rate estimate)
    alpha_trace : torch.Tensor or None
        Decay factor for activity filter
    """

    def __init__(
        self,
        eta: float = 1e-4,
        beta: float = 1e-4,
        tau_trace: float = 100e-3,
        dt: float = 1e-3,
        w_min: float = 0.0,
        w_max: float = 1.0
    ):
        self.eta = eta
        self.beta = beta
        self.tau_trace = tau_trace
        self.dt = dt
        self.w_min = w_min
        self.w_max = w_max

        # State variables (allocated in bind())
        self.y_bar = None
        self.alpha_trace = None

    def bind(self, conn: StaticDense) -> None:
        """Initialize filtered activity state."""
        num_post = conn.pos.size
        device = conn.device

        # Allocate filtered postsynaptic activity trace
        self.y_bar = torch.zeros(num_post, dtype=torch.float32, device=device)

        # Compute decay factor
        self.alpha_trace = torch.tensor(
            math.exp(-self.dt / self.tau_trace),
            dtype=torch.float32,
            device=device
        )

    def apply(
        self,
        conn: StaticDense,
        eligibility: torch.Tensor,
        learning_signal: torch.Tensor,
        modulators: Optional[Dict[str, Any]] = None
    ) -> None:
        """Apply Oja's learning rule weight update.

        Parameters
        ----------
        conn : StaticDense
            The connection to update
        eligibility : torch.Tensor
            Eligibility traces e, shape (num_pre_neurons,)
        learning_signal : torch.Tensor
            Learning signal L' per postsynaptic neuron, shape (num_post_neurons,)
        modulators : dict, optional
            Additional modulation signals (not used in basic Oja rule)
        """
        # Update filtered postsynaptic activity (spike rate estimate)
        y_post = conn.pos.get_spikes().float()  # shape: (num_post,)
        self.y_bar = self.alpha_trace * self.y_bar + (1 - self.alpha_trace) * y_post

        # Hebbian term: η · outer(e, L')
        # Shape: (num_pre, 1) × (1, num_post) → (num_pre, num_post)
        hebbian = self.eta * torch.outer(eligibility, learning_signal)

        # Normalization term: -β · w · ȳ²
        # Uses filtered activity, NOT learning signal
        # Broadcast ȳ² across presynaptic dimension
        y_bar_sq = self.y_bar * self.y_bar  # (num_post,)
        normalization = -self.beta * conn.weight * y_bar_sq.unsqueeze(0)

        # Total update
        dw = hebbian + normalization

        # Update weights
        conn.weight += dw

        # Clamp weights to valid range
        conn.weight.clamp_(self.w_min, self.w_max)
