"""
Vogels inhibitory STDP update policy.

Implements the Vogels et al. (2011) inhibitory STDP rule:
    Δw_ij = η · x_pre · (z_post - ρ₀)

where:
    x_pre = presynaptic eligibility trace
    z_post = postsynaptic firing rate (smoothed)
    ρ₀ = target firing rate
    η = learning rate

This is an anti-Hebbian rule with homeostatic regulation, designed
for inhibitory synapses to maintain balanced network activity.

Reference:
    Vogels, T. P., Sprekeler, H., Zenke, F., Clopath, C., & Gerstner, W. (2011).
    Inhibitory plasticity balances excitation and inhibition in sensory pathways
    and memory networks. Science, 334(6062), 1569-1573.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Dict, Any
import torch

if TYPE_CHECKING:
    from ...sparse_connections import StaticSparse
    from ...dense_connections import StaticDense

from ..base import UpdatePolicyBase


class UpdateVogelsSparse(UpdatePolicyBase):
    """Vogels inhibitory STDP update for sparse connections.

    Applies anti-Hebbian weight updates with homeostatic regulation:
        Δw = η · x_pre · L'
    where L' = z_post - ρ₀ (from SignalRateHomeostasis)

    Parameters
    ----------
    eta : float
        Learning rate. Default: 1e-4
    w_min : float
        Minimum weight value (clamp). Default: 0.0
    w_max : float
        Maximum weight value (clamp). Default: 1.0

    Attributes
    ----------
    eta : float
        Learning rate
    w_min : float
        Minimum weight bound
    w_max : float
        Maximum weight bound
    """

    def __init__(self, eta: float = 1e-4, w_min: float = 0.0, w_max: float = 1.0):
        self.eta = eta
        self.w_min = w_min
        self.w_max = w_max

    def apply(
        self,
        conn: StaticSparse,
        eligibility: torch.Tensor,
        learning_signal: torch.Tensor,
        modulators: Optional[Dict[str, Any]] = None
    ) -> None:
        """Apply Vogels iSTDP weight update.

        Parameters
        ----------
        conn : StaticSparse
            The connection to update
        eligibility : torch.Tensor
            Presynaptic traces x_pre, shape (num_synapses,)
        learning_signal : torch.Tensor
            Homeostatic signal L' per postsynaptic neuron, shape (num_post_neurons,)
        modulators : dict, optional
            Additional modulation signals (not used in basic Vogels rule)
        """
        # Get learning signal for each synapse based on postsynaptic neuron
        L_prime_per_synapse = learning_signal[conn.idx_pos]

        # Apply update: Δw = η · x_pre · L'
        dw = self.eta * eligibility * L_prime_per_synapse

        # Update weights
        conn.weight += dw

        # Clamp weights to valid range
        conn.weight.clamp_(self.w_min, self.w_max)


class UpdateVogelsDense(UpdatePolicyBase):
    """Vogels inhibitory STDP update for dense connections.

    Applies anti-Hebbian weight updates with homeostatic regulation:
        Δw = η · x_pre · L'
    where L' = z_post - ρ₀ (from SignalRateHomeostasis)

    For dense connections, eligibility is per-neuron (not per-synapse),
    so we use outer product to compute per-synapse updates.

    Parameters
    ----------
    eta : float
        Learning rate. Default: 1e-4
    w_min : float
        Minimum weight value (clamp). Default: 0.0
    w_max : float
        Maximum weight value (clamp). Default: 1.0

    Attributes
    ----------
    eta : float
        Learning rate
    w_min : float
        Minimum weight bound
    w_max : float
        Maximum weight bound
    """

    def __init__(self, eta: float = 1e-4, w_min: float = 0.0, w_max: float = 1.0):
        self.eta = eta
        self.w_min = w_min
        self.w_max = w_max

    def apply(
        self,
        conn: StaticDense,
        eligibility: torch.Tensor,
        learning_signal: torch.Tensor,
        modulators: Optional[Dict[str, Any]] = None
    ) -> None:
        """Apply Vogels iSTDP weight update.

        Parameters
        ----------
        conn : StaticDense
            The connection to update
        eligibility : torch.Tensor
            Presynaptic traces x_pre, shape (num_pre_neurons,)
        learning_signal : torch.Tensor
            Homeostatic signal L' per postsynaptic neuron, shape (num_post_neurons,)
        modulators : dict, optional
            Additional modulation signals (not used in basic Vogels rule)
        """
        # Compute outer product: dw[i,j] = η · x_pre[i] · L'[j]
        # Shape: (num_pre, 1) × (1, num_post) → (num_pre, num_post)
        dw = self.eta * torch.outer(eligibility, learning_signal)

        # Update weights
        conn.weight += dw

        # Clamp weights to valid range
        conn.weight.clamp_(self.w_min, self.w_max)
