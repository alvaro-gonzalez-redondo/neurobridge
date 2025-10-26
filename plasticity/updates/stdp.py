"""
STDP weight update policies.

Implements the classic STDP weight update with optional Oja normalization:

    Δw_ij = A_plus · x_pre,ij · spike_post,j      (potentiation)
          + A_minus · x_post,ij · spike_pre,ij    (depression)
          - λ_oja · x_post,ij² · w_ij              (Oja normalization)

with hard bounds: w_min ≤ w_ij ≤ w_max
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Tuple, Optional

if TYPE_CHECKING:
    from ...sparse_connections import StaticSparse
    from ...dense_connections import StaticDense
    import torch

from ..base import UpdatePolicyBase
import torch


class UpdateSTDPSparse(UpdatePolicyBase):
    """STDP update policy for sparse connections.

    Parameters
    ----------
    A_plus : float
        Learning rate for potentiation (typically positive, e.g., 1e-4).
    A_minus : float
        Learning rate for depression (typically negative, e.g., -1.2e-4).
    w_min : float
        Minimum weight value (hard clamp).
    w_max : float
        Maximum weight value (hard clamp).
    oja_decay : float
        Oja normalization factor (set to 0 to disable). Default: 1e-5.
    """

    def __init__(
        self,
        A_plus: float = 1e-4,
        A_minus: float = -1.2e-4,
        w_min: float = 0.0,
        w_max: float = 1.0,
        oja_decay: float = 1e-5
    ):
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.w_min = w_min
        self.w_max = w_max
        self.oja_decay = oja_decay

    def apply(
        self,
        conn: StaticSparse,
        eligibility: Tuple[torch.Tensor, torch.Tensor],
        learning_signal: torch.Tensor,
        modulators: Optional[dict] = None
    ) -> None:
        """Apply STDP weight updates.

        Parameters
        ----------
        conn : StaticSparse
            Sparse connection to update.
        eligibility : tuple of (x_pre, x_post)
            Both tensors of shape (num_synapses,) from EligibilitySTDPSparse.
        learning_signal : torch.Tensor
            Postsynaptic spike indicators, shape (num_post_neurons,).
        modulators : dict, optional
            Not used in basic STDP.
        """
        x_pre, x_post = eligibility

        # Get per-synapse learning signals (index into postsynaptic neurons)
        L_prime_per_synapse = learning_signal[conn.idx_pos]

        # Get per-synapse presynaptic spikes
        pre_spikes = conn.pre.get_spikes_at(conn.delay, conn.idx_pre).float()

        # Compute weight updates
        # Potentiation: A_plus · x_pre · spike_post
        potentiation = self.A_plus * x_pre * L_prime_per_synapse

        # Depression: A_minus · x_post · spike_pre
        depression = self.A_minus * x_post * pre_spikes

        # Homeostasis: -λ · x_post² · w (Oja normalization)
        homeostasis = -self.oja_decay * (x_post * x_post) * conn.weight

        # Apply updates
        conn.weight += potentiation + depression + homeostasis

        # Hard clamp
        conn.weight.clamp_(self.w_min, self.w_max)


class UpdateSTDPDense(UpdatePolicyBase):
    """STDP update policy for dense connections.

    Parameters
    ----------
    A_plus : float
        Learning rate for potentiation (typically positive, e.g., 1e-4).
    A_minus : float
        Learning rate for depression (typically negative, e.g., -1.2e-4).
    w_min : float
        Minimum weight value (hard clamp).
    w_max : float
        Maximum weight value (hard clamp).
    oja_decay : float
        Oja normalization factor (set to 0 to disable). Default: 1e-5.
    """

    def __init__(
        self,
        A_plus: float = 1e-4,
        A_minus: float = -1.2e-4,
        w_min: float = 0.0,
        w_max: float = 1.0,
        oja_decay: float = 1e-5
    ):
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.w_min = w_min
        self.w_max = w_max
        self.oja_decay = oja_decay

    def apply(
        self,
        conn: StaticDense,
        eligibility: Tuple[torch.Tensor, torch.Tensor],
        learning_signal: torch.Tensor,
        modulators: Optional[dict] = None
    ) -> None:
        """Apply STDP weight updates using outer products.

        Parameters
        ----------
        conn : StaticDense
            Dense connection to update.
        eligibility : tuple of (x_pre, x_post)
            x_pre: shape (num_pre_neurons,)
            x_post: shape (num_post_neurons,)
        learning_signal : torch.Tensor
            Postsynaptic spike indicators, shape (num_post_neurons,).
        modulators : dict, optional
            Not used in basic STDP.
        """
        x_pre, x_post = eligibility

        # Get current presynaptic spikes
        from ... import globals
        t = globals.simulator.local_circuit.t
        t_idx = (t - conn.delay) % conn.pre.delay_max
        pre_spikes = conn.pre._spike_buffer[:, t_idx].float().squeeze(-1)

        # Compute weight updates via outer products
        # Potentiation: A_plus · outer(x_pre, spike_post)
        potentiation = self.A_plus * torch.outer(x_pre, learning_signal)

        # Depression: A_minus · outer(spike_pre, x_post)
        depression = self.A_minus * torch.outer(pre_spikes, x_post)

        # Oja normalization: -λ · x_post² · W (broadcast)
        homeostasis = -self.oja_decay * (x_post * x_post) * conn.weight

        # Apply updates only where connections exist (respect mask)
        delta_w = (potentiation + depression + homeostasis) * conn.mask
        conn.weight += delta_w

        # Hard clamp
        conn.weight.clamp_(self.w_min, self.w_max)
