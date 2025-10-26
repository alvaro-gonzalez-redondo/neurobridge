"""
Lipshutz voltage-based learning signal for SFA using simple derivative.

Implements the postsynaptic component of Lipshutz et al. (2020) SFA using
temporal derivative of membrane voltage:
    L'_j(t) = V_j(t) - V_j(t-1) = dV_j/dt

where V_j is the membrane voltage of postsynaptic neuron j.

Combined with the presynaptic temporal difference in the eligibility trace,
this implements a derivative-based SFA rule:

    Δw_ij = η · (x̄_i - x_i) · (dV_j/dt)

This version is designed for continuous voltage neurons (e.g., StochasticIF
without hard resets). The temporal derivative is negative when voltage is
decreasing (stable, slow features) and positive when increasing (transient).

For smoothed versions with slow traces, use the SFA implementations in
recipes/factories.py instead.

Advantages:
- Simple and direct (no normalization or filtering)
- Biologically plausible (dendrites measure voltage directly)
- Works well with continuous voltage neurons

Reference:
----------
Lipshutz, D., et al. (2020). "A biologically plausible neural network for
slow feature analysis." NeurIPS.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from ...sparse_connections import StaticSparse
    from ...dense_connections import StaticDense

from ..base import LearningSignalBase


class SignalLipshutzVoltage(LearningSignalBase):
    """Lipshutz voltage-based learning signal using simple derivative.

    Computes the temporal derivative of postsynaptic membrane voltage:
        L'_j(t) = V_j(t) - V_j(t-1) = dV_j/dt

    This signal is negative when voltage is decreasing (stable, slow features)
    and positive when voltage is increasing (transient, fast features).

    Designed for continuous voltage neurons (e.g., StochasticIF without hard
    resets). For smoothed versions with slow traces, use the SFA
    implementations in recipes/factories.py instead.

    Attributes
    ----------
    V_prev : torch.Tensor or None
        Previous voltage values, shape (num_post_neurons,)
        None on first step (returns zeros)
    """

    def __init__(self):
        # State variable (allocated in bind())
        self.V_prev = None

    def bind(self, conn: StaticSparse | StaticDense) -> None:
        """Initialize state for derivative computation."""
        # Reset previous voltage (will be set on first step)
        self.V_prev = conn.pos.V.clone()

    def step(self, conn: StaticSparse | StaticDense) -> torch.Tensor:
        """Compute temporal derivative of voltage: L' = dV/dt = V(t) - V(t-1).

        Returns
        -------
        torch.Tensor
            Learning signal per postsynaptic neuron, shape (num_post_neurons,).
            Negative when voltage decreasing (stable), positive when increasing.
        """
        # Get current postsynaptic membrane voltage
        V_post = conn.pos.V

        # Compute temporal derivative (simple difference)
        # Negative = decreasing (stable, slow features)
        # Positive = increasing (transient, fast features)
        learning_signal = V_post - self.V_prev

        # Update previous voltage for next step
        self.V_prev = V_post.clone()

        return learning_signal
