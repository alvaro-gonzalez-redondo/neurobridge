"""
Constant learning signal for autoorganized plasticity rules.

Used in plasticity rules where the learning dynamics are entirely captured
in the eligibility trace, and the learning signal is simply a constant
scaling factor.

Examples:
---------
- Lipshutz (2020) SFA: All temporal structure in eligibility e = (ȳ-y)(x̄-x)
- Eigendecomposition rules: Weight updates driven by covariance structure
- Oja's rule (pure form): Δw = η·y·x without additional modulation
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from ...sparse_connections import StaticSparse
    from ...dense_connections import StaticDense

from ..base import LearningSignalBase


class SignalConstant(LearningSignalBase):
    """Constant learning signal (always returns 1.0).

    This signal is used for plasticity rules where all the learning dynamics
    are captured in the eligibility trace itself, and the "learning signal"
    is simply a constant scaling factor.

    The weight update becomes:
        Δw_ij = η · e_ij · L'_j

    where L'_j = 1.0 (constant), so:
        Δw_ij = η · e_ij

    This is particularly useful for:
    1. **Lipshutz (2020) SFA**: The eligibility e = (ȳ-y)(x̄-x) contains
       all the temporal structure. No additional learning signal needed.

    2. **Eigendecomposition rules**: Weight updates are driven purely by
       the covariance structure encoded in eligibility.

    3. **Pure Hebbian/Oja**: Δw = η·y·x without additional modulation.

    Advantages:
    -----------
    - Simplifies implementation of autoorganized learning rules
    - No additional state variables needed
    - Minimal computational overhead
    - Cleaner separation of concerns (temporal structure in eligibility)

    Parameters
    ----------
    None. This signal has no parameters or state.
    """

    def __init__(self):
        # No parameters needed for constant signal
        pass

    def bind(self, conn: StaticSparse | StaticDense) -> None:
        """Initialize (no-op for constant signal).

        Constant signals don't need state, but we implement this method
        for compatibility with the LearningSignalBase interface.
        """
        pass

    def step(self, conn: StaticSparse | StaticDense) -> torch.Tensor:
        """Return constant learning signal (all ones).

        Returns
        -------
        torch.Tensor
            Constant signal of shape (num_post_neurons,), filled with 1.0
        """
        num_post = conn.pos.size
        device = conn.device
        return torch.ones(num_post, dtype=torch.float32, device=device)
