"""
Postsynaptic spike-based learning signal.

This is the simplest learning signal: just the current postsynaptic spike activity.
Used in basic STDP and many other rules.

L'_j(t) = z_post,j(t)

where z_post,j is the spike indicator (0 or 1) for postsynaptic neuron j.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...sparse_connections import StaticSparse
    from ...dense_connections import StaticDense
    import torch

from ..base import LearningSignalBase


class SignalPostSpikes(LearningSignalBase):
    """Learning signal based on postsynaptic spikes.

    This is agnostic to Sparse/Dense (operates on neuron groups, not synapses).

    Returns the current spike activity of postsynaptic neurons as the learning signal.
    """

    def step(self, conn: StaticSparse | StaticDense) -> torch.Tensor:
        """Get postsynaptic spikes as learning signal.

        Parameters
        ----------
        conn : StaticSparse or StaticDense
            Connection to get postsynaptic spikes from.

        Returns
        -------
        torch.Tensor
            Postsynaptic spike indicators, shape (num_post_neurons,)
            Values are 0.0 (no spike) or 1.0 (spike).
        """
        return conn.pos.get_spikes().float()
