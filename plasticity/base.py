"""
Base interfaces for the modular plasticity framework.

This module defines the abstract base classes for the three-component
plasticity system based on e-prop: Eligibility, LearningSignal, and UpdatePolicy.

The core idea is that any plasticity rule can be expressed as:
    Δw_ij(t) = L'_j(t) · e_ij(t)

where:
- e_ij(t) is the eligibility trace (local to the synapse)
- L'_j(t) is the learning signal (accessible at the postsynaptic neuron)
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Union
from ..sparse_connections import StaticSparse
from ..dense_connections import StaticDense

if TYPE_CHECKING:
    import torch

Connection = StaticSparse | StaticDense


class EligibilityBase(ABC):
    """Base class for eligibility trace computation.

    Eligibility traces represent the local synaptic history that determines
    which synapses are "eligible" for weight changes. Different rules use
    different eligibility computations:

    - STDP: Pre/post spike traces
    - Vogels: Presynaptic traces only
    - SFA: Difference-of-Exponentials (fast - slow traces)
    - etc.

    For Sparse connections, eligibilities are per-synapse tensors.
    For Dense connections, eligibilities are typically per-neuron (pre/post).
    """

    @abstractmethod
    def bind(self, conn: Connection) -> None:
        """Initialize state variables for this connection.

        Called once during connection initialization to allocate buffers
        for traces and other state variables.

        Parameters
        ----------
        conn : StaticSparse or StaticDense
            The connection this eligibility module is bound to.
        """
        pass

    @abstractmethod
    def step(self, conn: Connection) -> torch.Tensor:
        """Compute eligibility traces for the current timestep.

        This method should:
        1. Update internal traces (decay, add new spikes, etc.)
        2. Return the current eligibility values

        Parameters
        ----------
        conn : StaticSparse or StaticDense
            The connection to compute eligibilities for.

        Returns
        -------
        torch.Tensor
            Eligibility values. Shape depends on connection type:
            - Sparse: (num_synapses,) or tuple of (pre_trace, post_trace)
            - Dense: tuple of (pre_neurons, post_neurons) for outer product
        """
        pass


class LearningSignalBase(ABC):
    """Base class for learning signal computation.

    The learning signal L'_j(t) represents the "teaching" or "error" signal
    available at each postsynaptic neuron. Different rules use different signals:

    - STDP: Postsynaptic spike activity
    - Vogels: (z_post - ρ_0) homeostatic signal
    - SFA: High-pass filtered output (detects change)
    - Urbanczik & Senn: Soma-dendrite mismatch
    - etc.

    Learning signals are typically agnostic to Sparse/Dense (they operate on
    neuron groups, not individual synapses).
    """

    def bind(self, conn: Connection) -> None:
        """Initialize state variables for this connection (optional).

        Most learning signals don't need state (they compute from current
        neuron variables), but some (like HPF filters) do.

        Parameters
        ----------
        conn : StaticSparse or StaticDense
            The connection this signal module is bound to.
        """
        pass

    @abstractmethod
    def step(self, conn: Connection) -> torch.Tensor:
        """Compute learning signal for the current timestep.

        Parameters
        ----------
        conn : StaticSparse or StaticDense
            The connection to compute signals for. Typically uses conn.pos
            to access postsynaptic neuron states.

        Returns
        -------
        torch.Tensor
            Learning signal values of shape (num_post_neurons,)
        """
        pass


class UpdatePolicyBase(ABC):
    """Base class for weight update policies.

    The update policy determines how eligibility and learning signal are
    combined to produce weight changes. It also handles:

    - Weight dependence (soft bounds)
    - Hard clamping (w_min, w_max)
    - Synaptic scaling / normalization
    - Heterosynaptic competition
    - etc.

    Update policies need Sparse/Dense variants because they apply updates
    differently (index_add_ vs broadcasting).
    """

    def bind(self, conn: Connection) -> None:
        """Initialize state variables for this connection (optional).

        Most update policies don't need state, but some (like Oja's rule
        with filtered activity) do.

        Parameters
        ----------
        conn : StaticSparse or StaticDense
            The connection this update policy is bound to.
        """
        pass

    @abstractmethod
    def apply(
        self,
        conn: Connection,
        eligibility: torch.Tensor,
        learning_signal: torch.Tensor,
        modulators: Optional[dict] = None
    ) -> None:
        """Apply weight updates based on eligibility and learning signal.

        This method modifies conn.weight in-place.

        Parameters
        ----------
        conn : StaticSparse or StaticDense
            The connection whose weights to update.
        eligibility : torch.Tensor
            Eligibility traces from EligibilityBase.step()
        learning_signal : torch.Tensor
            Learning signals from LearningSignalBase.step()
        modulators : dict, optional
            Optional neuromodulatory signals (dopamine, acetylcholine, etc.)
            that can gate or scale plasticity.
        """
        pass
