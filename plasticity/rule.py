"""
PlasticityRule: Orchestrator for modular plasticity components.

This module defines the PlasticityRule class that combines Eligibility,
LearningSignal, and UpdatePolicy into a complete plasticity rule.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Union
from ..sparse_connections import StaticSparse
from ..dense_connections import StaticDense

if TYPE_CHECKING:
    from .base import EligibilityBase, LearningSignalBase, UpdatePolicyBase

Connection = StaticSparse | StaticDense


class PlasticityRule:
    """Orchestrates eligibility, learning signal, and update policy.

    This class implements the core e-prop update:
        Δw_ij(t) = L'_j(t) · e_ij(t)

    by delegating to three modular components:
    - eligibility: Computes e_ij(t)
    - signal: Computes L'_j(t)
    - update: Applies Δw with constraints

    Examples
    --------
    >>> # STDP rule (constructed manually)
    >>> rule = PlasticityRule(
    ...     eligibility=EligibilitySTDPSparse(tau_pre=20e-3, tau_post=20e-3),
    ...     signal=SignalPostSpikes(),
    ...     update=UpdateSTDPSparse(A_plus=1e-4, A_minus=-1.2e-4)
    ... )
    >>>
    >>> # Usually created via factory
    >>> rule = _build_rule_for_sparse({"name": "stdp", "params": {...}})
    """

    def __init__(
        self,
        eligibility: EligibilityBase,
        signal: LearningSignalBase,
        update: UpdatePolicyBase,
        modulators: Optional[list] = None
    ):
        """Initialize a plasticity rule.

        Parameters
        ----------
        eligibility : EligibilityBase
            Component that computes eligibility traces.
        signal : LearningSignalBase
            Component that computes learning signals.
        update : UpdatePolicyBase
            Component that applies weight updates.
        modulators : list, optional
            List of neuromodulator components (not used in Fase 1).
        """
        self.eligibility = eligibility
        self.signal = signal
        self.update = update
        self.modulators = modulators or []

    def init_state(self, conn: Connection) -> None:
        """Initialize state for all components.

        Called once when the connection is created.

        Parameters
        ----------
        conn : StaticSparse or StaticDense
            The connection this rule will operate on.
        """
        self.eligibility.bind(conn)
        self.signal.bind(conn)
        self.update.bind(conn)  # Some update policies need state (e.g., Oja's filtered activity)

    def step(self, conn: Connection) -> None:
        """Execute one plasticity update step.

        This is called from PlasticSparse._update() or PlasticDense._update()
        each timestep.

        Parameters
        ----------
        conn : StaticSparse or StaticDense
            The connection to update.
        """
        # 1. Compute eligibility traces
        e = self.eligibility.step(conn)

        # 2. Compute learning signal
        L_prime = self.signal.step(conn)

        # 3. Save for debugging/monitoring (these can be accessed by VariableMonitor)
        # Handle tuple eligibilities (e.g., STDP returns (x_pre, x_post))
        if isinstance(e, tuple):
            import torch
            # Stack tuple into tensor with shape (num_components, ...)
            # E.g., STDP: (x_pre, x_post) → shape (2, num_neurons) or (2, num_synapses)
            conn.last_eligibility = torch.stack(e, dim=0)
        else:
            # Single tensor (e.g., Vogels returns only x_pre)
            # Wrap in extra dimension for consistency: shape (1, ...)
            conn.last_eligibility = e.unsqueeze(0)

        conn.last_learning_signal = L_prime

        # 4. Apply update (this modifies conn.weight in-place)
        # TODO: Add modulator support in later phases
        self.update.apply(conn, e, L_prime, modulators=None)
