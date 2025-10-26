"""
Plastic connection classes with modular plasticity rules.

This module provides PlasticSparse and PlasticDense connections that extend
their static counterparts with plasticity capabilities using the modular
e-prop framework.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Union, Dict, Optional

if TYPE_CHECKING:
    from .connection import ConnectionSpec

from .sparse_connections import StaticSparse
from .dense_connections import StaticDense
from .plasticity.rule import PlasticityRule
from .plasticity.recipes.factories import build_rule_for_sparse, build_rule_for_dense


class PlasticSparse(StaticSparse):
    """Sparse connection with modular plasticity.

    Extends StaticSparse by adding a plasticity rule that updates weights
    according to local eligibility traces and learning signals.

    The plasticity rule can be specified in three ways:
    1. Dict configuration (built by factories)
    2. Pre-constructed PlasticityRule object
    3. None (no plasticity initially, can be added later with set_plasticity)

    Examples
    --------
    # Method 1: Dict configuration (factory-built)
    >>> spec = ConnectionSpec(
    ...     pre=source, pos=target,
    ...     src_idx=idx_pre, tgt_idx=idx_pos,
    ...     weight=weights, delay=delays,
    ...     params={
    ...         "plasticity": {
    ...             "name": "stdp",
    ...             "params": {
    ...                 "A_plus": 1e-4,
    ...                 "A_minus": -1.2e-4,
    ...                 "tau_plus": 20e-3,
    ...                 "tau_post": 20e-3,
    ...                 "w_min": 0.0,
    ...                 "w_max": 1.0
    ...             }
    ...         }
    ...     }
    ... )
    >>> conn = PlasticSparse(spec)

    # Method 2: Pre-constructed rule (manual composition)
    >>> signal = MultiplySignals([SignalLipshutzVoltage(), SignalReward()])
    >>> rule = PlasticityRule(
    ...     eligibility=EligibilityLipshutzVoltageSparse(...),
    ...     signal=signal,
    ...     update=UpdateOjaSparse(...)
    ... )
    >>> spec = ConnectionSpec(..., params={"plasticity": rule})
    >>> conn = PlasticSparse(spec)

    # Method 3: No plasticity initially, add later
    >>> spec = ConnectionSpec(..., params={"plasticity": None})
    >>> conn = PlasticSparse(spec)
    >>> # ... later ...
    >>> conn.set_plasticity(my_rule)

    Notes
    -----
    **Multi-GPU restriction**: PlasticSparse requires pre and pos neurons to be
    on the same device. This is a permanent design restriction for efficiency
    (see IMPLEMENTACIÓN DE EPROP.md, section 4.1).

    If you need learning across GPUs, use Bridge neurons as intermediaries.
    """

    rule: PlasticityRule

    def __init__(self, spec: ConnectionSpec):
        """Initialize a plastic sparse connection.

        Parameters
        ----------
        spec : ConnectionSpec
            Connection specification. Must include spec.params["plasticity"]
            which can be:
            - Dict: Factory configuration (e.g., {"name": "stdp", "params": {...}})
            - PlasticityRule: Pre-constructed rule object
            - None: No plasticity initially (can add later with set_plasticity)

        Raises
        ------
        RuntimeError
            If pre and pos neurons are on different devices (multi-GPU restriction).
        ValueError
            If spec.params["plasticity"] is missing.
        TypeError
            If spec.params["plasticity"] is not dict, PlasticityRule, or None.
        """
        # Validate multi-GPU restriction BEFORE calling super().__init__
        if spec.pre.device != spec.pos.device:
            raise RuntimeError(
                f"PlasticSparse requires pre and pos neurons on the same device. "
                f"Got pre on {spec.pre.device} and pos on {spec.pos.device}.\n"
                f"This is a permanent design restriction for efficiency "
                f"(plasticity requires local information only).\n"
                f"If you need learning across GPUs, use Bridge neurons as intermediaries."
            )

        # Initialize static connection
        super().__init__(spec)

        # Build or assign plasticity rule
        if "plasticity" not in spec.params:
            raise ValueError(
                "PlasticSparse requires spec.params['plasticity']. "
                "Can be dict (factory config), PlasticityRule (pre-built), or None."
            )

        plasticity = spec.params["plasticity"]

        if isinstance(plasticity, dict):
            # Build rule from factory configuration
            self.rule = build_rule_for_sparse(plasticity)
        elif isinstance(plasticity, PlasticityRule):
            # Use pre-constructed rule directly
            self.rule = plasticity
        elif plasticity is None:
            # No plasticity initially (can be added later)
            self.rule = None
        else:
            raise TypeError(
                f"spec.params['plasticity'] must be dict, PlasticityRule, or None. "
                f"Got {type(plasticity).__name__}"
            )

        # Initialize rule state if rule exists
        if self.rule is not None:
            self.rule.init_state(self)

        # Initialize monitoring variables (populated during first step)
        self.last_eligibility = None
        self.last_learning_signal = None

    def _update(self) -> None:
        """Apply plasticity rule to update weights.

        Called automatically each timestep from _process().
        Delegates to the modular plasticity rule.
        """
        if self.rule is not None:
            self.rule.step(self)

    def set_plasticity(self, rule: PlasticityRule) -> None:
        """Set or replace the plasticity rule.

        Useful for:
        - Adding plasticity to a connection created with plasticity=None
        - Switching between different learning rules dynamically
        - Composing custom rules manually

        Parameters
        ----------
        rule : PlasticityRule
            The plasticity rule to use. Must be a PlasticityRule instance
            with eligibility, signal, and update components.

        Examples
        --------
        # Add plasticity after creation:
        >>> conn = PlasticSparse(ConnectionSpec(..., params={"plasticity": None}))
        >>> conn.set_plasticity(my_rule)

        # Switch to different rule:
        >>> conn.set_plasticity(exploration_rule)  # exploration phase
        >>> # ... train ...
        >>> conn.set_plasticity(exploitation_rule)  # exploitation phase

        # Manual composition:
        >>> signal = MultiplySignals([SignalLipshutzVoltage(), SignalReward()])
        >>> rule = PlasticityRule(
        ...     eligibility=EligibilityLipshutzVoltageSparse(...),
        ...     signal=signal,
        ...     update=UpdateOjaSparse(...)
        ... )
        >>> conn.set_plasticity(rule)
        """
        self.rule = rule
        self.rule.init_state(self)


class PlasticDense(StaticDense):
    """Dense connection with modular plasticity.

    Extends StaticDense by adding a plasticity rule that updates weights
    according to local eligibility traces and learning signals.

    The plasticity rule can be specified in three ways:
    1. Dict configuration (built by factories)
    2. Pre-constructed PlasticityRule object
    3. None (no plasticity initially, can be added later with set_plasticity)

    Examples
    --------
    # Method 1: Dict configuration (factory-built)
    >>> spec = ConnectionSpec(
    ...     pre=source, pos=target,
    ...     src_idx=idx_pre, tgt_idx=idx_pos,
    ...     weight=weights, delay=delay,  # uniform delay for Dense
    ...     params={
    ...         "plasticity": {
    ...             "name": "stdp",
    ...             "params": {...}
    ...         }
    ...     }
    ... )
    >>> conn = PlasticDense(spec)

    # Method 2: Pre-constructed rule (manual composition)
    >>> signal = MultiplySignals([SignalLipshutzVoltage(), SignalReward()])
    >>> rule = PlasticityRule(
    ...     eligibility=EligibilityLipshutzVoltageDense(...),
    ...     signal=signal,
    ...     update=UpdateOjaDense(...)
    ... )
    >>> spec = ConnectionSpec(..., params={"plasticity": rule})
    >>> conn = PlasticDense(spec)

    # Method 3: No plasticity initially, add later
    >>> spec = ConnectionSpec(..., params={"plasticity": None})
    >>> conn = PlasticDense(spec)
    >>> # ... later ...
    >>> conn.set_plasticity(my_rule)

    Notes
    -----
    **Multi-GPU restriction**: PlasticDense requires pre and pos neurons to be
    on the same device. This is a permanent design restriction for efficiency
    (see IMPLEMENTACIÓN DE EPROP.md, section 4.1).

    If you need learning across GPUs, use Bridge neurons as intermediaries.
    """

    rule: PlasticityRule

    def __init__(self, spec: ConnectionSpec):
        """Initialize a plastic dense connection.

        Parameters
        ----------
        spec : ConnectionSpec
            Connection specification. Must include spec.params["plasticity"]
            which can be:
            - Dict: Factory configuration (e.g., {"name": "stdp", "params": {...}})
            - PlasticityRule: Pre-constructed rule object
            - None: No plasticity initially (can add later with set_plasticity)

        Raises
        ------
        RuntimeError
            If pre and pos neurons are on different devices (multi-GPU restriction).
        ValueError
            If spec.params["plasticity"] is missing.
        TypeError
            If spec.params["plasticity"] is not dict, PlasticityRule, or None.
        """
        # Validate multi-GPU restriction BEFORE calling super().__init__
        if spec.pre.device != spec.pos.device:
            raise RuntimeError(
                f"PlasticDense requires pre and pos neurons on the same device. "
                f"Got pre on {spec.pre.device} and pos on {spec.pos.device}.\n"
                f"This is a permanent design restriction for efficiency "
                f"(plasticity requires local information only).\n"
                f"If you need learning across GPUs, use Bridge neurons as intermediaries."
            )

        # Initialize static connection
        super().__init__(spec)

        # Build or assign plasticity rule
        if "plasticity" not in spec.params:
            raise ValueError(
                "PlasticDense requires spec.params['plasticity']. "
                "Can be dict (factory config), PlasticityRule (pre-built), or None."
            )

        plasticity = spec.params["plasticity"]

        if isinstance(plasticity, dict):
            # Build rule from factory configuration
            self.rule = build_rule_for_dense(plasticity)
        elif isinstance(plasticity, PlasticityRule):
            # Use pre-constructed rule directly
            self.rule = plasticity
        elif plasticity is None:
            # No plasticity initially (can be added later)
            self.rule = None
        else:
            raise TypeError(
                f"spec.params['plasticity'] must be dict, PlasticityRule, or None. "
                f"Got {type(plasticity).__name__}"
            )

        # Initialize rule state if rule exists
        if self.rule is not None:
            self.rule.init_state(self)

        # Initialize monitoring variables (populated during first step)
        self.last_eligibility = None
        self.last_learning_signal = None

    def _update(self) -> None:
        """Apply plasticity rule to update weights.

        Called automatically each timestep from _process().
        Delegates to the modular plasticity rule.
        """
        if self.rule is not None:
            self.rule.step(self)

    def set_plasticity(self, rule: PlasticityRule) -> None:
        """Set or replace the plasticity rule.

        Useful for:
        - Adding plasticity to a connection created with plasticity=None
        - Switching between different learning rules dynamically
        - Composing custom rules manually

        Parameters
        ----------
        rule : PlasticityRule
            The plasticity rule to use. Must be a PlasticityRule instance
            with eligibility, signal, and update components.

        Examples
        --------
        # Add plasticity after creation:
        >>> conn = PlasticDense(ConnectionSpec(..., params={"plasticity": None}))
        >>> conn.set_plasticity(my_rule)

        # Switch to different rule:
        >>> conn.set_plasticity(exploration_rule)  # exploration phase
        >>> # ... train ...
        >>> conn.set_plasticity(exploitation_rule)  # exploitation phase

        # Manual composition:
        >>> signal = MultiplySignals([SignalLipshutzVoltage(), SignalReward()])
        >>> rule = PlasticityRule(
        ...     eligibility=EligibilityLipshutzVoltageDense(...),
        ...     signal=signal,
        ...     update=UpdateOjaDense(...)
        ... )
        >>> conn.set_plasticity(rule)
        """
        self.rule = rule
        self.rule.init_state(self)
