"""
Composite learning signals for modular composition.

These classes allow combining multiple learning signals together using
arithmetic operations, enabling complex plasticity rules from simple components.

Examples
--------
# Reinforcement Learning SFA:
>>> signal = MultiplySignals([
...     SignalLipshutzVoltage(),  # temporal structure
...     SignalReward()            # task performance
... ])

# Multi-objective learning:
>>> signal = AddSignals([
...     MultiplySignals([SignalLipshutzVoltage(), SignalReward()]),
...     SignalRateHomeostasis(target=10.0)
... ], weights=[1.0, 0.1])

# Three-factor rule (dopamine-modulated STDP):
>>> signal = MultiplySignals([
...     SignalPostSpikes(),
...     SignalDopamine()
... ])
"""

from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional
import torch

if TYPE_CHECKING:
    from ...sparse_connections import StaticSparse
    from ...dense_connections import StaticDense

from ..base import LearningSignalBase


class MultiplySignals(LearningSignalBase):
    """Multiply multiple learning signals element-wise.

    Computes the element-wise product of multiple signals:
        L'(t) = signal₁(t) × signal₂(t) × ... × signalₙ(t)

    This is useful for:
    - **Gating**: One signal gates another (e.g., reward gates temporal learning)
    - **Modulation**: Neuromodulators modulate plasticity (e.g., dopamine-modulated STDP)
    - **Three-factor rules**: Combine pre, post, and modulatory signals

    Mathematically, for the weight update:
        Δw_ij = η · e_ij · (L'₁ × L'₂ × ... × L'ₙ)

    Parameters
    ----------
    signals : List[LearningSignalBase]
        List of learning signal objects to multiply together.
        Each signal must implement the LearningSignalBase interface.

    Attributes
    ----------
    signals : List[LearningSignalBase]
        The component signals

    Examples
    --------
    # Reward-modulated temporal learning:
    >>> signal = MultiplySignals([
    ...     SignalLipshutzVoltage(tau_slow=2500e-3),
    ...     SignalReward()
    ... ])
    >>> rule = PlasticityRule(
    ...     eligibility=EligibilityLipshutzVoltageDense(...),
    ...     signal=signal,
    ...     update=UpdateOjaDense(...)
    ... )

    # Dopamine-modulated STDP:
    >>> signal = MultiplySignals([
    ...     SignalPostSpikes(),
    ...     SignalDopamine()
    ... ])
    >>> rule = PlasticityRule(
    ...     eligibility=EligibilitySTDPDense(...),
    ...     signal=signal,
    ...     update=UpdateSTDPDense(...)
    ... )
    """

    def __init__(self, signals: List[LearningSignalBase]):
        if len(signals) < 2:
            raise ValueError(
                f"MultiplySignals requires at least 2 signals, got {len(signals)}"
            )
        self.signals = signals

    def bind(self, conn: StaticSparse | StaticDense) -> None:
        """Bind all component signals to the connection."""
        for signal in self.signals:
            signal.bind(conn)

    def step(self, conn: StaticSparse | StaticDense) -> torch.Tensor:
        """Compute element-wise product of all signals.

        Returns
        -------
        torch.Tensor
            Product of all signals, shape (num_post_neurons,)
        """
        # Start with ones
        result = torch.ones(conn.pos.size, dtype=torch.float32, device=conn.device)

        # Multiply by each signal
        for signal in self.signals:
            result = result * signal.step(conn)

        return result


class AddSignals(LearningSignalBase):
    """Add multiple learning signals with optional weights.

    Computes the weighted sum of multiple signals:
        L'(t) = w₁·signal₁(t) + w₂·signal₂(t) + ... + wₙ·signalₙ(t)

    This is useful for:
    - **Multi-objective learning**: Balance multiple learning objectives
    - **Ensemble methods**: Combine different learning rules
    - **Regularization**: Add homeostatic or normalization terms

    Mathematically, for the weight update:
        Δw_ij = η · e_ij · (w₁·L'₁ + w₂·L'₂ + ... + wₙ·L'ₙ)

    Parameters
    ----------
    signals : List[LearningSignalBase]
        List of learning signal objects to add together.
        Each signal must implement the LearningSignalBase interface.
    weights : List[float], optional
        Weight coefficients for each signal. If None, all weights are 1.0.
        Must have same length as signals list.

    Attributes
    ----------
    signals : List[LearningSignalBase]
        The component signals
    weights : List[float]
        Weight coefficients for each signal

    Examples
    --------
    # Combine RL with homeostasis:
    >>> signal = AddSignals([
    ...     MultiplySignals([SignalLipshutzVoltage(), SignalReward()]),
    ...     SignalRateHomeostasis(target=10.0)
    ... ], weights=[1.0, 0.1])
    >>> rule = PlasticityRule(
    ...     eligibility=EligibilityLipshutzVoltageDense(...),
    ...     signal=signal,
    ...     update=UpdateOjaDense(...)
    ... )

    # Ensemble of different learning signals:
    >>> signal = AddSignals([
    ...     SignalHPFPost(tau_hpf=100e-3),
    ...     SignalTemporalSurrogate(gamma=1.0),
    ...     SignalRateHomeostasis(target=20.0)
    ... ], weights=[0.5, 0.3, 0.2])
    """

    def __init__(
        self,
        signals: List[LearningSignalBase],
        weights: Optional[List[float]] = None
    ):
        if len(signals) < 2:
            raise ValueError(
                f"AddSignals requires at least 2 signals, got {len(signals)}"
            )

        self.signals = signals

        # Default weights to 1.0 if not provided
        if weights is None:
            self.weights = [1.0] * len(signals)
        else:
            if len(weights) != len(signals):
                raise ValueError(
                    f"Number of weights ({len(weights)}) must match "
                    f"number of signals ({len(signals)})"
                )
            self.weights = weights

    def bind(self, conn: StaticSparse | StaticDense) -> None:
        """Bind all component signals to the connection."""
        for signal in self.signals:
            signal.bind(conn)

    def step(self, conn: StaticSparse | StaticDense) -> torch.Tensor:
        """Compute weighted sum of all signals.

        Returns
        -------
        torch.Tensor
            Weighted sum of all signals, shape (num_post_neurons,)
        """
        # Start with zeros
        result = torch.zeros(conn.pos.size, dtype=torch.float32, device=conn.device)

        # Add each weighted signal
        for signal, weight in zip(self.signals, self.weights):
            result = result + weight * signal.step(conn)

        return result
