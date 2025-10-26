"""
Factory functions for building plasticity rules.

These functions automatically select the appropriate Sparse or Dense variants
of eligibility, signal, and update components based on the connection type.
"""

from __future__ import annotations
from typing import Dict, Any

from ..rule import PlasticityRule
from ..eligibility.stdp import EligibilitySTDPSparse, EligibilitySTDPDense
from ..eligibility.vogels import EligibilityPreSparse, EligibilityPreDense
from ..eligibility.sfa import EligibilityDoESparse, EligibilityDoEDense
from ..eligibility.lipshutz import EligibilityLipshutzVoltageSparse, EligibilityLipshutzVoltageDense
from ..signals.post_spikes import SignalPostSpikes
from ..signals.homeostasis import SignalRateHomeostasis
from ..signals.hpf import SignalHPFPost, SignalHPFVoltage, SignalTemporalSurrogate
from ..signals.constant import SignalConstant
from ..signals.lipshutz import SignalLipshutzVoltage
from ..updates.stdp import UpdateSTDPSparse, UpdateSTDPDense
from ..updates.vogels import UpdateVogelsSparse, UpdateVogelsDense
from ..updates.oja import UpdateOjaSparse, UpdateOjaDense


def build_rule_for_sparse(config: Dict[str, Any]) -> PlasticityRule:
    """Build a plasticity rule for sparse connections.

    Parameters
    ----------
    config : dict
        Configuration dictionary with keys:
        - "name": str, rule name (e.g., "stdp", "vogels", "sfa")
        - "params": dict, optional parameters for the rule components

    Returns
    -------
    PlasticityRule
        Configured rule with Sparse-compatible components.

    Raises
    ------
    ValueError
        If rule name is unknown or config is invalid.

    Examples
    --------
    >>> config = {
    ...     "name": "stdp",
    ...     "params": {
    ...         "A_plus": 1e-4,
    ...         "tau_plus": 20e-3
    ...     }
    ... }
    >>> rule = build_rule_for_sparse(config)
    """
    rule_name = config.get("name")
    params = config.get("params", {})

    if rule_name == "stdp":
        return _build_stdp_sparse(params)
    elif rule_name == "vogels":
        return _build_vogels_sparse(params)
    elif rule_name == "sfa":
        return _build_sfa_sparse(params)
    elif rule_name == "lipshutz_voltage":
        return _build_lipshutz_voltage_sparse(params)
    else:
        raise ValueError(
            f"Unknown plasticity rule: '{rule_name}'. "
            f"Available rules: 'stdp', 'vogels', 'sfa', 'lipshutz_voltage'"
        )


def build_rule_for_dense(config: Dict[str, Any]) -> PlasticityRule:
    """Build a plasticity rule for dense connections.

    Parameters
    ----------
    config : dict
        Configuration dictionary with keys:
        - "name": str, rule name (e.g., "stdp", "vogels", "sfa")
        - "params": dict, optional parameters for the rule components

    Returns
    -------
    PlasticityRule
        Configured rule with Dense-compatible components.

    Raises
    ------
    ValueError
        If rule name is unknown or config is invalid.

    Examples
    --------
    >>> config = {
    ...     "name": "stdp",
    ...     "params": {
    ...         "A_plus": 1e-4,
    ...         "tau_plus": 20e-3
    ...     }
    ... }
    >>> rule = build_rule_for_dense(config)
    """
    rule_name = config.get("name")
    params = config.get("params", {})

    if rule_name == "stdp":
        return _build_stdp_dense(params)
    elif rule_name == "vogels":
        return _build_vogels_dense(params)
    elif rule_name == "sfa":
        return _build_sfa_dense(params)
    elif rule_name == "lipshutz_voltage":
        return _build_lipshutz_voltage_dense(params)
    else:
        raise ValueError(
            f"Unknown plasticity rule: '{rule_name}'. "
            f"Available rules: 'stdp', 'vogels', 'sfa', 'lipshutz_voltage'"
        )


def _build_stdp_sparse(params: Dict[str, Any]) -> PlasticityRule:
    """Build STDP rule for sparse connections.

    Parameters
    ----------
    params : dict
        Can include:
        - tau_pre, tau_post: Trace time constants (default: 20e-3)
        - A_plus, A_minus: Learning rates (default: 1e-4, -1.2e-4)
        - w_min, w_max: Weight bounds (default: 0.0, 1.0)
        - oja_decay: Oja normalization (default: 1e-5)
        - dt: Timestep (default: 1e-3)

    Returns
    -------
    PlasticityRule
        STDP rule with Sparse-compatible components.
    """
    # Extract parameters with defaults
    tau_pre = params.get("tau_pre", 20e-3)
    tau_post = params.get("tau_post", 20e-3)
    dt = params.get("dt", 1e-3)

    A_plus = params.get("A_plus", 1e-4)
    A_minus = params.get("A_minus", -1.2e-4)
    w_min = params.get("w_min", 0.0)
    w_max = params.get("w_max", 1.0)
    oja_decay = params.get("oja_decay", 1e-5)

    # Build components
    eligibility = EligibilitySTDPSparse(tau_pre=tau_pre, tau_post=tau_post, dt=dt)
    signal = SignalPostSpikes()
    update = UpdateSTDPSparse(
        A_plus=A_plus,
        A_minus=A_minus,
        w_min=w_min,
        w_max=w_max,
        oja_decay=oja_decay
    )

    return PlasticityRule(eligibility=eligibility, signal=signal, update=update)


def _build_stdp_dense(params: Dict[str, Any]) -> PlasticityRule:
    """Build STDP rule for dense connections.

    Parameters
    ----------
    params : dict
        Can include:
        - tau_pre, tau_post: Trace time constants (default: 20e-3)
        - A_plus, A_minus: Learning rates (default: 1e-4, -1.2e-4)
        - w_min, w_max: Weight bounds (default: 0.0, 1.0)
        - oja_decay: Oja normalization (default: 1e-5)
        - dt: Timestep (default: 1e-3)

    Returns
    -------
    PlasticityRule
        STDP rule with Dense-compatible components.
    """
    # Extract parameters with defaults
    tau_pre = params.get("tau_pre", 20e-3)
    tau_post = params.get("tau_post", 20e-3)
    dt = params.get("dt", 1e-3)

    A_plus = params.get("A_plus", 1e-4)
    A_minus = params.get("A_minus", -1.2e-4)
    w_min = params.get("w_min", 0.0)
    w_max = params.get("w_max", 1.0)
    oja_decay = params.get("oja_decay", 1e-5)

    # Build components
    eligibility = EligibilitySTDPDense(tau_pre=tau_pre, tau_post=tau_post, dt=dt)
    signal = SignalPostSpikes()
    update = UpdateSTDPDense(
        A_plus=A_plus,
        A_minus=A_minus,
        w_min=w_min,
        w_max=w_max,
        oja_decay=oja_decay
    )

    return PlasticityRule(eligibility=eligibility, signal=signal, update=update)


def _build_vogels_sparse(params: Dict[str, Any]) -> PlasticityRule:
    """Build Vogels iSTDP rule for sparse connections.

    Parameters
    ----------
    params : dict
        Can include:
        - tau_pre: Presynaptic trace time constant (default: 20e-3)
        - target_rate: Target firing rate in Hz (default: 10.0)
        - tau_rate: Rate estimation time constant (default: 1.0)
        - eta: Learning rate (default: 1e-4)
        - w_min, w_max: Weight bounds (default: 0.0, 1.0)
        - dt: Timestep (default: 1e-3)

    Returns
    -------
    PlasticityRule
        Vogels iSTDP rule with Sparse-compatible components.
    """
    # Extract parameters with defaults
    tau_pre = params.get("tau_pre", 20e-3)
    target_rate = params.get("target_rate", 10.0)
    tau_rate = params.get("tau_rate", 1.0)
    eta = params.get("eta", 1e-4)
    w_min = params.get("w_min", 0.0)
    w_max = params.get("w_max", 1.0)
    dt = params.get("dt", 1e-3)

    # Build components
    eligibility = EligibilityPreSparse(tau_pre=tau_pre, dt=dt)
    signal = SignalRateHomeostasis(target_rate=target_rate, tau_rate=tau_rate, dt=dt)
    update = UpdateVogelsSparse(eta=eta, w_min=w_min, w_max=w_max)

    return PlasticityRule(eligibility=eligibility, signal=signal, update=update)


def _build_vogels_dense(params: Dict[str, Any]) -> PlasticityRule:
    """Build Vogels iSTDP rule for dense connections.

    Parameters
    ----------
    params : dict
        Can include:
        - tau_pre: Presynaptic trace time constant (default: 20e-3)
        - target_rate: Target firing rate in Hz (default: 10.0)
        - tau_rate: Rate estimation time constant (default: 1.0)
        - eta: Learning rate (default: 1e-4)
        - w_min, w_max: Weight bounds (default: 0.0, 1.0)
        - dt: Timestep (default: 1e-3)

    Returns
    -------
    PlasticityRule
        Vogels iSTDP rule with Dense-compatible components.
    """
    # Extract parameters with defaults
    tau_pre = params.get("tau_pre", 20e-3)
    target_rate = params.get("target_rate", 10.0)
    tau_rate = params.get("tau_rate", 1.0)
    eta = params.get("eta", 1e-4)
    w_min = params.get("w_min", 0.0)
    w_max = params.get("w_max", 1.0)
    dt = params.get("dt", 1e-3)

    # Build components
    eligibility = EligibilityPreDense(tau_pre=tau_pre, dt=dt)
    signal = SignalRateHomeostasis(target_rate=target_rate, tau_rate=tau_rate, dt=dt)
    update = UpdateVogelsDense(eta=eta, w_min=w_min, w_max=w_max)

    return PlasticityRule(eligibility=eligibility, signal=signal, update=update)


def _build_sfa_sparse(params: Dict[str, Any]) -> PlasticityRule:
    """Build SFA rule for sparse connections.

    Parameters
    ----------
    params : dict
        Can include:
        - tau_fast, tau_slow: DoE trace time constants (default: 10e-3, 100e-3)
        - tau_z: Postsynaptic activity smoothing (default: 20e-3)
        - tau_hpf: High-pass filter time constant (default: 100e-3)
        - eta: Hebbian learning rate (default: 1e-4)
        - beta: Normalization coefficient (default: 1e-4)
        - w_min, w_max: Weight bounds (default: 0.0, 1.0)
        - dt: Timestep (default: 1e-3)
        - signal_type: Learning signal type: "hpf", "voltage", "surrogate" (default: "hpf")
        - use_voltage: Legacy param, same as signal_type="voltage" (overridden by signal_type)
        - v_scale: Voltage scaling factor (default: 1.0)
        - gamma: Surrogate scaling (default: 1.0, only for signal_type="surrogate")
        - delta: Surrogate width (default: 0.1, only for signal_type="surrogate")
        - surrogate_type: "tanh"/"triangular"/"sigmoid" (default: "tanh")

    Returns
    -------
    PlasticityRule
        SFA rule with Sparse-compatible components.
    """
    # Extract parameters with defaults
    tau_fast = params.get("tau_fast", 10e-3)
    tau_slow = params.get("tau_slow", 100e-3)
    tau_z = params.get("tau_z", 20e-3)
    tau_hpf = params.get("tau_hpf", 100e-3)
    eta = params.get("eta", 1e-4)
    beta = params.get("beta", 1e-4)
    w_min = params.get("w_min", 0.0)
    w_max = params.get("w_max", 1.0)
    dt = params.get("dt", 1e-3)

    # Learning signal parameters
    signal_type = params.get("signal_type", None)
    use_voltage = params.get("use_voltage", False)
    v_scale = params.get("v_scale", 1.0)
    gamma = params.get("gamma", 1.0)
    delta = params.get("delta", 0.1)
    surrogate_type = params.get("surrogate_type", "tanh")

    # Determine signal type (backward compatibility with use_voltage)
    if signal_type is None:
        signal_type = "voltage" if use_voltage else "hpf"

    # Build components
    eligibility = EligibilityDoESparse(tau_fast=tau_fast, tau_slow=tau_slow, dt=dt)

    # Choose learning signal based on signal_type
    if signal_type == "surrogate":
        # Temporal surrogate gradient (most advanced, prevents dead neurons)
        signal = SignalTemporalSurrogate(
            tau_smooth=tau_z,
            gamma=gamma,
            delta=delta,
            surrogate_type=surrogate_type,
            use_voltage=True,
            v_scale=v_scale,
            dt=dt
        )
    elif signal_type == "voltage":
        # Voltage-based HPF (prevents dead neuron problem)
        signal = SignalHPFVoltage(tau_v=tau_z, tau_hpf=tau_hpf, dt=dt, v_scale=v_scale)
    elif signal_type == "hpf":
        # Spike-based HPF (original SFA)
        signal = SignalHPFPost(tau_z=tau_z, tau_hpf=tau_hpf, dt=dt)
    else:
        raise ValueError(
            f"Invalid signal_type '{signal_type}'. "
            f"Must be one of: 'hpf', 'voltage', 'surrogate'"
        )

    update = UpdateOjaSparse(eta=eta, beta=beta, w_min=w_min, w_max=w_max)

    return PlasticityRule(eligibility=eligibility, signal=signal, update=update)


def _build_sfa_dense(params: Dict[str, Any]) -> PlasticityRule:
    """Build SFA rule for dense connections.

    Parameters
    ----------
    params : dict
        Can include:
        - tau_fast, tau_slow: DoE trace time constants (default: 10e-3, 100e-3)
        - tau_z: Postsynaptic activity smoothing (default: 20e-3)
        - tau_hpf: High-pass filter time constant (default: 100e-3)
        - eta: Hebbian learning rate (default: 1e-4)
        - beta: Normalization coefficient (default: 1e-4)
        - w_min, w_max: Weight bounds (default: 0.0, 1.0)
        - dt: Timestep (default: 1e-3)
        - signal_type: Learning signal type: "hpf", "voltage", "surrogate" (default: "hpf")
        - use_voltage: Legacy param, same as signal_type="voltage" (overridden by signal_type)
        - v_scale: Voltage scaling factor (default: 1.0)
        - gamma: Surrogate scaling (default: 1.0, only for signal_type="surrogate")
        - delta: Surrogate width (default: 0.1, only for signal_type="surrogate")
        - surrogate_type: "tanh"/"triangular"/"sigmoid" (default: "tanh")

    Returns
    -------
    PlasticityRule
        SFA rule with Dense-compatible components.
    """
    # Extract parameters with defaults
    tau_fast = params.get("tau_fast", 10e-3)
    tau_slow = params.get("tau_slow", 100e-3)
    tau_z = params.get("tau_z", 20e-3)
    tau_hpf = params.get("tau_hpf", 100e-3)
    eta = params.get("eta", 1e-4)
    beta = params.get("beta", 1e-4)
    w_min = params.get("w_min", 0.0)
    w_max = params.get("w_max", 1.0)
    dt = params.get("dt", 1e-3)

    # Learning signal parameters
    signal_type = params.get("signal_type", None)
    use_voltage = params.get("use_voltage", False)
    v_scale = params.get("v_scale", 1.0)
    gamma = params.get("gamma", 1.0)
    delta = params.get("delta", 0.1)
    surrogate_type = params.get("surrogate_type", "tanh")

    # Determine signal type (backward compatibility with use_voltage)
    if signal_type is None:
        signal_type = "voltage" if use_voltage else "hpf"

    # Build components
    eligibility = EligibilityDoEDense(tau_fast=tau_fast, tau_slow=tau_slow, dt=dt)

    # Choose learning signal based on signal_type
    if signal_type == "surrogate":
        # Temporal surrogate gradient (most advanced, prevents dead neurons)
        signal = SignalTemporalSurrogate(
            tau_smooth=tau_z,
            gamma=gamma,
            delta=delta,
            surrogate_type=surrogate_type,
            use_voltage=True,
            v_scale=v_scale,
            dt=dt
        )
    elif signal_type == "voltage":
        # Voltage-based HPF (prevents dead neuron problem)
        signal = SignalHPFVoltage(tau_v=tau_z, tau_hpf=tau_hpf, dt=dt, v_scale=v_scale)
    elif signal_type == "hpf":
        # Spike-based HPF (original SFA)
        signal = SignalHPFPost(tau_z=tau_z, tau_hpf=tau_hpf, dt=dt)
    else:
        raise ValueError(
            f"Invalid signal_type '{signal_type}'. "
            f"Must be one of: 'hpf', 'voltage', 'surrogate'"
        )

    update = UpdateOjaDense(eta=eta, beta=beta, w_min=w_min, w_max=w_max)

    return PlasticityRule(eligibility=eligibility, signal=signal, update=update)


def _build_lipshutz_voltage_sparse(params: Dict[str, Any]) -> PlasticityRule:
    """Build Lipshutz voltage-based SFA rule for sparse connections.

    Implements Lipshutz et al. (2020) SFA adapted for spiking neurons using
    membrane voltage as the postsynaptic signal. The weight update is:
        Δw_ij = η · (x̄_i - x_i) × (V̄_j - V_j)

    This directly implements the mexican-hat STDP kernel from Lipshutz (2020):
    - Eligibility: e_i = x̄_i - x_i (presynaptic temporal difference)
    - Learning signal: L'_j = V̄_j - V_j (postsynaptic temporal difference)

    Parameters
    ----------
    params : dict
        Can include:
        - tau_slow_pre: Presynaptic slow trace time constant (default: 2500e-3)
        - tau_slow_post: Postsynaptic slow voltage trace time constant (default: 2500e-3)
        - v_rest: Resting potential for voltage normalization (default: -70.0)
        - v_scale: Voltage scaling factor (default: 30.0)
        - eta: Hebbian learning rate (default: 1e-5)
        - beta: Oja normalization coefficient (default: 0.0)
        - w_min, w_max: Weight bounds (default: 0.0, 1.0)
        - dt: Timestep (default: 1e-3)

    Returns
    -------
    PlasticityRule
        Lipshutz voltage-based SFA rule with Sparse-compatible components.

    Reference
    ---------
    Lipshutz, D., et al. (2020). "A biologically plausible neural network for
    slow feature analysis." NeurIPS.
    """
    # Extract parameters with defaults
    tau_slow_pre = params.get("tau_slow_pre", 2500e-3)
    tau_slow_post = params.get("tau_slow_post", 2500e-3)
    v_rest = params.get("v_rest", -70.0)
    v_scale = params.get("v_scale", 30.0)
    eta = params.get("eta", 1e-5)
    beta = params.get("beta", 0.0)
    w_min = params.get("w_min", 0.0)
    w_max = params.get("w_max", 1.0)
    dt = params.get("dt", 1e-3)

    # Build components
    eligibility = EligibilityLipshutzVoltageSparse(
        tau_slow_pre=tau_slow_pre,
        dt=dt
    )
    signal = SignalLipshutzVoltage(
        tau_slow=tau_slow_post,
        v_rest=v_rest,
        v_scale=v_scale,
        dt=dt
    )
    update = UpdateOjaSparse(eta=eta, beta=beta, w_min=w_min, w_max=w_max)

    return PlasticityRule(eligibility=eligibility, signal=signal, update=update)


def _build_lipshutz_voltage_dense(params: Dict[str, Any]) -> PlasticityRule:
    """Build Lipshutz voltage-based SFA rule for dense connections.

    Implements Lipshutz et al. (2020) SFA adapted for spiking neurons using
    membrane voltage as the postsynaptic signal. The weight update is:
        Δw_ij = η · (x̄_i - x_i) × (V̄_j - V_j)

    This directly implements the mexican-hat STDP kernel from Lipshutz (2020):
    - Eligibility: e_i = x̄_i - x_i (presynaptic temporal difference)
    - Learning signal: L'_j = V̄_j - V_j (postsynaptic temporal difference)

    Parameters
    ----------
    params : dict
        Can include:
        - tau_slow_pre: Presynaptic slow trace time constant (default: 2500e-3)
        - tau_slow_post: Postsynaptic slow voltage trace time constant (default: 2500e-3)
        - v_rest: Resting potential for voltage normalization (default: -70.0)
        - v_scale: Voltage scaling factor (default: 30.0)
        - eta: Hebbian learning rate (default: 1e-5)
        - beta: Oja normalization coefficient (default: 0.0)
        - w_min, w_max: Weight bounds (default: 0.0, 1.0)
        - dt: Timestep (default: 1e-3)

    Returns
    -------
    PlasticityRule
        Lipshutz voltage-based SFA rule with Dense-compatible components.

    Reference
    ---------
    Lipshutz, D., et al. (2020). "A biologically plausible neural network for
    slow feature analysis." NeurIPS.
    """
    # Extract parameters with defaults
    tau_slow_pre = params.get("tau_slow_pre", 2500e-3)
    eta = params.get("eta", 1e-5)
    beta = params.get("beta", 0.0)
    w_min = params.get("w_min", 0.0)
    w_max = params.get("w_max", 1.0)
    dt = params.get("dt", 1e-3)

    # Build components
    eligibility = EligibilityLipshutzVoltageDense(
        tau_slow_pre=tau_slow_pre,
        dt=dt
    )
    signal = SignalLipshutzVoltage()
    update = UpdateOjaDense(eta=eta, beta=beta, w_min=w_min, w_max=w_max)

    return PlasticityRule(eligibility=eligibility, signal=signal, update=update)
