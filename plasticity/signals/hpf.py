"""
High-pass filtered learning signal for SFA.

Used in Slow Feature Analysis to extract temporal derivatives:
    L'(t) = z(t) - z_low(t)

where:
    z_low(t) = α_hpf · z_low(t-1) + (1 - α_hpf) · z(t)
    z(t) = low-pass filtered postsynaptic activity

The high-pass filter emphasizes changes in the signal, making the learning
focus on slow, temporally stable features.

Two versions are provided:
- SignalHPFPost: Uses spike output (binary 0/1)
- SignalHPFVoltage: Uses membrane voltage (continuous)

The voltage version prevents "dead neuron" problem by maintaining learning
signal even when neuron doesn't spike (as long as voltage changes).
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import torch
import math

if TYPE_CHECKING:
    from ...sparse_connections import StaticSparse
    from ...dense_connections import StaticDense

from ..base import LearningSignalBase


class SignalHPFPost(LearningSignalBase):
    """High-pass filtered postsynaptic learning signal for SFA.

    Computes L' = z - z_low, where z is the current postsynaptic activity
    and z_low is a low-pass filtered version. This emphasizes temporal
    changes in the postsynaptic signal.

    The learning signal drives synapses to respond to slowly varying
    features in the input, implementing Slow Feature Analysis.

    Parameters
    ----------
    tau_z : float
        Time constant for postsynaptic activity smoothing (in seconds).
        Default: 20e-3 (20ms)
    tau_hpf : float
        Time constant for high-pass filter (in seconds). Default: 100e-3 (100ms)
    dt : float
        Simulation timestep (in seconds). Default: 1e-3 (1ms)

    Attributes
    ----------
    z_post : torch.Tensor
        Smoothed postsynaptic activity, shape (num_post_neurons,)
    z_low : torch.Tensor
        Low-pass filtered postsynaptic activity, shape (num_post_neurons,)
    alpha_z : torch.Tensor
        Decay factor for postsynaptic activity smoothing
    alpha_hpf : torch.Tensor
        Decay factor for low-pass filter
    """

    def __init__(self, tau_z: float = 20e-3, tau_hpf: float = 100e-3, dt: float = 1e-3):
        self.tau_z = tau_z
        self.tau_hpf = tau_hpf
        self.dt = dt

        # State variables (allocated in bind())
        self.z_post = None
        self.z_low = None
        self.alpha_z = None
        self.alpha_hpf = None

    def bind(self, conn: StaticSparse | StaticDense) -> None:
        """Allocate state buffers for learning signal computation."""
        num_post = conn.pos.size
        device = conn.device

        # Allocate activity tracking (per postsynaptic neuron)
        self.z_post = torch.zeros(num_post, dtype=torch.float32, device=device)
        self.z_low = torch.zeros(num_post, dtype=torch.float32, device=device)

        # Compute decay factors
        self.alpha_z = torch.tensor(
            math.exp(-self.dt / self.tau_z),
            dtype=torch.float32,
            device=device
        )
        self.alpha_hpf = torch.tensor(
            math.exp(-self.dt / self.tau_hpf),
            dtype=torch.float32,
            device=device
        )

    def step(self, conn: StaticSparse | StaticDense) -> torch.Tensor:
        """Compute high-pass filtered learning signal L' = z - z_low.

        Returns
        -------
        torch.Tensor
            Learning signal per postsynaptic neuron, shape (num_post_neurons,)
        """
        # 1. Get current postsynaptic spikes
        post_spikes = conn.pos.get_spikes().float()

        # 2. Update smoothed postsynaptic activity with normalized exponential filter
        # Standard form: z(t) = α·z(t-1) + (1-α)·spike(t)
        self.z_post = self.alpha_z * self.z_post + (1.0 - self.alpha_z) * post_spikes

        # 3. Update low-pass filtered version
        # z_low(t) = α · z_low(t-1) + (1-α) · z(t)
        self.z_low = self.alpha_hpf * self.z_low + (1.0 - self.alpha_hpf) * self.z_post

        # 4. Compute high-pass filtered signal: L' = z - z_low
        # Note: HPF signal has implicit negative sign from filter dynamics
        # (acts as derivative, naturally provides correct sign for SFA)
        return self.z_post - self.z_low


class SignalHPFVoltage(LearningSignalBase):
    """High-pass filtered voltage-based learning signal for SFA.

    Similar to SignalHPFPost, but uses membrane voltage instead of spikes.
    This prevents the "dead neuron" problem: learning continues as long as
    the voltage changes, even if the neuron doesn't spike.

    Computes L' = V_smooth - V_low, where V is the membrane voltage.

    Advantages over spike-based version:
    - Continuous signal (no binary 0/1)
    - Learning signal exists even for silent neurons
    - More robust to low firing rates

    Parameters
    ----------
    tau_v : float
        Time constant for voltage smoothing (in seconds).
        Default: 20e-3 (20ms)
    tau_hpf : float
        Time constant for high-pass filter (in seconds).
        Default: 100e-3 (100ms)
    dt : float
        Simulation timestep (in seconds). Default: 1e-3 (1ms)
    v_scale : float
        Scaling factor for voltage normalization. Default: 1.0
        Can be used to normalize voltage to similar range as spike traces.

    Attributes
    ----------
    v_post : torch.Tensor
        Smoothed postsynaptic voltage, shape (num_post_neurons,)
    v_low : torch.Tensor
        Low-pass filtered voltage, shape (num_post_neurons,)
    alpha_v : torch.Tensor
        Decay factor for voltage smoothing
    alpha_hpf : torch.Tensor
        Decay factor for low-pass filter
    """

    def __init__(
        self,
        tau_v: float = 20e-3,
        tau_hpf: float = 100e-3,
        dt: float = 1e-3,
        v_scale: float = 1.0
    ):
        self.tau_v = tau_v
        self.tau_hpf = tau_hpf
        self.dt = dt
        self.v_scale = v_scale

        # State variables (allocated in bind())
        self.v_post = None
        self.v_low = None
        self.alpha_v = None
        self.alpha_hpf = None

    def bind(self, conn: StaticSparse | StaticDense) -> None:
        """Allocate state buffers for learning signal computation."""
        num_post = conn.pos.size
        device = conn.device

        # Allocate voltage tracking (per postsynaptic neuron)
        self.v_post = torch.zeros(num_post, dtype=torch.float32, device=device)
        self.v_low = torch.zeros(num_post, dtype=torch.float32, device=device)

        # Compute decay factors
        self.alpha_v = torch.tensor(
            math.exp(-self.dt / self.tau_v),
            dtype=torch.float32,
            device=device
        )
        self.alpha_hpf = torch.tensor(
            math.exp(-self.dt / self.tau_hpf),
            dtype=torch.float32,
            device=device
        )

    def step(self, conn: StaticSparse | StaticDense) -> torch.Tensor:
        """Compute high-pass filtered voltage learning signal.

        Returns
        -------
        torch.Tensor
            Learning signal per postsynaptic neuron, shape (num_post_neurons,)
        """
        # 1. Get current postsynaptic membrane voltage
        # Assuming postsynaptic neurons have a 'V' attribute (true for IFNeurons)
        voltage = conn.pos.V * self.v_scale

        # 2. Update smoothed voltage with normalized exponential filter
        # v_smooth(t) = α·v_smooth(t-1) + (1-α)·V(t)
        self.v_post = self.alpha_v * self.v_post + (1.0 - self.alpha_v) * voltage

        # 3. Update low-pass filtered version
        # v_low(t) = α · v_low(t-1) + (1-α) · v_smooth(t)
        self.v_low = self.alpha_hpf * self.v_low + (1.0 - self.alpha_hpf) * self.v_post

        # 4. Compute high-pass filtered signal: L' = v_smooth - v_low
        # Note: HPF signal has implicit negative sign from filter dynamics
        # (acts as derivative, naturally provides correct sign for SFA)
        return self.v_post - self.v_low


class SignalTemporalSurrogate(LearningSignalBase):
    """Temporal surrogate gradient learning signal for SFA.

    Implements a smooth approximation of the temporal derivative that penalizes
    rapid changes in postsynaptic activity, analogous to how e-prop uses spatial
    surrogate gradients for the spike function.

    The learning signal is:
        L'(t) = γ · φ'(Δy) where Δy = y(t) - y(t-1)

    where φ' is a smooth function (e.g., tanh, triangular, sigmoid).

    This approach:
    - Penalizes rapid changes (high-frequency components) in output
    - Provides differentiable, causal learning signal
    - Prevents dead neuron problem (signal exists whenever activity changes)
    - Combines naturally with DoE eligibility for band-pass filtering

    Advantages over HPF:
    - No explicit filtering needed (derivative is implicit)
    - Surrogate handles discontinuities smoothly
    - Can use voltage or spike rate

    Parameters
    ----------
    tau_smooth : float
        Time constant for smoothing postsynaptic signal (in seconds).
        Default: 20e-3 (20ms)
    gamma : float
        Scaling factor for the surrogate gradient. Default: 1.0
    delta : float
        Width parameter for the surrogate function. Smaller = steeper.
        Default: 0.1
    surrogate_type : str
        Type of surrogate function: "tanh", "triangular", "sigmoid".
        Default: "tanh"
    use_voltage : bool
        If True, use membrane voltage; if False, use spike rate.
        Default: True
    v_scale : float
        Voltage scaling factor (only used if use_voltage=True). Default: 1.0
    dt : float
        Simulation timestep (in seconds). Default: 1e-3 (1ms)

    Attributes
    ----------
    y_smooth : torch.Tensor
        Smoothed postsynaptic signal (voltage or spike rate), shape (num_post_neurons,)
    y_prev : torch.Tensor
        Previous value of smoothed signal, shape (num_post_neurons,)
    alpha_smooth : torch.Tensor
        Decay factor for smoothing
    """

    def __init__(
        self,
        tau_smooth: float = 20e-3,
        gamma: float = 1.0,
        delta: float = 0.1,
        surrogate_type: str = "tanh",
        use_voltage: bool = True,
        v_scale: float = 1.0,
        dt: float = 1e-3
    ):
        self.tau_smooth = tau_smooth
        self.gamma = gamma
        self.delta = delta
        self.surrogate_type = surrogate_type
        self.use_voltage = use_voltage
        self.v_scale = v_scale
        self.dt = dt

        # Validate surrogate type
        valid_types = ["tanh", "triangular", "sigmoid"]
        if surrogate_type not in valid_types:
            raise ValueError(
                f"Invalid surrogate_type '{surrogate_type}'. "
                f"Must be one of: {valid_types}"
            )

        # State variables (allocated in bind())
        self.y_smooth = None
        self.y_prev = None
        self.alpha_smooth = None

    def bind(self, conn: StaticSparse | StaticDense) -> None:
        """Allocate state buffers for learning signal computation."""
        num_post = conn.pos.size
        device = conn.device

        # Allocate smoothed signal tracking
        self.y_smooth = torch.zeros(num_post, dtype=torch.float32, device=device)
        self.y_prev = torch.zeros(num_post, dtype=torch.float32, device=device)

        # Compute decay factor
        self.alpha_smooth = torch.tensor(
            math.exp(-self.dt / self.tau_smooth),
            dtype=torch.float32,
            device=device
        )

    def step(self, conn: StaticSparse | StaticDense) -> torch.Tensor:
        """Compute temporal surrogate gradient learning signal.

        Returns
        -------
        torch.Tensor
            Learning signal per postsynaptic neuron, shape (num_post_neurons,)
        """
        # 1. Get current postsynaptic activity (voltage or spike rate)
        if self.use_voltage:
            # Use membrane voltage
            y_raw = conn.pos.V * self.v_scale
        else:
            # Use spike rate (convert spikes to float)
            y_raw = conn.pos.get_spikes().float()

        # 2. Update smoothed signal
        # y_smooth(t) = α·y_smooth(t-1) + (1-α)·y_raw(t)
        self.y_smooth = self.alpha_smooth * self.y_smooth + (1.0 - self.alpha_smooth) * y_raw

        # 3. Compute temporal change
        delta_y = self.y_smooth - self.y_prev

        # 4. Apply surrogate gradient function
        # IMPORTANT: Negative sign for gradient DESCENT on slowness objective
        # We want to MINIMIZE temporal variations: J = E[(dy/dt)²]
        # Therefore: Δw ∝ -∂J/∂w ∝ -(dy/dt) · eligibility
        if self.surrogate_type == "tanh":
            # φ'(Δy) = -γ · tanh(Δy/δ)  [negative for gradient descent]
            L_prime = -self.gamma * torch.tanh(delta_y / self.delta)

        elif self.surrogate_type == "triangular":
            # φ'(Δy) = -γ · max(0, 1 - |Δy|/δ) · sign(Δy)  [negative for gradient descent]
            # Note: triangular surrogate should also penalize changes in either direction
            L_prime = -self.gamma * torch.sign(delta_y) * torch.clamp(1.0 - torch.abs(delta_y) / self.delta, min=0.0)

        elif self.surrogate_type == "sigmoid":
            # φ'(Δy) = -γ · (sigmoid(Δy/δ) - 0.5) (centered, negative for gradient descent)
            L_prime = -self.gamma * (torch.sigmoid(delta_y / self.delta) - 0.5)

        # 5. Store current value for next step
        self.y_prev = self.y_smooth.clone()

        return L_prime
