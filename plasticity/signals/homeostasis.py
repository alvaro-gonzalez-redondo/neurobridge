"""
Homeostatic learning signal for rate regulation.

Used in Vogels inhibitory STDP (iSTDP) to maintain target firing rates:
    L'(t) = z_post(t) - ρ₀

where:
    z_post(t) = α_rate · z_post(t-1) + spike_post(t)
    ρ₀ = target firing rate

The learning signal drives weights to maintain the postsynaptic firing rate
at the target level through homeostatic regulation.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import torch
import math

if TYPE_CHECKING:
    from ...sparse_connections import StaticSparse
    from ...dense_connections import StaticDense

from ..base import LearningSignalBase


class SignalRateHomeostasis(LearningSignalBase):
    """Homeostatic learning signal based on firing rate regulation.

    Computes L' = z_post - ρ₀, where z_post is a low-pass filtered version
    of the postsynaptic spike train, and ρ₀ is the target firing rate.

    This signal is used in Vogels inhibitory STDP to maintain postsynaptic
    neurons at a target firing rate through homeostatic regulation.

    Parameters
    ----------
    target_rate : float
        Target firing rate ρ₀ (in Hz). Default: 10.0 Hz
    tau_rate : float
        Time constant for firing rate estimation (in seconds). Default: 1.0s
    dt : float
        Simulation timestep (in seconds). Default: 1e-3 (1ms)

    Attributes
    ----------
    z_post : torch.Tensor
        Low-pass filtered postsynaptic firing rates, shape (num_post_neurons,)
    alpha_rate : torch.Tensor
        Decay factor for rate estimation
    rho_target : torch.Tensor
        Target firing rate (converted to spikes per timestep)
    """

    def __init__(self, target_rate: float = 10.0, tau_rate: float = 1.0, dt: float = 1e-3):
        self.target_rate = target_rate  # Hz
        self.tau_rate = tau_rate
        self.dt = dt

        # State variables (allocated in bind())
        self.z_post = None
        self.alpha_rate = None
        self.rho_target = None

    def bind(self, conn: StaticSparse | StaticDense) -> None:
        """Allocate state buffers for learning signal computation."""
        num_post = conn.pos.size
        device = conn.device

        # Allocate firing rate tracking (per postsynaptic neuron)
        self.z_post = torch.zeros(num_post, dtype=torch.float32, device=device)

        # Compute decay factor for rate estimation
        self.alpha_rate = torch.tensor(
            math.exp(-self.dt / self.tau_rate),
            dtype=torch.float32,
            device=device
        )

        # Convert target rate from Hz to spikes per timestep
        # target_rate [Hz] * dt [s] = expected spikes per timestep
        self.rho_target = torch.tensor(
            self.target_rate * self.dt,
            dtype=torch.float32,
            device=device
        )

    def step(self, conn: StaticSparse | StaticDense) -> torch.Tensor:
        """Compute homeostatic learning signal L' = z_post - ρ₀.

        Returns
        -------
        torch.Tensor
            Learning signal per postsynaptic neuron, shape (num_post_neurons,)
        """
        # 1. Get current postsynaptic spikes
        post_spikes = conn.pos.get_spikes().float()

        # 2. Update rate estimate with normalized exponential filter
        # This maintains z_post in the correct scale: z ∈ [0, 1]
        # Standard form: z(t) = α·z(t-1) + (1-α)·spike(t)
        self.z_post = self.alpha_rate * self.z_post + (1.0 - self.alpha_rate) * post_spikes

        # 3. Clamp to prevent unrealistic rates (safety mechanism)
        # Max ~100 Hz → z_post_max = 100 Hz * dt = 0.1 (for dt=1ms)
        self.z_post.clamp_(max=0.1)

        # 4. Compute learning signal: L' = z_post - ρ₀
        return (self.z_post - self.rho_target)
