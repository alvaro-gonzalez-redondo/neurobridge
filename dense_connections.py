from __future__ import annotations

from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from .neurons import NeuronGroup
from . import globals
from .dense import Dense
from .utils import resolve_param
from .connection import ConnectionSpec

import torch


class StaticDense(Dense):
    """Dense connection between two neuron groups.
    
    Represents a fully-matrix synaptic connection. Efficient but limited:
    - All weights stored in a dense [Npre, Npos] matrix
    - Only supports a single uniform delay (int)

    Attributes
    ----------
    pre : NeuronGroup
        Pre-synaptic (source) neuron group.
    pos : NeuronGroup
        Post-synaptic (target) neuron group.
    mask : torch.Tensor
        Boolean mask indicating existing connections between neurons.
    weight : torch.Tensor
        Synaptic weights for each connection.
    delay : Int
        Synaptic delays for all connection, in time steps.
    """
    pre: NeuronGroup
    pos: NeuronGroup
    mask: torch.Tensor
    weight: torch.Tensor
    delay: torch.Tensor
    channel: int


    def __init__(self, spec: ConnectionSpec):
        super().__init__((spec.pre.size, spec.pos.size), spec.pre.device)
        self.pre = spec.pre
        self.pos = spec.pos

        # --- delay check ---
        if spec.delay is None:
            self.delay = 0
        elif isinstance(spec.delay, torch.Tensor):
            all_equal = (spec.delay.min() == spec.delay.max()).item()
            if all_equal:
                self.delay = int(spec.delay[0].item())
            else:
                raise ValueError(
                    f"Dense connections only support a uniform scalar delay, got {spec.delay}"
                )
        elif isinstance(spec.delay, (int, float)):
            self.delay = int(spec.delay)
        else:
            raise TypeError(f"Invalid delay type for Dense connection: {type(spec.delay)}")

        # --- weight matrix ---
        self.weight = torch.zeros(
            (self.pre.size, self.pos.size),
            dtype=torch.float32,
            device=self.pre.device,
        )
        if spec.weight is not None:
            self.weight[spec.src_idx, spec.tgt_idx] = spec.weight

        # Optional: binary mask of existing connections
        self.mask = torch.zeros_like(self.weight, dtype=torch.bool)
        self.mask[spec.src_idx, spec.tgt_idx] = True

        self.channel = spec.params.get("channel", 0)

        # Sanity checks
        if self.delay >= self.pre.delay_max:
            raise ValueError(
                f"Dense connection delay {self.delay} must be less than pre.delay_max {self.pre.delay_max}"
            )
        if self.channel >= self.pos.n_channels:
            raise ValueError(
                f"Channel {self.channel} not valid for post-synaptic group with {self.pos.n_channels} channels"
            )

    def _process(self):
        super()._process()
        self._propagate()
        self._update()

    def _propagate(self):
        """Propagate spikes using dense weight matrix and uniform delay."""
        t = globals.simulator.local_circuit.t
        t_idx = (t - self.delay) % self.pre.delay_max

        pre_spikes = self.pre._spike_buffer[:, t_idx].to(self.weight.dtype).squeeze(-1)  # [Npre]
        effective_weight = self.weight.masked_fill(~self.mask, 0)
        contrib = pre_spikes @ effective_weight
        self.pos.inject_currents(contrib, self.channel)

    def _update(self):
        """To be implemented by subclasses (plasticity)."""
        pass


class STDPDense(StaticDense):
    """Spike-Timing-Dependent Plasticity (STDP) synaptic connections for dense networks.

    Implements STDP, a biologically-inspired learning rule where synaptic
    weights are modified based on the relative timing of pre- and post-synaptic spikes.
    This version is optimized for dense connectivity patterns using matrix operations.

    Attributes
    ----------
    A_plus : torch.Tensor
        Learning rate for potentiation (when pre-synaptic spike precedes post-synaptic).
    A_minus : torch.Tensor
        Learning rate for depression (when post-synaptic spike precedes pre-synaptic).
    tau_plus : torch.Tensor
        Time constant for pre-synaptic trace decay.
    tau_minus : torch.Tensor
        Time constant for post-synaptic trace decay.
    w_min : torch.Tensor
        Minimum allowed weight value.
    w_max : torch.Tensor
        Maximum allowed weight value.
    x_pre : torch.Tensor
        Pre-synaptic spike traces for each pre-synaptic neuron.
    x_pos : torch.Tensor
        Post-synaptic spike traces for each post-synaptic neuron.
    alpha_pre : torch.Tensor
        Decay factor for pre-synaptic traces.
    alpha_pos : torch.Tensor
        Decay factor for post-synaptic traces.
    """
    A_plus: torch.Tensor
    A_minus: torch.Tensor
    tau_plus: torch.Tensor
    tau_minus: torch.Tensor
    w_min: torch.Tensor
    w_max: torch.Tensor
    oja_decay: torch.Tensor
    x_pre: torch.Tensor
    x_pos: torch.Tensor
    alpha_pre: torch.Tensor
    alpha_pos: torch.Tensor


    def __init__(self, spec: ConnectionSpec):
        super().__init__(spec)

        device = self.pre.device
        params = spec.params

        # STDP parameters
        self.A_plus = torch.tensor(params.get("A_plus", 1e-4), device=device)
        self.A_minus = torch.tensor(params.get("A_minus", -1.2e-4), device=device)
        self.tau_plus = torch.tensor(params.get("tau_plus", 20e-3), device=device)
        self.tau_minus = torch.tensor(params.get("tau_minus", 20e-3), device=device)
        self.w_min = torch.tensor(params.get("w_min", 0.0), device=device)
        self.w_max = torch.tensor(params.get("w_max", 1.0), device=device)
        self.oja_decay = torch.tensor(spec.params.get("oja_decay", 1e-5), device=self.device)

        # Traces
        self.x_pre = torch.zeros(self.pre.size, dtype=torch.float32, device=device)
        self.x_pos = torch.zeros(self.pos.size, dtype=torch.float32, device=device)

        # Decay factors
        self.alpha_pre = torch.exp(-1e-3 / self.tau_plus)
        self.alpha_pos = torch.exp(-1e-3 / self.tau_minus)

    def _update(self):
        # Decay traces
        self.x_pre *= self.alpha_pre
        self.x_pos *= self.alpha_pos

        # Get current spikes
        t = globals.simulator.local_circuit.t
        t_idx = (t - self.delay) % self.pre.delay_max
        pre_spikes = self.pre._spike_buffer[:, t_idx].float().squeeze(-1)  # [Npre]
        pos_spikes = self.pos.get_spikes().float()             # [Npos]

        # Update traces
        self.x_pre += pre_spikes
        self.x_pos += pos_spikes

        # Potentiation: pre spikes × pos traces
        potentiation = torch.outer(pre_spikes, self.x_pos) * self.A_plus
        # Depression: pre traces × pos spikes
        depression = torch.outer(self.x_pre, pos_spikes) * self.A_minus
        # Homeostasis: Oja's normalization rule
        homeostasis = self.x_pos*self.x_pos * self.weight * self.oja_decay

        # Apply update only where connections exist
        self.weight += (potentiation + depression - homeostasis) * self.mask
        self.weight.clamp_(self.w_min, self.w_max)