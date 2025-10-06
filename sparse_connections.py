from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .neurons import NeuronGroup
from .group import Group
from .connection import ConnectionSpec
from .utils import resolve_param, block_distance_connect

from typing import Any

import torch


class StaticSparse(Group):
    pre: NeuronGroup
    pos: NeuronGroup
    idx_pre: torch.Tensor
    idx_pos: torch.Tensor
    weight: torch.Tensor
    delay: torch.Tensor
    channel: int
    _current_buffer: torch.Tensor

    def __init__(self, spec: ConnectionSpec):
        """
        Initialize a sparse connection using a ConnectionSpec.

        Parameters
        ----------
        spec : ConnectionSpec
            Specification object with all details of the connection.
        """
        if spec.pre.device != spec.pos.device:
            raise RuntimeError("Connected populations must be on the same device.")

        super().__init__(size=spec.src_idx.numel(), device=spec.pre.device)

        self.pre = spec.pre
        self.pos = spec.pos
        self.idx_pre = spec.src_idx
        self.idx_pos = spec.tgt_idx
        self.weight = spec.weight.to(device=self.pre.device, dtype=torch.float32)
        self.delay  = spec.delay.to(device=self.pre.device, dtype=torch.long)
        self.channel = spec.params.get("channel", 0)

        if torch.any(self.delay >= self.pre.delay_max):
            raise ValueError(
                f"Connection delay too large (max {torch.max(self.delay)}) "
                f"for pre.delay_max={self.pre.delay_max}"
            )

        self._current_buffer = torch.zeros(
            self.pos.size, dtype=torch.float32, device=self.pre.device
        )


    def _process(self):
        super()._process()
        self._propagate()
        self._update()


    def _propagate(self):
        """Propagate spikes from pre-synaptic to post-synaptic neurons.

        Retrieves pre-synaptic spikes with appropriate delays, multiplies by weights,
        and injects the resulting currents into post-synaptic neurons.
        """
        spikes_mask = self.pre.get_spikes_at(self.delay, self.idx_pre)
        contrib = self.weight * spikes_mask.to(self.weight.dtype)
        self._current_buffer.zero_()
        self._current_buffer.index_add_(0, self.idx_pos, contrib)
        self.pos.inject_currents(self._current_buffer, self.channel)


    def _update(self) -> None:
        # Por defecto, nada (estático). Subclases sobreescriben.
        pass


class STDPSparse(StaticSparse):
    """Spike-Timing-Dependent Plasticity (STDP) synaptic connections.

    Implements STDP, a biologically-inspired learning rule where synaptic
    weights are modified based on the relative timing of pre- and post-synaptic spikes.

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
        Pre-synaptic spike traces for each connection.
    x_pos : torch.Tensor
        Post-synaptic spike traces for each connection.
    alpha_pre : torch.Tensor
        Decay factor for pre-synaptic traces.
    alpha_pos : torch.Tensor
        Decay factor for post-synaptic traces.
    _delay_1 : torch.Tensor
        Constant tensor of ones for accessing post-synaptic spikes.
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

        # Parámetros STDP (se pueden pasar en spec.params)
        self.A_plus = torch.tensor(spec.params.get("A_plus",    1e-4), device=self.device)
        self.A_minus = torch.tensor(spec.params.get("A_minus", -1.2e-4), device=self.device)
        self.tau_plus = torch.tensor(spec.params.get("tau_plus", 20e-3), device=self.device)
        self.tau_minus = torch.tensor(spec.params.get("tau_minus", 20e-3), device=self.device)
        self.w_min = torch.tensor(spec.params.get("w_min", 0.0), device=self.device)
        self.w_max = torch.tensor(spec.params.get("w_max", 1.0), device=self.device)
        self.oja_decay = torch.tensor(spec.params.get("oja_decay", 1e-5), device=self.device)

        # Trazas sinápticas
        n_edges = len(self.idx_pre)
        self.x_pre = torch.zeros(n_edges, dtype=torch.float32, device=self.device)
        self.x_pos = torch.zeros(n_edges, dtype=torch.float32, device=self.device)

        self.alpha_pre = torch.exp(-1e-3 / self.tau_plus)
        self.alpha_pos = torch.exp(-1e-3 / self.tau_minus)


    def _update(self) -> None:
        """Update synaptic weights according to the STDP rule."""
        # 1. Decaimiento
        self.x_pre *= self.alpha_pre
        self.x_pos *= self.alpha_pos

        # 2. Spikes actuales
        pre_spikes = self.pre.get_spikes_at(self.delay, self.idx_pre)
        pos_spikes = self.pos.get_spikes_at(1, self.idx_pos)

        self.x_pre += pre_spikes.to(torch.float32)
        self.x_pos += pos_spikes.to(torch.float32)

        # 3. STDP updates
        dw_plus = self.A_plus * self.x_pre * pos_spikes
        dw_minus = self.A_minus * self.x_pos * pre_spikes
        dw_oja = self.x_pos*self.x_pos * self.weight * self.oja_decay
        self.weight += dw_plus + dw_minus - dw_oja

        self.weight.clamp_(self.w_min, self.w_max)
