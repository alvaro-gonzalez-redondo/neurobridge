from __future__ import annotations
from typing import Optional
import torch
from torch import Tensor
from neurobridge.core.neuron_group import NeuronGroup


class IandF(NeuronGroup):
    v: Tensor
    threshold: float
    _input_currents: Tensor
    _input_spikes: Tensor

    def __init__(self, size: int, delay: int, device: Optional[str] = None, threshold: float = 1.0) -> None:
        super().__init__(size, delay, device)
        self.v = torch.zeros(size, dtype=torch.float32, device=self.device)
        self.threshold = threshold
        self._input_currents = torch.zeros(size, dtype=torch.float32, device=self.device)
        self._input_spikes = torch.zeros(size, dtype=torch.bool, device=self.device)

    def inject_currents(self, I: Tensor) -> None:
        self._input_currents += I

    def inject_spikes(self, spikes: Tensor) -> None:
        self._input_spikes |= spikes.bool()

    def step(self) -> None:
        self.v += self._input_currents

        spikes = self.v >= self.threshold
        spikes |= self._input_spikes
        self._register_spikes(spikes)

        self.v = self.v * (~spikes).float()  # Reset if spiked
        self._input_currents.zero_()
        self._input_spikes.zero_()
        self.t += 1
