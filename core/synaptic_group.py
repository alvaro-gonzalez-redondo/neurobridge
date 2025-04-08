from __future__ import annotations
from abc import ABC, abstractmethod
import torch
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from neurobridge.core.neuron_group import NeuronGroup


class SynapticGroup(ABC):
    pre: NeuronGroup
    pos: NeuronGroup
    idx_pre: torch.Tensor
    idx_pos: torch.Tensor
    delay: torch.Tensor
    device: torch.device
    _all_indices: torch.Tensor
    _valid_indices: Optional[torch.Tensor]

    def __init__(
        self,
        pre: NeuronGroup,
        pos: NeuronGroup,
        idx_pre: torch.Tensor,
        idx_pos: torch.Tensor,
        delay: torch.Tensor,
    ) -> None:
        """
        pre, pos: objetos NeuronGroup
        idx_pre, idx_pos: índices de conexiones (shape: [num_synapses])
        delay: tensor de delays por sinapsis (shape: [num_synapses])
        """
        self.pre = pre
        self.pos = pos
        self.idx_pre = idx_pre
        self.idx_pos = idx_pos
        self.delay = delay
        self.device = pre.device

        self._all_indices = torch.arange(self.idx_pre.shape[0], device=self.device)
        self._valid_indices = None

    def step(self, t: int) -> None: # t no usado, mantenido por compatibilidad
        self.cache_active_indices()
        self.propagate()
        self.update()
        self._valid_indices = None

    def get_active_indices(self, filtered: bool = True) -> Optional[torch.Tensor]:
        """
        Devuelve un tensor 1D con los índices de las conexiones activas
        (aquellas cuyo pre-sináptico ha disparado con su delay correspondiente).
        """
        if not filtered:
            return self._all_indices
        spikes = self.pre.get_spikes_at(self.delay, self.idx_pre)
        return spikes.nonzero(as_tuple=True)[0] if torch.any(spikes) else None

    def cache_active_indices(self, filtered: bool = True) -> None:
        """Calcula y guarda los índices de sinapsis activas en este paso."""
        self._valid_indices = self.get_active_indices(filtered)

    @abstractmethod
    def propagate(self) -> None:
        pass

    @abstractmethod
    def update(self) -> None:
        pass
