from abc import ABC, abstractmethod
import torch


class SynapticGroup(ABC):

    def __init__(self, pre, pos, idx_pre, idx_pos, delay):
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
    

    def step(self, t):
        self.cache_active_indices()
        self.propagate()
        self.update()
        self._valid_indices = None


    def get_active_indices(self, filtered=True) -> torch.Tensor:
        """
        Devuelve un tensor 1D con los índices de las conexiones activas
        (aquellas cuyo pre-sináptico ha disparado con su delay correspondiente).
        """
        if not filtered:
            return self._all_indices
        spikes = self.pre.get_spikes_at(self.delay, self.idx_pre)
        return spikes.nonzero(as_tuple=True)[0] if torch.any(spikes) else None


    def cache_active_indices(self, filtered=True):
        """Calcula y guarda los índices de sinapsis activas en este paso."""
        self._valid_indices = self.get_active_indices(filtered)
        

    @abstractmethod
    def propagate(self): pass

    @abstractmethod
    def update(self): pass