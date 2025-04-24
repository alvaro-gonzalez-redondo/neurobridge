from __future__ import annotations

from . import globals
from .groups import _SpatialGroup
from .synapses import _ConnectionOperator

import torch


class NeuronGroup(_SpatialGroup):
    delay_max: torch.Tensor
    _spike_buffer: torch.Tensor
    _input_currents: torch.Tensor
    _input_spikes: torch.Tensor


    def __init__(self, device:str, n_neurons:int, spatial_dimensions:int=2, delay_max:int=20):
        super().__init__(device, n_neurons, spatial_dimensions)
        self.delay_max = torch.tensor([delay_max], dtype=torch.int, device=self.device)
        self._spike_buffer = torch.zeros((n_neurons, delay_max), dtype=torch.bool, device=self.device)
        self._input_currents = torch.zeros(n_neurons, dtype=torch.float32, device=self.device)
        self._input_spikes = torch.zeros(n_neurons, dtype=torch.bool, device=self.device)


    def get_spike_buffer(self):
        return self._spike_buffer


    def inject_currents(self, I: torch.Tensor) -> None:
        assert I.shape[0] == self.size
        self._input_currents += I


    def inject_spikes(self, spikes: torch.Tensor) -> None:
        """Forces the neurons to spike, independently of weights."""
        assert spikes.shape[0] == self.size
        self._input_spikes |= spikes.bool()
    

    def get_spikes_at(self, delays: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        Devuelve el spike de cada neurona en `indices` con un retraso `delays`.
        
        Args:
            delays: Tensor 1D de enteros con los delays (uno por índice).
            indices: Tensor 1D con los índices de neuronas a consultar.
        
        Returns:
            Tensor booleano con el spike registrado en t-delay para cada índice.
        """
        assert delays.shape == indices.shape, "Delays and indices must match in shape"

        t_indices = (globals.engine.local_circuit.t - delays) % self.delay_max
        return self._spike_buffer[indices, t_indices]


    def __rshift__(self, other) -> _ConnectionOperator:
        """Implementa el operador >> para crear conexiones."""
        return _ConnectionOperator(self, other)
    

# Esta neurona funciona como un simple repetidor
class ParrotNeurons(NeuronGroup):

    def _process(self) -> None:
        super()._process()

        # Limpiamos los spikes que hubiera en este instante temporal
        t_idx = globals.engine.local_circuit.t % self.delay_max
        self._spike_buffer.index_fill_(1, t_idx, 0)

        # Procesar cualquier spike inyectado
        # Guardar el spike en el búfer para t actual
        self._spike_buffer.index_copy_(1, t_idx, (
            self._spike_buffer.index_select(1, t_idx) | self._input_spikes.unsqueeze(1)
            ))
        # Limpiar spikes inyectados
        self._input_spikes.fill_(False)
            
        # Procesar corrientes de entrada
        # Generar spikes para las neuronas que reciben corriente positiva
        spikes = self._input_currents > 0
        self._spike_buffer.index_copy_(1, t_idx, (
            self._spike_buffer.index_select(1, t_idx) | spikes.unsqueeze(1)
            ))
        # Limpiar corrientes
        self._input_currents.fill_(0.0)


class IFNeurons(NeuronGroup):
    V: torch.Tensor
    threshold: torch.Tensor
    decay: torch.Tensor


    def __init__(self, device: str, n_neurons: int, spatial_dimensions: int = 2, delay_max: int = 20, threshold: float = 1.0, tau: float = 0.1):
        super().__init__(device, n_neurons, spatial_dimensions, delay_max)
        self.V = torch.zeros(n_neurons, dtype=torch.float32, device=self.device)
        self.threshold = torch.tensor([threshold], dtype=torch.float32, device=device)
        self.decay = torch.exp(torch.tensor(-1e-3 / tau, dtype=torch.float32, device=device))


    def _process(self):
        super()._process()
        t_idx = globals.engine.local_circuit.t % self.delay_max

        # Update potential with decay and input
        self.V *= self.decay
        self.V += self._input_currents
        self._input_currents.fill_(0.0)

        # Determine which neurons spike
        spikes = self.V >= self.threshold
        self._spike_buffer.index_copy_(1, t_idx, spikes.unsqueeze(1))
        self.V[spikes] = 0.0  # Reset membrane potential


class RandomSpikeNeurons(NeuronGroup):
    firing_rate: torch.Tensor  # En Hz
    probabilities: torch.Tensor

    def __init__(
            self,
            device: str,
            n_neurons: int,
            firing_rate: float = 10.0,
            spatial_dimensions: int = 2,
            delay_max: int = 20
    ):
        super().__init__(device, n_neurons, spatial_dimensions=spatial_dimensions, delay_max=delay_max)
        self.firing_rate = torch.tensor(firing_rate*1e-3, dtype=torch.float32, device=device)
        self.probabilities = torch.zeros(n_neurons, dtype=torch.float32, device=device)


    def _process(self):
        super()._process()
        t_idx = globals.engine.local_circuit.t % self.delay_max

        self.probabilities.uniform_()
        spikes = self.probabilities < self.firing_rate
        self._spike_buffer.index_copy_(1, t_idx, spikes.unsqueeze(1))