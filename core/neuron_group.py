from __future__ import annotations
from abc import ABC, abstractmethod
import torch
from neurobridge.core.connections import ConnectionOperator
from neurobridge.engine import SimulatorEngine
from typing import Optional, Callable, Self
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from neurobridge.core.typing_aliases import *


class NeuronGroup(ABC):
    # Atributos de instancia anotados (NO son variables de clase)
    device: torch.device
    size: int
    delay_max: int
    spike_buffer: torch.Tensor
    t: int
    filter: Optional[torch.Tensor]
    positions: Optional[torch.Tensor]


    def __init__(self, size: int, delay_max: int, device: str = None, topological_dims: Optional[int] = None) -> None:
        if device is None:
            circuit: Optional[LocalCircuit] = SimulatorEngine.get_current_device()
            if circuit is None:
                raise RuntimeError("No se ha especificado un dispositivo y no hay contexto activo")
            self.device = circuit.device
            circuit._add_neuron_group(self)
        else:
            self.device = torch.device(device)

        self.size = size
        self.delay_max = delay_max
        self.spike_buffer = torch.zeros((delay_max, size), dtype=torch.bool, device=self.device)
        self.t = 0
        
        self.filter = None  # Puede usarse para seleccionar subgrupos
        
        self.positions = (
            torch.rand((size, topological_dims), dtype=torch.float32, device=self.device)
            if topological_dims is not None
            else None
        )


    def _register_spikes(self, spikes: torch.Tensor) -> None:
        self.spike_buffer[self.t % self.delay_max] = spikes.bool()


    def get_spikes_at(self, delay_steps: torch.Tensor, neuron_indices: torch.Tensor) -> torch.Tensor:
        time_indices = (self.t - delay_steps) % self.delay_max
        return self.spike_buffer[time_indices, neuron_indices]


    @abstractmethod
    def step(self) -> None:
        pass


    def reset(self) -> None:
        self.spike_buffer.zero_()
        self.t = 0


    def __rshift__(self, other) -> ConnectionOperator:
        """Implementa el operador >> para crear conexiones."""
        return ConnectionOperator(self, other)


    def where_id(self, condition: Callable[..., bool]) -> NeuronGroup:
        """
        Aplica un filtro basado en su índice.

        Args:
            condition: Función que toma un índice y devuelve True/False

        Returns:
            A sí mismo con el parámetro `filter` modificado
        """
        if self.filter is None:
            self.filter = torch.ones(self.size, dtype=torch.bool, device=self.device)

        for i in range(self.size):
            self.filter[i] &= condition(i)

        return self
    

    def where_pos(self, condition: Callable[..., bool]) -> Self:
        """
        Aplica un filtro basado en su posición.

        Args:
            condition: Función que toma coordenadas (float) y devuelve True/False

        Returns:
            A sí mismo con el parámetro `filter` modificado
        """
        if self.positions is None:
            raise RuntimeError("Este grupo no tiene posiciones definidas.")
        if self.filter is None:
            self.filter = torch.ones(self.size, dtype=torch.bool, device=self.device)
        for i in range(self.size):
            self.filter[i] &= condition(*self.positions[i].tolist())
        return self