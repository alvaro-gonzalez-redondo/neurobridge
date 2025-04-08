from __future__ import annotations
import torch
from neurobridge.bridges import AxonalBridge
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from neurobridge.core.typing_aliases import *


class LocalCircuit:
    # Atributos de instancia (solo anotación, no inicialización de variables de clase)
    device: torch.device
    rank: int
    world_size: int
    n_bridge_steps: int
    neuron_groups: List[NeuronGroup]
    synaptic_groups: List[SynapticGroup]
    bridge: Optional[AxonalBridge]

    def __init__(self, device: str, rank: int, world_size: int, n_bridge_steps: int, bridge_size: int = 0) -> None:
        """
        device: 'cpu' o 'cuda:X'
        rank: índice del proceso (0 si único)
        world_size: número total de GPUs o 1
        n_bridge_steps: ciclo de comunicación si se usa AxonalBridge
        bridge_size: número de neuronas puente (N), o 0 si no se usa bridge
        """
        self.device = torch.device(device)
        self.rank = rank
        self.world_size = world_size
        self.n_bridge_steps = n_bridge_steps

        self.neuron_groups = []
        self.synaptic_groups = []

        self.bridge = None
        if bridge_size > 0:
            self.bridge = AxonalBridge(
                size=bridge_size,
                n_steps=n_bridge_steps,
                rank=rank,
                world_size=world_size,
                device=self.device,
            )


    def _add_neuron_group(self, group:NeuronGroup) -> None:
        self.neuron_groups.append(group)


    def add_synaptic_group(self, group:SynapticGroup) -> None:
        self.synaptic_groups.append(group)


    def inject_from_bridge(self, target_group:NeuronGroup, bridge_indices:torch.Tensor) -> None:
        """
        bridge_indices: lista o tensor de índices de neuronas puente conectadas a target_group
        """
        if self.bridge is None:
            return
        spikes :torch.Tensor = self.bridge.read_spikes()
        spikes_to_inject :torch.Tensor = spikes[bridge_indices]
        target_group.inject_spikes(spikes_to_inject)


    def export_to_bridge(self, source_group:NeuronGroup, bridge_indices:torch.Tensor) -> None:
        """
        bridge_indices: lista o tensor con mapeo local→puente
        """
        if self.bridge is None:
            return
        spikes :torch.Tensor = source_group.spike_buffer[source_group.t % source_group.delay_max]
        spikes_to_write :torch.Tensor = torch.zeros(self.bridge.size, dtype=torch.bool, device=self.device)
        spikes_to_write[bridge_indices] = spikes
        self.bridge.write_spikes(spikes_to_write)


    def step(self) -> None:
        # Actualizar dinámicas neuronales
        for group in self.neuron_groups:
            group.step()

        # Propagar spikes a través de sinapsis
        for syn in self.synaptic_groups:
            syn.step()

        # Avanzar bridge si está presente
        if self.bridge is not None:
            self.bridge.step()
