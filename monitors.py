from __future__ import annotations

from . import globals
from .core import _Node
from .groups import _Group
from .neurons import NeuronGroup

from typing import List, Dict

import torch


class SpikeMonitor(_Node):
    groups: List[NeuronGroup]
    filters: List[torch.Tensor]
    recorded_spikes: List[List[torch.Tensor]]  # Por grupo: lista de tensores [N_spikes, 2] (neuron_idx, t)


    def __init__(self, groups: List[NeuronGroup]):
        super().__init__()
        self.groups = groups
        self.filters = [group.filter.nonzero(as_tuple=True)[0] for group in groups]
        self.recorded_spikes = [[] for _ in groups]


    def _process(self):
        super()._process()
        t = globals.engine.local_circuit.t.item()
        for i, (group, filter) in enumerate(zip(self.groups, self.filters)):
            delay_max = group.delay_max

            if (t % delay_max) != (delay_max - 1):
                continue

            buffer = group.get_spike_buffer()  # shape: [N, D]
            spike_indices = buffer.nonzero(as_tuple=False)  # shape: [N_spikes, 2]
            if spike_indices.numel() == 0:
                continue

            neuron_ids = spike_indices[:, 0]
            is_filtered = torch.isin(neuron_ids, filter)
            neuron_ids = neuron_ids[is_filtered]

            delay_slots = spike_indices[:, 1][is_filtered]
            times = globals.engine.local_circuit.t - delay_max + delay_slots

            spikes_tensor = torch.stack([neuron_ids, times], dim=1)  # shape: [N_spikes, 2]
            self.recorded_spikes[i].append(spikes_tensor)


    def get_spike_tensor(self, group_index: int, to_cpu: bool = True) -> torch.Tensor:
        """
        Devuelve un tensor [N_total_spikes, 2] con (neuron_id, time) para un grupo dado.
        """
        device = self.groups[group_index].device if not to_cpu else 'cpu'
        if not self.recorded_spikes[group_index]:
            return torch.empty((0, 2), dtype=torch.long, device=device)
        return torch.cat(self.recorded_spikes[group_index], dim=0).to(device)


class VariableMonitor(_Node):
    groups: List[_Group]
    filters: List[torch.Tensor]
    variable_names: List[str]
    recorded_values: List[Dict[str, List[torch.Tensor]]]  # [group_idx][var_name] = list of tensors over time

    def __init__(self, groups: List[_Group], variable_names: List[str]):
        super().__init__()
        self.groups = groups
        self.filters = [group.filter.nonzero(as_tuple=True)[0] for group in groups]
        self.variable_names = variable_names

        # recorded_values[i][var] = lista de tensores por tiempo
        self.recorded_values = [
            {var_name: [] for var_name in variable_names}
            for _ in groups
        ]


    def _process(self):
        super()._process()
        for i, (group, filter) in enumerate(zip(self.groups, self.filters)):
            for var_name in self.variable_names:
                value = getattr(group, var_name, None)
                if value is None:
                    raise AttributeError(f"Group {i} does not have variable '{var_name}'.")

                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"Monitored variable '{var_name}' is not a torch.Tensor.")

                # Copiar para evitar aliasing
                self.recorded_values[i][var_name].append(value[filter].detach().clone())


    def get_variable_tensor(self, group_index: int, var_name: str, to_cpu: bool = True) -> torch.Tensor:
        """
        Devuelve un tensor [T, N] con la evolución temporal de la variable dada.
        """
        values = self.recorded_values[group_index][var_name]
        device = self.groups[group_index].device if not to_cpu else 'cpu'
        if not values:
            return torch.empty((0, self.groups[group_index].size), device=device)
        return torch.stack(values, dim=0).to(device)
