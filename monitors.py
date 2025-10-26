from __future__ import annotations

from . import globals
from .core import Node
from .group import Group
from .neurons import NeuronGroup

from typing import List, Dict

import torch
import re


class SpikeMonitor(Node):
    """Records and stores spike events from neuron groups.

    This monitor tracks spike events from one or more neuron groups,
    filtering them according to each group's current filter. The recorded
    spikes can be retrieved as tensors for analysis or visualization.

    Attributes
    ----------
    groups : List[NeuronGroup]
        List of neuron groups being monitored.
    filters : List[torch.Tensor]
        List of filters (indices of neurons to monitor) for each group.
    recorded_spikes : List[List[torch.Tensor]]
        Nested list of recorded spike tensors, organized by group.
        Each spike tensor has shape [N_spikes, 2] with columns (neuron_idx, time).
    """

    groups: List[NeuronGroup]
    filters: List[torch.Tensor]
    recorded_spikes: List[
        List[torch.Tensor]
    ]  # Per group: list of tensors [N_spikes, 2] (neuron_idx, t)

    def __init__(self, groups: List[NeuronGroup]):
        """Initialize a spike monitor for the given neuron groups.

        Parameters
        ----------
        groups : List[NeuronGroup]
            List of neuron groups to monitor.
        """
        super().__init__()
        self.groups = groups
        self.filters = [group.filter.nonzero(as_tuple=True)[0] for group in groups]
        self.recorded_spikes = [[] for _ in groups]

    def _process(self):
        """Process the monitor for the current time step.

        Checks for spikes in the monitored groups and records them if they
        match the current filters. Spikes are recorded at regular intervals
        based on each group's delay_max parameter.
        """
        super()._process()
        t = globals.simulator.local_circuit.t.item()
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
            times = globals.simulator.local_circuit.t - delay_max + delay_slots +1 #TODO: Check if this +1 is really necessary to avoid negative time values

            spikes_tensor = torch.stack(
                [neuron_ids, times], dim=1
            )  # shape: [N_spikes, 2]
            self.recorded_spikes[i].append(spikes_tensor)

    def get_spike_tensor(self, group_index: int, to_cpu: bool = True) -> torch.Tensor:
        """Get a tensor of all recorded spikes for a specific group.

        Parameters
        ----------
        group_index : int
            Index of the group in the monitor's groups list.
        to_cpu : bool, optional
            Whether to move the result to CPU memory, by default True.

        Returns
        -------
        torch.Tensor
            Tensor of shape [N_total_spikes, 2] with columns (neuron_id, time_step).
            If no spikes were recorded, returns an empty tensor of shape (0, 2).
        """
        device = self.groups[group_index].device if not to_cpu else "cpu"
        if not self.recorded_spikes[group_index]:
            return torch.empty((0, 2), dtype=torch.long, device=device)
        return torch.cat(self.recorded_spikes[group_index], dim=0).to(device)


class VariableMonitor(Node):
    """Records and stores the evolution of variables from groups over time.

    This monitor tracks specified variables from one or more groups,
    filtering them according to each group's current filter. The recorded
    values can be retrieved as tensors for analysis or visualization.

    Attributes
    ----------
    groups : List[Group]
        List of groups being monitored.
    filters : List[torch.Tensor]
        List of filters (indices of elements to monitor) for each group.
    variable_names : List[str]
        Names of the variables to monitor.
    recorded_values : List[Dict[str, List[torch.Tensor]]]
        Nested structure of recorded values, organized by group and variable.
    """

    groups: List[Group]
    filters: List[torch.Tensor]
    variable_names: List[str]
    recorded_values: List[
        Dict[str, List[torch.Tensor]]
    ]  # [group_idx][var_name] = list of tensors over time
    neuron_filters: List[Dict[str, torch.Tensor]]  # Neuron-level filters deduced from connection filters

    def __init__(self, groups: List[Group], variable_names: List[str]):
        """Initialize a variable monitor for the given groups and variables.

        Parameters
        ----------
        groups : List[Group]
            List of groups to monitor.
        variable_names : List[str]
            Names of variables to monitor in each group.

        Notes
        -----
        For connection groups, the filter is used for per-synapse variables (like weight).
        For per-neuron trace variables (like last_eligibility, last_learning_signal),
        the neuron indices are automatically deduced from which rows/columns of the
        connection filter matrix have True values:
        - Presynaptic neurons: rows with any True value
        - Postsynaptic neurons: columns with any True value
        """
        super().__init__()
        self.groups = groups
        self.filters = [group.filter.nonzero(as_tuple=True)[0] for group in groups]
        self.variable_names = variable_names

        # Deduce neuron-level filters from connection filters
        self.neuron_filters = []
        for group in groups:
            # Check if group has 2D filter (connection) or 1D filter (neuron group)
            if group.filter.ndim == 2:
                # Connection filter: deduce pre/post neurons from rows/columns with True
                rows_with_true = group.filter.any(dim=1).nonzero(as_tuple=True)[0]
                cols_with_true = group.filter.any(dim=0).nonzero(as_tuple=True)[0]
                self.neuron_filters.append({
                    'pre': rows_with_true,
                    'post': cols_with_true
                })
            else:
                # Neuron group: use the filter as-is (though traces don't apply here)
                neuron_idx = group.filter.nonzero(as_tuple=True)[0]
                self.neuron_filters.append({
                    'pre': neuron_idx,
                    'post': neuron_idx
                })

        # recorded_values[i][var] = list of tensors over time
        self.recorded_values = [
            {var_name: [] for var_name in variable_names} for _ in groups
        ]


    def _process(self):
        super()._process()
        for i, (group, filt) in enumerate(zip(self.groups, self.filters)):
            for var_name in self.variable_names:
                # --- detectar patrón "variable@índice" ---
                match = re.match(r"^([\w\.]+)@(\d+)$", var_name)
                if match:
                    base_name, sub_index = match.group(1), int(match.group(2))
                else:
                    base_name, sub_index = var_name, None

                # --- obtener la variable ---
                value = getattr(group, base_name, None)
                if value is None:
                    raise AttributeError(
                        f"Group {i} does not have variable '{base_name}'."
                    )
                if not isinstance(value, torch.Tensor):
                    raise TypeError(
                        f"Monitored variable '{base_name}' is not a torch.Tensor."
                    )

                # --- indexar dimensión extra si se pidió ---
                if sub_index is not None:
                    if value.ndim < 2:
                        raise ValueError(
                            f"Variable '{base_name}' has no extra dimension to index with '@'."
                        )
                    if sub_index >= value.shape[1]:
                        raise IndexError(
                            f"Index {sub_index} out of range for '{base_name}' (shape {tuple(value.shape)})."
                        )
                    value = value[:, sub_index]

                # --- aplicar filtro y almacenar ---
                # Special handling for per-neuron traces (not per-synapse variables)
                # These variables have shape (num_components, num_neurons) and should not
                # be filtered like weight matrices
                if base_name == 'last_eligibility':
                    # Eligibility traces: shape (num_components, num_neurons)
                    # For STDP: (2, num_neurons) where [0]=x_pre, [1]=x_post
                    # For Vogels: (1, num_neurons) where [0]=x_pre

                    # Apply neuron-level filters deduced from connection filter
                    pre_filter = self.neuron_filters[i]['pre']
                    post_filter = self.neuron_filters[i]['post']

                    if value.shape[0] == 1:
                        # Vogels: only x_pre
                        filtered_value = torch.stack([
                            value[0, pre_filter].detach().clone()
                        ], dim=0)
                    elif value.shape[0] == 2:
                        # STDP: x_pre and x_post
                        filtered_value = torch.stack([
                            value[0, pre_filter].detach().clone(),
                            value[1, post_filter].detach().clone()
                        ], dim=0)
                    else:
                        raise ValueError(f"Unexpected eligibility shape: {value.shape}")

                    self.recorded_values[i][var_name].append(filtered_value)

                elif base_name == 'last_learning_signal':
                    # Learning signal: shape (num_post_neurons,)
                    post_filter = self.neuron_filters[i]['post']
                    filtered_value = value[post_filter].detach().clone()
                    self.recorded_values[i][var_name].append(filtered_value)

                else:
                    # Apply connection filter to per-synapse variables like weight
                    self.recorded_values[i][var_name].append(
                        value[filt].detach().clone().squeeze()
                    )

    def get_variable_tensor(
        self, group_index: int, var_name: str, to_cpu: bool = True
    ) -> torch.Tensor:
        """Get a tensor of all recorded values for a specific group and variable.

        Parameters
        ----------
        group_index : int
            Index of the group in the monitor's groups list.
        var_name : str
            Name of the variable to retrieve.
        to_cpu : bool, optional
            Whether to move the result to CPU memory, by default True.

        Returns
        -------
        torch.Tensor
            Tensor of shape [T, N] with the evolution of the variable over time.
            T is the number of recorded time steps and N is the number of filtered elements.
            If no values were recorded, returns an empty tensor.
        """
        values = self.recorded_values[group_index][var_name]
        device = self.groups[group_index].device if not to_cpu else "cpu"
        if not values:
            return torch.empty((0, self.groups[group_index].size), device=device)
        return torch.stack(values, dim=0).to(device)


class RingBufferSpikeMonitor(Node):
    """Spike monitor with fixed-size ring buffer, optimized for CUDA graph recording.

    This monitor records spikes from one or more neuron groups into preallocated
    ring buffers on GPU memory. It supports efficient extraction of spikes to CPU
    memory using pinned memory and non-blocking transfers.

    Attributes
    ----------
    groups : List[NeuronGroup]
        List of neuron groups being monitored.
    filter_masks : List[torch.Tensor]
        Boolean masks indicating which neurons are monitored in each group.
    spike_buffers : List[torch.Tensor]
        Fixed-size spike buffers per group, storing (neuron_id, time) pairs.
    write_indices : List[int]
        Write pointers for each group's ring buffer.
    total_spikes : List[int]
        Total number of spikes recorded for each group (used to handle overflows).
    """

    def __init__(self, groups: List[NeuronGroup], max_spikes: int = 1_000_000):
        """Initialize the monitor with the specified neuron groups and buffer size.

        Parameters
        ----------
        groups : List[NeuronGroup]
            List of neuron groups to monitor.
        max_spikes : int, optional
            Maximum number of spikes to store per group, by default 1_000_000.
        """
        super().__init__()
        self.groups = groups
        self.max_spikes = max_spikes

        self.device = groups[0].device
        self.num_groups = len(groups)

        # Create a filter mask per group (boolean tensor on device)
        self.filter_masks = [
            torch.zeros(group.size, dtype=torch.bool, device=self.device).scatter_(
                0, group.filter, True
            )
            for group in groups
        ]

        # Preallocate spike buffers per group
        self.spike_buffers = [
            torch.empty((max_spikes, 2), dtype=torch.long, device=self.device)
            for _ in groups
        ]
        self.write_indices = [0 for _ in groups]
        self.total_spikes = [0 for _ in groups]

    def _process(self):
        """Process the monitor at the current time step.

        This method checks for new spikes in the monitored groups and records
        the spikes that match the group's filter into the corresponding ring buffer.
        Compatible with CUDA graph capture (no CPU operations inside).
        """
        super()._process()

        t = (
            globals.simulator.local_circuit.t
        )  # Keep t as tensor for CUDA graph compatibility

        for i, (group, mask) in enumerate(zip(self.groups, self.filter_masks)):
            delay_max = group.delay_max

            if (t % delay_max) != (delay_max - 1):
                continue

            buffer = group.get_spike_buffer()  # shape [N, D]
            spike_indices = buffer.nonzero(as_tuple=False)  # [N_spikes, 2]
            if spike_indices.numel() == 0:
                continue

            neuron_ids = spike_indices[:, 0]
            delay_slots = spike_indices[:, 1]

            valid = mask[neuron_ids]
            neuron_ids = neuron_ids[valid]
            delay_slots = delay_slots[valid]

            if neuron_ids.numel() == 0:
                continue

            times = t - delay_max + delay_slots
            new_spikes = torch.stack([neuron_ids, times], dim=1)  # [M, 2]
            M = new_spikes.shape[0]

            start = self.write_indices[i]
            end = (start + M) % self.max_spikes

            if end < start:  # Wrap around
                self.spike_buffers[i][start:] = new_spikes[: self.max_spikes - start]
                self.spike_buffers[i][:end] = new_spikes[self.max_spikes - start :]
            else:
                self.spike_buffers[i][start:end] = new_spikes

            self.write_indices[i] = end
            self.total_spikes[i] += M

    def get_spike_tensor(
        self, group_index: int, to_cpu: bool = True, pin_memory: bool = True
    ) -> torch.Tensor:
        """Retrieve recorded spikes for a given group.

        Parameters
        ----------
        group_index : int
            Index of the group in the monitor's groups list.
        to_cpu : bool, optional
            Whether to move the tensor to CPU memory, by default True.
        pin_memory : bool, optional
            Whether to use pinned memory when moving to CPU, by default True.

        Returns
        -------
        torch.Tensor
            Tensor of shape [N_spikes, 2] with columns (neuron_id, time).
            If no spikes were recorded, returns an empty tensor.
        """
        count = min(self.total_spikes[group_index], self.max_spikes)
        buffer = self.spike_buffers[group_index]
        start = (self.write_indices[group_index] - count) % self.max_spikes

        if count == 0:
            device = "cpu" if to_cpu else self.device
            return torch.empty((0, 2), dtype=torch.long, device=device)

        if start + count <= self.max_spikes:
            out = buffer[start : start + count]
        else:
            part1 = buffer[start:]
            part2 = buffer[: (start + count) % self.max_spikes]
            out = torch.cat([part1, part2], dim=0)

        if to_cpu:
            return out.to("cpu", non_blocking=True) if pin_memory else out.cpu()
        return out
