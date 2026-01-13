from __future__ import annotations

from . import globals
from .core import Node
from .group import Group
from .neurons import NeuronGroup
from .utils import VisualizerClient

from typing import List, Dict, Optional

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
    recorded_spikes: List[List[torch.Tensor]]  # Per group: list of tensors [N_spikes, 2] (neuron_idx, t)

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
        current_step = globals.simulator.local_circuit.current_step
        for i, (group, filter) in enumerate(zip(self.groups, self.filters)):
            delay_max = group.delay_max

            if (current_step % delay_max) != (delay_max - 1):
                continue

            buffer = group.get_spike_buffer()  # shape: [N, D]
            spike_indices = buffer.nonzero(as_tuple=False)  # shape: [N_spikes, 2]
            if spike_indices.numel() == 0:
                continue

            neuron_ids = spike_indices[:, 0]
            is_filtered = torch.isin(neuron_ids, filter)
            neuron_ids = neuron_ids[is_filtered]

            delay_slots = spike_indices[:, 1][is_filtered]
            times = globals.simulator.local_circuit.current_step - delay_max + delay_slots +1 #TODO: Check if this +1 is really necessary to avoid negative time values

            spikes_tensor = torch.stack(
                [neuron_ids, times], dim=1
            )  # shape: [N_spikes, 2]
            self.recorded_spikes[i].append(spikes_tensor)

    def get_spike_tensor(self, group_index: int = 0, to_cpu: bool = True) -> torch.Tensor:
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

    def __init__(self, groups: List[Group], variable_names: List[str]):
        """Initialize a variable monitor for the given groups and variables.

        Parameters
        ----------
        groups : List[Group]
            List of groups to monitor.
        variable_names : List[str]
            Names of variables to monitor in each group.
        """
        super().__init__()
        self.groups = groups
        self.filters = [group.filter.nonzero(as_tuple=True) for group in groups]
        self.variable_names = variable_names

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
    filters : List[torch.Tensor]
        Boolean masks indicating which neurons are monitored in each group.
    spike_buffers : List[torch.Tensor]
        Fixed-size spike buffers per group, storing (neuron_id, time) pairs.
    write_indices : List[int]
        Write pointers for each group's ring buffer.
    total_spikes : List[int]
        Total number of spikes recorded for each group (used to handle overflows).
    """
    groups: List[NeuronGroup]
    filters: List[torch.Tensor]
    spike_buffers: List[torch.Tensor]  # Per group: list of tensors [N_spikes, 2] (neuron_idx, t)
    write_indices: List
    total_spikes: List


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
        self.filters = [
            torch.zeros(group.size, dtype=torch.bool, device=self.device).scatter_(
                0, group.filter.nonzero(as_tuple=True)[0], True
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

        current_step = globals.simulator.local_circuit.current_step
        for i, (group, mask) in enumerate(zip(self.groups, self.filters)):
            delay_max = group.delay_max

            if (current_step % delay_max) != (delay_max - 1):
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

            times = current_step - delay_max + delay_slots
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


class RingBufferVariableMonitor(Node):
    """
    Ring-buffer monitor for continuous variables (voltages, weights, etc).

    Stores T time steps, each containing the filtered variable values
    for each monitored variable of each group.

    Structure:
        buffer[group_idx][var_name]   → tensor [T, N_filtered]
    """

    def __init__(
        self,
        groups: List,
        variable_names: List[str],
        max_steps: int = 100_000,
        pin_memory: bool = True,
    ):
        """
        Parameters
        ----------
        groups : List[Group]
        variable_names : List[str]
            Variables to monitor, possibly with indexing "var@k".
        max_steps : int
            Ring buffer temporal length.
        """
        super().__init__()

        self.groups = groups
        self.variable_names = variable_names
        self.max_steps = max_steps
        self.pin_memory = pin_memory

        self.num_groups = len(groups)
        self.device = groups[0].device

        # --- Precompute boolean filters ---
        self.filters = [
            group.filter.nonzero(as_tuple=True)[0].to(self.device)
            for group in groups
        ]
        self.sizes_filtered = [f.numel() for f in self.filters]

        # --- Preallocate ring buffers ---
        # buffers[g][var] = tensor [T, N_filtered]
        self.buffers: List[Dict[str, torch.Tensor]] = []
        self.write_indices = []     # pointer per group
        self.total_steps = []       # steps recorded

        for g, filt in enumerate(self.filters):
            group_buffers = {}

            N_filtered = filt.numel()

            for var_name in variable_names:
                # We will discover dtype/shape at runtime → allocate after 1st step
                group_buffers[var_name] = None

            self.buffers.append(group_buffers)
            self.write_indices.append(0)
            self.total_steps.append(0)


    # ----------------------------------------------------------
    # Utility: extract variable + optional @index
    # ----------------------------------------------------------
    def _extract_var(self, group, var_name):
        match = re.match(r"^([\w\.]+)@(\d+)$", var_name)
        if match:
            base, idx = match.group(1), int(match.group(2))
        else:
            base, idx = var_name, None

        value = getattr(group, base, None)
        if value is None:
            raise AttributeError(f"Group has no variable '{base}'")
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Variable '{base}' is not a torch.Tensor.")

        if idx is not None:
            if value.ndim < 2:
                raise ValueError(f"Variable '{base}' does not support indexing '@'.")
            if idx >= value.shape[1]:
                raise IndexError(f"Index {idx} out of range for '{base}'")
            value = value[:, idx]

        return value


    # ----------------------------------------------------------
    # Processing step
    # ----------------------------------------------------------
    def _process(self):
        super()._process()

        for g, (group, filt) in enumerate(zip(self.groups, self.filters)):
            N = filt.numel()
            write_ptr = self.write_indices[g]

            for var_name in self.variable_names:

                # ---- Get variable (with possible "@k" extraction)
                raw_val = self._extract_var(group, var_name)
                val = raw_val[filt].squeeze().detach()   # shape [N], ensure no graph

                # ---- Allocate buffer on first use
                if self.buffers[g][var_name] is None:
                    dtype = val.dtype
                    device = val.device
                    self.buffers[g][var_name] = torch.empty(
                        (self.max_steps, N),
                        dtype=dtype,
                        device=device
                    )

                # ---- Write into ring buffer
                self.buffers[g][var_name][write_ptr] = val

            # Advance write index
            self.write_indices[g] = (write_ptr + 1) % self.max_steps
            self.total_steps[g] += 1


    # ----------------------------------------------------------
    # Get recorded variable tensor
    # ----------------------------------------------------------
    def get_variable_tensor(
        self,
        group_index: int,
        var_name: str,
        to_cpu: bool = True,
    ):
        """
        Returns tensor [T, N_filtered], with correct ring-buffer ordering.
        """
        buffer = self.buffers[group_index][var_name]
        if buffer is None:
            # No data recorded yet
            device = "cpu" if to_cpu else self.device
            return torch.empty((0, 0), device=device)

        total = min(self.total_steps[group_index], self.max_steps)
        w = self.write_indices[group_index]

        # Start index = w - total (mod max_steps)
        start = (w - total) % self.max_steps

        if start + total <= self.max_steps:
            out = buffer[start : start + total]
        else:
            # wrap around
            part1 = buffer[start:]
            part2 = buffer[: (start + total) % self.max_steps]
            out = torch.cat([part1, part2], dim=0)

        if to_cpu:
            return out.to("cpu", non_blocking=True) if self.pin_memory else out.cpu()
        return out


class RealtimeSpikeMonitor(Node):
    def __init__(self, groups: List[NeuronGroup], viz_client: VisualizerClient, plot_id: str, 
                 group_names: Optional[List[str]] = None, rollover_spikes: int=50000, rollover_lines: int=200, smooth_tau: float=0.1, rate_interval: int=20):
        super().__init__()
        self.groups = groups
        self.filters = [group.filter.nonzero(as_tuple=True)[0] for group in groups]
        self.viz = viz_client
        self.plot_id = plot_id
        self.rate_plot_id = f"{plot_id}_rates" # ID derivada para la gráfica de abajo
        
        self.group_names = [group.name for group in groups] if group_names is None else group_names
        self.smooth_tau = smooth_tau
        self.rate_interval = rate_interval 
        
        self.current_rates = [0.0] * len(groups)
        self.n_neurons = [filt.numel() for filt in self.filters]

        # 1. CREAR RASTER (Arriba)
        self.viz.create_raster(self.plot_id, title=f"Raster: {plot_id}", rollover=rollover_spikes)
        
        # 2. CREAR RATES (Abajo)
        # Usamos un rollover menor por defecto (1000) porque es una linea continua
        self.viz.create_lineplot(self.rate_plot_id, series_names=self.group_names, 
                                 title=f"Firing Rates: {plot_id}", rollover=rollover_lines)
        
        self.offsets = [0]
        for g in groups[:-1]:
            self.offsets.append(self.offsets[-1] + g.filter.sum().item())

    def _process(self):
        super()._process()
        current_step = globals.simulator.local_circuit.current_step
        dt = 1e-3 
        t_now = current_step * dt

        # --- RECOLECCIÓN SPIKES ---
        ids_batch = []
        times_batch = []
        groups_batch = []
        step_counts = [0] * len(self.groups)

        for group_idx, (group, filter_tensor) in enumerate(zip(self.groups, self.filters)):
            delay_max = group.delay_max
            if (current_step % delay_max) != (delay_max - 1): continue

            buffer = group.get_spike_buffer()
            spike_indices = buffer.nonzero(as_tuple=False)
            if spike_indices.numel() == 0: continue

            neuron_ids = spike_indices[:, 0]
            is_filtered = torch.isin(neuron_ids, filter_tensor)
            valid_neuron_ids = neuron_ids[is_filtered]
            
            count = valid_neuron_ids.numel()
            step_counts[group_idx] = count 

            if count == 0: continue
            
            delay_slots = spike_indices[:, 1][is_filtered]
            spike_steps = current_step - delay_max + delay_slots + 1
            spike_times = spike_steps.float() * dt
            final_ids = valid_neuron_ids + self.offsets[group_idx]
            
            ids_batch.extend(final_ids.cpu().tolist())
            times_batch.extend(spike_times.cpu().tolist())
            groups_batch.extend([group_idx] * len(final_ids))

        if ids_batch:
            self.viz.push_spikes(self.plot_id, ids_batch, times_batch, groups_batch)

        # --- CÁLCULO RATES ---
        alpha = dt / self.smooth_tau if self.smooth_tau > 0 else 1.0
        rates_current_step = []
        for i, count in enumerate(step_counts):
            inst_rate = (count / self.n_neurons[i]) / dt if self.n_neurons[i] > 0 else 0.0
            self.current_rates[i] = alpha * inst_rate + (1 - alpha) * self.current_rates[i]
            rates_current_step.append(self.current_rates[i])

        # --- ENVÍO RATES (A la gráfica secundaria) ---
        if current_step % self.rate_interval == 0:
            self.viz.push_values(self.rate_plot_id, t_now, rates_current_step)

# La clase RealtimeVariableMonitor se queda igual que en la versión anterior
class RealtimeVariableMonitor(Node):
    def __init__(self, groups: List[Group], variable_names: List[str], viz_client: VisualizerClient, plot_id: str, interval: int = 10, rollover: int = 1000):
        super().__init__()
        self.groups = groups
        self.filters = [group.filter.nonzero(as_tuple=True)[0] for group in groups]
        self.variable_names = variable_names
        self.viz = viz_client
        self.plot_id = plot_id
        self.interval = interval

        self.series_labels = []
        self.monitor_config = []

        for g_i, (group, filt) in enumerate(zip(self.groups, self.filters)):
            indices = filt.tolist()
            for var_name in variable_names:
                match = re.match(r"^([\w\.]+)@(\d+)$", var_name)
                base_name, sub_index = (match.group(1), int(match.group(2))) if match else (var_name, None)
                
                for n_i in indices:
                    label = f"G{g_i}_N{n_i}_{base_name}"
                    if sub_index is not None: label += f"@{sub_index}"
                    self.series_labels.append(label)
                    self.monitor_config.append({"group": group, "idx": n_i, "var": base_name, "sub": sub_index})

        self.viz.create_lineplot(plot_id, series_names=self.series_labels, title=f"Monitor: {plot_id}", rollover=rollover)

    def _process(self):
        super()._process()
        current_step = globals.simulator.local_circuit.current_step
        if current_step % self.interval != 0: return

        t_now = current_step * 1e-3 
        current_values = []
        for cfg in self.monitor_config:
            try:
                val_tensor = getattr(cfg["group"], cfg["var"])
                if cfg["sub"] is not None: val = val_tensor[cfg["idx"], cfg["sub"]]
                else: val = val_tensor[cfg["idx"]]
                current_values.append(val.item())
            except AttributeError:
                pass


        self.viz.push_values(self.plot_id, t_now, current_values)