from __future__ import annotations

from . import globals
from .neurons import NeuronGroup
from .utils import is_distributed

from typing import List, Optional

import torch
import torch.distributed as dist


class BridgeNeuronGroup(NeuronGroup):
    """Bridge for inter-GPU communication of spikes.

    This specialized neuron group enables communication of spikes between
    neural populations on different GPUs in a distributed simulation.
    It collects spikes over a predefined number of steps and then
    efficiently transfers them to all other GPUs in the simulation.

    Attributes
    ----------
    n_local_neurons : int
        Number of neurons per GPU in the bridge.
    rank : int
        Rank (GPU index) of the current process.
    n_bridge_steps : int
        Number of time steps to collect before synchronizing.
    _write_buffer : torch.Tensor
        Buffer for collecting outgoing spikes.
    _gathered : List[torch.Tensor]
        Buffers for receiving spikes from all GPUs.
    _time_range : torch.Tensor
        Range of time indices for scheduling future spikes.
    _comm_req : Optional[dist.Work]
        Handle for asynchronous communication operation.
    _comm_result : Optional[torch.Tensor]
        Result tensor for asynchronous communication.
    """

    n_local_neurons: int
    rank: int
    n_bridge_steps: int
    _write_buffer: torch.Tensor
    _gathered: List[torch.Tensor]
    _time_range: torch.Tensor
    _comm_req: Optional[dist.Work]
    _comm_result: Optional[torch.Tensor]

    def __init__(
        self,
        device: torch.device,
        rank: int,
        world_size: int,
        n_local_neurons: int,
        n_bridge_steps: int,
        spatial_dimensions: int = 2,
        delay_max: int = 20,
    ):
        """Initialize a bridge neuron group for inter-GPU communication.

        Parameters
        ----------
        device : torch.device
            GPU device for this process.
        rank : int
            Rank (GPU index) of the current process.
        world_size : int
            Total number of GPUs in the simulation.
        n_local_neurons : int
            Number of bridge neurons per GPU.
        n_bridge_steps : int
            Number of time steps to collect before synchronizing.
        spatial_dimensions : int, optional
            Number of spatial dimensions, by default 2.
        delay_max : int, optional
            Maximum delay in time steps for spike propagation, by default 20.
            Must be greater than n_bridge_steps.

        Raises
        ------
        AssertionError
            If n_bridge_steps is not less than delay_max.
        """
        super().__init__(
            device,
            n_local_neurons * world_size,
            spatial_dimensions=spatial_dimensions,
            delay_max=delay_max,
        )
        self.n_local_neurons = n_local_neurons
        self.n_bridge_steps = n_bridge_steps
        self.rank = rank
        self.n_bits = n_local_neurons * n_bridge_steps
        self._write_buffer = torch.zeros(
            (n_local_neurons, n_bridge_steps), dtype=torch.bool, device=device
        )
        # Crear buffer para comunicación
        self._bool2uint8_weights = torch.tensor(
            [1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8, device=device
        )
        dummy_packed = self._bool_to_uint8(
            torch.zeros(
                n_local_neurons * n_bridge_steps, dtype=torch.bool, device=device
            )
        )
        self._gathered = [torch.empty_like(dummy_packed) for _ in range(world_size)]
        self._time_range = torch.arange(
            self.n_bridge_steps, dtype=torch.long, device=device
        )
        assert (
            n_bridge_steps < delay_max
        ), "Bridge steps must be lower than the bridge neuron population's max delay."
        self._comm_req: Optional[dist.Work] = None
        self._comm_result: Optional[torch.Tensor] = None

    def inject_currents(self, I: torch.Tensor):
        """Inject input currents into the bridge neurons.

        Bridge neurons will spike in response to any positive input current.
        Only neurons corresponding to the current GPU's rank are affected.

        Parameters
        ----------
        I : torch.Tensor
            Float tensor of shape (n_neurons,) containing input currents.

        Raises
        ------
        AssertionError
            If the shape of I doesn't match the total number of neurons.
        """
        assert I.shape[0] == self.size
        from_id = self.rank * self.n_local_neurons
        to_id = (self.rank + 1) * self.n_local_neurons
        subset = I[from_id:to_id]  # [n_local_neurons]

        t_mod = globals.engine.local_circuit.t % self.n_bridge_steps
        mask = subset > 0  # [n_local_neurons]
        self._write_buffer.index_copy_(
            1, t_mod, (self._write_buffer.index_select(1, t_mod) | mask.unsqueeze(1))
        )

    def inject_spikes(self, spikes: torch.Tensor):
        """Inject spikes directly into the bridge neurons.

        Only neurons corresponding to the current GPU's rank are affected.

        Parameters
        ----------
        spikes : torch.Tensor
            Boolean tensor of shape (n_neurons,) indicating which neurons should spike.

        Raises
        ------
        AssertionError
            If the shape of spikes doesn't match the total number of neurons.
        """
        assert spikes.shape[0] == self.size
        from_id = self.rank * self.n_local_neurons
        to_id = (self.rank + 1) * self.n_local_neurons
        subset = spikes[from_id:to_id].bool()

        t_mod = globals.engine.local_circuit.t % self.n_bridge_steps
        mask = subset > 0  # [n_local_neurons]
        self._write_buffer.index_copy_(
            1, t_mod, (self._write_buffer.index_select(1, t_mod) | mask.unsqueeze(1))
        )

    def _process(self):
        """Process the bridge for the current time step.

        Collects spikes from local neurons and periodically synchronizes with other GPUs.
        This is implemented differently depending on whether the simulation is distributed:
        - In distributed mode: Uses asynchronous all_gather to efficiently exchange spike data
        - In non-distributed mode: Simply copies spikes to future time steps
        """
        super()._process()

        t = globals.engine.local_circuit.t
        phase = t % self.n_bridge_steps

        if is_distributed():
            # --- At the end of the block (n_bridge_steps-1): pack, clean, and launch async gather ---
            if phase == self.n_bridge_steps - 1:
                # 1) Flatten and pack
                write_buffer_flat = self._write_buffer.flatten()
                packed = self._bool_to_uint8(write_buffer_flat)

                # 2) Clean for the next block
                self._write_buffer.fill_(False)

                # 3) Async Gather
                self._comm_req = dist.all_gather(self._gathered, packed, async_op=True)

            # --- At the beginning of next block: wait, rebuild and load to the buffer ---
            elif phase == 0 and getattr(self, "_comm_req", None) is not None:
                # 4) Wait the gather to finish
                self._comm_req.wait()

                # 5) Rebuild the tensor [n_total_neurons x n_bridge_steps]
                bool_list = []
                for p in self._gathered:
                    unpacked = self._uint8_to_bool(p, self.n_bits)
                    reshaped = unpacked.view(self.n_local_neurons, self.n_bridge_steps)
                    bool_list.append(reshaped)
                result = torch.cat(bool_list, dim=0)

                # 6) Set the spikes in the future
                time_indices = (t + self._time_range + 1) % self.delay_max
                self._spike_buffer.index_copy_(1, time_indices, result)

                # 7) Clean handle for the next block
                self._comm_req = None

        else:
            # Non-distributed mode (just load to the buffer)
            if phase == self.n_bridge_steps - 1:
                time_indices = (t + 1 + self._time_range) % self.delay_max
                self._spike_buffer.index_copy_(1, time_indices, self._write_buffer)
                self._write_buffer.fill_(False)

    def where_rank(self, rank: int) -> BridgeNeuronGroup:
        """Filter the bridge group to select neurons from a specific GPU.

        Parameters
        ----------
        rank : int
            The rank (GPU index) to filter for.

        Returns
        -------
        BridgeNeuronGroup
            A new bridge group with only neurons from the specified rank selected.

        Raises
        ------
        ValueError
            If the rank is out of the valid range.

        Examples
        --------
        >>> # Connect to neurons on GPU 1
        >>> (neurons >> bridge.where_rank(1))(pattern='one-to-one', weight=1.0)
        """
        clone = self._clone_with_new_filter()

        if rank < 0 or rank >= clone.size // clone.n_local_neurons:
            raise ValueError(f"El rank {rank} está fuera del rango válido.")

        start = rank * clone.n_local_neurons
        end = (rank + 1) * clone.n_local_neurons
        idx = torch.arange(clone.size, device=clone.device)
        mask = (idx >= start) & (idx < end)
        clone.filter &= mask
        return clone

    def _bool_to_uint8(self, x: torch.Tensor) -> torch.Tensor:
        """Pack boolean values into uint8 for efficient communication.

        Packs 8 boolean values into each byte to reduce communication overhead.

        Parameters
        ----------
        x : torch.Tensor
            Boolean tensor to pack.

        Returns
        -------
        torch.Tensor
            Packed uint8 tensor.
        """
        x_flat = x.flatten().to(torch.uint8)
        pad_len = (8 - x_flat.numel() % 8) % 8
        if pad_len:
            x_flat = torch.cat(
                [x_flat, torch.zeros(pad_len, dtype=torch.uint8, device=x.device)]
            )
        x_flat = x_flat.reshape(-1, 8)
        return (x_flat * self._bool2uint8_weights).sum(dim=1)

    def _uint8_to_bool(self, x: torch.Tensor, num_bits: int) -> torch.Tensor:
        """Unpack uint8 values back to booleans after communication.

        Parameters
        ----------
        x : torch.Tensor
            Packed uint8 tensor.
        num_bits : int
            Number of boolean values to extract.

        Returns
        -------
        torch.Tensor
            Unpacked boolean tensor.
        """
        x = x.flatten()
        bits = ((x.unsqueeze(1) >> torch.arange(8, device=x.device)) & 1).to(torch.bool)
        return bits.flatten()[:num_bits]
