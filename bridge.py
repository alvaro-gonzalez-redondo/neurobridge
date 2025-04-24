from __future__ import annotations

from . import globals
from .neurons import NeuronGroup
from .utils import is_distributed

from typing import List, Optional

import torch
import torch.distributed as dist



class _BridgeNeuronGroup(NeuronGroup):
    n_local_neurons: int
    rank: int
    n_bridge_steps: int
    _write_buffer: torch.Tensor
    _gathered: List[torch.Tensor]
    _time_range: torch.Tensor
    _comm_req: Optional[dist.Work]
    _comm_result: Optional[torch.Tensor]


    def __init__(self, device: torch.device, rank: int, world_size: int, n_local_neurons: int, n_bridge_steps: int, spatial_dimensions: int=2, delay_max: int=20):
        super().__init__(device, n_local_neurons*world_size, spatial_dimensions=spatial_dimensions, delay_max=delay_max)
        self.n_local_neurons = n_local_neurons
        self.n_bridge_steps = n_bridge_steps
        self.rank = rank
        self.n_bits = n_local_neurons * n_bridge_steps
        self._write_buffer = torch.zeros((n_local_neurons, n_bridge_steps), dtype=torch.bool, device=device)
        # Crear buffer para comunicación
        self._bool2uint8_weights = torch.tensor([1,2,4,8,16,32,64,128], dtype=torch.uint8, device=device)
        dummy_packed = self._bool_to_uint8(torch.zeros(n_local_neurons * n_bridge_steps, dtype=torch.bool, device=device))
        self._gathered = [torch.empty_like(dummy_packed) for _ in range(world_size)]
        self._time_range = torch.arange(self.n_bridge_steps, dtype=torch.long, device=device)
        assert n_bridge_steps<delay_max, "Bridge steps must be lower than the bridge neuron population's max delay."
        self._comm_req: Optional[dist.Work] = None
        self._comm_result: Optional[torch.Tensor] = None


    # Any positive input to a bridge neuron will automatically generate a spike
    def inject_currents(self, I: torch.Tensor):
        assert I.shape[0] == self.size
        from_id = self.rank * self.n_local_neurons
        to_id = (self.rank + 1) * self.n_local_neurons
        subset = I[from_id:to_id]  # [n_local_neurons]

        t_mod = globals.engine.local_circuit.t % self.n_bridge_steps
        mask = subset > 0  # [n_local_neurons]
        self._write_buffer.index_copy_(1, t_mod, (
            self._write_buffer.index_select(1, t_mod) | mask.unsqueeze(1)
            ))


    # Any injected spike to a bridge neuron will automatically generate a spike
    def inject_spikes(self, spikes: torch.Tensor):
        assert spikes.shape[0] == self.size
        from_id = self.rank * self.n_local_neurons
        to_id = (self.rank+1) * self.n_local_neurons
        subset = spikes[from_id:to_id].bool()
        
        #indices = subset.nonzero(as_tuple=True)[0]
        t_mod = globals.engine.local_circuit.t % self.n_bridge_steps
        mask = subset > 0  # [n_local_neurons]
        self._write_buffer.index_copy_(1, t_mod, (
            self._write_buffer.index_select(1, t_mod) | mask.unsqueeze(1)
            ))


    def _process(self):
        super()._process()

        t = globals.engine.local_circuit.t
        phase = t % self.n_bridge_steps

        if is_distributed():
            # --- al final del bloque: empaquetar, limpiar y lanzar gather asíncrono ---
            if phase == self.n_bridge_steps - 1:
                # 1) Aplana y empaqueta
                write_buffer_flat = self._write_buffer.flatten()
                packed = self._bool_to_uint8(write_buffer_flat)

                # 2) Limpia para el siguiente bloque
                self._write_buffer.fill_(False)

                # 3) Gather asíncrono
                self._comm_req = dist.all_gather(self._gathered, packed, async_op=True)

            # --- al inicio del bloque siguiente: esperar, reconstruir y volcar al buffer ---
            elif phase == 0 and getattr(self, "_comm_req", None) is not None:
                # 4) Espera a que termine el gather
                self._comm_req.wait()

                # 5) Reconstruye el tensor [n_total_neurons x n_bridge_steps]
                bool_list = []
                for p in self._gathered:
                    unpacked = self._uint8_to_bool(p, self.n_bits)
                    reshaped = unpacked.view(self.n_local_neurons, self.n_bridge_steps)
                    bool_list.append(reshaped)
                result = torch.cat(bool_list, dim=0)

                # 6) Programa esos spikes en el futuro
                time_indices = (t + self._time_range + 1) % self.delay_max
                self._spike_buffer.index_copy_(1, time_indices, result)

                # 7) Limpia el handle para el próximo bloque
                self._comm_req = None

        else:
            # Modo no distribuido (igual que antes)
            if phase == self.n_bridge_steps - 1:
                time_indices = (t + 1 + self._time_range) % self.delay_max
                self._spike_buffer.index_copy_(1, time_indices, self._write_buffer)
                self._write_buffer.fill_(False)


    
    def where_rank(self, rank: int) -> _BridgeNeuronGroup:
        """
        Aplica un filtro que selecciona únicamente las neuronas puente asociadas a un rank específico.
    
        Args:
            rank: Índice del proceso (GPU) cuyos axones puente se desean filtrar.
    
        Returns:
            El propio grupo, con el filtro actualizado.
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
        # Aplanar el tensor primero
        x_flat = x.flatten().to(torch.uint8)
        pad_len = (8 - x_flat.numel() % 8) % 8
        if pad_len:
            x_flat = torch.cat([x_flat, torch.zeros(pad_len, dtype=torch.uint8, device=x.device)])
        x_flat = x_flat.reshape(-1, 8)
        return (x_flat * self._bool2uint8_weights).sum(dim=1)


    def _uint8_to_bool(self, x: torch.Tensor, num_bits: int) -> torch.Tensor:
        # Asegurar que x sea un tensor 1D
        x = x.flatten()
        bits = ((x.unsqueeze(1) >> torch.arange(8, device=x.device)) & 1).to(torch.bool)
        return bits.flatten()[:num_bits]
