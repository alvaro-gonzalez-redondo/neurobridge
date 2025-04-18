from __future__ import annotations
import os
import random
import numpy as np
import torch
import torch.distributed as dist
#import torch_scatter
import logging
from typing import List, Optional, Callable, ClassVar, Any, Type, Union, Dict

import sys
import matplotlib
import matplotlib.pyplot as plt

import contextlib, contextvars


class ParentStack(contextlib.AbstractContextManager):
    _stack_var = contextvars.ContextVar("_stack", default=[])

    def __init__(self, parent: Node):
        self._parent = parent
        self._token  = None

    def __enter__(self):
        stack = self._stack_var.get()
        self._token = self._stack_var.set(stack + [self._parent])
        return self._parent

    def __exit__(self, exc_type, exc, tb):
        self._stack_var.reset(self._token)
        return False          # no suprime excepciones

    @staticmethod
    def current_parent() -> Optional[Node]:
        stack = ParentStack._stack_var.get()
        return stack[-1] if stack else None


class Node:
    children: List[Node]
    parent: Optional[Node]

    def __init__(self):
        self.children = []
        self.parent = None

        parent = ParentStack.current_parent()
        if parent is not None:
            parent.add_child(self)


    def add_child(self, node:Node) -> None:
        # Nodes should be unique in the scene tree
        if node.parent is not None:
            node.parent.remove_child(node)
        
        self.children.append(node)
        node.parent = self
    
    def remove_child(self, node:Node):
        self.children.remove(node)

    def _call_ready(self) -> None:
        for child in self.children:
            child._call_ready()
        self._ready()

    def _ready(self) -> None:
        """Override this to set up the node after all children are ready."""
        pass

    def _call_process(self) -> None:
        for child in self.children:
            child._call_process()
        self._process()

    def _process(self) -> None:
        """Override this to define what the node does each step."""
        pass


class GPUNode(Node):
    """A node attached to a GPU."""
    device: torch.device

    def __init__(self, device:str):
        super().__init__()
        self.device = torch.device(device)


class Group(GPUNode):
    size: int
    filter: torch.Tensor

    def __init__(self, device:str, size:int):
        super().__init__(device)
        self.size = size
        self.filter = torch.ones(self.size, dtype=torch.bool, device=self.device)


    def where_id(self, condition: Callable[[torch.Tensor], torch.Tensor]) -> Group:
        """
        Aplica un filtro basado en los índices (vectorizado).
    
        Args:
            condition: Función que recibe un tensor de índices y devuelve una máscara booleana.
    
        Returns:
            A sí mismo con el filtro actualizado.
        """
        idx = torch.arange(self.size, device=self.device)
        mask = condition(idx)
        if mask.shape != (self.size,) or mask.dtype != torch.bool:
            raise ValueError("La función debe devolver una máscara booleana del mismo tamaño que el grupo.")
        self.filter &= mask
        return self
        

    def reset_filter(self) -> None:
        self.filter.fill_(True)


class SpatialGroup(Group):
    spatial_dimensions: torch.Tensor
    positions: torch.Tensor


    def __init__(self, device:str, size:int, spatial_dimensions:int=2):
        super().__init__(device, size)
        self.spatial_dimensions = torch.tensor(spatial_dimensions, dtype=torch.int32, device=self.device)
        self.positions = torch.randn((self.size, self.spatial_dimensions), device=self.device)


    def where_pos(self, condition: Callable[[torch.Tensor], torch.Tensor]) -> SpatialGroup:
        """
        Aplica un filtro basado en las posiciones (vectorizado).
    
        Args:
            condition: Función que toma un tensor de posiciones (n x d) y devuelve una máscara booleana.
    
        Returns:
            A sí mismo con el filtro actualizado.
        """
        if self.positions is None:
            raise RuntimeError("Este grupo no tiene posiciones definidas.")
        
        mask = condition(self.positions)
        if mask.shape != (self.size,) or mask.dtype != torch.bool:
            raise ValueError("La función debe devolver una máscara booleana del mismo tamaño que el grupo.")
        
        self.filter &= mask
        return self
    

class NeuronGroup(SpatialGroup):
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

        t_indices = (SimulatorEngine.engine.local_circuit.t - delays) % self.delay_max
        return self._spike_buffer[indices, t_indices]


    def __rshift__(self, other) -> ConnectionOperator:
        """Implementa el operador >> para crear conexiones."""
        return ConnectionOperator(self, other)
    

# Esta neurona funciona como un simple repetidor
class ParrotGroup(NeuronGroup):

    def _process(self) -> None:
        super()._process()

        # Limpiamos los spikes que hubiera en este instante temporal
        t_idx = SimulatorEngine.engine.local_circuit.t % self.delay_max
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


class IFNeuronGroup(NeuronGroup):
    V: torch.Tensor
    threshold: torch.Tensor
    decay: torch.Tensor


    def __init__(self, device: str, n_neurons: int, spatial_dimensions: int = 2, delay_max: int = 20, threshold: float = 1.0, tau: float = 0.1):
        super().__init__(device, n_neurons, spatial_dimensions, delay_max)
        self.V = torch.zeros(n_neurons, dtype=torch.float32, device=self.device)
        self.threshold = torch.tensor([threshold], dtype=torch.float32, device=device)
        decay = np.exp(-1e-3/tau)
        self.decay = torch.tensor(decay, dtype=torch.float32, device=device)


    def _process(self):
        super()._process()
        t_idx = SimulatorEngine.engine.local_circuit.t % self.delay_max

        # Update potential with decay and input
        self.V *= self.decay
        self.V += self._input_currents
        self._input_currents.fill_(0.0)

        # Determine which neurons spike
        spikes = self.V >= self.threshold
        self._spike_buffer.index_copy_(1, t_idx, spikes.unsqueeze(1))
        self.V[spikes] = 0.0  # Reset membrane potential


class RandomSpikeGenerator(NeuronGroup):
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
        t_idx = SimulatorEngine.engine.local_circuit.t % self.delay_max

        self.probabilities.uniform_()
        spikes = self.probabilities < self.firing_rate
        self._spike_buffer.index_copy_(1, t_idx, spikes.unsqueeze(1))


class ConnectionOperator:
    pre: NeuronGroup
    pos: NeuronGroup
    pattern: Optional[str]
    kwargs: dict[str, Any]


    def __init__(self, pre: NeuronGroup, pos: NeuronGroup) -> None:
        if pre.device != pos.device:
            raise RuntimeError("No se pueden conectar poblaciones en diferentes GPUs directamente.")
        self.pre = pre
        self.pos = pos
        self.device = self.pre.device
        self.pattern = None
        self.kwargs = {}
    

    def __call__(
        self,
        pattern: str = 'all-to-all',
        synapse_class: Optional[Type[SynapticGroup]] = None,
        **kwargs: Any
    ) -> SynapticGroup:

        self.pattern = pattern
        self.kwargs = kwargs

        # Generar subconjuntos filtrados (o completos)
        valid_pre = self.pre.filter.nonzero(as_tuple=True)[0]
        valid_pos = self.pos.filter.nonzero(as_tuple=True)[0]

        if pattern == 'all-to-all':
            grid_pre, grid_pos = torch.meshgrid(valid_pre, valid_pos, indexing='ij')
            source_indices = grid_pre.flatten()
            target_indices = grid_pos.flatten()

        elif pattern == 'specific':
            try:
                source_indices = self._compute_parameter(kwargs['idx_pre'], kwargs['idx_pre'], kwargs['idx_pre'])
                target_indices = self._compute_parameter(kwargs['idx_pos'], kwargs['idx_pos'], kwargs['idx_pos'])
            except KeyError:
                raise RuntimeError("Faltan 'idx_pre' o 'idx_pos' en los parámetros para el patrón 'specific'.")
        
        elif pattern == 'one-to-one':
            assert valid_pre.numel() == valid_pos.numel()
            source_indices = valid_pre.clone()
            target_indices = valid_pos.clone()

        else:
            raise NotImplementedError(f"Patrón de conexión '{pattern}' no implementado.")

        # Parámetros comunes a todas las sinapsis
        delay = self._compute_parameter(kwargs.get('delay', 0), source_indices, target_indices).to(torch.long)
        weight = self._compute_parameter(kwargs.get('weight', 0.0), source_indices, target_indices)

        # Crear objeto sináptico
        if synapse_class is None or synapse_class is StaticSynapse:
            connection = StaticSynapse(
                pre=self.pre,
                pos=self.pos,
                idx_pre=source_indices,
                idx_pos=target_indices,
                delay=delay,
                weight=weight
            )

        elif synapse_class is STDPSynapse:
            connection = STDPSynapse(
                pre=self.pre,
                pos=self.pos,
                idx_pre=source_indices,
                idx_pos=target_indices,
                delay=delay,
                weight=weight,
                A_plus=kwargs.get('A_plus', 0.01),
                A_minus=kwargs.get('A_minus', 0.012),
                tau_plus=kwargs.get('tau_plus', 20.0),
                tau_minus=kwargs.get('tau_minus', 20.0),
                dt=kwargs.get('dt', 1.0),
                w_min=kwargs.get('w_min', 0.0),
                w_max=kwargs.get('w_max', 1.0)
            )

        else:
            raise NotImplementedError(f"Clase de sinapsis '{synapse_class}' no soportada.")

        # Limpiar filtros tras conectar
        self.pre.reset_filter()
        self.pos.reset_filter()

        return connection


    def _compute_parameter(
        self,
        param: Any,
        idx_pre: torch.Tensor,
        idx_post: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcula un tensor de parámetros por conexión a partir de:
        - Escalar: se replica.
        - Tensor: se usa directamente o se expande.
        - Lista: se convierte a tensor.
        - Función: se aplica vectorizadamente a idx_pre, idx_post (ambos tensores).
        """
        n = len(idx_pre)
    
        if callable(param):
            values = param(idx_pre, idx_post)
            if not isinstance(values, torch.Tensor):
                raise TypeError("Las funciones deben devolver un tensor.")
            if values.shape[0] != n:
                raise ValueError(f"El tensor devuelto debe tener tamaño {n}, pero tiene {values.shape[0]}.")
            return values.to(device=self.device)
    
        elif isinstance(param, torch.Tensor):
            if param.numel() == 1:
                return torch.full((n,), param.item(), device=self.device)
            if param.numel() != n:
                raise ValueError(f"Expected a tensor of length {n}, got {param.numel()}.")
            return param.to(device=self.device)
    
        elif isinstance(param, list):
            param = torch.tensor(param, device=self.device)
            return self._compute_parameter(param, idx_pre, idx_post)
    
        else:  # Escalar
            return torch.full((n,), float(param), device=self.device)



class SynapticGroup(Group):
    pre: NeuronGroup
    pos: NeuronGroup
    idx_pre: torch.Tensor
    idx_pos: torch.Tensor
    weight: torch.Tensor
    delay: torch.Tensor
    _current_buffer: torch.Tensor


    def __init__(self, pre: NeuronGroup, pos: NeuronGroup, idx_pre: torch.Tensor, idx_pos: torch.Tensor, weight: torch.Tensor, delay: torch.Tensor):
        if pre.device != pos.device:
            raise RuntimeError("Connected populations must be from the same device.")
        device = pre.device
        
        if idx_pre.numel() != idx_pos.numel():
            raise RuntimeError(f"The number of sources ({idx_pre.numel()}) and targets ({idx_pos.numel()}) do not match.")
        size = idx_pre.numel()

        super().__init__(device, size)

        self.pre = pre
        self.pos = pos
        self.idx_pre = idx_pre
        self.idx_pos = idx_pos
        self.weight = weight.to(device=pre.device, dtype=torch.float32)
        self.delay = delay

        self._current_buffer = torch.zeros(self.pos.size, dtype=torch.float32, device=self.device)


    def _process(self):
        super()._process()
        self._propagate()
        self._update()


    def _propagate(self):
        spikes_mask = self.pre.get_spikes_at(self.delay, self.idx_pre)
        mask_f = spikes_mask.to(self.weight.dtype)
        contrib = self.weight * mask_f
        self._current_buffer.zero_()
        self._current_buffer.index_add_(0, self.idx_pos, contrib)
        self.pos.inject_currents(self._current_buffer)


    def _update(self) -> None:
        """
        Time-driven learning rules are implemented here
        """
        raise NotImplementedError("`update` method in `SynapticGroup` must be implemented.")


class StaticSynapse(SynapticGroup):

    def __init__(self, pre, pos, idx_pre, idx_pos, weight, delay):
        super().__init__(pre, pos, idx_pre, idx_pos, weight, delay)

    def _update(self) -> None:
        pass
        

class STDPSynapse(SynapticGroup):
    A_plus: torch.Tensor
    A_minus: torch.Tensor
    tau_plus: torch.Tensor
    tau_minus: torch.Tensor
    w_min: torch.Tensor
    w_max: torch.Tensor
    x_pre: torch.Tensor
    x_pos: torch.Tensor
    alpha_pre: torch.Tensor
    alpha_pos: torch.Tensor
    _delay_1: torch.Tensor

    def __init__(
        self,
        pre: NeuronGroup,
        pos: NeuronGroup,
        idx_pre: torch.Tensor,
        idx_pos: torch.Tensor,
        delay: torch.Tensor,
        weight: Union[float, torch.Tensor],
        A_plus: float = 1e-3,
        A_minus: float = 2e-3,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
        dt: float = 1.0,
        w_min: float = 0.0,
        w_max: float = 1.0
    ):
        super().__init__(pre, pos, idx_pre, idx_pos, weight, delay)
        self.weight = (
            torch.full((len(idx_pre),), float(weight), dtype=torch.float32, device=self.device)
            if isinstance(weight, (int, float))
            else weight.to(device=self.device)
        )
        # Optimization attempts to reduce cache misses
        #sorted_indices = torch.argsort(self.idx_pre)
        #self.idx_pre = self.idx_pre[sorted_indices]
        #self.idx_pos = self.idx_pos[sorted_indices]
        #self.weight = self.weight[sorted_indices]
        
        self.A_plus = torch.tensor(A_plus, device=self.device)
        self.A_minus = torch.tensor(A_minus, device=self.device)
        self.tau_plus = torch.tensor(tau_plus, device=self.device)
        self.tau_minus = torch.tensor(tau_minus, device=self.device)
        self.w_min = torch.tensor(w_min, device=self.device)
        self.w_max = torch.tensor(w_max, device=self.device)

        self.x_pre = torch.zeros(len(idx_pre), dtype=torch.float32, device=self.device)
        self.x_pos = torch.zeros(len(idx_pos), dtype=torch.float32, device=self.device)

        self.alpha_pre = torch.exp(torch.tensor(-dt / tau_plus, device=self.device))
        self.alpha_pos = torch.exp(torch.tensor(-dt / tau_minus, device=self.device))

        self._delay_1 = torch.full_like(self.idx_pos, 1, device=self.device)


    def _update(self) -> None:
        self.x_pre *= self.alpha_pre
        self.x_pos *= self.alpha_pos

        # Spikes relevantes con los delays correctos
        pre_spikes = self.pre.get_spikes_at(self.delay, self.idx_pre)
        pos_spikes = self.pos.get_spikes_at(self._delay_1, self.idx_pos)

        # Actualización de trazas
        self.x_pre += pre_spikes.to(torch.float32)
        self.x_pos += pos_spikes.to(torch.float32)

        # STDP - pre dispara antes que post
        dw = self.A_plus * self.x_pos * pre_spikes
        self.weight += dw

        # STDP - post dispara después que pre
        dw = -self.A_minus * self.x_pre * pos_spikes
        self.weight += dw

        self.weight.clamp_(self.w_min, self.w_max)


class BridgeNeuronGroup(NeuronGroup):
    n_local_neurons: int
    rank: int
    n_bridge_steps: int
    write_buffer: torch.Tensor
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
        self.write_buffer = torch.zeros((n_local_neurons, n_bridge_steps), dtype=torch.bool, device=device)
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

        if False:
            t_mod = SimulatorEngine.engine.local_circuit.t % self.n_bridge_steps
            current_column = self.write_buffer[:, t_mod].squeeze(1)

            mask = subset > 0  # [n_local_neurons]
            current_column |= mask  # In-place bitwise OR
        else:
            t_mod = SimulatorEngine.engine.local_circuit.t % self.n_bridge_steps
            mask = subset > 0  # [n_local_neurons]
            self.write_buffer.index_copy_(1, t_mod, (
                self.write_buffer.index_select(1, t_mod) | mask.unsqueeze(1)
                ))


    # Any injected spike to a bridge neuron will automatically generate a spike
    def inject_spikes(self, spikes: torch.Tensor):
        assert spikes.shape[0] == self.size
        from_id = self.rank * self.n_local_neurons
        to_id = (self.rank+1) * self.n_local_neurons
        subset = spikes[from_id:to_id].bool()
        indices = subset.nonzero(as_tuple=True)[0]
        self.write_buffer[indices, SimulatorEngine.engine.local_circuit.t % self.n_bridge_steps] = True


    def _process(self):
        super()._process()
    
        result = None
        phase = SimulatorEngine.engine.local_circuit.t % self.n_bridge_steps
    
        if is_distributed():
            if phase == self.n_bridge_steps - 1:
                # Lanzar comunicación asíncrona
                write_buffer_flat = self.write_buffer.flatten()
                packed = self._bool_to_uint8(write_buffer_flat)
                self._comm_req = dist.all_gather(self._gathered, packed, async_op=True)
    
            elif phase == 0 and self._comm_req is not None:
                # Esperar a que termine y procesar
                self._comm_req.wait()
    
                bool_list = []
                for p in self._gathered:
                    unpacked = self._uint8_to_bool(p, self.n_bits)
                    reshaped = unpacked.reshape(self.n_local_neurons, self.n_bridge_steps)
                    bool_list.append(reshaped)
    
                result = torch.cat(bool_list, dim=0)
                self._comm_req = None  # Limpiar
    
                # Limpiar buffer tras uso
                self.write_buffer.fill_(False)
    
        else:
            # Modo no distribuido, mismo tratamiento pero inmediato
            if phase == 0:
                result = self.write_buffer.clone()
                self.write_buffer.fill_(False)
    
        # AÑADIR SPIKES FUTUROS
        if result is not None:
            time_indices = (SimulatorEngine.engine.local_circuit.t + self._time_range) % self.delay_max
            self._spike_buffer.index_copy_(1, time_indices, result)

    
    def where_rank(self, rank: int) -> BridgeNeuronGroup:
        """
        Aplica un filtro que selecciona únicamente las neuronas puente asociadas a un rank específico.
    
        Args:
            rank: Índice del proceso (GPU) cuyos axones puente se desean filtrar.
    
        Returns:
            El propio grupo, con el filtro actualizado.
        """
        if rank < 0 or rank >= self.size // self.n_local_neurons:
            raise ValueError(f"El rank {rank} está fuera del rango válido.")
        
        start = rank * self.n_local_neurons
        end = (rank + 1) * self.n_local_neurons
        idx = torch.arange(self.size, device=self.device)
        mask = (idx >= start) & (idx < end)
        self.filter &= mask
        return self


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



class SpikeMonitor(Node):
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
        for i, (group, filter) in enumerate(zip(self.groups, self.filters)):
            delay_max = group.delay_max

            if SimulatorEngine.engine.local_circuit.t % delay_max != delay_max-1:
                continue

            buffer = group.get_spike_buffer()  # shape: [N, D]
            spike_indices = buffer.nonzero(as_tuple=False)  # shape: [N_spikes, 2]
            if spike_indices.numel() == 0:
                continue

            neuron_ids = spike_indices[:, 0]
            is_filtered = torch.isin(neuron_ids, filter)
            neuron_ids = neuron_ids[is_filtered]

            delay_slots = spike_indices[:, 1][is_filtered]
            times = SimulatorEngine.engine.local_circuit.t - delay_max + delay_slots

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


class VariableMonitor(Node):
    groups: List[Group]
    filters: List[torch.Tensor]
    variable_names: List[str]
    recorded_values: List[Dict[str, List[torch.Tensor]]]  # [group_idx][var_name] = list of tensors over time

    def __init__(self, groups: List[Group], variable_names: List[str]):
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


class CUDAGraphSubTree(GPUNode):
    pass


class LocalCircuit(GPUNode):
    """
    Contains all updatable objects of this GPU.
    Used for structural organization. It might contain future extensions for analyses and serialization/deserialization.
    """
    t: torch.Tensor

    graph_root: CUDAGraphSubTree
    graph: torch.cuda.CUDAGraph
    graph_stream: torch.cuda.Stream
    
    bridge: Optional[BridgeNeuronGroup]


    def __init__(self, device):
        super().__init__(device)
        self.t = torch.zeros(1, dtype=torch.long, device=device)
        self.graph_root = CUDAGraphSubTree(device=device)
        self.graph = torch.cuda.CUDAGraph()
        self.graph_stream = torch.cuda.Stream()
        self.bridge = None

    
    def get_statistics(self):
        """Recopila estadísticas de actividad de todos los componentes."""
        stats = {}
        for child in self.children:
            if hasattr(child, 'get_activity_stats'):
                stats[child.name] = child.get_activity_stats()
        return stats


    def save_state(self, path):
        """Guarda el estado completo del circuito."""
        raise NotImplementedError("Not implemented yet.")


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


# ANSI 24-bit escape codes para C0–C9 (matplotlib default color cycle)
def rgb_escape(r, g, b):
    return f"\033[38;2;{r};{g};{b}m"

MATPLOTLIB_RGB = [
    (31, 119, 180),   # C0
    (255, 127, 14),   # C1
    (44, 160, 44),    # C2
    (214, 39, 40),    # C3
    (148, 103, 189),  # C4
    (140, 86, 75),    # C5
    (227, 119, 194),  # C6
    (127, 127, 127),  # C7
    (188, 189, 34),   # C8
    (23, 190, 207),   # C9
]
RESET = "\033[0m"

class RankColorFormatter(logging.Formatter):
    def __init__(self, rank: int, fmt: str):
        super().__init__(fmt)
        rgb = MATPLOTLIB_RGB[rank % 10]
        self.color = rgb_escape(*rgb)

    def format(self, record):
        message = super().format(record)
        return f"{self.color}{message}{RESET}"

def setup_logger(rank: int) -> logging.Logger:
    logger = logging.getLogger(f"Rank{rank}")
    logger.setLevel(logging.INFO)

    # Formato base
    fmt = '%(asctime)s - [%(name)s] %(message)s'

    # Salida a fichero (sin color)
    file_formatter = logging.Formatter(fmt)
    fh = logging.FileHandler(f"log_rank{rank}.txt")
    fh.setFormatter(file_formatter)
    logger.addHandler(fh)

    # Salida a consola (con color según rank)
    console_formatter = RankColorFormatter(rank, fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(console_formatter)
    logger.addHandler(ch)

    return logger



class SimulatorEngine(Node):
    engine: Optional[SimulatorEngine] = None
    logger: ClassVar[Optional[logging.Logger]] = None

    t: int
    n_gpus: int
    world_size: int
    rank: int
    local_circuit: LocalCircuit


    def __enter__(self):
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if is_distributed():
            dist.destroy_process_group()

    def autoparent(self, mode: str = "normal") -> ParentStack:
        if mode == "graph":
            target = self.local_circuit.graph_root
        elif mode == "normal":
            target = self.local_circuit
        else:
            raise RuntimeError(f"Invalid autoparent mode: {mode}.")
        return ParentStack(target)


    def __init__(self):
        super().__init__()

        SimulatorEngine.engine = self
        
        # Seguridad: Verificar GPUs disponibles
        if not torch.cuda.is_available(): ## ¿Tenemos CUDA?
            raise RuntimeError("CUDA is not available.")
        
        self.n_gpus = torch.cuda.device_count() ## ¿Número de GPUs?
        if "WORLD_SIZE" in os.environ:
            self.world_size = int(os.environ["WORLD_SIZE"])
            if self.world_size > self.n_gpus:
                raise RuntimeError(f"It is required to have {self.world_size} GPUs but there are only {self.n_gpus} available.")
        else:
            self.world_size = 1

        # Inicialización del circuito local
        if "RANK" in os.environ:
            dist.init_process_group(backend="nccl") #, init_method="env://") #Asumimos siempre que se ejecutará con `torchrun`
            self.rank = dist.get_rank()
        else:
            self.rank = 0 #Modo no distribuido
        
        device = torch.device(f"cuda:{self.rank % self.n_gpus}")
        torch.cuda.set_device(device)

        self.local_circuit = LocalCircuit(device)
        self.add_child(self.local_circuit)

        # Logger
        SimulatorEngine.logger = setup_logger(self.rank)

        # Reproducibilidad
        self.set_random_seeds(42 + self.rank)

        # Creamos el modelo personalizado del usuario
        log("#################################################################")
        log("Neurobridge initialized. Building user network...")
        self.build_user_network(self.rank, self.world_size)
        log("User network built successfully.")
    
    
    def set_random_seeds(self, seed):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
    

    def initialize(self):
        # Make sure the bridge is the last Node updated in each simulation step.
        # This will be useful to allow an async `all_gather`.
        if self.local_circuit.bridge:
            self.local_circuit.add_child(self.local_circuit.bridge)
        self._call_ready()
        
        # Warming up the CUDA graph
        self.local_circuit.graph_root._call_ready()
        self.local_circuit.graph_root._call_process()
        self.local_circuit.t += 1
        self.local_circuit.graph_root._call_process()
        self.local_circuit.t += 1

        # Capturing the graph
        with torch.cuda.graph(self.local_circuit.graph, stream=self.local_circuit.graph_stream):
            self.local_circuit.graph_root._call_process()
        self.local_circuit.t += 1
    

    def step(self):
        
        #with torch.cuda.stream(self.local_circuit.graph_stream):
            #self.local_circuit.graph.replay()
        #torch.cuda.current_stream().wait_stream(self.local_circuit.graph_stream)

        self.local_circuit.graph.replay()
        self._call_process()
        self.local_circuit.t += 1

    

    def add_default_bridge(self, n_local_neurons: int, n_steps: int):
        bridge = BridgeNeuronGroup(
            device = self.local_circuit.device,
            rank = self.rank,
            world_size = self.world_size,
            n_local_neurons = n_local_neurons,
            n_bridge_steps = n_steps,
            spatial_dimensions=2,
            delay_max=n_steps+1)
        self.local_circuit.bridge = bridge
        self.local_circuit.add_child(bridge)

    
    def build_user_network(self, rank: int, world_size: int) -> None:
        """
        User must overwrite this method to build a custom network.
        """
        raise NotImplementedError("`build_user_network` in `SimulatorEngine` must be implemented.")


def log(msg: str) -> None:
    if SimulatorEngine.logger:
        SimulatorEngine.logger.info(msg)
    else:
        print(msg)


def log_error(msg: str) -> None:
    if SimulatorEngine.logger:
        SimulatorEngine.logger.error(msg)
    else:
        print(f"ERROR: {msg}")


def can_display_graphics():
    # Lista de backends que podrían ser interactivos
    interactive_backends = [backend.lower() for backend in [
        'GTK3Agg', 'GTK3Cairo', 'MacOSX', 'nbAgg', 'Qt4Agg',
        'Qt5Agg', 'QtAgg', 'TkAgg', 'TkCairo', 'WebAgg', 'WX', 'WXAgg'
    ]]
    backend = matplotlib.get_backend()

    # En Unix, requiere DISPLAY; en Windows/Mac suele funcionar siempre
    has_display = (
        sys.platform.startswith('win') or
        sys.platform == 'darwin' or
        os.environ.get("DISPLAY") is not None
    )

    return backend.lower() in interactive_backends and has_display


def show_or_save_plot(filename="output.png", log=None):
    if can_display_graphics():
        plt.show()
    else:
        plt.savefig(filename)
        if log:
            log(f"Gráfico guardado como '{filename}'")
        else:
            print(f"Gráfico guardado como '{filename}'")