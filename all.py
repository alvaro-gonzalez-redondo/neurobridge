from __future__ import annotations
import os
import random
import numpy as np
import torch
import torch.distributed as dist
import logging
from typing import List, Optional, Callable, ClassVar, Any, Type, Union


class Node:
    auto_parent: Optional[Node] = None

    children: List[Node]
    parent: Optional[Node]
    t: int

    def __init__(self):
        self.children = []
        self.t = 0
        self.parent = None

        if Node.auto_parent:
            Node.auto_parent.add_child(self)

    def add_child(self, node:Node) -> None:
        # Nodes should be unique in the scene tree
        if node.parent is not None:
            node.parent.remove_child(node)
        
        self.children.append(node)
        node.parent = self
    
    def remove_child(self, node:Node) -> Node:
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
        self.t += 1


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


    def where_id(self, condition: Callable[..., bool]) -> Group:
        """
        Aplica un filtro basado en su índice.

        Args:
            condition: Función que toma un índice y devuelve True/False

        Returns:
            A sí mismo con el parámetro `filter` modificado
        """

        for i in range(self.size):
            self.filter[i] &= condition(i)

        return self

    def reset_filter(self) -> None:
        self.filter.fill_(True)


class SpatialGroup(Group):
    spatial_dimensions: int
    positions: torch.Tensor


    def __init__(self, device:str, size:int, spatial_dimensions:int=2):
        super().__init__(device, size)
        self.spatial_dimensions = spatial_dimensions
        self.positions = torch.randn((self.size, spatial_dimensions), device=self.device)


    def where_pos(self, condition: Callable[..., bool]) -> SpatialGroup:
        """
        Aplica un filtro basado en su posición.

        Args:
            condition: Función que toma coordenadas (float) y devuelve True/False

        Returns:
            A sí mismo con el parámetro `filter` modificado
        """
        if self.positions is None:
            raise RuntimeError("Este grupo no tiene posiciones definidas.")
        for i in range(self.size):
            self.filter[i] &= condition(*self.positions[i].tolist())
        return self
    

class NeuronGroup(SpatialGroup):
    delay_max: int
    spike_buffer: torch.Tensor
    _input_currents: torch.Tensor
    _input_spikes: torch.Tensor


    def __init__(self, device:str, n_neurons:int, spatial_dimensions:int=2, delay_max:int=20):
        super().__init__(device, n_neurons, spatial_dimensions)
        self.delay_max = delay_max
        self.spike_buffer = torch.zeros((n_neurons, delay_max), dtype=torch.bool, device=self.device)
        self._input_currents = torch.zeros(n_neurons, dtype=torch.float32, device=self.device)
        self._input_spikes = torch.zeros(n_neurons, dtype=torch.bool, device=self.device)


    def inject_currents(self, I: torch.Tensor) -> None:
        assert I.shape[0] == self.size
        self._input_currents += I


    def inject_spikes(self, spikes: torch.Tensor) -> None:
        """Forces the neurons to spike, independently of weights."""
        assert spikes.shape[0] == self.size
        self._input_spikes |= spikes.bool()
    

    def get_spikes_at(self, delays: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        assert delays.shape[0] == indices.shape[0]
        assert delays.shape[0] == self.size

        t_indices = (self.t - delays) % self.delay_max
        return self.spike_buffer[indices, t_indices]


    def __rshift__(self, other) -> ConnectionOperator:
        """Implementa el operador >> para crear conexiones."""
        return ConnectionOperator(self, other)
    

# Esta neurona funciona como un simple repetidor
class ParrotGroup(NeuronGroup):

    def _process(self) -> None:
        super()._process()

        # Limpiamos los spikes que hubiera en este instante temporal
        t_idx = self.t % self.delay_max
        self.spike_buffer[:, t_idx] = False

        # Procesar cualquier spike inyectado
        if torch.any(self._input_spikes):
            # Guardar el spike en el búfer para t actual
            self.spike_buffer[:, t_idx] |= self._input_spikes
            # Limpiar spikes inyectados
            self._input_spikes.fill_(False)
            
        # Procesar corrientes de entrada (si superan un umbral, generar spike)
        if torch.any(self._input_currents > 0):  # Umbral simple: cualquier corriente positiva
            # Generar spikes para las neuronas que reciben corriente positiva
            spikes = self._input_currents > 0
            self.spike_buffer[:, t_idx] |= spikes
            # Limpiar corrientes
            self._input_currents.fill_(0.0)


class IFNeuronGroup(NeuronGroup):
    V: torch.Tensor
    threshold: torch.Tensor
    decay: torch.Tensor


    def __init__(self, device: str, n_neurons: int, spatial_dimensions: int = 2, delay_max: int = 20, threshold: float = 1.0, decay: float = 0.95):
        super().__init__(device, n_neurons, spatial_dimensions, delay_max)
        self.V = torch.zeros(n_neurons, dtype=torch.float32, device=self.device)
        self.threshold = torch.Tensor([threshold], dtype=torch.float, device=device)
        self.decay = torch.Tensor([decay], dtype=torch.float, device=device)


    def _process(self):
        super()._process()
        t_idx = self.t % self.delay_max

        # Update potential with decay and input
        self.V *= self.decay
        self.V += self._input_currents
        self._input_currents.fill_(0.0)

        # Determine which neurons spike
        spikes = self.V >= self.threshold
        self.spike_buffer[:, t_idx] = spikes
        self.V[spikes] = 0.0  # Reset membrane potential


class RandomSpikeGenerator(NeuronGroup):
    firing_rate: torch.Tensor  # En Hz

    def __init__(self, device: str, n_neurons: int, firing_rate: float = 10.0, spatial_dimensions: int = 2, delay_max: int = 20):
        super().__init__(device, n_neurons, spatial_dimensions=spatial_dimensions, delay_max=delay_max)
        self.firing_rate = torch.Tensor([firing_rate], dtype=torch.float, device=device)


    def _process(self):
        super()._process()
        t_idx = self.t % self.delay_max

        p = self.firing_rate * 1e-3 #dt=0.001 seconds
        spikes = torch.rand(self.size, device=self.device) < p
        self.spike_buffer[:, t_idx] = spikes


class ConnectionOperator:
    current_circuit: Optional[LocalCircuit] = None

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
                source_indices = self._compute_parameter(kwargs['idx_pre'])
                target_indices = self._compute_parameter(kwargs['idx_pos'])
            except KeyError:
                raise RuntimeError("Faltan 'idx_pre' o 'idx_pos' en los parámetros para el patrón 'specific'.")
        
        # TODO: elif pattern == '...'

        else:
            raise NotImplementedError(f"Patrón de conexión '{pattern}' no implementado.")

        # Crear objeto sináptico
        if synapse_class is None or synapse_class is StaticSynapse:
            try:
                delay = self._compute_parameter(kwargs['delay'], source_indices, target_indices).to(torch.long)
                weight = self._compute_parameter(kwargs['weight'], source_indices, target_indices)
            except KeyError:
                raise RuntimeError("Faltan 'delay' o 'weight' en los parámetros para StaticSynapse.")
            
            connection = StaticSynapse(
                pre=self.pre,
                pos=self.pos,
                idx_pre=source_indices,
                idx_pos=target_indices,
                delay=delay,
                weight=weight
            )

        else:
            raise NotImplementedError(f"Clase de sinapsis '{synapse_class}' no implementada o no especificada.")

        # Limpiar filtros tras conectar
        self.pre.reset_filter()
        self.pos.reset_filter()

        return connection


    def _compute_parameter(
        self,
        param: Any,
        idx_pre: Optional[torch.Tensor] = None,
        idx_post: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calcula un tensor de parámetros por conexión a partir de:
        - Un valor escalar: se replica.
        - Un tensor: se usa directamente o se expande.
        - Una función: se evalúa por conexión usando (idx_pre, idx_post).
        """
        if callable(param):
            if idx_pre is None or idx_post is None:
                raise ValueError("Se requieren 'idx_pre' y 'idx_post' si el parámetro es una función.")
            n = len(idx_pre)
            values = torch.zeros(n, device=self.device)
            for i in range(n):
                values[i] = param(idx_pre[i].item(), idx_post[i].item())
        
        elif isinstance(param, torch.Tensor):
            n = len(idx_pre) if idx_pre is not None else param.shape[0]
            if param.numel() == 1:
                values = torch.full((n,), param.item(), device=self.device)
            else:
                if param.numel() != n:
                    raise ValueError(f"Expected a tensor of length {n}, but got {param.numel()}.")
                values = param.to(device=self.device)
        
        elif isinstance(param, List):
            param = torch.tensor(param, device=self.device)
            values = self._compute_parameter(param)
        
        else:
            n = len(idx_pre) if idx_pre is not None else 1
            values = torch.full((n,), float(param), device=self.device)

        return values


class SynapticGroup(Group):
    pre: NeuronGroup
    pos: NeuronGroup
    idx_pre: torch.Tensor
    idx_pos: torch.Tensor
    delay: torch.Tensor
    _spiked: Optional[torch.Tensor]


    def __init__(self, pre: NeuronGroup, pos: NeuronGroup, idx_pre: torch.Tensor, idx_pos: torch.Tensor, delay: torch.Tensor):
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
        self.delay = delay

        self._spiked = None


    def _process(self):
        super()._process()
        self._spiked = self.get_active_indices()
        self._propagate()
        self._update()


    def get_active_indices(self) -> Optional[torch.Tensor]:
        """
        Devuelve un tensor 1D con los índices de las conexiones activas
        (aquellas cuyo pre-sináptico ha disparado con su delay correspondiente).
        """
        spikes = self.pre.get_spikes_at(self.delay, self.idx_pre)
        active_indices = None
        if torch.any(spikes):
            active_indices = spikes.nonzero(as_tuple=True)[0]
        return active_indices
    

    def _propagate(self) -> None:
        """
        """
        raise NotImplementedError("`propagate` method in `SynapticGroup` must be implemented.")


    def _update(self) -> None:
        """
        Time-driven learning rules are implemented here
        """
        raise NotImplementedError("`update` method in `SynapticGroup` must be implemented.")


class StaticSynapse(SynapticGroup):
    weight: torch.Tensor

    def __init__(self, pre, pos, idx_pre, idx_pos, weight, delay):
        super().__init__(pre, pos, idx_pre, idx_pos, delay)
        self.weight = weight.to(device=pre.device)


    def _propagate(self) -> None:
        if self._spiked is None or self._spiked.numel() == 0:
            return
        
        tgt = self.idx_pos[self._spiked]
        wgt = self.weight[self._spiked]
        net_current = torch.zeros(self.pos.size, dtype=torch.float32, device=self.device)
        net_current.index_add_(0, tgt, wgt)
        self.pos.inject_currents(net_current)


    def _update(self) -> None:
        pass
        

def bool_to_uint8(x: torch.Tensor) -> torch.Tensor:
    # Aplanar el tensor primero
    x_flat = x.flatten().to(torch.uint8)
    pad_len = (8 - x_flat.numel() % 8) % 8
    if pad_len:
        x_flat = torch.cat([x_flat, torch.zeros(pad_len, dtype=torch.uint8, device=x.device)])
    x_flat = x_flat.reshape(-1, 8)
    weights = torch.tensor([1,2,4,8,16,32,64,128], dtype=torch.uint8, device=x.device)
    return (x_flat * weights).sum(dim=1)


def uint8_to_bool(x: torch.Tensor, num_bits: int) -> torch.Tensor:
    # Asegurar que x sea un tensor 1D
    x = x.flatten()
    bits = ((x.unsqueeze(1) >> torch.arange(8, device=x.device)) & 1).to(torch.bool)
    return bits.flatten()[:num_bits]


class BridgeNeuronGroup(NeuronGroup):
    n_local_neurons: int
    rank: int
    n_bridge_steps: int
    write_buffer: torch.Tensor
    _gathered: List[torch.Tensor]
    _time_range: torch.Tensor


    def __init__(self, device: torch.device, rank: int, world_size: int, n_local_neurons: int, n_bridge_steps: int, spatial_dimensions: int=2, delay_max: int=20):
        super().__init__(device, n_local_neurons*world_size, spatial_dimensions=spatial_dimensions, delay_max=delay_max)
        self.n_local_neurons = n_local_neurons
        self.n_bridge_steps = n_bridge_steps
        self.rank = rank
        self.n_bits = n_local_neurons * n_bridge_steps
        self.write_buffer = torch.zeros((n_local_neurons, n_bridge_steps), dtype=torch.bool, device=device)
        # Crear buffer para comunicación
        dummy_packed = bool_to_uint8(torch.zeros(n_local_neurons * n_bridge_steps, dtype=torch.bool, device=device))
        self._gathered = [torch.empty_like(dummy_packed) for _ in range(world_size)]
        self._time_range = torch.arange(self.n_bridge_steps, dtype=torch.long, device=device)
        assert n_bridge_steps<delay_max, "Bridge steps must be lower than the bridge neuron population's max delay."


    # Any positive input to a bridge neuron will automatically generate a spike
    def inject_currents(self, I: torch.Tensor):
        assert I.shape[0] == self.size
        from_id = self.rank * self.n_local_neurons
        to_id = (self.rank+1) * self.n_local_neurons
        subset = I[from_id:to_id]
        self.write_buffer[subset>0, self.t % self.n_bridge_steps] = True


    # Any injected spike to a bridge neuron will automatically generate a spike
    def inject_spikes(self, spikes: torch.Tensor):
        assert spikes.shape[0] == self.size
        from_id = self.rank * self.n_local_neurons
        to_id = (self.rank+1) * self.n_local_neurons
        subset = spikes[from_id:to_id].bool()
        indices = subset.nonzero(as_tuple=True)[0]
        if indices.numel() > 0:
            self.write_buffer[indices, self.t % self.n_bridge_steps] = True


    def _process(self):
        super()._process()
    
        # COMUNICACIÓN
        if self.t % self.n_bridge_steps == 0:
            result = None

            if is_distributed():
                # Aplanar y empaquetar el buffer de escritura para comunicación
                write_buffer_flat = self.write_buffer.flatten()
                packed = bool_to_uint8(write_buffer_flat)
                
                # Sincronizar entre GPUs
                dist.all_gather(self._gathered, packed)
                
                # Desempaquetar y reshapear
                bool_list = []
                for p in self._gathered:
                    # Desempaquetar a booleanos
                    unpacked = uint8_to_bool(p, self.n_bits)
                    # Reshapear a la forma original
                    reshaped = unpacked.reshape(self.n_local_neurons, self.n_bridge_steps)
                    bool_list.append(reshaped)
                
                # Concatenar resultados de todas las GPUs
                result = torch.cat(bool_list, dim=0)
                
                # Limpiar buffer después de comunicar
                self.write_buffer.fill_(False)
                dist.barrier()
            else:
                # Modo no distribuido, usar buffer local
                result = self.write_buffer.clone()
                self.write_buffer.fill_(False)
        
            # AÑADIR SPIKES FUTUROS
            if result is not None:
                time_indices = (self.t + self._time_range) % self.delay_max
                self.spike_buffer.index_copy_(1, time_indices, result)


class LocalCircuit(GPUNode):
    """
    Contains all updatable objects of this GPU.
    Used for structural organization. It might contain future extensions for analyses and serialization/deserialization.
    """
    
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


import logging

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
    logger: ClassVar[Optional[logging.Logger]] = None

    n_gpus: int
    world_size: int
    rank: int
    local_circuit: LocalCircuit
    bridge: Optional[BridgeNeuronGroup]


    def __init__(self, autoparenting_nodes=False):
        super().__init__()
        
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

        ConnectionOperator.current_circuit = self.local_circuit
        if autoparenting_nodes:
            Node.auto_parent = self.local_circuit

        self.bridge = None

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
        #if self.bridge:
            #self.local_circuit.add_child(self.bridge)
        self._call_ready()
    

    def step(self):
        self._call_process()
    

    def add_default_bridge(self, n_local_neurons: int, n_steps: int):
        bridge = BridgeNeuronGroup(
            device = self.local_circuit.device,
            rank = self.rank,
            world_size = self.world_size,
            n_local_neurons = n_local_neurons,
            n_bridge_steps = n_steps,
            spatial_dimensions=2,
            delay_max=n_steps+1)
        self.local_circuit.add_child(bridge)
        return bridge

    
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


class PingPongRingEngine(SimulatorEngine):

    def __init__(self):
        super().__init__(autoparenting_nodes=True)

    def build_user_network(self, rank: int, world_size: int):
        n_neurons = 10

        # Crear un puente neuronal (permite la comunicación entre GPUs)
        bridge = self.add_default_bridge(n_local_neurons=n_neurons, n_steps=20)

        # Crear un grupo neuronal local
        local_neurons = ParrotGroup(self.local_circuit.device, n_neurons)

        # Envía a la siguiente GPU (o a sí misma si está sola)
        tgt = (torch.arange(n_neurons) + rank*n_neurons) % (n_neurons*world_size)
        (local_neurons >> bridge)(
            pattern = 'specific',
            idx_pre = torch.arange(n_neurons),
            idx_pos = tgt,
            delay   = 0,
            weight  = 1.0,
        )

        # Recibe de la GPU anterior (o de sí misma si está sola)
        src = (torch.arange(n_neurons) + (rank-1)*n_neurons) % (n_neurons*world_size)
        (bridge >> local_neurons)(
            pattern = 'specific',
            idx_pre = src,
            idx_pos = torch.arange(n_neurons),
            delay   = 0,
            weight  = 1.0,
        )
                      
        # Registramos las neuronas para poder monitorearlas
        self.local_neurons = local_neurons
        self.bridge = bridge


# Monitorear actividad
def monitor(engine:PingPongRingEngine):
    step = engine.t
    t = engine.local_neurons.t % engine.local_neurons.delay_max
    spikes = ["|" if n else "_" for n in engine.local_neurons.spike_buffer[:, t]]
    spikes = ''.join(spikes)
    log(f"t={step:<6d} {spikes}")


def feed_input(engine:PingPongRingEngine):
    # En la primera neurona (rank 0), inyectar un spike inicial para comenzar la actividad
    if engine.rank == 0:
        if engine.t < engine.local_neurons.size+1:
            initial_spikes = torch.zeros(engine.local_neurons.size, dtype=torch.bool, device=engine.local_circuit.device)
            initial_spikes[engine.t-1] = True
            engine.local_neurons.inject_spikes(initial_spikes)



def main():
    try:
        engine = PingPongRingEngine()
        engine.initialize()

        # Ejecutar por más tiempo para asegurar que vemos varios ciclos
        for _ in range(150):
            engine.step()
            feed_input(engine)
            monitor(engine)

    except Exception as e:
        log_error(f"ERROR: {e}")
        import traceback
        log_error(traceback.format_exc())

    finally:
        if is_distributed():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()