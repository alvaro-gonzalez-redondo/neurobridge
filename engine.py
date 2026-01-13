from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .bridge import BridgeNeuronGroup
from typing import Optional, Type, Union, Callable

import os
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
import numpy as np
import random

from . import globals
from .core import Node, GPUNode, ParentStack
from .group import SpatialGroup
from .dense import Dense
from .utils import _setup_logger, log, is_distributed, can_use_torch_compile, to_tensor, block_distance_connect, resolve_param
from .connection import ConnectionSpec


class CUDAGraphSubTree(GPUNode):
    """A subtree of nodes that will be captured in a CUDA graph.

    This is a specialized node for organizing computation that will be
    optimized using CUDA graphs, which can significantly improve performance
    by pre-recording GPU operations.
    """

    pass


class LocalCircuit(GPUNode):
    """Container for all neural components on a single GPU.

    The LocalCircuit represents the complete neural circuit running on one GPU.
    It organizes components, manages time, and handles CUDA graph optimization.

    Attributes
    ----------
    t : torch.Tensor
        Current simulation time step.
    graph_root : CUDAGraphSubTree
        Root node for components that will be captured in the CUDA graph.
    graph : torch.cuda.CUDAGraph
        CUDA graph for optimized execution.
    graph_stream : torch.cuda.Stream
        CUDA stream for graph execution.
    bridge : Optional[BridgeNeuronGroup]
        Bridge for inter-GPU communication, if any.
    """

    current_step: int
    
    rank: int

    graph_root: CUDAGraphSubTree
    graph: torch.cuda.CUDAGraph
    graph_stream: torch.cuda.Stream

    bridge: Optional[BridgeNeuronGroup]

    def __init__(self, device, rank: int):
        """Initialize a local circuit on the specified device.

        Parameters
        ----------
        device : str or torch.device
            The GPU device for this circuit.
        rank : int
            The rank of the GPU for this circuit.
        """
        super().__init__(device)
        self.current_step = 0
        self.rank = rank
        self.graph_root = CUDAGraphSubTree(device=self.device)
        self.graph = torch.cuda.CUDAGraph()
        self.graph_stream = torch.cuda.Stream()
        self.bridge = None

    def get_statistics(self):
        """Initialize a local circuit on the specified device.

        Parameters
        ----------
        device : str or torch.device
            The GPU device for this circuit.
        """
        stats = {}
        for child in self.children:
            if hasattr(child, "get_activity_stats"):
                stats[child.name] = child.get_activity_stats()
        return stats

    def save_state(self, path):
        """Save the complete state of the circuit to a file.

        Parameters
        ----------
        path : str
            Path where the state should be saved.

        Raises
        ------
        NotImplementedError
            This feature is not yet implemented.
        """
        raise NotImplementedError("Not implemented yet.")


class Simulator(Node):
    """Main engine for running neural simulations.

    The SimulatorEngine is the central controller for the simulation,
    managing timing, GPU allocation, initialization, and stepping.
    It also provides utilities for building and organizing neural networks.

    Attributes
    ----------
    engine : Optional[SimulatorEngine]
        Class variable referencing the singleton instance.
    logger : ClassVar[Optional[logging.Logger]]
        Class variable for logging.
    t : int
        Current global simulation time step.
    n_gpus : int
        Number of available GPUs.
    world_size : int
        Number of processes in the distributed setup.
    local_circuit : LocalCircuit
        The neural circuit for this GPU.
    """

    #t: int
    dt: float
    n_gpus: int
    world_size: int
    local_circuit: LocalCircuit

    def close(self):
        """Finish the simulation.

        Cleans up resources.
        If running in distributed mode, destroys the process group.
        """
        if is_distributed():
            dist.destroy_process_group()

    def autoparent(self, mode: str = "normal") -> ParentStack:
        """Create a ParentStack context with the appropriate parent node.

        Parameters
        ----------
        mode : str, optional
            The type of parent node to use, by default "normal".
            Options:
                - "normal": Use the local circuit as parent.
                - "graph": Use the CUDA graph root as parent.

        Returns
        -------
        ParentStack
            A context manager that sets the appropriate parent node.

        Raises
        ------
        RuntimeError
            If an invalid mode is specified.

        Examples
        --------
        >>> # Add components to the normal circuit
        >>> with engine.autoparent("normal"):
        ...     monitor = SpikeMonitor([neurons])
        >>>
        >>> # Add components to the CUDA graph for optimization
        >>> with engine.autoparent("graph"):
        ...     neurons = IFNeuronGroup(device, 100)
        """
        if mode == "graph":
            target = self.local_circuit.graph_root
        elif mode == "normal":
            target = self.local_circuit
        else:
            raise RuntimeError(f"Invalid autoparent mode: {mode}.")
        return ParentStack(target)

    def __init__(self, seed=42):
        """Initialize the simulator engine.

        Sets up the simulation environment, including GPU detection,
        distributed processing configuration, and local circuit creation.
        Creates a logger and sets random seeds for reproducibility.

        Raises
        ------
        RuntimeError
            If CUDA is not available or if there are insufficient GPUs.
        """
        super().__init__()

        globals.simulator = self

        self.dt = 1e-3 #This value is a design decision.

        # Safety checks: Available GPUs
        if not torch.cuda.is_available():  ## Do we have CUDA?
            raise RuntimeError("CUDA is not available.")

        self.n_gpus = torch.cuda.device_count()  ## Number of GPUs?
        if "WORLD_SIZE" in os.environ:
            self.world_size = int(os.environ["WORLD_SIZE"])
            if self.world_size > self.n_gpus:
                raise RuntimeError(
                    f"It is required to have {self.world_size} GPUs but there are only {self.n_gpus} available."
                )
        else:
            self.world_size = 1

        # Initializing local circuit
        if "RANK" in os.environ:
            dist.init_process_group(
                backend="nccl"
            )  # , init_method="env://") #We assume user will always use `torchrun`
            rank = dist.get_rank()
        else:
            rank = 0  # Non-distributed mode

        device = torch.device(f"cuda:{rank % self.n_gpus}")
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True

        self.local_circuit = LocalCircuit(device=device, rank=rank)
        self.add_child(self.local_circuit)

        # Logger
        globals.logger = _setup_logger(rank)

        # Reproducibility
        self.seed = seed
        self.set_random_seeds(seed + rank)
        
    def set_random_seeds(self, seed):
        """Set random seeds for reproducibility.

        Sets seeds for PyTorch, Python's random module, and NumPy.

        Parameters
        ----------
        seed : int
            Base seed value. For distributed simulations, the rank is added
            to ensure different but deterministic sequences per GPU.
        """
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    def initialize(self):
        """Initialize the simulation.

        This method must be called before starting the simulation.
        It finalizes the network structure, calls _ready() methods,
        and captures CUDA graphs for optimized execution.
        """
        if self.local_circuit.bridge:
            self.local_circuit.add_child(self.local_circuit.bridge)
        self._call_ready()

        # Compilar la parte optimizada antes de calentar y capturar
        if can_use_torch_compile():
            self.local_circuit.graph_root._call_process = torch.compile(
                self.local_circuit.graph_root._call_process,
                mode="reduce-overhead"  # Opcionalmente "max-autotune" si quieres máxima aceleración
            )
        else:
            print("torch.compile no compatible con esta GPU o falta triton; ejecutando sin compilar.")

        # Warming up the CUDA graph
        self.local_circuit.graph_root._call_ready()
        self.local_circuit.graph_root._call_process()
        self.local_circuit.current_step += 1
        self.local_circuit.graph_root._call_process()
        self.local_circuit.current_step += 1

        # Capturing the graph
        with torch.cuda.graph(self.local_circuit.graph, stream=self.local_circuit.graph_stream):
            self.local_circuit.graph_root._call_process()
        self.local_circuit.current_step = 0

    def step(self):
        """Advance the simulation by one time step.

        Executes one simulation step by replaying the CUDA graph and
        calling _process() on all nodes that are not part of the graph.
        Increments the local circuit's time counter.
        """
        self.local_circuit.graph.replay()
        self._call_process()
        self.local_circuit.current_step += 1

    def connect_edges(
        self,
        pre: GPUNode, pos: GPUNode,
        connection_type: Type[GPUNode],
        src_idx: torch.LongTensor, tgt_idx: torch.LongTensor,
        weight: Optional[Union[torch.Tensor, float]] = None,
        delay: Optional[Union[torch.Tensor, int]] = None,
        **kwargs,
    ) -> GPUNode:
        assert isinstance(pre, GPUNode), f"pre must be GPUNode, got {type(pre)}"
        assert isinstance(pos, GPUNode), f"pos must be GPUNode, got {type(pos)}"

        # normalize tensors
        src_idx = to_tensor(src_idx, dtype=torch.long)
        tgt_idx = to_tensor(tgt_idx, dtype=torch.long)
        weight  = to_tensor(weight, dtype=torch.float32, device=pre.device)
        delay   = to_tensor(delay, dtype=torch.long, device=pre.device)

        spec = ConnectionSpec(
            pre=pre, pos=pos,
            src_idx=src_idx, tgt_idx=tgt_idx,
            weight=weight, delay=delay,
            connection_type=connection_type,
            params=kwargs,
        )

        if connection_type is None:
            raise ValueError("Must provide connection_type")

        # Instanciar la conexión final
        conn = connection_type(spec)
        self.local_circuit.add_child(conn)
        return conn
    

    def connect(
        self,
        pre: Union[GPUNode, SpatialGroup],
        pos: Union[GPUNode, SpatialGroup],
        connection_type: Type[GPUNode],
        pattern: Union[str, Dense, Group, GPUNode] = "all-to-all",
        weight: Optional[Union[Callable, torch.Tensor, float, tuple]] = 1.0,
        delay: Optional[Union[Callable, torch.Tensor, int, tuple]] = 0,
        **kwargs,
    ) -> Optional[GPUNode]:
        """
        Establece conexiones sinápticas entre subconjuntos activos de poblaciones neuronales.

        Calcula los índices de conectividad basándose en un patrón generativo (string)
        o copiando la topología de una conexión existente (objeto) y sus retardos.

        Parameters
        ----------
        pre : GPUNode
            Grupo presináptico (fuente).
        pos : GPUNode
            Grupo postsináptico (objetivo).
        connection_type : Type[GPUNode]
            Clase para instanciar la conexión (e.g., StaticDense, StaticSparse).
        pattern : str | Dense | Group
            - ``str``: Genera nueva topología ('all-to-all', 'random', 'distance').
            - ``Dense``: Copia la máscara 2D (mask & filter).
            - ``StaticSparse/Group``: Copia los índices explícitos activos (idx_pre/pos & filter).
        weight, delay : ...
            Parámetros sinápticos.

        Returns
        -------
        GPUNode | None
            Objeto de conexión o None si no hay sinapsis resultantes.
        """
        device: torch.device = pre.device

        # =========================================================================
        # 1. Determinación del Subespacio Activo
        # =========================================================================
        valid_pre_idx: torch.Tensor = pre.filter.nonzero(as_tuple=True)[0]
        valid_pos_idx: torch.Tensor = pos.filter.nonzero(as_tuple=True)[0]

        n_pre_eff: int = len(valid_pre_idx)
        n_pos_eff: int = len(valid_pos_idx)

        if n_pre_eff == 0 or n_pos_eff == 0:
            return None

        src_global: Optional[torch.Tensor] = None
        tgt_global: Optional[torch.Tensor] = None
        
        # Variable para almacenar los delays extraídos del objeto patrón (si existen)
        inherited_delay: Optional[torch.Tensor] = None

        # =========================================================================
        # 2. Resolución de Topología (y extracción de propiedades)
        # =========================================================================

        # CASO A: Copia de Topología (Objeto existente)
        if not isinstance(pattern, str):
            
            s_glob_raw: torch.Tensor
            t_glob_raw: torch.Tensor
            raw_delay_vals: Optional[torch.Tensor] = None

            # --- Subcaso A.1: Conexión Densa (Matriz) ---
            if hasattr(pattern, "mask") and hasattr(pattern, "filter") and pattern.mask.ndim == 2:
                if pattern.shape != (pre.size, pos.size):
                    raise ValueError(f"Topology mismatch: Pattern {pattern.shape} != Groups ({pre.size}, {pos.size})")
                
                # Topología efectiva
                effective_topology = pattern.mask & pattern.filter
                s_glob_raw, t_glob_raw = effective_topology.nonzero(as_tuple=True)

                # Intentar extraer delays si existen en formato denso
                if hasattr(pattern, "delay") and isinstance(pattern.delay, torch.Tensor):
                    if pattern.delay.shape == pattern.shape:
                        # Extraemos los valores correspondientes a las conexiones activas
                        raw_delay_vals = pattern.delay[s_glob_raw, t_glob_raw]

            # --- Subcaso A.2: Conexión Dispersa (Lista de aristas / Group) ---
            elif hasattr(pattern, "idx_pre") and hasattr(pattern, "idx_pos") and hasattr(pattern, "filter"):
                # Filtro 1D sobre aristas
                active_indices = pattern.filter.nonzero(as_tuple=True)[0]
                
                s_glob_raw = pattern.idx_pre[active_indices]
                t_glob_raw = pattern.idx_pos[active_indices]

                # Intentar extraer delays si existen en formato disperso (1D)
                if hasattr(pattern, "delay") and isinstance(pattern.delay, torch.Tensor):
                    if pattern.delay.ndim == 1 and pattern.delay.shape[0] == pattern.idx_pre.shape[0]:
                        raw_delay_vals = pattern.delay[active_indices]

            else:
                raise TypeError(
                    f"Pattern object {type(pattern)} unsupported. Must have 'mask' (Dense) or 'idx_pre' (Sparse)."
                )

            # --- Filtrado de Intersección (Source active AND Target active) ---
            valid_mask: torch.Tensor = torch.isin(s_glob_raw, valid_pre_idx) & \
                                       torch.isin(t_glob_raw, valid_pos_idx)
            
            src_global = s_glob_raw[valid_mask]
            tgt_global = t_glob_raw[valid_mask]

            # Si extrajimos delays crudos, aplicamos la misma máscara de validación
            if raw_delay_vals is not None:
                inherited_delay = raw_delay_vals[valid_mask]

        # CASO B: Generación Nueva
        else:
            src_local: torch.Tensor
            tgt_local: torch.Tensor

            if pattern == "all-to-all":
                grid_src, grid_tgt = torch.meshgrid(
                    torch.arange(n_pre_eff, device=device),
                    torch.arange(n_pos_eff, device=device),
                    indexing="ij",
                )
                src_local, tgt_local = grid_src.reshape(-1), grid_tgt.reshape(-1)

            elif pattern == "one-to-one":
                if n_pre_eff != n_pos_eff:
                    raise ValueError(f"Size mismatch 'one-to-one': {n_pre_eff} != {n_pos_eff}")
                src_local = torch.arange(n_pre_eff, device=device)
                tgt_local = torch.arange(n_pos_eff, device=device)

            elif pattern == "random":
                p = kwargs.get("p")
                fanin = kwargs.get("fanin")
                fanout = kwargs.get("fanout")

                if p is not None:
                    mask = (torch.rand((n_pre_eff, n_pos_eff), device=device) < p)
                    src_local, tgt_local = mask.nonzero(as_tuple=True)
                elif fanin is not None and fanout is not None:
                    if kwargs.get("allow_multiedges", True):
                        src_local, tgt_local = _stub_matching(n_pre_eff, n_pos_eff, fanout, fanin, device)
                    else:
                        src_local, tgt_local = _stub_matching_with_rejection(n_pre_eff, n_pos_eff, fanout, fanin, device)
                elif fanin is not None:
                    rand = torch.rand((n_pre_eff, n_pos_eff), device=device)
                    idx = torch.topk(rand, k=fanin, dim=0).indices
                    src_local, tgt_local = idx.reshape(-1), torch.arange(n_pos_eff, device=device).repeat_interleave(fanin)
                elif fanout is not None:
                    rand = torch.rand((n_pre_eff, n_pos_eff), device=device)
                    idx = torch.topk(rand, k=fanout, dim=1).indices
                    src_local, tgt_local = torch.arange(n_pre_eff, device=device).repeat_interleave(fanout), idx.reshape(-1)
                else:
                    raise ValueError("Random requires p, fanin, or fanout")

            elif pattern == "distance":
                 # (Asumiendo implementación existente de block_distance_connect)
                 src_local, tgt_local = block_distance_connect(
                    pre.positions[valid_pre_idx], pos.positions[valid_pos_idx],
                    **kwargs
                )
            else:
                raise ValueError(f"Unknown pattern: {pattern}")

            # Mapeo Local -> Global
            src_global = valid_pre_idx[src_local]
            tgt_global = valid_pos_idx[tgt_local]

        # =========================================================================
        # 3. Finalización y Parámetros
        # =========================================================================
        
        if src_global.numel() == 0:
            return None

        # Autapses
        if pre is pos and not kwargs.get("autapses", True):
            mask_no_self = (src_global != tgt_global)
            src_global = src_global[mask_no_self]
            tgt_global = tgt_global[mask_no_self]
            if inherited_delay is not None:
                inherited_delay = inherited_delay[mask_no_self]

            if src_global.numel() == 0:
                return None

        # --- Lógica de Prioridad de Delays ---
        # Si hemos heredado delays del patrón, estos tienen prioridad sobre el valor por defecto (0),
        # pero asumimos que si el usuario pasa explícitamente un valor distinto en 'delay',
        # podría querer sobrescribirlo. Aquí priorizamos la herencia si 'delay' es el default.
        
        final_delay_arg = delay
        if inherited_delay is not None:
            # Usamos el delay heredado. resolve_param debería ser capaz de aceptar
            # un tensor pre-calculado que ya coincide con el tamaño de src_global.
            final_delay_arg = inherited_delay

        # Resolución
        w_tens = resolve_param(weight, src_idx=src_global, tgt_idx=tgt_global, src=pre, tgt=pos, 
                               default_val=kwargs.get('default_weight', 0.0), dtype=torch.float32)
        
        d_tens = resolve_param(final_delay_arg, src_idx=src_global, tgt_idx=tgt_global, src=pre, tgt=pos, 
                               default_val=kwargs.get('default_delay', 0), dtype=torch.long)

        return self.connect_edges(
            pre=pre, pos=pos, connection_type=connection_type,
            src_idx=src_global, tgt_idx=tgt_global,
            weight=w_tens, delay=d_tens, **kwargs
        )


def _stub_matching(n_pre, n_post, fanout, fanin, device):
    src = torch.arange(n_pre, device=device).repeat_interleave(fanout)
    tgt = torch.arange(n_post, device=device).repeat_interleave(fanin)
    perm = torch.randperm(tgt.numel(), device=device)
    return src, tgt[perm]


def _stub_matching_with_rejection(
    n_pre,
    n_post,
    fanout,
    fanin,
    device,
    max_retries=10,
):
    num_edges = n_pre * fanout

    src_base = torch.arange(n_pre, device=device).repeat_interleave(fanout)
    tgt_base = torch.arange(n_post, device=device).repeat_interleave(fanin)

    for attempt in range(max_retries):
        perm = torch.randperm(num_edges, device=device)
        src = src_base
        tgt = tgt_base[perm]

        # Encode pairs as unique integers: src * n_post + tgt
        pairs = src * n_post + tgt

        # Check uniqueness
        if torch.unique(pairs).numel() == num_edges:
            return src, tgt

    raise RuntimeError(
        f"Failed to generate fanin/fanout graph without multiedges "
        f"after {max_retries} attempts. "
        f"Try allow_multiedges=True or relax degrees."
    )
