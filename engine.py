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
from .utils import _setup_logger, log, is_distributed, can_use_torch_compile, to_tensor, block_distance_connect, resolve_param
from .sparse_connections import StaticSparse
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

    current_step: torch.Tensor
    
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
        self.current_step = torch.zeros(1, dtype=torch.long, device=self.device)
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
        self.local_circuit.current_step.zero_()

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
        pre: Union[GPUNode, SpatialGroup], pos: Union[GPUNode, SpatialGroup],
        connection_type: Type[GPUNode],
        pattern: str = "all_to_all",
        weight: Optional[Union[Callable, torch.Tensor, float]] = 1.0,
        delay: Optional[Union[Callable, torch.Tensor, int]] = 0,
        **kwargs,
    ) -> GPUNode:
        """
        High-level connect method using patterns.

        Parameters
        ----------
        pre : GPUNode
            Source group.
        pos : GPUNode
            Target group.
        connection_type : Type[GPUNode]
            The connection class to instantiate (e.g. SparseStaticConnection).
        pattern : str
            Pattern type: 'one_to_one', 'all_to_all', 'random'.
        weight : scalar or tensor
            Synaptic weight(s).
        delay : scalar or tensor
            Synaptic delay(s).
        kwargs : dict
            Pattern-specific arguments.
        """
        Npre, Npos = pre.size, pos.size
        device = pre.device

        if kwargs is None:
            kwargs = {}

        # --- Pattern handling ---
        if pattern == "distance":
            src_idx, tgt_idx = block_distance_connect(
                pre.positions, pos.positions,
                block=kwargs.get("block_size", 1024),
                sigma=kwargs.get("sigma", None),
                p_max=kwargs.get("p_max", 1.0),
                max_distance=kwargs.get("max_distance", None),
                fanin=kwargs.get("fanin", None),
                fanout=kwargs.get("fanout", None),
                prob_func=kwargs.get("prob_func", None),
            )
        
        elif pattern == "one-to-one":
            if Npre != Npos:
                raise ValueError("one_to_one requires pre.size == pos.size")
            src_idx = torch.arange(Npre, device=device, dtype=torch.long)
            tgt_idx = torch.arange(Npos, device=device, dtype=torch.long)

        elif pattern == "all-to-all":
            src_idx, tgt_idx = torch.meshgrid(
                torch.arange(Npre, device=device),
                torch.arange(Npos, device=device),
                indexing="ij",
            )
            src_idx = src_idx.reshape(-1)
            tgt_idx = tgt_idx.reshape(-1)

        elif pattern == "random":
            p = kwargs.get("p", None)
            fanin = kwargs.get("fanin", None)
            fanout = kwargs.get("fanout", None)

            if fanin is not None and fanout is not None:
                total_edges = fanin * Npos
                if total_edges != fanout * Npre:
                    raise ValueError(
                        f"Inconsistent fanin/fanout: fanin*Npos={fanin*Npos} != fanout*Npre={fanout*Npre}"
                    )

                # Inicializamos contadores de cuántas conexiones lleva cada pre/post
                pre_remaining  = torch.full((Npre,),  fanout, dtype=torch.int32, device=device)
                post_remaining = torch.full((Npos,), fanin,  dtype=torch.int32, device=device)

                src_list, tgt_list = [], []

                # Mientras quede demanda de posts
                for j in torch.randperm(Npos, device=device):
                    need = post_remaining[j].item()
                    if need <= 0:
                        continue
                    # Candidatos disponibles en pres con hueco
                    candidates = (pre_remaining > 0).nonzero(as_tuple=True)[0]
                    if len(candidates) < need:
                        raise RuntimeError("No hay suficientes pres disponibles para cumplir restricciones")
                    chosen = candidates[torch.randperm(len(candidates), device=device)[:need]]
                    src_list.append(chosen)
                    tgt_list.append(torch.full((need,), j, device=device))
                    pre_remaining[chosen] -= 1
                    post_remaining[j] = 0

                src_idx = torch.cat(src_list)
                tgt_idx = torch.cat(tgt_list)

            elif p is not None:
                # Caso 1: probabilidad por par
                mask = (torch.rand((Npre, Npos), device=device) < p)
                src_idx, tgt_idx = mask.nonzero(as_tuple=True)

            elif fanin is not None:
                # Caso 2: cada postsináptica recibe exactamente `fanin` conexiones aleatorias
                if fanin > Npre:
                    raise ValueError("fanin cannot exceed number of presynaptic neurons")
                # Para cada columna (target), escoger fanin pres al azar
                rand = torch.rand((Npre, Npos), device=device)
                idx = torch.topk(rand, k=fanin, dim=0).indices  # [fanin, Npos]
                tgt_idx = torch.arange(Npos, device=device).repeat(fanin, 1)
                src_idx = idx
                src_idx, tgt_idx = src_idx.reshape(-1), tgt_idx.reshape(-1)

            elif fanout is not None:
                # Caso 3: cada presináptica conecta a exactamente `fanout` posts al azar
                if fanout > Npos:
                    raise ValueError("fanout cannot exceed number of postsynaptic neurons")
                rand = torch.rand((Npre, Npos), device=device)
                idx = torch.topk(rand, k=fanout, dim=1).indices  # [Npre, fanout]
                src_idx = torch.arange(Npre, device=device).unsqueeze(1).repeat(1, fanout)
                tgt_idx = idx
                src_idx, tgt_idx = src_idx.reshape(-1), tgt_idx.reshape(-1)

        else:
            raise ValueError(f"Unsupported connection pattern: {pattern}")
        
        # --- Resolve params ---

        weight = resolve_param(weight, src_idx=src_idx, tgt_idx=tgt_idx, src=pre, tgt=pos, default_val=kwargs.get('default_weight', 0.0), dtype=torch.float32)
        delay = resolve_param(delay, src_idx=src_idx, tgt_idx=tgt_idx, src=pre, tgt=pos, default_val=kwargs.get('default_delay', 0), dtype=torch.long)


        # --- Delegate to connect_edges ---
        conn = self.connect_edges(
            pre=pre,
            pos=pos,
            connection_type=connection_type,
            src_idx=src_idx,
            tgt_idx=tgt_idx,
            weight=weight,
            delay=delay,
            **kwargs,
        )

        return conn