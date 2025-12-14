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
        pre: Union["GPUNode", "SpatialGroup"],
        pos: Union["GPUNode", "SpatialGroup"],
        connection_type: Type["GPUNode"],
        pattern: str = "all-to-all",
        weight: Optional[Union[Callable, torch.Tensor, float, tuple]] = 1.0,
        delay: Optional[Union[Callable, torch.Tensor, int, tuple]] = 0,
        **kwargs,
    ) -> Optional["GPUNode"]:
        """
        Creates a synaptic connection between two groups based on a connectivity pattern.
        
        This method respects the current 'filter' state of the pre- and post-synaptic 
        groups. Only neurons where group.filter is True will be considered for connection.
        Indices generated by patterns are mapped back to the global indices of the groups.

        Parameters
        ----------
        pre : GPUNode or SpatialGroup
            Source neuron group.
        pos : GPUNode or SpatialGroup
            Target neuron group.
        connection_type : Type[GPUNode]
            The connection class to instantiate (e.g. StaticDense).
        pattern : str, optional
            Connectivity pattern. Options: 'all-to-all', 'one-to-one', 'random', 'distance'.
            Default is 'all-to-all'.
        weight : float, tensor, tuple, or callable, optional
            Synaptic weight specification. Can be a scalar, a specific tensor, a 
            distribution tuple (min, max), or a function receiving connection indices.
        delay : int, tensor, tuple, or callable, optional
            Synaptic delay specification. Same flexibility as weight.
        **kwargs : dict
            Pattern-specific arguments (e.g., 'p', 'fanin' for random, 'sigma' for distance).

        Returns
        -------
        GPUNode or None
            The instantiated connection object, or None if the filtered groups are empty.

        Raises
        ------
        ValueError
            If pattern requirements (like shape mismatches) are not met.
        """
        device = pre.device
        
        # =========================================================================
        # 1. Effective Indices (Filter Handling)
        # =========================================================================
        # Retrieve the global indices of neurons that are currently active/selected.
        # We use as_tuple=True to get a 1D tensor of indices.
        valid_pre_idx = pre.filter.nonzero(as_tuple=True)[0]
        valid_pos_idx = pos.filter.nonzero(as_tuple=True)[0]

        n_pre_eff = len(valid_pre_idx)
        n_pos_eff = len(valid_pos_idx)

        if n_pre_eff == 0 or n_pos_eff == 0:
            # No connections possible if one of the subgroups is empty.
            return None

        # =========================================================================
        # 2. Pattern Generation (Local/Effective Space)
        # =========================================================================
        # Generated indices (src_local, tgt_local) are relative to the filtered subset.
        # Range: 0 to n_pre_eff-1, 0 to n_pos_eff-1.
        
        src_local, tgt_local = None, None

        if pattern == "all-to-all":
            # Generate Cartesian product of all effective indices
            src_local, tgt_local = torch.meshgrid(
                torch.arange(n_pre_eff, device=device),
                torch.arange(n_pos_eff, device=device),
                indexing="ij",
            )
            src_local = src_local.reshape(-1)
            tgt_local = tgt_local.reshape(-1)

        elif pattern == "one-to-one":
            if n_pre_eff != n_pos_eff:
                raise ValueError(
                    f"Pattern 'one-to-one' requires equal filtered sizes. "
                    f"Pre: {n_pre_eff}, Pos: {n_pos_eff}"
                )
            src_local = torch.arange(n_pre_eff, device=device)
            tgt_local = torch.arange(n_pos_eff, device=device)

        elif pattern == "random":
            p = kwargs.get("p", None)
            fanin = kwargs.get("fanin", None)
            fanout = kwargs.get("fanout", None)

            if p is not None:
                # Probability-based connection
                mask = (torch.rand((n_pre_eff, n_pos_eff), device=device) < p)
                src_local, tgt_local = mask.nonzero(as_tuple=True)

            elif fanin is not None:
                # Fixed number of pre-synaptic partners per post-synaptic neuron
                if fanin > n_pre_eff:
                    raise ValueError(f"fanin ({fanin}) cannot exceed available source neurons ({n_pre_eff})")
                
                # Use topk for efficient selection without sorting entire array
                rand_matrix = torch.rand((n_pre_eff, n_pos_eff), device=device)
                indices = torch.topk(rand_matrix, k=fanin, dim=0).indices  # Shape [fanin, n_pos_eff]
                
                src_local = indices.reshape(-1)
                tgt_local = torch.arange(n_pos_eff, device=device).repeat_interleave(fanin)

            elif fanout is not None:
                # Fixed number of post-synaptic partners per pre-synaptic neuron
                if fanout > n_pos_eff:
                    raise ValueError(f"fanout ({fanout}) cannot exceed available target neurons ({n_pos_eff})")
                
                rand_matrix = torch.rand((n_pre_eff, n_pos_eff), device=device)
                indices = torch.topk(rand_matrix, k=fanout, dim=1).indices  # Shape [n_pre_eff, fanout]
                
                src_local = torch.arange(n_pre_eff, device=device).repeat_interleave(fanout)
                tgt_local = indices.reshape(-1)
                
            else:
                raise ValueError("Random pattern requires 'p', 'fanin', or 'fanout'.")

        elif pattern == "distance":
            if not (hasattr(pre, "positions") and hasattr(pos, "positions")):
                raise RuntimeError("Pattern 'distance' requires 'positions' attribute on both groups.")
            
            # Pass only the filtered positions to the distance algorithm.
            # The algorithm returns indices relative to the tensors passed.
            src_local, tgt_local = block_distance_connect(
                pre.positions[valid_pre_idx], 
                pos.positions[valid_pos_idx],
                block=kwargs.get("block_size", 1024),
                sigma=kwargs.get("sigma", None),
                p_max=kwargs.get("p_max", 1.0),
                max_distance=kwargs.get("max_distance", None),
                fanin=kwargs.get("fanin", None),
                fanout=kwargs.get("fanout", None),
                prob_func=kwargs.get("prob_func", None),
            )

        else:
            raise ValueError(f"Unknown connection pattern: '{pattern}'")

        # =========================================================================
        # 3. Global Index Mapping
        # =========================================================================
        # Convert local relative indices back to global indices of the original groups.
        
        src_global = valid_pre_idx[src_local]
        tgt_global = valid_pos_idx[tgt_local]

        # =========================================================================
        # 4. Constraints (Autapses)
        # =========================================================================
        # Remove self-connections if requested and if groups are the same object.
        
        if pre is pos and not kwargs.get("autapses", True):
            non_self_mask = (src_global != tgt_global)
            src_global = src_global[non_self_mask]
            tgt_global = tgt_global[non_self_mask]

        if src_global.numel() == 0:
            return None

        # =========================================================================
        # 5. Parameter Resolution
        # =========================================================================
        # Resolve weights and delays using the global indices. This allows lambda 
        # functions to access the correct spatial properties or IDs.

        weight_tensor = resolve_param(
            weight, 
            src_idx=src_global, 
            tgt_idx=tgt_global, 
            src=pre, 
            tgt=pos, 
            default_val=kwargs.get('default_weight', 0.0), 
            dtype=torch.float32
        )
        
        delay_tensor = resolve_param(
            delay, 
            src_idx=src_global, 
            tgt_idx=tgt_global, 
            src=pre, 
            tgt=pos, 
            default_val=kwargs.get('default_delay', 0), 
            dtype=torch.long
        )

        # =========================================================================
        # 6. Connection Instantiation
        # =========================================================================
        
        conn = self.connect_edges(
            pre=pre,
            pos=pos,
            connection_type=connection_type,
            src_idx=src_global,
            tgt_idx=tgt_global,
            weight=weight_tensor,
            delay=delay_tensor,
            **kwargs,
        )

        return conn