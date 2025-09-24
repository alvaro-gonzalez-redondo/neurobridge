from __future__ import annotations

from . import globals
from .core import Node, GPUNode, ParentStack
from .utils import _setup_logger, log, is_distributed, can_use_torch_compile
from .group_connections import StaticConnection

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .bridge import BridgeNeuronGroup

from typing import Optional

import os

import torch
import torch.distributed as dist
import numpy as np
import random


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

    t: torch.Tensor
    
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
        self.t = torch.zeros(1, dtype=torch.long, device=self.device)
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

    def __init__(self):
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
        self.set_random_seeds(42 + rank)
        
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
        self.local_circuit.t += 1
        self.local_circuit.graph_root._call_process()
        self.local_circuit.t += 1

        # Capturing the graph
        with torch.cuda.graph(self.local_circuit.graph, stream=self.local_circuit.graph_stream):
            self.local_circuit.graph_root._call_process()
        self.local_circuit.t.zero_()

    def step(self):
        """Advance the simulation by one time step.

        Executes one simulation step by replaying the CUDA graph and
        calling _process() on all nodes that are not part of the graph.
        Increments the local circuit's time counter.
        """
        self.local_circuit.graph.replay()
        self._call_process()
        self.local_circuit.t += 1

    def connect(self, pre, pos, 
                pattern: str = "all_to_all",
                weight=1.0,
                delay=1,
                synapse=None,
                **kwargs):
        """
        Create a connection between populations.

        Parameters
        ----------
        pre : NeuronGroup
            Source population.
        pos : NeuronGroup
            Target population.
        pattern : str, optional
            Connection pattern: 'all_to_all', 'one_to_one', 'random'.
        weight : float, tensor or callable
            Synaptic weight(s).
        delay : int, tensor or callable
            Synaptic delay(s).
        synapse : class, optional
            Synapse class to instantiate (default: StaticConnection).
        kwargs : dict
            Pattern-specific parameters (e.g., p, fanin, fanout).

        Returns
        -------
        Connection
            Connection object.
        """
        device = pre.device  # asumimos ambos en mismo device

        # --- 1. Generate mask of connections ---
        if pattern == "all_to_all":
            mask = torch.ones((pre.size, pos.size), dtype=torch.bool, device=device)

        elif pattern == "one_to_one":
            if pre.size != pos.size:
                raise ValueError("one_to_one requires pre.size == post.size")
            mask = torch.eye(pre.size, dtype=torch.bool, device=device)

        elif pattern == "random":
            p = kwargs.get("p", None)
            fanin = kwargs.get("fanin", None)
            fanout = kwargs.get("fanout", None)

            if p is not None:
                # Each pair connected with prob p
                mask = (torch.rand((pre.size, pos.size), device=device) < p)

            elif fanin is not None:
                mask = torch.zeros((pre.size, pos.size), dtype=torch.bool, device=device)
                for j in range(pos.size):
                    idx = torch.randperm(pre.size, device=device)[:fanin]
                    mask[idx, j] = True

            elif fanout is not None:
                mask = torch.zeros((pre.size, pos.size), dtype=torch.bool, device=device)
                for i in range(pre.size):
                    idx = torch.randperm(pos.size, device=device)[:fanout]
                    mask[i, idx] = True
            else:
                raise ValueError("random requires one of p, fanin, fanout")

        elif pattern == "distance":
            # Required args
            max_distance = kwargs.get("max_distance", None)
            sigma = kwargs.get("sigma", None)
            p_max = kwargs.get("p_max", 1.0)
            fanin = kwargs.get("fanin", None)
            fanout = kwargs.get("fanout", None)
            prob_func = kwargs.get("prob_func", None)

            if not hasattr(pre, "positions") or not hasattr(pos, "positions"):
                raise ValueError("distance pattern requires populations with positions")

            pre_pos = pre.positions.to(device)   # [Npre, dim]
            pos_pos = pos.positions.to(device) # [Npost, dim]

            # Compute pairwise distances
            dists = torch.cdist(pre_pos, pos_pos)  # [Npre, Npost]

            if prob_func is not None:
                # User-defined probability function prob_func(src, tgt)
                # Vectorized: apply on all pairs
                pre_exp = pre_pos[:, None, :]  # [Npre, 1, dim]
                post_exp = pos_pos[None, :, :] # [1, Npost, dim]
                probs = prob_func(pre_exp, post_exp)
            else:
                if sigma is None:
                    raise ValueError("distance pattern requires either prob_func or sigma")
                probs = p_max * torch.exp(-(dists**2) / (2 * sigma**2))

            # Apply distance cutoff
            if max_distance is not None:
                probs = probs * (dists <= max_distance)

            if fanin is not None:
                mask = torch.zeros_like(probs, dtype=torch.bool)
                for j in range(pos.size):
                    idx = torch.argsort(dists[:, j])[:fanin]
                    mask[idx, j] = True
            elif fanout is not None:
                mask = torch.zeros_like(probs, dtype=torch.bool)
                for i in range(pre.size):
                    idx = torch.argsort(dists[i, :])[:fanout]
                    mask[i, idx] = True
            else:
                mask = (torch.rand_like(probs) < probs)

        else:
            raise NotImplementedError(f"Pattern '{pattern}' not supported yet")

        # --- 2. Generate weights ---
        if callable(weight):
            # function of (pre_idx, post_idx)
            pre_idx, post_idx = mask.nonzero(as_tuple=True)
            w = weight(pre_idx, post_idx)
        else:
            w = torch.full(mask.shape, float(weight), device=device)

        # --- 3. Generate delays ---
        if callable(delay):
            pre_idx, post_idx = mask.nonzero(as_tuple=True)
            d = delay(pre_idx, post_idx)
        else:
            d = torch.full(mask.shape, int(delay), device=device)

        # --- 4. Select synapse class ---
        SynapseClass = synapse or StaticConnection  # usar la que tengas por defecto

        # --- 5. Instantiate connection ---
        conn = SynapseClass(pre, pos)
        conn._establish_connection(
            pattern=pattern,
            weight=weight,
            delay=delay,
            **kwargs
        )

        # --- 6. Register connection in network ---
        self.local_circuit.add_child(conn)

        return conn
