from __future__ import annotations

from . import globals
from .core import Node, GPUNode, ParentStack
from .bridge import BridgeNeuronGroup
from .utils import _setup_logger, log, is_distributed, can_use_torch_compile

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

    graph_root: CUDAGraphSubTree
    graph: torch.cuda.CUDAGraph
    graph_stream: torch.cuda.Stream

    bridge: Optional[BridgeNeuronGroup]

    def __init__(self, device):
        """Initialize a local circuit on the specified device.

        Parameters
        ----------
        device : str or torch.device
            The GPU device for this circuit.
        """
        super().__init__(device)
        self.t = torch.zeros(1, dtype=torch.long, device=device)
        self.graph_root = CUDAGraphSubTree(device=device)
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


class SimulatorEngine(Node):
    """Main engine for running neural simulations.

    The SimulatorEngine is the central controller for the simulation,
    managing timing, GPU allocation, initialization, and stepping.
    It also provides utilities for building and organizing neural networks.

    This is a singleton class - only one instance should exist at a time.

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
    rank : int
        Rank (process index) in the distributed setup.
    local_circuit : LocalCircuit
        The neural circuit for this GPU.
    """

    t: int
    n_gpus: int
    world_size: int
    rank: int
    local_circuit: LocalCircuit

    def __enter__(self):
        """Context manager entry point.

        Initializes the simulator when entering a with block.

        Returns
        -------
        SimulatorEngine
            The initialized simulator instance.
        """
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Context manager exit point.

        Cleans up resources when exiting a with block.
        If running in distributed mode, destroys the process group.

        Parameters
        ----------
        exc_type : type
            The type of exception raised, if any.
        exc_value : Exception
            The exception raised, if any.
        traceback : traceback
            The traceback, if an exception was raised.
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

        globals.engine = self

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
            self.rank = dist.get_rank()
        else:
            self.rank = 0  # Non-distributed mode

        device = torch.device(f"cuda:{self.rank % self.n_gpus}")
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True

        self.local_circuit = LocalCircuit(device)
        self.add_child(self.local_circuit)

        # Logger
        globals.logger = _setup_logger(self.rank)

        # Reproducibility
        self.set_random_seeds(42 + self.rank)

        # Creates the custom user model
        log("#################################################################")
        log("Neurobridge initialized. Building user network...")
        self.build_user_network(self.rank, self.world_size)
        log("User network built successfully.")

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
            print("torch.compile no compatible con esta GPU, ejecutando sin compilar.")

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

    def add_default_bridge(self, n_local_neurons: int, n_steps: int):
        """Add a default bridge for inter-GPU communication.

        Creates a BridgeNeuronGroup and adds it to the local circuit.

        Parameters
        ----------
        n_local_neurons : int
            Number of bridge neurons per GPU.
        n_steps : int
            Number of time steps to collect before synchronizing.
        """
        bridge = BridgeNeuronGroup(
            device=self.local_circuit.device,
            rank=self.rank,
            world_size=self.world_size,
            n_local_neurons=n_local_neurons,
            n_bridge_steps=n_steps,
            spatial_dimensions=2,
            delay_max=n_steps + 1,
        )
        self.local_circuit.bridge = bridge
        self.local_circuit.add_child(bridge)

    def build_user_network(self, rank: int, world_size: int) -> None:
        """Build the user-defined neural network.

        This method must be implemented by subclasses to define the
        specific neural network structure for the simulation.

        Parameters
        ----------
        rank : int
            Rank (process index) in the distributed setup.
        world_size : int
            Number of processes in the distributed setup.

        Raises
        ------
        NotImplementedError
            If not implemented by a subclass.
        """
        raise NotImplementedError(
            "`build_user_network` in `SimulatorEngine` must be implemented."
        )
