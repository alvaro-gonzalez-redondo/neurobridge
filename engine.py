from __future__ import annotations

from . import globals
from .core import _Node, _GPUNode, _ParentStack
from .bridge import _BridgeNeuronGroup
from .utils import _setup_logger, log, is_distributed

from typing import Optional

import os

import torch
import torch.distributed as dist
import numpy as np
import random


class _CUDAGraphSubTree(_GPUNode):
    pass


class _LocalCircuit(_GPUNode):
    """
    Contains all updatable objects of this GPU.
    Used for structural organization. It might contain future extensions for analyses and serialization/deserialization.
    """
    t: torch.Tensor

    graph_root: _CUDAGraphSubTree
    graph: torch.cuda.CUDAGraph
    graph_stream: torch.cuda.Stream
    
    bridge: Optional[_BridgeNeuronGroup]


    def __init__(self, device):
        super().__init__(device)
        self.t = torch.zeros(1, dtype=torch.long, device=device)
        self.graph_root = _CUDAGraphSubTree(device=device)
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


class SimulatorEngine(_Node):
    t: int
    n_gpus: int
    world_size: int
    rank: int
    local_circuit: _LocalCircuit


    def __enter__(self):
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if is_distributed():
            dist.destroy_process_group()

    def autoparent(self, mode: str = "normal") -> _ParentStack:
        if mode == "graph":
            target = self.local_circuit.graph_root
        elif mode == "normal":
            target = self.local_circuit
        else:
            raise RuntimeError(f"Invalid autoparent mode: {mode}.")
        return _ParentStack(target)


    def __init__(self):
        super().__init__()

        globals.engine = self
        
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

        self.local_circuit = _LocalCircuit(device)
        self.add_child(self.local_circuit)

        # Logger
        globals.logger = _setup_logger(self.rank)

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
        self.local_circuit.t.zero_()
    

    def step(self):
        self.local_circuit.graph.replay()
        self._call_process()
        self.local_circuit.t += 1

    
    def add_default_bridge(self, n_local_neurons: int, n_steps: int):
        bridge = _BridgeNeuronGroup(
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
