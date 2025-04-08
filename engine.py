from __future__ import annotations
import torch
import os
from contextlib import contextmanager
import torch.distributed as dist
import torch.multiprocessing as mp
from neurobridge.local_circuit import LocalCircuit

from typing import List, Optional, Generator, Dict, Any, Union, Type, Callable
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from neurobridge.core.typing_aliases import *


def setup_distributed(rank: int, world_size: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12400"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


class SimulatorEngine:
    # Atributo de clase
    current_circuit: Optional[LocalCircuit] = None

    # Atributos de instancia
    n_gpus: int
    world_size: int
    bridge_size: int
    n_bridge_steps: int
    circuits: List[LocalCircuit]
    t: int
    processes: Optional[List[mp.Process]]
    command_queues: Optional[List[mp.Queue]]
    response_queues: Optional[List[mp.Queue]]
    is_distributed_initialized: bool

    def __init__(self, n_gpus: int = 0, bridge_size: int = 0, n_bridge_steps: int = 5) -> None:
        """
        n_gpus: número de procesos (0=CPU, 1=una GPU, >=2 múltiples GPUs)
        bridge_size: el número de axones en el AxonalBridge
        n_bridge_steps: periodo de comunicación del AxonalBridge
        """
        if n_gpus < 0 or n_gpus >= torch.cuda.device_count():
            n_gpus = torch.cuda.device_count()
        self.n_gpus = n_gpus
        self.world_size = max(1, self.n_gpus)

        self.bridge_size = bridge_size
        self.n_bridge_steps = n_bridge_steps

        self.circuits = []  # 1 o tantos como GPUs haya
        self.t = 0  # Paso actual de la simulación

        self.processes = None
        self.command_queues = None
        self.response_queues = None
        self.is_distributed_initialized = False

        self._build_circuits()
        SimulatorEngine.current_circuit = self.circuits[0]


    def _build_circuits(self) -> None:
        if self.n_gpus == 0:
            self.circuits.append(
                LocalCircuit(
                    device = 'cpu', 
                    rank = 0,
                    world_size = 1,
                    n_bridge_steps = self.n_bridge_steps,
                    bridge_size = None
                )
            )
        else:
            for rank in range(self.n_gpus):
                self.circuits.append(
                    LocalCircuit(
                        device = f"cuda:{rank}",
                        rank = rank,
                        world_size = self.n_gpus,
                        n_bridge_steps = self.n_bridge_steps,
                        bridge_size = self.bridge_size
                    )
                )
            
        # Si es multi-GPU, iniciar los procesos ahora
        if self.n_gpus > 1:
            # Iniciar un contexto para compartir y comunicar entre procesos
            self.start_multi_gpu_simulation()
    

    def start_multi_gpu_simulation(self) -> None:
        # Iniciar procesos para simulación multi-GPU
        if self.n_gpus > 1 and not self.is_distributed_initialized:
            # Usar Queue o similar para comunicar comandos a los procesos
            self.command_queues = [mp.Queue() for _ in range(self.n_gpus)]
            self.response_queues = [mp.Queue() for _ in range(self.n_gpus)]
            
            # Iniciar procesos que permanecerán activos durante toda la simulación
            self.processes = []
            for rank in range(self.n_gpus):
                p = mp.Process(
                    target=self._gpu_worker_process, 
                    args=(rank, self.n_gpus, self.command_queues[rank], self.response_queues[rank])
                )
                p.start()
                self.processes.append(p)
                
            self.is_distributed_initialized = True
    

    def _gpu_worker_process(self, rank: int, world_size: int, command_queue: mp.Queue, response_queue: mp.Queue) -> None:
        # Este proceso permanece vivo durante toda la simulación
        setup_distributed(rank, world_size)
        
        # Bucle de procesamiento de comandos
        while True:
            command = command_queue.get()
            
            if command == "STEP":
                # Ejecutar un paso de simulación
                self._run_circuit(self.circuits[rank], rank=rank)
                response_queue.put("DONE")
            elif command == "EXIT":
                # Terminar el proceso
                cleanup_distributed()
                response_queue.put("EXITED")
                break
            # Se podrían añadir más comandos según sea necesario


    def step(self) -> None:
        if self.n_gpus <= 1:
            # Ejecución simple en CPU o una GPU
            self._run_circuit(self.circuits[0])

        else:
            # Enviar comando de paso a todos los procesos
            for q in self.command_queues:
                q.put("STEP")
                
            # Esperar a que todos los procesos terminen
            for q in self.response_queues:
                q.get()  # Esperar "DONE"
        
        self.t += 1
    

    def cleanup(self) -> None:
        # Limpiar recursos al finalizar la simulación
        if self.is_distributed_initialized:
            # Enviar comando de salida a todos los procesos
            for q in self.command_queues:
                q.put("EXIT")
                
            # Esperar a que todos los procesos terminen
            for q in self.response_queues:
                q.get()  # Esperar "EXITED"
                
            # Esperar a que los procesos terminen
            for p in self.processes:
                p.join()
                
            self.is_distributed_initialized = False
    

    @contextmanager
    def device(self, gpu_id: int) -> Generator[None, None, None]:
        """
        Gestor de contexto que establece el dispositivo/GPU actual para las operaciones.
        
        Parámetros:
        gpu_id (int): Índice de la GPU a utilizar (0 a n_gpus-1)
        """
        if gpu_id < 0 or gpu_id >= self.n_gpus:
            raise ValueError(f"GPU ID {gpu_id} fuera de rango (0 a {self.n_gpus-1})")
        
        old_circuit = SimulatorEngine.current_circuit
        SimulatorEngine.current_circuit = self.circuits[gpu_id]

        yield

        SimulatorEngine.current_circuit = old_circuit


    @classmethod
    def get_current_device() -> LocalCircuit:
        """
        Obtiene el dispositivo actual del contexto.
        """
        return SimulatorEngine.current_circuit


    def build_user_network(self) -> None:
        """
        El usuario debe sobreescribir este método para construir su red personalizada.
        """
        raise NotImplementedError("Debes sobreescribir build_user_network(...) para definir tu red.")


    def _run_circuit(self, circuit: LocalCircuit) -> None:
        # Estimulación personalizada (si existe)
        if hasattr(circuit, "stimulate"):
            circuit.stimulate(self.t)

        # Comunicación con el AxonalBridge
        if circuit.bridge is not None:
            if hasattr(circuit, "export_map"):
                for group, bridge_indices in circuit.export_map:
                    circuit.export_to_bridge(group, bridge_indices)

            if hasattr(circuit, "inject_map"):
                for bridge_indices, group in circuit.inject_map:
                    circuit.inject_from_bridge(group, bridge_indices)

        # Avanzar simulación
        circuit.step()

        # Mostrar actividad
        print(f"[t={self.t}] [GPU {circuit.rank}]")
        for i, group in enumerate(circuit.neuron_groups):
            spikes = group.spike_buffer[(group.t-1) % group.delay_max].to(torch.uint8).tolist()
            print(f"  NeuronGroup[{i}] spikes: {spikes}")
