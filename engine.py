import torch
import os
import torch.distributed as dist
import torch.multiprocessing as mp


def setup_distributed(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12400"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


class SimulatorEngine:

    def __init__(self, n_gpus=0, n_bridge_steps=3):
        """
        n_gpus: número de procesos (0=CPU, 1=una GPU, >=2 múltiples GPUs)
        n_bridge_steps: periodo de comunicación del AxonalBridge
        """
        if n_gpus<0 or n_gpus>=torch.cuda.device_count():
            n_gpus = torch.cuda.device_count()
        self.n_gpus = n_gpus
        self.world_size = max(1, self.n_gpus)
        self.n_bridge_steps = n_bridge_steps
        self.circuits = [] #1 or as many as GPUs
        self.t = 0 #Current simulation step


    def _run_single_cpu_or_gpu(self):
        self._run_circuit(self.circuits[0])


    def _run_multi_gpu_worker(self, rank):
        setup_distributed(rank, self.world_size)
        self._run_circuit(self.circuits[rank], rank=rank)
        cleanup_distributed()
    
    
    def build(self):
        if self.n_gpus == 0:
            self.circuits = [self.build_user_circuit(rank=0, world_size=1, device='cpu')]            
        else:
            self.circuits = [
                self.build_user_circuit(rank=rank, world_size=self.n_gpus, device=f"cuda:{rank}") 
                for rank in range(self.n_gpus)
            ]        


    def step(self):
        if self.n_gpus <= 1:
            self._run_single_cpu_or_gpu()
            
        else:
            mp.spawn(self._run_multi_gpu_worker, args=(self.n_gpus,), nprocs=self.n_gpus, join=True)
        
        self.t += 1


    def build_user_circuit(self, rank, world_size, device):
        """
        El usuario debe sobreescribir este método para construir su circuito personalizado.
        Debe devolver un objeto `Circuit` ya configurado con grupos y sinapsis.
        También puede definir:
            - circuit.export_map: [(group, bridge_indices)]
            - circuit.inject_map: [(bridge_indices, group)]
            - circuit.stimulate(t): función opcional para inyectar actividad
        """
        raise NotImplementedError("Debes sobreescribir build_user_circuit(...) para definir tu red.")


    def _run_circuit(self, circuit):
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
