import torch
from neurobridge.engine import SimulatorEngine
from neurobridge.local_circuit import LocalCircuit
from neurobridge.neuron_groups.if_neuron_group import IandF
from neurobridge.synaptic_groups.static_synapse import StaticSynapse


class MyEngine(SimulatorEngine):

    def build_user_network(self, rank, world_size, device):
        # Crear circuito
        circuit = LocalCircuit(
            device=device,
            rank=rank,
            world_size=world_size,
            n_bridge_steps=self.n_bridge_steps,
            bridge_size=12,
        )

        # Crear poblaciones
        popA = IandF(size=12, delay=5, device=device)
        popB = IandF(size=12, delay=5, device=device)

        # Crear sinapsis A → B
        idx_pre = torch.arange(6, device=device)
        idx_pos = torch.arange(6, device=device)
        delay = torch.tensor(
            [1, 2, 3, 1, 2, 3], device=device
        )  # torch.randint(1, 4, (6,), device=device)
        weight = torch.ones(6, device=device) * 1

        syn = StaticSynapse(
            pre=popA,
            pos=popB,
            idx_pre=idx_pre,
            idx_pos=idx_pos,
            delay=delay,
            weight=weight,
        )

        # Añadir al circuito
        circuit._add_neuron_group(popA)
        circuit._add_neuron_group(popB)
        circuit.add_synaptic_group(syn)

        # Comunicación puente: grupo A exporta, grupo B recibe
        bridge_indices = torch.arange(12, device=device)
        circuit.export_map = [(popA, bridge_indices)]
        circuit.inject_map = [(bridge_indices.flip(dims=(0,)), popB)]

        # Estimulación artificial (cada 3 pasos, activar primeras neuronas de A)
        def stimulate(t):
            if t % 3 == 0:
                spikes = torch.zeros(popA.size, dtype=torch.bool, device=device)
                spikes[0:3] = 1
                popA.inject_spikes(spikes)

        circuit.stimulate = stimulate

        return circuit


if __name__ == "__main__":
    engine = MyEngine(
        n_gpus=-1, n_bridge_steps=3
    )  # Cambia a n=0 (CPU) o n>=2 (multi-GPU) si quieres (-1 significa "todas las que haya")
    engine.build()
    for _ in range(15):
        engine.step()
