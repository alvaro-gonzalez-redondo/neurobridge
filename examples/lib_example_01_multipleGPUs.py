# torchrun --standalone --nnodes=1 --nproc_per_node=2 main01.py

import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from neurobridge import (
    Simulator,
    Experiment,
    NeuronGroup,
    ParrotNeurons,
    SpikeMonitor,
    show_or_save_plot,
    log,
)


class PingPongRingExperiment(Experiment):
    local_neurons: NeuronGroup
    spike_monitor: SpikeMonitor

    start_at: int = 10


    def build_network(self):
        n_neurons = 20

        with self.sim.autoparent("normal"):
            # Crear un puente neuronal (permite la comunicación entre GPUs)
            self.add_default_bridge(n_local_neurons=n_neurons, n_steps=10)
            bridge: NeuronGroup = self.sim.local_circuit.bridge

        with self.sim.autoparent("graph"):
            # Crear un grupo neuronal local
            local_neurons = ParrotNeurons(n_neurons, delay_max=20)

            # Envía a la siguiente GPU (o a sí misma si está sola)
            (local_neurons >> bridge.where_rank(self.local_rank))(
                pattern="one-to-one",
                delay=0,
                weight=1.0,
            )

            # Recibe de la GPU anterior (o de sí misma si está sola)
            (bridge.where_rank((self.local_rank - 1) % self.world_size) >> local_neurons)(
                pattern="one-to-one",
                delay=0,
                weight=1.0,
            )

            # Registramos las neuronas para poder meterle entradas
            self.local_neurons = local_neurons

        with self.sim.autoparent("normal"):
            # Añadimos un monitor
            self.spike_monitor = SpikeMonitor([self.local_neurons])


    def pre_step(self):
        # En la primera neurona (rank 0), inyectar un spike inicial para comenzar la actividad
        if self.local_rank == 0:
            if (
                self.time >= self.start_at
                and self.time < self.local_neurons.size + self.start_at
            ):
                initial_spikes = torch.zeros(
                    self.local_neurons.size,
                    dtype=torch.bool,
                    device=self.local_device,
                )
                initial_spikes[self.time - self.start_at] = True
                self.local_neurons.inject_spikes(initial_spikes)


    def pos_step(self):
        # Imprimimos los últimos spikes de la población neuronal
        spk_buf = self.local_neurons.get_spike_buffer()
        phase = (self.time-1) % self.local_neurons.delay_max
        spks = spk_buf[:, phase].squeeze().tolist()
        spks_str = "".join(["|" if spk else "_" for spk in spks])
        log(f"t={self.time:<5}: {spks_str}")


    def on_finish(self):
        monitor: SpikeMonitor = self.spike_monitor
        cpu_spikes = monitor.get_spike_tensor(0).cpu()
        ot, oi = cpu_spikes[:, 1], cpu_spikes[:, 0]
        plt.scatter(ot, oi, s=4)
        show_or_save_plot(filename=f"rank{self.local_rank}_output.png", log=log)

    
# Main

if __name__ == "__main__":
    exp = PingPongRingExperiment(sim=Simulator(), start_at=10)
    exp.run(steps=100)  # se cierra solo al final