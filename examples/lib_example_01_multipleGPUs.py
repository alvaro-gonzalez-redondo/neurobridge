import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from neurobridge import (
    SimulatorEngine,
    NeuronGroup,
    ParrotNeurons,
    SpikeMonitor,
    show_or_save_plot,
    log,
    log_error,
)


class PingPongRingSimulation(SimulatorEngine):
    local_neurons: NeuronGroup
    spike_monitor: SpikeMonitor

    def build_user_network(self, rank: int, world_size: int):
        n_neurons = 20

        with self.autoparent("normal"):
            # Crear un puente neuronal (permite la comunicación entre GPUs)
            self.add_default_bridge(n_local_neurons=n_neurons, n_steps=10)
            bridge = self.local_circuit.bridge

        with self.autoparent("graph"):
            # Crear un grupo neuronal local
            local_neurons = ParrotNeurons(
                self.local_circuit.device, n_neurons, delay_max=20
            )

            # Envía a la siguiente GPU (o a sí misma si está sola)
            if True:
                (local_neurons >> bridge.where_rank(rank))(
                    pattern="one-to-one",
                    delay=0,
                    weight=1.0,
                )
            else:
                (local_neurons >> local_neurons)(
                    pattern="one-to-one",
                    delay=10,
                    weight=1.0,
                )

            # Recibe de la GPU anterior (o de sí misma si está sola)
            (bridge.where_rank((rank - 1) % world_size) >> local_neurons)(
                pattern="one-to-one",
                delay=0,
                weight=1.0,
            )

            # Registramos las neuronas para poder meterle entradas
            self.local_neurons = local_neurons

        with self.autoparent("normal"):
            # Añadimos un monitor
            self.spike_monitor = SpikeMonitor([self.local_neurons])

    def feed_input(self, start_at=10):
        # En la primera neurona (rank 0), inyectar un spike inicial para comenzar la actividad
        if self.rank == 0:
            if (
                self.local_circuit.t >= start_at
                and self.local_circuit.t < self.local_neurons.size + start_at
            ):
                initial_spikes = torch.zeros(
                    self.local_neurons.size,
                    dtype=torch.bool,
                    device=self.local_circuit.device,
                )
                initial_spikes[self.local_circuit.t - start_at] = True
                self.local_neurons.inject_spikes(initial_spikes)

    def plot_spikes(self):
        monitor: SpikeMonitor = self.spike_monitor
        cpu_spikes = monitor.get_spike_tensor(0).cpu()
        ot, oi = cpu_spikes[:, 1], cpu_spikes[:, 0]
        plt.scatter(ot, oi, s=4)
        show_or_save_plot(filename=f"rank{self.rank}_output.png", log=log)


# Main

try:
    with PingPongRingSimulation() as engine:
        # simulation_length = 0.1
        # simulation_steps = int(simulation_length * 1000)
        simulation_steps = 100
        for step in tqdm(range(simulation_steps), disable=True):
            engine.feed_input()
            engine.step()

            # Imprimimos los últimos spikes de la población neuronal
            spk_buf = engine.local_neurons.get_spike_buffer()
            t = engine.local_circuit.t.item() - 1
            phase = t % engine.local_neurons.delay_max
            spks = spk_buf[:, phase].squeeze().tolist()
            spks_str = "".join(["|" if spk else "_" for spk in spks])
            log(f"t={step:<5}: {spks_str}")

        engine.plot_spikes()

except Exception as e:
    log_error(f"ERROR: {e}")
    import traceback

    log_error(traceback.format_exc())
