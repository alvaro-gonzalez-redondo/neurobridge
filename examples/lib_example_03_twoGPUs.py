from neurobridge import (
    SimulatorEngine,
    RandomSpikeNeurons,
    SimpleIFNeurons,
    STDPConnection,
    SpikeMonitor,
    VariableMonitor,
    show_or_save_plot,
    log,
    log_error,
)

import torch

from matplotlib import pyplot as plt
from tqdm import tqdm


class RandomInputSimulation(SimulatorEngine):

    def build_user_network(self, rank: int, world_size: int):
        n_src_neurons = 1_000
        n_tgt_neurons = 4_000
        self.is_monitoring = True

        n_local_bridge_neurons = n_src_neurons

        # Crear un puente neuronal (permite la comunicación entre GPUs)
        self.add_default_bridge(n_local_neurons=n_src_neurons, n_steps=20)
        bridge = self.local_circuit.bridge

        if rank == 0:

            with self.autoparent("graph"):
                src_neurons = RandomSpikeNeurons(
                    device=self.local_circuit.device,
                    n_neurons=n_src_neurons,
                    firing_rate=10.0,
                )

            with self.autoparent("normal"):
                _ = (src_neurons >> bridge.where_rank(0))(
                    pattern="one-to-one",
                    weight=1.0,
                )

                if self.is_monitoring:
                    self.spike_monitor = SpikeMonitor(
                        [src_neurons.where_id(lambda ids: ids < 20)]
                    )

        elif rank == 1:

            with self.autoparent("graph"):
                tgt_neurons = SimpleIFNeurons(
                    device=self.local_circuit.device,
                    n_neurons=n_tgt_neurons,
                )

                stdp_conns = (bridge.where_rank(0) >> tgt_neurons)(
                    pattern="all-to-all",
                    synapse_class=STDPConnection,
                    weight=lambda pre, pos: torch.rand(len(pre))
                    * (2.0 / n_src_neurons),
                )

            with self.autoparent("normal"):
                if self.is_monitoring:
                    self.spike_monitor = SpikeMonitor(
                        [tgt_neurons.where_id(lambda ids: ids < 20)]
                    )
                    self.voltage_monitor = VariableMonitor(
                        [tgt_neurons.where_id(lambda ids: ids < 20)], ["V"]
                    )
                    self.weight_monitor = VariableMonitor(
                        [stdp_conns.where_id(lambda ids: ids < 20)], ["weight"]
                    )

    def plot_spikes(self):

        if self.rank == 0:
            spks = self.spike_monitor.get_spike_tensor(0)
            ot, oi = spks[:, 1], spks[:, 0]
            plt.scatter(ot, oi, s=4)
            show_or_save_plot(filename=f"rank{self.rank}_src_spikes.png", log=log)

        elif self.rank == 1:
            spks = self.spike_monitor.get_spike_tensor(0)
            ot, oi = spks[:, 1], spks[:, 0]
            plt.scatter(ot, oi, s=4)
            show_or_save_plot(filename=f"rank{self.rank}_tgt_spikes.png", log=log)

            plt.figure()
            vals = self.voltage_monitor.get_variable_tensor(0, "V")
            plt.plot(vals)
            show_or_save_plot(filename=f"rank{self.rank}_v_tgt.png", log=log)

            plt.figure()
            vals = self.weight_monitor.get_variable_tensor(0, "weight")
            plt.plot(vals)
            show_or_save_plot(filename=f"rank{self.rank}_weight_sample.png", log=log)


# Main

try:
    with RandomInputSimulation() as engine:
        simulation_length = 10
        simulation_steps = simulation_length * 1000
        for _ in tqdm(range(simulation_steps), disable=(engine.rank != 0)):
            engine.step()

        if engine.is_monitoring:
            engine.plot_spikes()

except Exception as e:
    log_error(f"ERROR: {e}")
    import traceback

    log_error(traceback.format_exc())
