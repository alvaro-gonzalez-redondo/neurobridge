from matplotlib import pyplot as plt
from tqdm import tqdm

from neurobridge import (
    SimulatorEngine,
    RandomSpikeNeurons,
    SimpleIFNeurons,
    STDPConnection,
    SpikeMonitor,
    show_or_save_plot,
    log,
    log_error,
)

import torch


class RandomInputSimulation(SimulatorEngine):

    def build_user_network(self, rank: int, world_size: int):
        n_src_neurons = 1_000
        n_tgt_neurons = 150
        self.is_monitoring = True

        with self.autoparent("graph"):

            src_neurons = RandomSpikeNeurons(
                device=self.local_circuit.device,
                n_neurons=n_src_neurons,
                firing_rate=10.0,
            )

            tgt_neurons = SimpleIFNeurons(
                device=self.local_circuit.device,
                n_neurons=n_tgt_neurons,
            )

            stdp_conns = (src_neurons >> tgt_neurons)(
                pattern="all-to-all",
                synapse_class=STDPConnection,
                weight=lambda pre, pos: torch.rand(len(pre)) * 1.9e-3,
            )

        with self.autoparent("normal"):

            if self.is_monitoring:
                if True:
                    self.spike_monitor = SpikeMonitor(
                        [
                            src_neurons.where_id(
                                lambda i: i < 20
                            ),  # src_neurons.where_id(lambda i: i%10==0),
                            tgt_neurons.where_id(
                                lambda i: i < 20
                            ),  # tgt_neurons.where_pos(lambda p: p[:,0]>0.5)
                        ]
                    )
                if False:
                    self.voltage_monitor = VariableMonitor(
                        [tgt_neurons.where_id(lambda ids: ids < 100)], ["V"]
                    )
                    self.weight_monitor = VariableMonitor(
                        [stdp_conns.where_id(lambda ids: ids < 100)], ["weight"]
                    )

    def plot_spikes(self):

        # Source spikes
        if hasattr(self, "spike_monitor"):
            plt.figure()
            src_spikes = self.spike_monitor.get_spike_tensor(0)
            ot, oi = src_spikes[:, 1], src_spikes[:, 0]
            plt.scatter(ot, oi, s=4)

            tgt_spikes = self.spike_monitor.get_spike_tensor(1)
            ot, oi = tgt_spikes[:, 1], tgt_spikes[:, 0]
            plt.scatter(ot, oi, s=4)

            show_or_save_plot(filename=f"rank{self.rank}_output_1.png", log=log)

        # Target voltages
        if hasattr(self, "voltage_monitor"):
            plt.figure()
            v_values = self.voltage_monitor.get_variable_tensor(0, "V")
            plt.plot(v_values)
            show_or_save_plot(filename=f"rank{self.rank}_output_2.png", log=log)

        # Synaptic weights
        if hasattr(self, "weight_monitor"):
            plt.figure()
            w_values = self.weight_monitor.get_variable_tensor(0, "weight")
            plt.plot(w_values)
            show_or_save_plot(filename=f"rank{self.rank}_output_3.png", log=log)


# Main

try:
    with RandomInputSimulation() as engine:
        simulation_length = 10.0
        simulation_steps = int(simulation_length * 1000)
        for _ in tqdm(range(simulation_steps)):
            engine.step()

        if engine.is_monitoring:
            engine.plot_spikes()

except Exception as e:
    log_error(f"ERROR: {e}")
    import traceback

    log_error(traceback.format_exc())
