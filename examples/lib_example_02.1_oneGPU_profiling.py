# from neurobridge.all import *

import sys

sys.path.insert(0, "..")
from all import *

import torch
import torch.profiler
from matplotlib import pyplot as plt
from tqdm import tqdm


class RandomInputSimulation(SimulatorEngine):

    def build_user_network(self, rank: int, world_size: int):
        n_src_neurons = 1000
        n_tgt_neurons = 1210
        self.is_monitoring = True

        with self.autoparent("graph"):

            src_neurons = RandomSpikeGenerator(
                device=self.local_circuit.device,
                n_neurons=n_src_neurons,
                firing_rate=10.0,
            )

            tgt_neurons = IFNeuronGroup(
                device=self.local_circuit.device,
                n_neurons=n_tgt_neurons,
            )

            stdp_conns = (src_neurons >> tgt_neurons)(
                pattern="all-to-all",
                synapse_class=STDPSynapse,
                weight=lambda pre, pos: torch.rand(len(pre)) * 1.9e-3,
            )

        with self.autoparent():

            if self.is_monitoring:
                self.spike_monitor = SpikeMonitor(
                    [
                        src_neurons,  # src_neurons.where_id(lambda i: i%10==0),
                        tgt_neurons,  # tgt_neurons.where_pos(lambda p: p[:,0]>0.5)
                    ]
                )
                self.voltage_monitor = VariableMonitor(
                    [tgt_neurons.where_id(lambda ids: ids < 100)], ["V"]
                )
                self.weight_monitor = VariableMonitor(
                    [stdp_conns.where_id(lambda ids: ids < 100)], ["weight"]
                )

    def plot_spikes(self):

        # Source spikes
        plt.figure()
        src_spikes = self.spike_monitor.get_spike_tensor(0)
        ot, oi = src_spikes[:, 1], src_spikes[:, 0]
        plt.scatter(ot, oi, s=4)

        tgt_spikes = self.spike_monitor.get_spike_tensor(1)
        ot, oi = tgt_spikes[:, 1], tgt_spikes[:, 0]
        plt.scatter(ot, oi, s=4)

        show_or_save_plot(filename=f"rank{self.rank}_output_1.png", log=log)

        # Target voltages
        plt.figure()
        v_values = self.voltage_monitor.get_variable_tensor(0, "V")
        plt.plot(v_values)
        show_or_save_plot(filename=f"rank{self.rank}_output_2.png", log=log)

        # Synaptic weights
        plt.figure()
        w_values = self.weight_monitor.get_variable_tensor(0, "weight")
        plt.plot(w_values)
        show_or_save_plot(filename=f"rank{self.rank}_output_3.png", log=log)


# Main

try:
    with RandomInputSimulation() as engine:
        simulation_length = 1.0
        simulation_steps = int(simulation_length * 1000)

        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=100, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                f"./logs/traces_rank{engine.rank}"
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )

        for _ in tqdm(range(simulation_steps)):
            engine.step(prof)

        if engine.is_monitoring:
            engine.plot_spikes()

except Exception as e:
    log_error(f"ERROR: {e}")
    import traceback

    log_error(traceback.format_exc())
