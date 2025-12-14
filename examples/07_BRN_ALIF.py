from neurobridge import *
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm


class BalancedRandomNetworkExperiment(Experiment):
    n_total_neurons: int = 1_000
    exc_prop: float = 0.8
    conn_prob: float = 0.1
    n_noise_neurons: int = 100

    
    def build_network(self):
        n_excitatory_neurons = int(self.n_total_neurons * self.exc_prop)
        n_inhibitory_neurons = self.n_total_neurons - n_excitatory_neurons

        #with self.sim.autoparent("graph"):
        with self.sim.autoparent("normal"):
            self.noise = RandomSpikeNeurons(n_neurons=self.n_noise_neurons, firing_rate=5.0)
            self.exc_neurons = ALIFNeurons(n_neurons=n_excitatory_neurons)
            self.inh_neurons = LIFNeurons(n_neurons=n_inhibitory_neurons)

            self.n2e = self.sim.connect(
                self.noise, self.exc_neurons,
                connection_type=StaticDense, pattern="random", p=self.conn_prob,
                weight=Uniform(0.0, 2e-3), delay=0,
            )

            self.e2e_ampa = self.sim.connect(
                self.exc_neurons, self.exc_neurons,
                connection_type=STDPDense, pattern="random", p=self.conn_prob,
                weight=Uniform(0.0, 1e-6), delay=5, w_max=1e-6,
                A_plus=1e-8, A_minus=1.2e-8, oja_decay=3e-3,
                channel=0,
            )

            self.e2e_nmda = self.sim.connect(
                self.exc_neurons, self.exc_neurons,
                connection_type=STDPDense, pattern="random", p=self.conn_prob,
                weight=Uniform(0.0, 1e-6), delay=5, w_max=1e-6,
                A_plus=1e-8, A_minus=1.2e-8, oja_decay=3e-3,
                channel=2,
            )

            self.e2i = self.sim.connect(
                self.exc_neurons, self.inh_neurons,
                connection_type=StaticDense, pattern="random", p=self.conn_prob,
                weight=Uniform(0.0, 1e-4), delay=0,
            )

            self.i2e = self.sim.connect(
                self.inh_neurons, self.exc_neurons,
                connection_type=StaticDense, pattern="random", p=self.conn_prob,
                weight=Uniform(0.0, 1e-4), delay=0,
                channel=1,
            )
        
        with self.sim.autoparent("normal"):
            self.spike_monitor = SpikeMonitor(
                [
                    self.noise.where_idx(lambda i: i<100),
                    self.exc_neurons.where_idx(lambda i: i<100),
                    self.inh_neurons.where_idx(lambda i: i<100),
                ]
            )
    
    def on_finish(self):
        fig, ax0 = plt.subplots()
        ax1 = ax0.twinx()
        id_sum = 0
        for idx, label in enumerate(["Noise", "Exc", "Inh"]):
            spikes = self.spike_monitor.get_spike_tensor(idx).cpu()
            spk_steps, spk_neurons = spikes[:, 1], spikes[:, 0]
            spk_times = spk_steps*1e-3
            ax1.scatter(spk_times, spk_neurons+id_sum, s=1, label=label, c=f"C{idx}")
            n_neurons = int(self.spike_monitor.filters[idx].nonzero(as_tuple=True)[0][-1]) + 1
            id_sum += n_neurons
            times, rate = smooth_spikes(spk_steps, n_neurons=n_neurons, to_step=self.current_step, sigma=0.1)
            ax0.plot(times, rate, c=f"C{idx}")
        
        ax1.legend(loc="lower right")
        plt.title(f"Spikes from different subpopulations")
        plt.xlabel("Time (seconds)")
        ax0.set_ylabel("Spiking rate (Hz)")
        ax1.set_ylabel("Neuron ID")

        plt.show()


if __name__ == "__main__":
    exp = BalancedRandomNetworkExperiment(sim=Simulator(seed=0))
    simulation_time = 10.0
    exp.run(steps=int(1000*simulation_time))