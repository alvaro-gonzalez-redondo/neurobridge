from neurobridge import *
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np


class BalancedRandomNetworkExperiment(Experiment):
    # Estructura general de la red    
    n_total_neurons: int = 1_000
    exc_prop: float = 0.8
    n_noise_neurons: int = 100

    # Parámetros de las conexiones
    inp2exc_params = {'fanin':10, 'delay':0, 'weight':Uniform(2.5e-4, 5e-4)}
    inp2inh_params = {'fanin':10, 'delay':0, 'weight':Uniform(5e-4, 10e-4)}
    exc2exc_params = {'fanin':80, 'weight':Uniform(0.0, 1e-4), 'delay':UniformInt(0,19), 'w_max':0.5, 'A_plus':1e-5, 'A_minus':1.2e-5, 'oja_decay':1e-4}
    exc2inh_params = {'fanin':80, 'weight':Constant(2e-4), 'delay':0}
    inh2exc_params = {'fanin':160, 'weight':Constant(0.0), 'delay':0, 'w_max':10.0, 'eta':1e-3, 'target_rate':10.0}
    inh2inh_params = {'fanin':20, 'weight':Constant(1e-3), 'delay':0}
    
    def build_network(self):
        n_excitatory_neurons = int(self.n_total_neurons * self.exc_prop)
        n_inhibitory_neurons = self.n_total_neurons - n_excitatory_neurons

        with self.sim.autoparent("normal"):
            # Capas
            self.inp_neurons = RandomSpikeNeurons(n_neurons=self.n_noise_neurons, firing_rate=15.0) # Subido a 15Hz            
            self.exc_neurons = LIFNeurons(n_neurons=n_excitatory_neurons, tau_refrac=3e-3)
            self.inh_neurons = LIFNeurons(n_neurons=n_inhibitory_neurons, tau_membrane=1e-2, tau_refrac=1e-3)

            # Conexiones
            self.inp2exc = self.sim.connect(self.inp_neurons, self.exc_neurons, connection_type=StaticSparse, pattern="random", **self.inp2exc_params)
            self.inp2inh = self.sim.connect(self.inp_neurons, self.inh_neurons, connection_type=StaticSparse, pattern="random", **self.inp2inh_params)
            self.exc2exc = self.sim.connect(self.exc_neurons, self.exc_neurons, connection_type=STDPSparse, pattern="random", channel=0, autapses=False, **self.exc2exc_params)
            self.exc2inh = self.sim.connect(self.exc_neurons, self.inh_neurons, connection_type=StaticSparse, pattern="random", **self.exc2inh_params)
            self.inh2exc = self.sim.connect(self.inh_neurons, self.exc_neurons, connection_type=VogelsSparse, pattern="random", channel=1, **self.inh2exc_params)
            self.inh2inh = self.sim.connect(self.inh_neurons, self.inh_neurons, connection_type=StaticSparse, pattern="random", channel=1, autapses=False, **self.inh2inh_params)

        with self.sim.autoparent("normal"):
            self.spike_monitor = SpikeMonitor(
                [
                    self.inp_neurons.where_idx(lambda i: i<100),
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
            print(f"\nAnalysis of {label} population:")
            analyze_network_state(spikes)

            # Graficas
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


def analyze_network_state(spike_tensor, dt=1e-3, bin_size=0.01):
    """
    spike_tensor: Tensor [N_spikes, 2] donde col 0 es neuron_id, col 1 es step
    """
    neuron_ids = spike_tensor[:, 0].long()
    spike_times = spike_tensor[:, 1].float() * dt
    n_neurons = neuron_ids.max() + 1
    
    # --- 1. Calcular CV (Coefficient of Variation) ---
    cv_list = []
    for i in range(n_neurons):
        # Extraer tiempos de esta neurona
        times = spike_times[neuron_ids == i]
        if len(times) > 2:
            isi = torch.diff(times)
            mean_isi = torch.mean(isi)
            std_isi = torch.std(isi)
            if mean_isi > 0:
                cv_list.append((std_isi / mean_isi).item())
    
    mean_cv = np.mean(cv_list)
    print(f"Mean CV (Target ~1.0): {mean_cv:.3f}")

    # --- 2. Calcular Correlación Promedio ---
    # Convertir a matriz densa binned (Rates)
    t_max = spike_times.max()
    n_bins = int(t_max / bin_size) + 1
    
    # Matriz [N_neurons, N_bins]
    spike_counts = torch.zeros((n_neurons, n_bins))
    
    # Indices de bin para cada spike
    bin_indices = (spike_times / bin_size).long()
    
    # Rellenar (esto es lento en python puro, pero ok para validación)
    # Forma vectorizada rápida:
    indices = torch.stack([neuron_ids, bin_indices], dim=0)
    values = torch.ones_like(neuron_ids, dtype=torch.float)
    spike_counts.index_put_(tuple(indices), values, accumulate=True)
    
    # Calcular correlación de Pearson matrix
    # Centrar datos
    means = spike_counts.mean(dim=1, keepdim=True)
    stds = spike_counts.std(dim=1, keepdim=True)
    
    # Evitar div por cero en neuronas mudas
    valid = stds.squeeze() > 0
    spike_counts = spike_counts[valid]
    means = means[valid]
    stds = stds[valid]
    
    spike_counts_norm = (spike_counts - means) / (stds + 1e-8)
    
    # Matriz de correlación (N x N)
    corr_matrix = (spike_counts_norm @ spike_counts_norm.T) / (n_bins - 1)
    
    # Media de elementos fuera de la diagonal
    off_diag = corr_matrix[~torch.eye(corr_matrix.shape[0], dtype=bool)]
    mean_corr = off_diag.mean().item()
    
    print(f"Mean Pairwise Correlation (Target ~0.0): {mean_corr:.4f}")
    
    return cv_list, off_diag


if __name__ == "__main__":
    exp = BalancedRandomNetworkExperiment(sim=Simulator(seed=0))
    simulation_time = 10.0
    exp.run(steps=int(1000*simulation_time))