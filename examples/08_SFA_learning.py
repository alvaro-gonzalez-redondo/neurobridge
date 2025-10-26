"""
Slow Feature Analysis (SFA) Learning

This example demonstrates Slow Feature Analysis, an unsupervised learning
algorithm that extracts slowly varying features from rapidly changing input.

Key features:
- Plastic connections using SFA rule
- Difference-of-exponentials (DoE) eligibility traces
- High-pass filtered learning signal
- Oja normalization for weight stability

The SFA principle:
    Δw = η · (x_fast - x_slow) · HPF(z_post) - β · w · HPF(z_post)²

where:
    - DoE eligibility: e = x_fast - x_slow (captures temporal structure)
    - HPF signal: emphasizes changes in postsynaptic activity
    - Oja term: prevents unbounded weight growth

Reference:
    Wiskott, L., & Sejnowski, T. J. (2002). Slow feature analysis: Unsupervised
    learning of invariances. Neural computation, 14(4), 715-770.

Network architecture:
- 200 input neurons (time-varying patterns)
- 100 output neurons (learn slow features)
- Plastic input→output connections (SFA rule)
"""

from neurobridge import *
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np


class SFAExperiment(Experiment):
    n_input: int = 200
    n_output: int = 100
    conn_prob: float = 0.2

    def build_network(self):
        # --- Network creation ---
        with self.sim.autoparent("normal"):
            # Input layer: modulated random spiking
            # We'll create temporal patterns by modulating firing rates
            self.input_neurons = RandomSpikeNeurons(
                n_neurons=self.n_input,
                firing_rate=20.0  # Base firing rate
            )

            # Output layer: integrate and fire neurons
            self.output_neurons = IFNeurons(n_neurons=self.n_output)

            # Plastic input → output connections (SFA rule)
            self.inp2out = self.sim.connect(
                self.input_neurons, self.output_neurons, PlasticDense,
                pattern="random", p=self.conn_prob,
                weight=(0, 2e-4),  # Initial weights
                delay=1,
                plasticity={
                    "name": "sfa",
                    "params": {
                        "tau_fast": 10e-3,      # Fast trace time constant (10ms)
                        "tau_slow": 100e-3,     # Slow trace time constant (100ms)
                        "tau_z": 20e-3,         # Postsynaptic smoothing (20ms)
                        "tau_hpf": 100e-3,      # High-pass filter (100ms)
                        "eta": 1e-5,            # Hebbian learning rate
                        "beta": 1e-3,           # Oja normalization coefficient
                        "w_min": 0.0,
                        "w_max": 5e-4,
                        "dt": 1e-3,
                    }
                }
            )

        # --- Monitor configuration ---
        with self.sim.autoparent("normal"):
            # Monitor spikes
            self.spike_monitor = SpikeMonitor(
                [
                    self.input_neurons.where_id(lambda i: i < 50),
                    self.output_neurons.where_id(lambda i: i < 50),
                ]
            )

            # Monitor membrane potential of first output neuron
            self.state_monitor = VariableMonitor(
                [self.output_neurons.where_id(lambda i: i < 1),],
                ['V', 'spikes']
            )

            # Monitor weight evolution
            self.inp2out.filter[:, :] = False
            self.inp2out.filter[:10, :10] = True  # Monitor 10x10 subset

            self.weight_monitor = VariableMonitor(
                [self.inp2out,],
                ["weight"]
            )

    def on_finish(self):
        """Visualization after simulation completes."""

        # --- Plot raster plot and firing rates ---
        fig, ax0 = plt.subplots(figsize=(12, 6))
        ax1 = ax0.twinx()
        id_sum = 0

        for idx, label in enumerate(["Input", "Output"]):
            spikes = self.spike_monitor.get_spike_tensor(idx).cpu()
            spk_steps, spk_neurons = spikes[:, 1], spikes[:, 0]
            spk_times = spk_steps * 1e-3  # Convert to seconds
            ax1.scatter(spk_times, spk_neurons + id_sum, s=1, label=label, c=f"C{idx}")

            n_neurons = int(self.spike_monitor.filters[idx].nonzero(as_tuple=True)[0][-1]) + 1
            id_sum += n_neurons

            # Smooth firing rate
            times, rate = smooth_spikes(spk_steps, n_neurons=n_neurons, to_step=self.step, sigma=0.1)
            ax0.plot(times, rate, c=f"C{idx}", linewidth=2, label=f"{label} rate")

        ax1.legend(loc='upper right')
        ax0.legend(loc='upper left')
        plt.title("SFA Learning - Raster plot and firing rates")
        plt.xlabel("Time (s)")
        ax0.set_ylabel("Firing rate (Hz)")
        ax1.set_ylabel("Neuron ID")
        plt.grid(True, alpha=0.3)

        # --- Plot membrane potential ---
        fig, ax = plt.subplots(figsize=(12, 4))
        V = self.state_monitor.get_variable_tensor(0, 'V')
        ax.plot(V, color='C0', label='V', linewidth=1)

        spikes = self.state_monitor.get_variable_tensor(0, 'spikes')
        spike_times = spikes.nonzero(as_tuple=True)[0]
        ax.vlines(spike_times, ymin=V.min(), ymax=V.max(), color='red',
                  alpha=0.3, linewidth=1, label='Spikes')

        plt.title("Membrane potential (first output neuron)")
        plt.xlabel("Time (steps)")
        plt.ylabel("Membrane potential (V)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # --- Plot weight evolution ---
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

        w_sfa = self.weight_monitor.get_variable_tensor(0, "weight")

        # Individual weight trajectories
        for i in range(min(10, w_sfa.shape[1])):
            for j in range(min(10, w_sfa.shape[2])):
                ax1.plot(w_sfa[:, i, j], alpha=0.5, linewidth=1)
        ax1.set_title("SFA - Individual weight trajectories")
        ax1.set_ylabel("Weight")
        ax1.grid(True, alpha=0.3)

        # Mean weight evolution
        mean_weights = w_sfa.mean(dim=(1, 2))
        std_weights = w_sfa.std(dim=(1, 2))
        steps = torch.arange(len(mean_weights))

        ax2.plot(steps, mean_weights, 'b-', linewidth=2, label='Mean weight')
        ax2.fill_between(steps, mean_weights - std_weights, mean_weights + std_weights,
                          alpha=0.3, label='±1 std')
        ax2.set_title("Mean weight evolution")
        ax2.set_ylabel("Weight")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Weight distribution at different time points
        n_timepoints = 5
        indices = np.linspace(0, len(w_sfa) - 1, n_timepoints, dtype=int)
        for idx in indices:
            weights_flat = w_sfa[idx].flatten().cpu().numpy()
            ax3.hist(weights_flat, bins=50, alpha=0.5,
                     label=f't={idx} ({idx/1000:.1f}s)')
        ax3.set_title("Weight distribution evolution")
        ax3.set_xlabel("Weight value")
        ax3.set_ylabel("Count")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


# --- Execution ---
if __name__ == "__main__":
    print("="*70)
    print("Slow Feature Analysis (SFA) Learning")
    print("="*70)
    print("\nThis example demonstrates SFA learning, which extracts slowly")
    print("varying features from rapidly changing sensory input.")
    print("\nKey features:")
    print("  - Plastic connections with SFA rule")
    print("  - Difference-of-exponentials (DoE) eligibility traces")
    print("  - High-pass filtered learning signal")
    print("  - Oja normalization for weight stability")
    print("\nThe SFA principle:")
    print("  Δw = η · (x_fast - x_slow) · HPF(z_post) - β · w · HPF(z_post)²")
    print("\nRunning simulation...\n")

    exp = SFAExperiment(sim=Simulator(seed=0))
    simulation_time = 10.0  # seconds
    exp.run(steps=int(1000 * simulation_time))
