"""
SFA Multi-Signal Diagnostic
============================

Diagnose why SFA is learning in the wrong direction.
Visualize the relationship between signal frequencies, neural activity, and weight changes.
"""

from neurobridge import *
import torch
import numpy as np
import matplotlib.pyplot as plt


class SFAMultiSignalDiagnostic(Experiment):
    n_input: int = 64
    n_output: int = 1
    base_rate: float = 20.0
    mod_amp: float = 15.0

    def build_network(self):
        np.random.seed(42)

        # Multiple slow signals with different frequencies
        self.slow_freqs = np.array([0.5, 1.0, 2.0])  # Hz
        self.n_signals = len(self.slow_freqs)

        # Each input neuron is tuned to a mixture of slow signals
        self.mod_weights = np.random.dirichlet([1.0] * self.n_signals, self.n_input)

        # Compute "slowness score" for each neuron
        slowness_weights = 1.0 / self.slow_freqs  # [2.0, 1.0, 0.5]
        slowness_weights /= slowness_weights.sum()
        self.slowness_score = self.mod_weights @ slowness_weights

        with self.sim.autoparent("normal"):
            self.input_neurons = RandomSpikeNeurons(
                n_neurons=self.n_input,
                firing_rate=self.base_rate
            )
            self.output_neurons = IFNeurons(n_neurons=self.n_output)

            self.conn = self.sim.connect(
                self.input_neurons, self.output_neurons, PlasticDense,
                pattern="all-to-all",
                weight=1e-4,
                delay=1,
                plasticity={
                    "name": "sfa",
                    "params": {
                        "tau_fast": 10e-3,
                        "tau_slow": 100e-3,
                        "tau_z": 20e-3,
                        "eta": 1e-3,
                        "beta": 0.0,
                        "w_min": 0.0,
                        "w_max": 1e-3,
                        "dt": 1e-3,
                        "signal_type": "surrogate",
                        "gamma": 1.0,
                        "delta": 0.1,
                        "surrogate_type": "tanh",
                        "v_scale": 0.1,
                    }
                }
            )

        with self.sim.autoparent("normal"):
            self.spikes = SpikeMonitor([self.input_neurons, self.output_neurons])
            self.weights = VariableMonitor([self.conn], ["weight"])

        # Track signals and output activity
        self.z_slow_history = []
        self.output_rate_history = []

    def pre_step(self):
        t = self.step * 1e-3

        # Compute all slow signals
        z_slow = np.array([np.sin(2 * np.pi * f * t) for f in self.slow_freqs])
        self.z_slow_history.append(z_slow)

        # Each neuron gets a linear mixture
        modulation = self.mod_amp * (self.mod_weights @ z_slow)
        firing_rates = np.clip(self.base_rate + modulation, 1.0, 50.0)

        self.input_neurons.firing_rate = torch.tensor(
            firing_rates, dtype=torch.float32, device=self.input_neurons.device
        )

    def on_finish(self):
        weights = self.weights.get_variable_tensor(0, "weight").cpu().numpy().squeeze()
        weights_initial = weights[0, :]
        weights_final = weights[-1, :]
        weights_change = weights_final - weights_initial

        # Compute output spike rate over time (binned)
        spikes_output = self.spikes.get_spike_tensor(1).cpu().numpy()
        bin_size = 100  # ms
        n_bins = len(self.z_slow_history) // bin_size
        output_rate_binned = []
        for i in range(n_bins):
            start_t = i * bin_size
            end_t = (i + 1) * bin_size
            spikes_in_bin = np.sum((spikes_output >= start_t) & (spikes_output < end_t))
            rate = spikes_in_bin / (bin_size * 1e-3)  # Hz
            output_rate_binned.append(rate)

        # Average z_slow over bins
        z_slow_binned = []
        for i in range(n_bins):
            start_idx = i * bin_size
            end_idx = (i + 1) * bin_size
            z_avg = np.mean(self.z_slow_history[start_idx:end_idx], axis=0)
            z_slow_binned.append(z_avg)
        z_slow_binned = np.array(z_slow_binned)  # (n_bins, 3)

        # Correlations
        corr_change = np.corrcoef(self.slowness_score, weights_change)[0, 1]

        # Analyze which frequency dominates output activity
        correlations_with_output = []
        for freq_idx in range(self.n_signals):
            corr = np.corrcoef(z_slow_binned[:, freq_idx], output_rate_binned)[0, 1]
            correlations_with_output.append(corr)

        print("\n" + "=" * 70)
        print("DIAGNOSTIC RESULTS")
        print("=" * 70)
        print(f"\nSlowness score vs weight change correlation: {corr_change:+.3f}")
        print(f"  → EXPECTED: positive (neurons tuned to slow signals should increase)")
        print(f"  → ACTUAL: {'CORRECT' if corr_change > 0 else 'WRONG DIRECTION'}")

        print(f"\nOutput rate correlations with each signal:")
        for freq_idx, (freq, corr) in enumerate(zip(self.slow_freqs, correlations_with_output)):
            print(f"  {freq} Hz: {corr:+.3f}")

        dominant_freq_idx = np.argmax(np.abs(correlations_with_output))
        print(f"\nDominant frequency in output: {self.slow_freqs[dominant_freq_idx]} Hz")

        # Analyze weight change patterns
        print(f"\nWeight change by frequency tuning:")
        for freq_idx, freq in enumerate(self.slow_freqs):
            # Find neurons with high weight on this frequency
            freq_weight = self.mod_weights[:, freq_idx]
            high_freq_neurons = freq_weight > 0.5
            if high_freq_neurons.any():
                avg_change = weights_change[high_freq_neurons].mean()
                print(f"  Neurons tuned to {freq} Hz: avg weight change = {avg_change:+.2e}")

        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot 1: Slowness score vs weight change
        ax = axes[0, 0]
        ax.scatter(self.slowness_score, weights_change, alpha=0.6, s=50)
        ax.set_xlabel("Slowness Score (higher = tuned to slow signals)")
        ax.set_ylabel("Weight Change")
        ax.set_title(f"Weight Change vs Slowness Score\nCorr = {corr_change:+.3f}")
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax.grid(True, alpha=0.3)

        # Plot 2: Frequency tuning distribution
        ax = axes[0, 1]
        colors = ['blue', 'green', 'red']
        for freq_idx, (freq, color) in enumerate(zip(self.slow_freqs, colors)):
            freq_weights = self.mod_weights[:, freq_idx]
            ax.hist(freq_weights, bins=20, alpha=0.5, label=f'{freq} Hz', color=color)
        ax.set_xlabel("Weight on frequency")
        ax.set_ylabel("Number of neurons")
        ax.set_title("Frequency Tuning Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Signals and output rate over time
        ax = axes[1, 0]
        time_binned = np.arange(n_bins) * bin_size * 1e-3  # seconds
        ax2 = ax.twinx()

        for freq_idx, (freq, color) in enumerate(zip(self.slow_freqs, colors)):
            ax.plot(time_binned, z_slow_binned[:, freq_idx],
                   label=f'{freq} Hz signal', color=color, alpha=0.7)
        ax2.plot(time_binned, output_rate_binned, 'k-', linewidth=2,
                label='Output rate', alpha=0.8)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Signal amplitude", color='b')
        ax2.set_ylabel("Output rate (Hz)", color='k')
        ax.set_title("Input Signals vs Output Activity")
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Plot 4: Weight change vs frequency preference
        ax = axes[1, 1]
        # Find which frequency each neuron prefers
        preferred_freq_idx = np.argmax(self.mod_weights, axis=1)
        for freq_idx, (freq, color) in enumerate(zip(self.slow_freqs, colors)):
            mask = preferred_freq_idx == freq_idx
            if mask.any():
                ax.scatter(np.ones(mask.sum()) * freq, weights_change[mask],
                          alpha=0.6, s=50, color=color, label=f'{freq} Hz')
        ax.set_xlabel("Preferred Frequency (Hz)")
        ax.set_ylabel("Weight Change")
        ax.set_title("Weight Change by Frequency Preference")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("sfa_multisignal_diagnostic.png", dpi=150)
        print(f"\nPlot saved: sfa_multisignal_diagnostic.png")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("SFA MULTI-SIGNAL DIAGNOSTIC")
    print("=" * 70)
    print("Running diagnostic with 3 slow signals (0.5, 1.0, 2.0 Hz)")
    print("=" * 70 + "\n")

    simulation_time = 50.0
    steps = int(1000 * simulation_time)

    exp = SFAMultiSignalDiagnostic(sim=Simulator(seed=0))
    exp.run(steps=steps)
