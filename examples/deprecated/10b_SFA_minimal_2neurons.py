"""
SFA Minimal Test - 2 Neurons
=============================

Minimal test to verify SFA learning direction.

Setup:
    - 2 input neurons with different mod_weights:
        Neuron 0: mod_weight = 0.1 (weak modulation)
        Neuron 1: mod_weight = 0.9 (strong modulation)
    - 1 output neuron with SFA plasticity

Expected Result:
    - Neuron 1 (strong modulation) should get HIGHER final weight
    - Positive correlation between mod_weight and final weight

This minimal setup makes it easy to see if SFA is learning the correct direction.
"""

from neurobridge import *
import torch
import matplotlib.pyplot as plt
import numpy as np


class SFAMinimalExperiment(Experiment):
    """Minimal 2-neuron SFA test."""

    n_input: int = 2
    n_output: int = 1
    freq_slow: float = 1.0  # Hz
    base_rate: float = 20.0
    mod_amp: float = 15.0

    def build_network(self):
        # Two neurons with very different mod_weights
        self.mod_weights = np.array([0.1, 0.9])

        print("\n" + "=" * 70)
        print("MINIMAL SFA TEST: 2 Neurons")
        print("=" * 70)
        print(f"Neuron 0: mod_weight = {self.mod_weights[0]:.1f} (WEAK modulation)")
        print(f"Neuron 1: mod_weight = {self.mod_weights[1]:.1f} (STRONG modulation)")
        print("\nExpected: Neuron 1 should get HIGHER final weight")
        print("=" * 70 + "\n")

        # Network
        with self.sim.autoparent("normal"):
            self.input_neurons = RandomSpikeNeurons(
                n_neurons=self.n_input,
                firing_rate=self.base_rate
            )
            self.output_neurons = IFNeurons(n_neurons=self.n_output)

            # SFA connection
            self.conn = self.sim.connect(
                self.input_neurons, self.output_neurons, PlasticDense,
                pattern="all-to-all",
                weight=1e-4,#(0, 2e-4),
                delay=1,
                plasticity={
                    "name": "sfa",
                    "params": {
                        "tau_fast": 10e-3,
                        "tau_slow": 100e-3,
                        "tau_z": 20e-3,
                        "eta": 1e-3,
                        "beta": 1e-5,
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

        # Monitors
        with self.sim.autoparent("normal"):
            self.spikes = SpikeMonitor([self.input_neurons, self.output_neurons])
            self.weights = VariableMonitor([self.conn], ["weight"])

    def pre_step(self):
        """Modulate input firing rates with slow signal."""
        t = self.step * 1e-3
        z_slow = np.sin(2 * np.pi * self.freq_slow * t)

        # Firing rates
        firing_rates = self.base_rate + self.mod_amp * self.mod_weights * z_slow
        firing_rates = np.clip(firing_rates, 1.0, 50.0)

        self.input_neurons.firing_rate = torch.tensor(
            firing_rates, dtype=torch.float32, device=self.input_neurons.device
        )

        # Store for analysis
        if not hasattr(self, "slow_trace"):
            self.slow_trace = []
        self.slow_trace.append(z_slow)

    def on_finish(self):
        """Analyze and visualize results."""
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)

        # Get weights
        weights = self.weights.get_variable_tensor(0, "weight").cpu().numpy()
        weights = weights.squeeze()  # [time, n_input]
        time_weights = np.arange(len(weights)) * 1e-3

        weights_initial = weights[0, :]
        weights_final = weights[-1, :]
        weights_change = weights_final - weights_initial

        # Print results
        print("\nINITIAL WEIGHTS:")
        for i in range(self.n_input):
            print(f"  Neuron {i} (mod={self.mod_weights[i]:.1f}): {weights_initial[i]:.6f}")

        print("\nFINAL WEIGHTS:")
        for i in range(self.n_input):
            print(f"  Neuron {i} (mod={self.mod_weights[i]:.1f}): {weights_final[i]:.6f}")

        print("\nWEIGHT CHANGES:")
        for i in range(self.n_input):
            direction = "↑" if weights_change[i] > 0 else "↓"
            print(f"  Neuron {i} (mod={self.mod_weights[i]:.1f}): {weights_change[i]:+.6f} {direction}")

        # Correlation
        corr = np.corrcoef(self.mod_weights, weights_final)[0, 1]
        print(f"\nCORRELATION (mod_weight vs final_weight): {corr:+.3f}")

        # Verdict
        print("\n" + "-" * 70)
        if weights_final[1] > weights_final[0]:
            print("✓ CORRECT: Neuron 1 (strong mod) has higher weight than Neuron 0 (weak mod)")
        else:
            print("✗ WRONG: Neuron 0 (weak mod) has higher weight than Neuron 1 (strong mod)")
            print("  → SFA is learning the OPPOSITE direction!")

        if corr > 0.5:
            print("✓ CORRECT: Strong positive correlation")
        elif corr > 0:
            print("⚠ WEAK: Positive but weak correlation")
        else:
            print("✗ WRONG: Negative correlation - SFA learning inverted!")

        print("=" * 70 + "\n")

        # === PLOTS ===

        # Plot 1: Weight Evolution
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(time_weights, weights[:, 0], 'b-', lw=2,
                label=f'Neuron 0 (mod={self.mod_weights[0]:.1f})')
        ax.plot(time_weights, weights[:, 1], 'r-', lw=2,
                label=f'Neuron 1 (mod={self.mod_weights[1]:.1f})')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Synaptic Weight")
        ax.set_title("Weight Evolution: Should Neuron 1 (red) increase more?")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # Plot 2: Output activity vs slow signal
        spikes_output = self.spikes.get_spike_tensor(1)
        time_array = np.arange(len(self.slow_trace)) * 1e-3
        z_slow_array = np.array(self.slow_trace)

        fig, ax1 = plt.subplots(figsize=(10, 4))

        # Output rate
        time_rate, rate_out = smooth_spikes(spikes_output[:, 1], to_step=self.step)
        ax1.plot(time_rate, rate_out, 'g-', lw=2, alpha=0.8, label='Output rate')
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Output rate (Hz)", color='g')
        ax1.tick_params(axis='y', labelcolor='g')
        ax1.grid(True, alpha=0.3)

        # Slow signal
        ax2 = ax1.twinx()
        ax2.plot(time_array, z_slow_array, 'b-', lw=1.5, alpha=0.7, label='z_slow')
        ax2.set_ylabel("z_slow", color='b')
        ax2.tick_params(axis='y', labelcolor='b')

        ax1.set_title("Output Activity vs Slow Signal")
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        plt.tight_layout()
        plt.show()

        # Plot 3: Final comparison
        fig, ax = plt.subplots(figsize=(6, 5))
        x = [0, 1]
        colors = ['blue', 'red']
        ax.bar(x, weights_final, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax.set_xticks(x)
        ax.set_xticklabels([f'Neuron 0\n(mod={self.mod_weights[0]:.1f})',
                            f'Neuron 1\n(mod={self.mod_weights[1]:.1f})'])
        ax.set_ylabel("Final Synaptic Weight")
        ax.set_title(f"Final Weights Comparison\nCorr = {corr:+.3f}")
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for i, v in enumerate(weights_final):
            ax.text(i, v + 0.01 * weights_final.max(), f'{v:.6f}',
                   ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    exp = SFAMinimalExperiment(sim=Simulator(seed=0))
    simulation_time = 100.0  # 100 seconds
    exp.run(steps=int(1000 * simulation_time))
