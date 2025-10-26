"""
SFA Diagnostic - Weight Change Investigation
=============================================

Diagnose WHY the correlation is always -0.687 regardless of beta.

Possible causes:
1. Weights are not changing at all (no learning)
2. Initial weight correlation is already negative
3. Output neuron has no activity (no learning signal)
4. Random initialization is always the same (seed issue)
"""

from neurobridge import *
import torch
import matplotlib.pyplot as plt
import numpy as np


class SFADiagnosticExperiment(Experiment):
    """Diagnostic experiment to investigate learning."""

    n_input: int = 64
    n_output: int = 1
    freq_slow: float = 1.0
    base_rate: float = 20.0
    mod_amp: float = 15.0
    beta_value: float = 0.0  # Test with NO Oja

    def build_network(self):
        np.random.seed(42)
        self.mod_weights = np.random.uniform(0.0, 1.0, self.n_input)

        print(f"\n[BUILD] Creating network with 64 neurons, beta = {self.beta_value:.0e}")

        with self.sim.autoparent("normal"):
            self.input_neurons = RandomSpikeNeurons(
                n_neurons=self.n_input,
                firing_rate=self.base_rate
            )
            self.output_neurons = IFNeurons(n_neurons=self.n_output)

            self.conn = self.sim.connect(
                self.input_neurons, self.output_neurons, PlasticDense,
                pattern="all-to-all",
                weight=1e-4,  # Fixed weight
                delay=1,
                plasticity={
                    "name": "sfa",
                    "params": {
                        "tau_fast": 10e-3,
                        "tau_slow": 100e-3,
                        "tau_z": 20e-3,
                        "eta": 1e-3,
                        "beta": self.beta_value,
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

        # Store initial weights for comparison
        self.weights_initial = None

    def pre_step(self):
        """Modulate inputs."""
        # Store initial weights on first step
        if self.step == 0:
            # Get initial weights
            w = self.conn.weight.cpu().numpy()
            self.weights_initial = w.copy()
            corr_initial = np.corrcoef(self.mod_weights, w.flatten())[0, 1]
            print(f"[INITIAL] Correlation (mod_weight vs initial_weight): {corr_initial:+.3f}")
            print(f"[INITIAL] Weight range: [{w.min():.6f}, {w.max():.6f}]")
            print(f"[INITIAL] Weight mean: {w.mean():.6f}")

        t = self.step * 1e-3
        z_slow = np.sin(2 * np.pi * self.freq_slow * t)

        firing_rates = self.base_rate + self.mod_amp * self.mod_weights * z_slow
        firing_rates = np.clip(firing_rates, 1.0, 50.0)

        self.input_neurons.firing_rate = torch.tensor(
            firing_rates, dtype=torch.float32, device=self.input_neurons.device
        )

        if not hasattr(self, "slow_trace"):
            self.slow_trace = []
        self.slow_trace.append(z_slow)

    def on_finish(self):
        """Detailed diagnostics."""
        print("\n" + "=" * 70)
        print("DIAGNOSTIC RESULTS")
        print("=" * 70)

        # Get final weights
        weights = self.weights.get_variable_tensor(0, "weight").cpu().numpy().squeeze()
        weights_initial = weights[0, :]
        weights_final = weights[-1, :]
        weights_change = weights_final - weights_initial

        # Correlations
        corr_initial = np.corrcoef(self.mod_weights, weights_initial)[0, 1]
        corr_final = np.corrcoef(self.mod_weights, weights_final)[0, 1]
        corr_change = np.corrcoef(self.mod_weights, weights_change)[0, 1]

        print(f"\n1. WEIGHT CORRELATIONS:")
        print(f"   Initial: {corr_initial:+.3f}")
        print(f"   Final:   {corr_final:+.3f}")
        print(f"   Change:  {corr_change:+.3f}")

        print(f"\n2. WEIGHT STATISTICS:")
        print(f"   Initial: mean={weights_initial.mean():.6f}, std={weights_initial.std():.6f}")
        print(f"   Final:   mean={weights_final.mean():.6f}, std={weights_final.std():.6f}")
        print(f"   Change:  mean={weights_change.mean():.6f}, std={weights_change.std():.6f}")
        print(f"   Max |change|: {np.abs(weights_change).max():.6f}")

        # Check if weights changed at all
        if np.abs(weights_change).max() < 1e-8:
            print("   ⚠️  WARNING: Weights barely changed! Learning not happening!")

        # Output activity
        spikes_output = self.spikes.get_spike_tensor(1).cpu().numpy()
        output_rate = len(spikes_output) / (len(self.slow_trace) * 1e-3)
        print(f"\n3. OUTPUT ACTIVITY:")
        print(f"   Total spikes: {len(spikes_output)}")
        print(f"   Firing rate: {output_rate:.2f} Hz")
        print(f"   Simulation time: {len(self.slow_trace) * 1e-3:.1f} s")

        if output_rate < 1.0:
            print("   ⚠️  WARNING: Very low output rate! SFA needs activity to learn.")

        # Input activity
        spikes_input = self.spikes.get_spike_tensor(0).cpu().numpy()
        n_input_spikes = len(spikes_input)
        input_rate_per_neuron = n_input_spikes / (self.n_input * len(self.slow_trace) * 1e-3)
        print(f"\n4. INPUT ACTIVITY:")
        print(f"   Total spikes: {n_input_spikes}")
        print(f"   Rate per neuron: {input_rate_per_neuron:.2f} Hz")

        print("=" * 70)

        # === PLOTS ===

        # Plot 1: Weight evolution (sample of neurons)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Top-left: Weight evolution for selected neurons
        ax = axes[0, 0]
        time_weights = np.arange(len(weights)) * 1e-3

        # Select 5 neurons with highest and lowest mod_weights
        idx_high = np.argsort(self.mod_weights)[-5:]
        idx_low = np.argsort(self.mod_weights)[:5]

        for i in idx_high:
            ax.plot(time_weights, weights[:, i], 'b-', alpha=0.6, lw=1.5)
        for i in idx_low:
            ax.plot(time_weights, weights[:, i], 'r-', alpha=0.6, lw=1.5)

        ax.plot([], [], 'b-', label='High mod_weight', lw=2)
        ax.plot([], [], 'r-', label='Low mod_weight', lw=2)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Synaptic Weight")
        ax.set_title("Weight Evolution")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Top-right: Initial vs final weights
        ax = axes[0, 1]
        ax.scatter(weights_initial, weights_final, alpha=0.6, s=50, c='green', edgecolors='k')
        ax.plot([weights_initial.min(), weights_initial.max()],
                [weights_initial.min(), weights_initial.max()],
                'k--', lw=2, alpha=0.5, label='No change line')
        ax.set_xlabel("Initial Weight")
        ax.set_ylabel("Final Weight")
        ax.set_title("Initial vs Final Weights")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Bottom-left: mod_weight vs initial weight
        ax = axes[1, 0]
        ax.scatter(self.mod_weights, weights_initial, alpha=0.6, s=50, c='blue', edgecolors='k')
        z_init = np.polyfit(self.mod_weights, weights_initial, 1)
        p_init = np.poly1d(z_init)
        x_fit = np.linspace(self.mod_weights.min(), self.mod_weights.max(), 100)
        ax.plot(x_fit, p_init(x_fit), 'r--', lw=2, alpha=0.8,
                label=f'Fit (r={corr_initial:+.3f})')
        ax.set_xlabel("mod_weight")
        ax.set_ylabel("Initial Weight")
        ax.set_title("Initial Weights vs mod_weight")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Bottom-right: mod_weight vs final weight
        ax = axes[1, 1]
        ax.scatter(self.mod_weights, weights_final, alpha=0.6, s=50, c='green', edgecolors='k')
        z_final = np.polyfit(self.mod_weights, weights_final, 1)
        p_final = np.poly1d(z_final)
        ax.plot(x_fit, p_final(x_fit), 'r--', lw=2, alpha=0.8,
                label=f'Fit (r={corr_final:+.3f})')
        ax.set_xlabel("mod_weight")
        ax.set_ylabel("Final Weight")
        ax.set_title("Final Weights vs mod_weight")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Plot 2: Spike raster (first 5 seconds)
        fig, ax = plt.subplots(figsize=(12, 6))

        spikes_input = self.spikes.get_spike_tensor(0).cpu().numpy()
        spikes_output = self.spikes.get_spike_tensor(1).cpu().numpy()

        # Only show first 5 seconds
        max_time = 5000  # ms
        spikes_input_clip = spikes_input[spikes_input[:, 1] < max_time]
        spikes_output_clip = spikes_output[spikes_output[:, 1] < max_time]

        ax.scatter(spikes_input_clip[:, 1] * 1e-3, spikes_input_clip[:, 0],
                   s=1, alpha=0.5, c='blue', label='Input spikes')
        ax.scatter(spikes_output_clip[:, 1] * 1e-3, spikes_output_clip[:, 0] + self.n_input,
                   s=20, alpha=0.8, c='green', marker='o', label='Output spikes')

        ax.axhline(self.n_input, color='red', ls='--', lw=1, alpha=0.5)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Neuron ID")
        ax.set_title("Spike Raster (first 5 seconds)")
        ax.set_xlim(0, 5)
        ax.legend()
        plt.tight_layout()
        plt.show()

        # Print diagnosis
        print("\n" + "=" * 70)
        print("DIAGNOSIS")
        print("=" * 70)

        if np.abs(corr_initial - corr_final) < 0.01:
            print("\n⚠️  PROBLEM IDENTIFIED:")
            print("  → Initial and final correlations are nearly identical!")
            print("  → This means weights are NOT changing significantly.")
            print("\nPossible causes:")
            print("  1. Output neuron has insufficient activity")
            print("  2. Weight initialization already creates the negative correlation")
            print("  3. Learning rate (eta) is too small")
            print("  4. Weights are hitting boundaries (w_min or w_max)")

        if corr_initial < -0.5:
            print("\n⚠️  INITIALIZATION PROBLEM:")
            print(f"  → Initial correlation is already strongly negative ({corr_initial:+.3f})")
            print("  → The random initialization happens to create a negative correlation!")
            print("\nSolution:")
            print("  - Use a different seed")
            print("  - Or initialize weights uniformly (not randomly)")

        print("=" * 70 + "\n")


if __name__ == "__main__":
    print("=" * 70)
    print("SFA DIAGNOSTIC EXPERIMENT")
    print("=" * 70)
    print("\nRunning 50s simulation with 64 neurons, beta=0")
    print("=" * 70)

    exp = SFADiagnosticExperiment(sim=Simulator(seed=0))
    exp.run(steps=50000)  # 50 seconds
