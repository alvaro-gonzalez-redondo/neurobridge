"""
SFA Beta (Oja Term) Test
========================

Test if the Oja normalization term (beta) is causing the learning inversion with 64 neurons.

Hypothesis:
    With 64 inputs, the output neuron has high activity → z² is large
    → Oja depression (β·w·z²) dominates over SFA potentiation
    → Strongest synapses (high mod_weight) get depressed most
    → Inverted learning (negative correlation)

Test:
    Run with 64 neurons and different beta values:
    - beta = 1e-5 (original, fails)
    - beta = 1e-6 (10x weaker)
    - beta = 1e-7 (100x weaker)
    - beta = 0     (no Oja)
"""

from neurobridge import *
import torch
import matplotlib.pyplot as plt
import numpy as np


class SFABetaExperiment(Experiment):
    """Test SFA with different Oja term strengths."""

    n_input: int = 64
    n_output: int = 1
    freq_slow: float = 1.0
    base_rate: float = 20.0
    mod_amp: float = 15.0
    beta_value: float = 1e-5  # Will be varied

    def build_network(self):
        np.random.seed(42)
        self.mod_weights = np.random.uniform(0.0, 1.0, self.n_input)

        print(f"  Testing beta = {self.beta_value:.0e}")

        with self.sim.autoparent("normal"):
            self.input_neurons = RandomSpikeNeurons(
                n_neurons=self.n_input,
                firing_rate=self.base_rate
            )
            self.output_neurons = IFNeurons(n_neurons=self.n_output)

            # SFA with variable beta
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
                        "beta": self.beta_value,  # Variable Oja strength
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

    def pre_step(self):
        """Modulate inputs with slow signal."""
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
        """Compute correlation."""
        weights = self.weights.get_variable_tensor(0, "weight").cpu().numpy().squeeze()
        weights_final = weights[-1, :]

        corr = np.corrcoef(self.mod_weights, weights_final)[0, 1]
        self.final_correlation = corr

        # Output activity
        spikes_output = self.spikes.get_spike_tensor(1).cpu().numpy()
        output_rate = len(spikes_output) / (len(self.slow_trace) * 1e-3)

        print(f"    → Correlation: {corr:+.3f}, Output rate: {output_rate:.1f} Hz")


def run_beta_test():
    """Test different beta (Oja term) values with 64 neurons."""

    print("\n" + "=" * 70)
    print("SFA BETA (OJA TERM) TEST - 64 Neurons")
    print("=" * 70)
    print("Testing if Oja term causes learning inversion")
    print("=" * 70)

    # Test different beta values
    beta_values = [1e-5, 5e-6, 1e-6, 5e-7, 1e-7, 0.0]
    correlations = []

    simulation_time = 50.0
    steps = int(1000 * simulation_time)

    print("\n")
    for beta in beta_values:
        exp = SFABetaExperiment(sim=Simulator(seed=0), beta_value=beta)
        exp.run(steps=steps)
        correlations.append(exp.final_correlation)

    # Plot results
    fig, ax = plt.subplots(figsize=(10, 6))

    # Use log scale for x-axis (handle beta=0 separately)
    beta_for_plot = [b if b > 0 else 1e-8 for b in beta_values]

    ax.semilogx(beta_for_plot, correlations, 'bo-', lw=2, markersize=10)

    # Mark beta=0 differently
    ax.scatter([beta_for_plot[-1]], [correlations[-1]],
               s=150, c='red', marker='*', zorder=5,
               label=f'β=0 (no Oja): {correlations[-1]:+.3f}')

    ax.axhline(0, color='black', linestyle='--', lw=1, alpha=0.5)
    ax.axhline(0.5, color='green', linestyle=':', lw=1, alpha=0.5,
               label='Target threshold (0.5)')

    # Vertical line at original beta
    ax.axvline(1e-5, color='red', linestyle=':', lw=2, alpha=0.5,
               label='Original β = 1e-5')

    ax.set_xlabel("Beta (Oja strength)", fontsize=12)
    ax.set_ylabel("Correlation (mod_weight vs final_weight)", fontsize=12)
    ax.set_title("SFA Learning Quality vs Oja Term Strength (64 neurons)",
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=10)

    # Custom x-tick labels
    ax.set_xticks(beta_for_plot)
    ax.set_xticklabels([f'{b:.0e}' if b > 0 else '0' for b in beta_values],
                       rotation=45, ha='right')

    plt.tight_layout()
    plt.show()

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY - 64 Neurons with Different Beta Values")
    print("=" * 70)
    for beta, corr in zip(beta_values, correlations):
        status = "✓" if corr > 0.3 else "✗"
        beta_str = f"{beta:.0e}" if beta > 0 else "0 (no Oja)"
        print(f"  β = {beta_str:12s}: correlation = {corr:+.3f} {status}")

    print("\n" + "=" * 70)

    # Conclusion
    if correlations[-1] > 0.5:  # beta = 0
        print("\n✓ HYPOTHESIS CONFIRMED:")
        print("  Removing Oja term (β=0) fixes the learning inversion!")
        print("  → The Oja normalization was too strong for 64 neurons.")
    elif correlations[0] < 0 and any(c > 0.3 for c in correlations[1:]):
        print("\n✓ HYPOTHESIS PARTIALLY CONFIRMED:")
        print("  Reducing beta improves learning.")
        print(f"  → Best beta value: {beta_values[np.argmax(correlations)]:.0e}")
    else:
        print("\n✗ HYPOTHESIS REJECTED:")
        print("  Beta (Oja term) is not the main problem.")
        print("  → Need to investigate other causes.")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_beta_test()
