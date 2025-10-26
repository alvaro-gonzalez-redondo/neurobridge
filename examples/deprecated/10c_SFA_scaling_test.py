"""
SFA Scaling Test
================

Test SFA with different numbers of input neurons to identify where the problem appears.

We'll run experiments with: 2, 4, 8, 16, 32, 64 neurons
And see at what point the correlation becomes negative.
"""

from neurobridge import *
import torch
import matplotlib.pyplot as plt
import numpy as np


class SFAScalingExperiment(Experiment):
    """Test SFA with varying number of input neurons."""

    n_input: int = 2  # Will be set dynamically
    n_output: int = 1
    freq_slow: float = 1.0
    base_rate: float = 20.0
    mod_amp: float = 15.0
    add_noise: bool = False  # Control noise
    noise_amp: float = 0.3

    def build_network(self):
        np.random.seed(42)

        # mod_weights uniformly in [0, 1]
        self.mod_weights = np.random.uniform(0.0, 1.0, self.n_input)

        print(f"\n[Setup] {self.n_input} neurons, noise={'ON' if self.add_noise else 'OFF'}")
        print(f"        mod_weights: [{self.mod_weights.min():.2f}, {self.mod_weights.max():.2f}]")

        with self.sim.autoparent("normal"):
            self.input_neurons = RandomSpikeNeurons(
                n_neurons=self.n_input,
                firing_rate=self.base_rate
            )
            self.output_neurons = IFNeurons(n_neurons=self.n_output)

            # SFA connection - same parameters as minimal example
            self.conn = self.sim.connect(
                self.input_neurons, self.output_neurons, PlasticDense,
                pattern="all-to-all",
                weight=1e-4,  # Fixed initial weight to ensure output activity
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

        with self.sim.autoparent("normal"):
            self.spikes = SpikeMonitor([self.input_neurons, self.output_neurons])
            self.weights = VariableMonitor([self.conn], ["weight"])

    def pre_step(self):
        """Modulate input with slow signal."""
        t = self.step * 1e-3
        z_slow = np.sin(2 * np.pi * self.freq_slow * t)

        # Optional noise
        if self.add_noise:
            if not hasattr(self, 'noise_filtered'):
                self.noise_filtered = np.zeros(self.n_input)
                self.noise_alpha = np.exp(-1e-3 / 0.05)

            noise_white = np.random.normal(0, 1.0, self.n_input)
            self.noise_filtered = (self.noise_alpha * self.noise_filtered +
                                  (1 - self.noise_alpha) * noise_white)
            noise = self.noise_amp * self.noise_filtered
        else:
            noise = 0

        firing_rates = self.base_rate + self.mod_amp * self.mod_weights * z_slow + noise
        firing_rates = np.clip(firing_rates, 1.0, 50.0)

        self.input_neurons.firing_rate = torch.tensor(
            firing_rates, dtype=torch.float32, device=self.input_neurons.device
        )

        if not hasattr(self, "slow_trace"):
            self.slow_trace = []
        self.slow_trace.append(z_slow)

    def on_finish(self):
        """Compute correlation only."""
        weights = self.weights.get_variable_tensor(0, "weight").cpu().numpy().squeeze()
        weights_final = weights[-1, :]

        corr = np.corrcoef(self.mod_weights, weights_final)[0, 1]

        # Store result
        self.final_correlation = corr

        # Output spike count
        spikes_output = self.spikes.get_spike_tensor(1).cpu().numpy()
        output_rate = len(spikes_output) / (len(self.slow_trace) * 1e-3)

        print(f"        → Correlation: {corr:+.3f}, Output rate: {output_rate:.1f} Hz")


def run_scaling_test():
    """Run experiments with increasing numbers of neurons."""

    print("\n" + "=" * 70)
    print("SFA SCALING TEST")
    print("=" * 70)
    print("Testing how correlation changes with number of input neurons")
    print("=" * 70)

    neuron_counts = [2, 4, 8, 16, 32, 64]
    correlations_no_noise = []
    correlations_with_noise = []

    simulation_time = 50.0  # seconds (reduced for faster testing)
    steps = int(1000 * simulation_time)

    # Test without noise
    print("\n--- WITHOUT NOISE ---")
    for n in neuron_counts:
        # Create experiment with correct n_input from the start
        exp = SFAScalingExperiment(sim=Simulator(seed=0), n_input=n, add_noise=False)
        exp.run(steps=steps)
        correlations_no_noise.append(exp.final_correlation)

    # Test with noise
    print("\n--- WITH NOISE ---")
    for n in neuron_counts:
        # Create experiment with correct n_input from the start
        exp = SFAScalingExperiment(sim=Simulator(seed=0), n_input=n, add_noise=True)
        exp.run(steps=steps)
        correlations_with_noise.append(exp.final_correlation)

    # Plot results
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(neuron_counts, correlations_no_noise, 'bo-', lw=2, markersize=8,
            label='Without noise')
    ax.plot(neuron_counts, correlations_with_noise, 'rs-', lw=2, markersize=8,
            label='With noise')

    ax.axhline(0, color='black', linestyle='--', lw=1, alpha=0.5)
    ax.axhline(0.5, color='green', linestyle=':', lw=1, alpha=0.5,
               label='Target threshold (0.5)')

    ax.set_xlabel("Number of Input Neurons", fontsize=12)
    ax.set_ylabel("Correlation (mod_weight vs final_weight)", fontsize=12)
    ax.set_title("SFA Learning Quality vs Network Size", fontsize=14, fontweight='bold')
    ax.set_xscale('log', base=2)
    ax.set_xticks(neuron_counts)
    ax.set_xticklabels([str(n) for n in neuron_counts])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.show()

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nWithout noise:")
    for n, corr in zip(neuron_counts, correlations_no_noise):
        status = "✓" if corr > 0.3 else "✗"
        print(f"  {n:3d} neurons: {corr:+.3f} {status}")

    print("\nWith noise:")
    for n, corr in zip(neuron_counts, correlations_with_noise):
        status = "✓" if corr > 0.3 else "✗"
        print(f"  {n:3d} neurons: {corr:+.3f} {status}")

    print("\n" + "=" * 70)

    # Identify breaking point
    breaking_point_no_noise = None
    breaking_point_with_noise = None

    for i, n in enumerate(neuron_counts):
        if correlations_no_noise[i] < 0 and breaking_point_no_noise is None:
            breaking_point_no_noise = n
        if correlations_with_noise[i] < 0 and breaking_point_with_noise is None:
            breaking_point_with_noise = n

    if breaking_point_no_noise:
        print(f"\n⚠ Without noise: Correlation becomes negative at {breaking_point_no_noise} neurons")
    else:
        print("\n✓ Without noise: All correlations positive!")

    if breaking_point_with_noise:
        print(f"⚠ With noise: Correlation becomes negative at {breaking_point_with_noise} neurons")
    else:
        print("✓ With noise: All correlations positive!")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_scaling_test()
