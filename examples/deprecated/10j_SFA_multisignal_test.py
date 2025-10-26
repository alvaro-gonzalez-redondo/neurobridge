"""
SFA Multi-Signal Test
=====================

Test SFA with multiple slow signals of different velocities.

This design avoids the cancellation problem from using ±versions of a single signal.
Each neuron is "tuned" to a mixture of slow signals with different timescales,
and SFA should learn to increase weights for neurons capturing the slowest features.
"""

from neurobridge import *
import torch
import numpy as np


class SFAMultiSignalTest(Experiment):
    n_input: int = 64
    n_output: int = 1
    base_rate: float = 20.0
    mod_amp: float = 15.0

    # Test parameters
    use_beta: bool = False
    use_random_init: bool = False

    def build_network(self):
        np.random.seed(42)

        # Multiple slow signals with different frequencies
        self.slow_freqs = np.array([0.5, 1.0, 2.0])  # Hz - slow to fast
        self.n_signals = len(self.slow_freqs)

        # Each input neuron is tuned to a mixture of slow signals
        # Using Dirichlet distribution ensures weights sum to 1
        self.mod_weights = np.random.dirichlet([1.0] * self.n_signals, self.n_input)
        # Shape: (n_input, n_signals), each row sums to 1

        # Compute "slowness score" for each neuron
        # Neurons with high weight on slow signals get high score
        slowness_weights = 1.0 / self.slow_freqs  # [2.0, 1.0, 0.5] - slower = higher
        slowness_weights /= slowness_weights.sum()  # normalize to [0.57, 0.29, 0.14]
        self.slowness_score = self.mod_weights @ slowness_weights  # (n_input,)

        with self.sim.autoparent("normal"):
            self.input_neurons = RandomSpikeNeurons(
                n_neurons=self.n_input,
                firing_rate=self.base_rate
            )
            self.output_neurons = IFNeurons(n_neurons=self.n_output)

            # Weight initialization
            if self.use_random_init:
                weight_init = (0, 2e-4)
            else:
                weight_init = 1e-4

            # Beta value
            beta_val = 1e-5 if self.use_beta else 0.0

            self.conn = self.sim.connect(
                self.input_neurons, self.output_neurons, PlasticDense,
                pattern="all-to-all",
                weight=weight_init,
                delay=1,
                plasticity={
                    "name": "sfa",
                    "params": {
                        "tau_fast": 200e-3,
                        "tau_slow": 2000e-3,
                        "tau_z": 20e-3,
                        "eta": 1e-3,
                        "beta": beta_val,
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
        t = self.step * 1e-3

        # Compute all slow signals
        z_slow = np.array([np.sin(2 * np.pi * f * t) for f in self.slow_freqs])
        # Shape: (n_signals,)

        # Each neuron gets a linear mixture of slow signals
        modulation = self.mod_amp * (self.mod_weights @ z_slow)
        # Shape: (n_input,)

        firing_rates = self.base_rate + modulation
        firing_rates = np.clip(firing_rates, 1.0, 50.0)

        self.input_neurons.firing_rate = torch.tensor(
            firing_rates, dtype=torch.float32, device=self.input_neurons.device
        )

        if not hasattr(self, "slow_trace"):
            self.slow_trace = []
        self.slow_trace.append(z_slow)

    def on_finish(self):
        weights = self.weights.get_variable_tensor(0, "weight").cpu().numpy().squeeze()
        weights_initial = weights[0, :]
        weights_final = weights[-1, :]
        weights_change = weights_final - weights_initial

        # Key metric: correlation between slowness_score and weight change
        # Neurons tuned to slow signals should increase weights more
        corr_initial = np.corrcoef(self.slowness_score, weights_initial)[0, 1]
        corr_final = np.corrcoef(self.slowness_score, weights_final)[0, 1]
        corr_change = np.corrcoef(self.slowness_score, weights_change)[0, 1]

        spikes_output = self.spikes.get_spike_tensor(1).cpu().numpy()
        output_rate = len(spikes_output) / (len(self.slow_trace) * 1e-3)

        self.results = {
            "corr_initial": corr_initial,
            "corr_final": corr_final,
            "corr_change": corr_change,
            "output_rate": output_rate,
            "weight_change_mean": weights_change.mean(),
            "weight_change_std": weights_change.std(),
            # Weight distribution statistics
            "w_init_mean": weights_initial.mean(),
            "w_init_std": weights_initial.std(),
            "w_init_min": weights_initial.min(),
            "w_init_max": weights_initial.max(),
            "w_final_mean": weights_final.mean(),
            "w_final_std": weights_final.std(),
            "w_final_min": weights_final.min(),
            "w_final_max": weights_final.max(),
            # Relative change
            "relative_change": np.abs(weights_change).mean() / (weights_initial.mean() + 1e-10),
        }


def run_multisignal_test():
    print("\n" + "=" * 70)
    print("SFA MULTI-SIGNAL TEST")
    print("=" * 70)
    print("Testing SFA with multiple slow signals (different velocities)")
    print("Frequencies: 0.5 Hz (slowest), 1.0 Hz, 2.0 Hz (fastest)")
    print("=" * 70 + "\n")

    simulation_time = 50.0
    steps = int(1000 * simulation_time)

    configurations = [
        {"use_beta": False, "use_random_init": False, "name": "Fixed weights, no Oja"},
        {"use_beta": True,  "use_random_init": False, "name": "Fixed weights, with Oja"},
        {"use_beta": False, "use_random_init": True,  "name": "Random weights, no Oja"},
        {"use_beta": True,  "use_random_init": True,  "name": "Random weights, with Oja"},
    ]

    results = []
    for config in configurations:
        print(f"Testing: {config['name']}")
        exp = SFAMultiSignalTest(
            sim=Simulator(seed=0),
            use_beta=config["use_beta"],
            use_random_init=config["use_random_init"]
        )
        exp.run(steps=steps)
        results.append({**config, **exp.results})
        print(f"  → Final corr: {exp.results['corr_final']:+.3f}, "
              f"Change corr: {exp.results['corr_change']:+.3f}, "
              f"Rate: {exp.results['output_rate']:.1f} Hz\n")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Configuration':<30} {'Corr(init)':<12} {'Corr(final)':<12} {'Corr(change)':<12} {'Status':<10}")
    print("-" * 70)

    for r in results:
        status = "✓" if r["corr_change"] > 0.3 else "✗"
        init_str = f"{r['corr_initial']:+.3f}" if not np.isnan(r['corr_initial']) else "NaN"
        final_str = f"{r['corr_final']:+.3f}"
        change_str = f"{r['corr_change']:+.3f}"
        print(f"{r['name']:<30} {init_str:<12} {final_str:<12} {change_str:<12} {status:<10}")

    # Weight distribution details
    print("\n" + "=" * 70)
    print("WEIGHT DISTRIBUTION DETAILS")
    print("=" * 70)

    for r in results:
        print(f"\n{r['name']}:")
        print(f"  Initial weights: mean={r['w_init_mean']:.2e}, std={r['w_init_std']:.2e}, "
              f"range=[{r['w_init_min']:.2e}, {r['w_init_max']:.2e}]")
        print(f"  Final weights:   mean={r['w_final_mean']:.2e}, std={r['w_final_std']:.2e}, "
              f"range=[{r['w_final_min']:.2e}, {r['w_final_max']:.2e}]")
        print(f"  Change:          mean={r['weight_change_mean']:+.2e}, std={r['weight_change_std']:.2e}")
        print(f"  Relative change: {r['relative_change']:.2%}")
        print(f"  Output rate:     {r['output_rate']:.1f} Hz")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    all_change_positive = all(r["corr_change"] > 0.3 for r in results)

    if all_change_positive:
        print("\n✓ SUCCESS: All configurations show positive correlation!")
        print("  → SFA correctly increases weights for neurons tuned to slow signals")
        print("  → Neurons capturing slower features (0.5 Hz) get stronger weights")
        print("  → Neurons capturing faster features (2.0 Hz) get weaker weights")
        print("\nThis demonstrates that SFA can extract slow features from")
        print("a complex mixture of signals at different timescales.")
    else:
        print("\n⚠ MIXED RESULTS:")
        for r in results:
            if r["corr_change"] < 0.3:
                print(f"\n  {r['name']}: corr_change = {r['corr_change']:+.3f}")

                # Diagnostic checks
                if r["output_rate"] > 400:
                    print(f"    → High output rate ({r['output_rate']:.1f} Hz) - possible saturation")
                if r["w_init_mean"] > 1.5e-4:
                    print(f"    → Initial weights may be too large ({r['w_init_mean']:.2e})")
                if r["relative_change"] < 0.1:
                    print(f"    → Very small relative change ({r['relative_change']:.2%}) - weak learning")
                if r["w_init_std"] > r["w_init_mean"] * 2:
                    print(f"    → High initial weight variance (std={r['w_init_std']:.2e}, "
                          f"mean={r['w_init_mean']:.2e})")

        print("\n  Possible issues to investigate:")
        print("  • Initial weights or learning rate may need adjustment")
        print("  • Signal amplitudes may need tuning")
        print("  • Consider longer simulation time for clearer differentiation")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_multisignal_test()
