"""
SFA Tuned Timescales Test
==========================

Test SFA with DoE eligibility timescales tuned to match the signal frequencies.

Previous tests used:
- tau_fast = 10 ms, tau_slow = 100 ms
- These are optimized for high-frequency signals (~10-50 Hz)

Our signals are much slower (0.5-2.0 Hz, periods 500-2000 ms), so we need:
- tau_fast = 200 ms
- tau_slow = 2000 ms

This should allow the DoE filter to properly distinguish slow from fast signals.
"""

from neurobridge import *
import torch
import numpy as np


class SFATunedTimescalesTest(Experiment):
    n_input: int = 64
    n_output: int = 1
    base_rate: float = 20.0
    mod_amp: float = 15.0

    # Test parameters
    use_beta: bool = False
    use_random_init: bool = False

    def build_network(self):
        np.random.seed(42)

        # Three frequency groups (slow, medium, fast)
        self.slow_freqs = np.array([0.5, 1.0, 2.0])  # Hz
        self.n_signals = len(self.slow_freqs)

        # Divide neurons into frequency-selective groups
        neurons_per_group = self.n_input // self.n_signals
        remainder = self.n_input % self.n_signals

        # Assign each neuron to ONE frequency
        self.frequency_assignment = []
        for freq_idx in range(self.n_signals):
            n_in_group = neurons_per_group + (1 if freq_idx < remainder else 0)
            self.frequency_assignment.extend([freq_idx] * n_in_group)
        self.frequency_assignment = np.array(self.frequency_assignment)

        # Compute slowness score
        self.slowness_score = 1.0 / self.slow_freqs[self.frequency_assignment]

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
                        # TUNED TIMESCALES - matched to signal frequencies
                        "tau_fast": 200e-3,   # 200 ms (was 10 ms)
                        "tau_slow": 2000e-3,  # 2000 ms (was 100 ms)
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

        # Each neuron responds ONLY to its assigned frequency
        firing_rates = np.zeros(self.n_input)
        for freq_idx in range(self.n_signals):
            mask = self.frequency_assignment == freq_idx
            modulation = self.mod_amp * z_slow[freq_idx]
            firing_rates[mask] = self.base_rate + modulation

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
        corr_initial = np.corrcoef(self.slowness_score, weights_initial)[0, 1]
        corr_final = np.corrcoef(self.slowness_score, weights_final)[0, 1]
        corr_change = np.corrcoef(self.slowness_score, weights_change)[0, 1]

        spikes_output = self.spikes.get_spike_tensor(1).cpu().numpy()
        output_rate = len(spikes_output) / (len(self.slow_trace) * 1e-3)

        # Analyze weight change by frequency group
        weight_change_by_freq = []
        for freq_idx in range(self.n_signals):
            mask = self.frequency_assignment == freq_idx
            avg_change = weights_change[mask].mean()
            weight_change_by_freq.append(avg_change)

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
            # Per-frequency changes
            "weight_change_0.5Hz": weight_change_by_freq[0],
            "weight_change_1.0Hz": weight_change_by_freq[1],
            "weight_change_2.0Hz": weight_change_by_freq[2],
        }


def run_tuned_timescales_test():
    print("\n" + "=" * 70)
    print("SFA TUNED TIMESCALES TEST")
    print("=" * 70)
    print("Testing SFA with DoE timescales tuned to signal frequencies")
    print("")
    print("Signal frequencies: 0.5 Hz, 1.0 Hz, 2.0 Hz (periods: 2000, 1000, 500 ms)")
    print("DoE timescales:     tau_fast=200ms, tau_slow=2000ms")
    print("                    (was: tau_fast=10ms, tau_slow=100ms)")
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
        exp = SFATunedTimescalesTest(
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

    # Weight change by frequency
    print("\n" + "=" * 70)
    print("WEIGHT CHANGE BY FREQUENCY GROUP")
    print("=" * 70)

    for r in results:
        print(f"\n{r['name']}:")
        print(f"  0.5 Hz (slow):   {r['weight_change_0.5Hz']:+.2e}")
        print(f"  1.0 Hz (medium): {r['weight_change_1.0Hz']:+.2e}")
        print(f"  2.0 Hz (fast):   {r['weight_change_2.0Hz']:+.2e}")

        # Check if ordering is correct
        slow_gt_fast = r['weight_change_0.5Hz'] > r['weight_change_2.0Hz']
        ratio = r['weight_change_0.5Hz'] / (r['weight_change_2.0Hz'] + 1e-10)

        if slow_gt_fast:
            print(f"  → ✓ Slow > Fast (ratio: {ratio:.2f}x)")
        else:
            print(f"  → ✗ Fast > Slow (ratio: {ratio:.2f}x)")

    # Weight distribution details
    print("\n" + "=" * 70)
    print("WEIGHT DISTRIBUTION DETAILS")
    print("=" * 70)

    for r in results:
        print(f"\n{r['name']}:")
        print(f"  Initial weights: mean={r['w_init_mean']:.2e}, std={r['w_init_std']:.2e}")
        print(f"  Final weights:   mean={r['w_final_mean']:.2e}, std={r['w_final_std']:.2e}")
        print(f"  Change:          mean={r['weight_change_mean']:+.2e}, std={r['weight_change_std']:.2e}")
        print(f"  Relative change: {r['relative_change']:.2%}")
        print(f"  Output rate:     {r['output_rate']:.1f} Hz")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    all_change_positive = all(r["corr_change"] > 0.3 for r in results)
    all_correct_ordering = all(r["weight_change_0.5Hz"] > r["weight_change_2.0Hz"] for r in results)

    if all_change_positive and all_correct_ordering:
        print("\n✓ SUCCESS: SFA correctly extracts slow features!")
        print("  → Strong positive correlation between slowness and weight change")
        print("  → Neurons responding to slow signals (0.5 Hz) increase weights MORE")
        print("  → Neurons responding to fast signals (2.0 Hz) increase weights LESS")
        print("\n✓ Tuning DoE timescales to match signal frequencies FIXED the problem!")
    elif all_correct_ordering:
        print("\n⚠ PARTIAL SUCCESS:")
        print("  → Correct ordering (slow > fast) achieved!")
        print("  → Correlation may be weak but direction is correct")
        print("  → Tuning timescales helped significantly")
    else:
        print("\n⚠ PROBLEM PERSISTS:")
        print("  → Tuning timescales alone did not fix the issue")
        print("  → May need to investigate other aspects of the implementation")

        for r in results:
            if r["weight_change_0.5Hz"] <= r["weight_change_2.0Hz"]:
                print(f"\n  {r['name']}: Still learning in wrong direction")
                print(f"    0.5Hz: {r['weight_change_0.5Hz']:+.2e}, 2.0Hz: {r['weight_change_2.0Hz']:+.2e}")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_tuned_timescales_test()
