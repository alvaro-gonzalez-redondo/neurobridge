"""
SFA Frequency-Selective Test
=============================

Test SFA with frequency-selective neurons.

Each neuron group responds to ONLY ONE frequency:
- Group 1: responds to 0.5 Hz (slowest)
- Group 2: responds to 1.0 Hz (medium)
- Group 3: responds to 2.0 Hz (fastest)

This allows SFA to selectively strengthen connections from slow-responding neurons
and weaken (or strengthen less) connections from fast-responding neurons.
"""

from neurobridge import *
import torch
import numpy as np


class SFAFrequencySelectiveTest(Experiment):
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

        # Verify correct assignment
        assert len(self.frequency_assignment) == self.n_input

        # Compute slowness score: neurons responding to slow signals get high score
        # 0.5 Hz → score = 2.0
        # 1.0 Hz → score = 1.0
        # 2.0 Hz → score = 0.5
        self.slowness_score = 1.0 / self.slow_freqs[self.frequency_assignment]

        # For reporting
        self.group_sizes = [np.sum(self.frequency_assignment == i) for i in range(self.n_signals)]

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
                        "tau_fast": 10e-3,
                        "tau_slow": 100e-3,
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
        # Neurons responding to slow signals should increase weights more
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


def run_frequency_selective_test():
    print("\n" + "=" * 70)
    print("SFA FREQUENCY-SELECTIVE TEST")
    print("=" * 70)
    print("Testing SFA with frequency-selective neurons")
    print("Each neuron responds to ONLY ONE frequency:")
    print("  - Group 1: 0.5 Hz (slowest)")
    print("  - Group 2: 1.0 Hz (medium)")
    print("  - Group 3: 2.0 Hz (fastest)")
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
        exp = SFAFrequencySelectiveTest(
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
        slow_gt_med = r['weight_change_0.5Hz'] > r['weight_change_1.0Hz']
        med_gt_fast = r['weight_change_1.0Hz'] > r['weight_change_2.0Hz']
        slow_gt_fast = r['weight_change_0.5Hz'] > r['weight_change_2.0Hz']

        if slow_gt_fast:
            print(f"  → ✓ Slow > Fast (correct ordering)")
        else:
            print(f"  → ✗ Fast > Slow (wrong ordering)")

        if slow_gt_med and med_gt_fast:
            print(f"  → ✓ Perfect ordering: Slow > Medium > Fast")

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
    all_correct_ordering = all(r["weight_change_0.5Hz"] > r["weight_change_2.0Hz"] for r in results)

    if all_change_positive and all_correct_ordering:
        print("\n✓ SUCCESS: SFA correctly extracts slow features!")
        print("  → Strong positive correlation between slowness and weight change")
        print("  → Neurons responding to slow signals (0.5 Hz) increase weights MORE")
        print("  → Neurons responding to fast signals (2.0 Hz) increase weights LESS")
        print("\nThis demonstrates that SFA can successfully extract slow features")
        print("from a population of frequency-selective neurons.")
    elif all_correct_ordering:
        print("\n⚠ PARTIAL SUCCESS:")
        print("  → Correct ordering (slow > fast) but weak correlation")
        print("  → SFA is learning in the right direction but signal may be weak")
    else:
        print("\n⚠ MIXED RESULTS:")
        for r in results:
            if r["corr_change"] < 0.3 or r["weight_change_0.5Hz"] <= r["weight_change_2.0Hz"]:
                print(f"\n  {r['name']}: corr_change = {r['corr_change']:+.3f}")
                print(f"    Weight changes: 0.5Hz={r['weight_change_0.5Hz']:+.2e}, "
                      f"2.0Hz={r['weight_change_2.0Hz']:+.2e}")

                # Diagnostic checks
                if r["output_rate"] > 400:
                    print(f"    → High output rate ({r['output_rate']:.1f} Hz) - possible saturation")
                if r["w_init_mean"] > 1.5e-4:
                    print(f"    → Initial weights may be too large ({r['w_init_mean']:.2e})")
                if r["relative_change"] < 0.1:
                    print(f"    → Very small relative change ({r['relative_change']:.2%}) - weak learning")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_frequency_selective_test()
