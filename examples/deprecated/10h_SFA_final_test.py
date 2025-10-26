"""
SFA Final Verification Test
============================

Comprehensive test of the sign correction with different configurations.
"""

from neurobridge import *
import torch
import numpy as np


class SFAFinalTest(Experiment):
    n_input: int = 64
    n_output: int = 1
    freq_slow: float = 1.0
    base_rate: float = 20.0
    mod_amp: float = 15.0

    # Test parameters
    use_beta: bool = False
    use_random_init: bool = False

    def build_network(self):
        np.random.seed(42)
        self.mod_weights = np.random.uniform(-1.0, 1.0, self.n_input)

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
        weights = self.weights.get_variable_tensor(0, "weight").cpu().numpy().squeeze()
        weights_initial = weights[0, :]
        weights_final = weights[-1, :]
        weights_change = weights_final - weights_initial

        corr_initial = np.corrcoef(np.abs(self.mod_weights), weights_initial)[0, 1]
        corr_final = np.corrcoef(np.abs(self.mod_weights), weights_final)[0, 1]
        corr_change = np.corrcoef(np.abs(self.mod_weights), weights_change)[0, 1]

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


def run_final_test():
    print("\n" + "=" * 70)
    print("SFA FINAL VERIFICATION TEST")
    print("=" * 70)
    print("Testing sign correction with different configurations")
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
        exp = SFAFinalTest(
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
        print("\n✓ SUCCESS: All configurations show positive weight change correlation!")
        print("  → The sign correction is working correctly.")
        print("  → SFA now properly minimizes temporal variations.")
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
        print("  • Random weight initialization may be too large (causing saturation)")
        print("  • High firing rates suggest neuron saturation")
        print("  • Consider reducing initial weight range or learning rate")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_final_test()
