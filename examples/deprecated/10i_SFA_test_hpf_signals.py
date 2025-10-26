"""
Test HPF Signal Types for SFA
==============================

Test if SignalHPFPost and SignalHPFVoltage also need sign correction.
"""

from neurobridge import *
import torch
import numpy as np


class SFASignalTypeTest(Experiment):
    n_input: int = 64
    n_output: int = 1
    freq_slow: float = 1.0
    base_rate: float = 20.0
    mod_amp: float = 15.0
    signal_type: str = "surrogate"  # Test parameter

    def build_network(self):
        np.random.seed(42)
        self.mod_weights = np.random.uniform(0.0, 1.0, self.n_input)

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
                        "tau_hpf": 100e-3,  # For HPF signals
                        "eta": 1e-3,
                        "beta": 0.0,
                        "w_min": 0.0,
                        "w_max": 1e-3,
                        "dt": 1e-3,
                        "signal_type": self.signal_type,
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
        weights_change = weights[-1, :] - weights[0, :]

        corr_change = np.corrcoef(self.mod_weights, weights_change)[0, 1]

        spikes_output = self.spikes.get_spike_tensor(1).cpu().numpy()
        output_rate = len(spikes_output) / (len(self.slow_trace) * 1e-3)

        self.results = {
            "corr_change": corr_change,
            "output_rate": output_rate,
        }


def run_signal_type_test():
    print("\n" + "=" * 70)
    print("SFA SIGNAL TYPE TEST")
    print("=" * 70)
    print("Testing if HPF signals also need sign correction")
    print("=" * 70 + "\n")

    simulation_time = 50.0
    steps = int(1000 * simulation_time)

    signal_types = ["surrogate", "hpf", "voltage"]
    results = []

    for signal_type in signal_types:
        print(f"Testing signal_type='{signal_type}'...")
        exp = SFASignalTypeTest(sim=Simulator(seed=0), signal_type=signal_type)
        exp.run(steps=steps)
        results.append({"signal_type": signal_type, **exp.results})
        print(f"  → Corr(change): {exp.results['corr_change']:+.3f}, "
              f"Rate: {exp.results['output_rate']:.1f} Hz\n")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Signal Type':<15} {'Corr(change)':<15} {'Status':<10}")
    print("-" * 70)

    for r in results:
        status = "✓" if r["corr_change"] > 0.3 else "✗"
        print(f"{r['signal_type']:<15} {r['corr_change']:+.3f} {'':>3} {status:<10}")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    surrogate_ok = results[0]["corr_change"] > 0.3
    hpf_ok = results[1]["corr_change"] > 0.3
    voltage_ok = results[2]["corr_change"] > 0.3

    if surrogate_ok and not hpf_ok and not voltage_ok:
        print("\n⚠ HPF SIGNALS NEED SIGN CORRECTION:")
        print("  - 'surrogate' works (already fixed)")
        print("  - 'hpf' fails (needs sign correction)")
        print("  - 'voltage' fails (needs sign correction)")
        print("\nAction needed:")
        print("  Add negative sign to SignalHPFPost.step() return")
        print("  Add negative sign to SignalHPFVoltage.step() return")
    elif surrogate_ok and hpf_ok and voltage_ok:
        print("\n✓ ALL SIGNAL TYPES WORK CORRECTLY!")
    else:
        print("\n⚠ UNEXPECTED RESULTS - needs investigation")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_signal_type_test()
