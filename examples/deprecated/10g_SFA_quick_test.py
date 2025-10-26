"""
Quick test of sign correction - NO PLOTS
"""

from neurobridge import *
import torch
import numpy as np


class QuickSFATest(Experiment):
    n_input: int = 64
    n_output: int = 1
    freq_slow: float = 1.0
    base_rate: float = 20.0
    mod_amp: float = 15.0

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
                        "eta": 1e-3,
                        "beta": 0.0,  # No Oja to isolate sign issue
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

        corr_initial = np.corrcoef(self.mod_weights, weights_initial)[0, 1]
        corr_final = np.corrcoef(self.mod_weights, weights_final)[0, 1]

        spikes_output = self.spikes.get_spike_tensor(1).cpu().numpy()
        output_rate = len(spikes_output) / (len(self.slow_trace) * 1e-3)

        print("\n" + "=" * 70)
        print("QUICK TEST RESULTS - 64 Neurons (Sign Corrected)")
        print("=" * 70)
        print(f"\nInitial correlation: {corr_initial:+.3f} (should be NaN, all weights equal)")
        print(f"Final correlation:   {corr_final:+.3f}")
        print(f"\nWeight statistics:")
        print(f"  Initial: mean={weights_initial.mean():.6f}, std={weights_initial.std():.6f}")
        print(f"  Final:   mean={weights_final.mean():.6f}, std={weights_final.std():.6f}")
        print(f"  Change:  mean={weights_change.mean():.6f}, std={weights_change.std():.6f}")
        print(f"  Max |change|: {np.abs(weights_change).max():.6f}")
        print(f"\nOutput rate: {output_rate:.2f} Hz")

        print("\n" + "=" * 70)
        if corr_final > 0.5:
            print("✓ SUCCESS: Strong positive correlation!")
            print("  → Sign correction FIXED the learning direction!")
        elif corr_final > 0.2:
            print("⚠ PARTIAL: Weak positive correlation")
            print("  → Sign correction helped, but may need tuning")
        elif corr_final > 0:
            print("⚠ WEAK: Very weak positive correlation")
        else:
            print("✗ FAILED: Still negative correlation")
            print("  → Sign correction did not fix the problem")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    print("\nRunning quick test with 64 neurons (50s simulation)...\n")
    exp = QuickSFATest(sim=Simulator(seed=0))
    exp.run(steps=50000)
