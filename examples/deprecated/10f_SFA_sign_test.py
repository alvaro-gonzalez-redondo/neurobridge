"""
SFA Sign Test - Testing Sign Correction Hypothesis
==================================================

Test if inverting the sign of the surrogate gradient fixes the learning direction.

Hypothesis:
    SFA should MINIMIZE temporal variations (slowness objective).
    The gradient is: ∂J/∂w ∝ (dy/dt) · eligibility
    For gradient DESCENT, we need: Δw ∝ -(dy/dt) · eligibility

    Current code does: Δw = η · e · L'  where L' = γ · tanh(Δy)
    This is gradient ASCENT (wrong!)

    Corrected: L' should be NEGATIVE: L' = -γ · tanh(Δy)

Test Plan:
    Run TWO experiments with 64 neurons:
    1. Original sign (current implementation) → expect correlation ≈ -0.687
    2. Inverted sign (corrected) → expect correlation > +0.5

We'll manually apply the sign correction in a custom plasticity update.
"""

from neurobridge import *
import torch
import matplotlib.pyplot as plt
import numpy as np


class SFASignTestExperiment(Experiment):
    """Test SFA with manually corrected sign."""

    n_input: int = 64
    n_output: int = 1
    freq_slow: float = 1.0
    base_rate: float = 20.0
    mod_amp: float = 15.0
    invert_sign: bool = False  # Test parameter

    def build_network(self):
        np.random.seed(42)
        self.mod_weights = np.random.uniform(0.0, 1.0, self.n_input)

        print(f"  Testing with sign inversion: {self.invert_sign}")

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
                weight=1e-4,
                delay=1,
                plasticity={
                    "name": "sfa",
                    "params": {
                        "tau_fast": 10e-3,
                        "tau_slow": 100e-3,
                        "tau_z": 20e-3,
                        "eta": 1e-3,
                        "beta": 0.0,  # Disable Oja to isolate sign issue
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
        """Modulate inputs."""
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

    def post_step(self):
        """Manually correct the sign if invert_sign is True."""
        if self.invert_sign:
            # Get the plasticity rule
            rule = self.conn.plasticity_rule

            # Access the learning signal component
            signal_component = rule.signal

            # Check if it's a temporal surrogate signal
            if hasattr(signal_component, 'y_smooth') and hasattr(signal_component, 'y_prev'):
                # We need to negate the learning signal that was already applied
                # The weight update that just happened was:
                #   Δw = η · outer(e, L')
                # We want to undo it and apply:
                #   Δw = η · outer(e, -L')
                # This is equivalent to applying an extra:
                #   Δw = -2 · η · outer(e, L')

                # Get the last eligibility and learning signal
                # (We'd need to recompute or store them, but that's complex)
                #
                # Simpler approach: We'll directly multiply the weight changes by -1
                # after plasticity runs, effectively inverting the update direction
                pass

    def on_finish(self):
        """Compute correlation."""
        weights = self.weights.get_variable_tensor(0, "weight").cpu().numpy().squeeze()
        weights_initial = weights[0, :]
        weights_final = weights[-1, :]
        weights_change = weights_final - weights_initial

        corr = np.corrcoef(self.mod_weights, weights_final)[0, 1]
        self.final_correlation = corr

        spikes_output = self.spikes.get_spike_tensor(1).cpu().numpy()
        output_rate = len(spikes_output) / (len(self.slow_trace) * 1e-3)

        print(f"    → Correlation: {corr:+.3f}, Output: {output_rate:.1f} Hz, "
              f"Weight change: {weights_change.mean():.2e} ± {weights_change.std():.2e}")


def run_sign_test():
    """Test original vs inverted sign."""

    print("\n" + "=" * 70)
    print("SFA SIGN TEST - Manual Sign Correction")
    print("=" * 70)
    print("Testing if inverting surrogate gradient sign fixes learning")
    print("=" * 70)

    simulation_time = 50.0
    steps = int(1000 * simulation_time)

    # Test 1: Original (wrong sign)
    print("\n1. ORIGINAL SIGN (current implementation):")
    exp1 = SFASignTestExperiment(sim=Simulator(seed=0), invert_sign=False)
    exp1.run(steps=steps)
    corr_original = exp1.final_correlation

    # Test 2: Inverted (corrected sign)
    # For this, we'll need to actually modify the code temporarily
    print("\n2. INVERTED SIGN (hypothesis: should be corrected):")
    print("   Note: Testing requires modifying hpf.py temporarily")
    print("   Skipping for now - will test after code modification")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nOriginal sign:  correlation = {corr_original:+.3f}")
    print("\nTo test inverted sign, we need to modify:")
    print("  File: neurobridge/plasticity/signals/hpf.py")
    print("  Line: 351")
    print("  Change: L_prime = self.gamma * torch.tanh(delta_y / self.delta)")
    print("  To:     L_prime = -self.gamma * torch.tanh(delta_y / self.delta)")
    print("\nExpected result with inverted sign: correlation > +0.5")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_sign_test()
