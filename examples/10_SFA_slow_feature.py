"""
SFA Slow Feature Extraction
============================

Demonstrates that SFA (Slow Feature Analysis) plasticity can extract slow
temporal features from a population of neurons encoding multiple signals.

Key Concepts:
-------------
- Input neurons respond to mixture of signals at different frequencies
- SFA should learn to increase weights for neurons tuned to SLOWER signals
- Output activity should track the slowest feature

Setup:
------
- Input: 64 Poisson neurons, each tuned to a mixture of 5 signals
  with frequencies: 0.5, 1.3, 3.4, 9.0, 23.5 Hz (irrational ratios φ²)

- Output: Single LIF neuron with SFA plasticity

- Learning rule: High-Pass Filtered (HPF) SFA
  • Filters spike trains to extract temporal structure
  • Minimal Oja regularization (β=1e-6) for stability
  • DoE timescales: tau_fast=1s, tau_slow=2.5s (optimized for 0.5 Hz)

Expected Result:
----------------
✓ Output should track 0.5 Hz signal (slowest)
✓ Weights for neurons tuned to slow signals > fast signals
✓ Positive correlation between slowness and final weights

Note:
-----
Uses irrational frequency ratios to avoid harmonic interference.
See docs/SFA_LESSONS_LEARNED.md for implementation details.

Reference:
----------
Sprekeler, H., et al. (2007). "Slowness: An objective for spike-timing-dependent
plasticity?" PLoS Computational Biology, 3(6), e112.
"""

from neurobridge import *
import torch
import matplotlib.pyplot as plt
import numpy as np


class SFASlowFeatureExperiment(Experiment):
    """Extract slow features from multi-frequency population code."""

    n_input: int = 64
    n_output: int = 1
    base_rate: float = 20.0
    mod_amp: float = 15.0

    def build_network(self):
        np.random.seed(42)

        # Multiple signals with irrational frequency ratios (golden ratio φ²)
        # This avoids harmonic interference
        phi = (1 + 5**0.5) / 2
        self.slow_freqs = np.array([0.5 * phi**(2*n) for n in range(5)])
        # ≈ [0.5, 1.31, 3.43, 8.97, 23.49] Hz
        self.n_signals = len(self.slow_freqs)

        # Each input neuron is tuned to a mixture of signals
        # Dirichlet distribution ensures mixture weights sum to 1
        self.mod_weights = np.random.dirichlet([1.0] * self.n_signals, self.n_input)

        # Compute "slowness score" for each neuron
        # Neurons tuned to slower signals get higher scores
        slowness_weights = 1.0 / self.slow_freqs
        slowness_weights /= slowness_weights.sum()
        self.slowness_score = self.mod_weights @ slowness_weights

        # Build network
        with self.sim.autoparent("normal"):
            self.input_neurons = RandomSpikeNeurons(
                n_neurons=self.n_input,
                firing_rate=self.base_rate
            )
            
            if True:
                self.output_neurons = IFNeurons(n_neurons=self.n_output)

                # Plastic connection with SFA learning rule
                self.conn = self.sim.connect(
                    self.input_neurons, self.output_neurons, PlasticDense,
                    pattern="all-to-all",
                    weight=5e-3/self.n_input,
                    delay=1,
                    plasticity={
                        "name": "sfa",
                        "params": {
                            # DoE eligibility timescales (optimized for 0.5 Hz discrimination)
                            "tau_fast": 1000e-3,    # 1000 ms → f_cutoff ≈ 0.16 Hz
                            "tau_slow": 2500e-3,    # 2500 ms → f_cutoff ≈ 0.06 Hz
                            "tau_z": 20e-3,         # Postsynaptic smoothing

                            # HPF learning signal (best performance for discrimination)
                            "signal_type": "hpf",
                            "tau_hpf": 1000e-3,     # Match tau_fast for consistency

                            # Learning parameters (tuned for stability)
                            "eta": 1e-9,            # Small learning rate for HPF
                            "beta": 1e-6,           # Light Oja regularization
                            "w_min": 0.0,
                            "w_max": 1e-1,
                            "dt": 1e-3,
                        }
                    }
                )

            elif False:
                self.output_neurons = IFNeurons(n_neurons=self.n_output)

                # Plastic connection with SFA learning rule
                self.conn = self.sim.connect(
                    self.input_neurons, self.output_neurons, PlasticDense,
                    pattern="all-to-all",
                    weight=5e-3/self.n_input,
                    delay=1,
                    plasticity={
                        "name": "sfa",
                        "params": {
                            "tau_fast": 1000e-3,    # Optimized for 0.5 Hz discrimination
                            "tau_slow": 2500e-3,
                            "tau_z": 20e-3,
                            "eta": 3e-5,
                            "beta": 1e-6,
                            "w_min": 0.0,
                            "w_max": 1e-1,
                            "dt": 1e-3,
                            "signal_type": "surrogate",
                            "gamma": 1.0,
                            "delta": 0.1,
                            "surrogate_type": "tanh",
                            "v_scale": 1.0,
                        }
                    }
                )

            elif False:
                self.output_neurons = StochasticIFNeurons(n_neurons=self.n_output, beta=200.0)

                # Plastic connection with SFA learning rule
                self.conn = self.sim.connect(
                    self.input_neurons, self.output_neurons, PlasticDense,
                    pattern="all-to-all",
                    weight=(0, 1e-1),
                    delay=1,
                    plasticity={
                        "name": "lipshutz_voltage",
                        "params": {
                            "tau_slow_pre": 1e-3,
                            "eta": 1e-2,
                            "beta": 1e-2,
                            "w_min": -0e-1, #0.0,
                            "w_max": 1e-1,
                            "dt": 1e-3,
                        }
                    }
                )

        # Monitors
        with self.sim.autoparent("normal"):
            self.spikes = SpikeMonitor([self.input_neurons, self.output_neurons])
            self.weights = VariableMonitor([self.conn], ["weight"])
            self.state_monitor = VariableMonitor([self.output_neurons,], ['V'])

    def pre_step(self):
        """Generate multi-frequency signals and modulate input firing rates."""
        # Save initial weights on first step
        if self.step == 0:
            self.weights_initial = self.conn.weight.detach().cpu().numpy().copy().squeeze()

        t = self.step * 1e-3

        # Compute all signals
        z_slow = np.array([np.sin(2 * np.pi * f * t) for f in self.slow_freqs])

        # Each neuron gets weighted mixture
        modulation = self.mod_amp * (self.mod_weights @ z_slow)
        firing_rates = np.clip(self.base_rate + modulation, 1.0, 50.0)

        self.input_neurons.firing_rate = torch.tensor(
            firing_rates, dtype=torch.float32, device=self.input_neurons.device
        )

        if not hasattr(self, "slow_trace"):
            self.slow_trace = []
        self.slow_trace.append(z_slow)

    def on_finish(self):
        """Analyze results and create visualizations."""
        print("\n" + "="*70)
        print("SFA LEARNING RESULTS")
        print("="*70)

        # Get final weights
        weights_final = self.conn.weight.detach().cpu().numpy().squeeze()
        weights_initial = self.weights_initial
        weights_change = weights_final - weights_initial

        # Analyze output activity
        spikes_output_torch = self.spikes.get_spike_tensor(1)
        spikes_output = spikes_output_torch.cpu().numpy()
        output_rate = len(spikes_output) / (len(self.slow_trace) * 1e-3)

        V = self.state_monitor.get_variable_tensor(0, 'V')

        # Correlation between slowness and final weights
        corr_slowness = np.corrcoef(self.slowness_score, weights_final)[0, 1]

        # Check which frequency dominates output
        z_slow_array = np.array(self.slow_trace)
        bin_size = 100
        n_bins = len(self.slow_trace) // bin_size

        output_rate_binned = []
        z_slow_binned = []

        for i in range(n_bins):
            # Output rate in bin
            start_t, end_t = i * bin_size, (i + 1) * bin_size
            spikes_in_bin = np.sum((spikes_output[:, 1] >= start_t) & (spikes_output[:, 1] < end_t))
            output_rate_binned.append(spikes_in_bin / (bin_size * 1e-3))

            # Average signal in bin
            z_slow_binned.append(np.mean(z_slow_array[start_t:end_t], axis=0))

        output_rate_binned = np.array(output_rate_binned)
        z_slow_binned = np.array(z_slow_binned)

        # Correlations between output and each signal
        output_signal_corrs = []
        for freq_idx in range(self.n_signals):
            if len(output_rate_binned) > 1:
                corr = np.corrcoef(output_rate_binned, z_slow_binned[:, freq_idx])[0, 1]
                output_signal_corrs.append(corr)

        dominant_freq_idx = np.argmax(np.abs(output_signal_corrs))

        # Weight distribution by frequency
        dominant_freq_assignment = np.argmax(self.mod_weights, axis=1)
        weight_by_freq = []
        for freq_idx in range(self.n_signals):
            mask = dominant_freq_assignment == freq_idx
            if mask.any():
                weight_by_freq.append(weights_final[mask].mean())

        # Print results
        print(f"\nOutput firing rate: {output_rate:.1f} Hz")
        print(f"Weight change: mean={weights_change.mean():.2e}, std={weights_change.std():.2e}")

        print(f"\n✓ Correlation(slowness, weights): {corr_slowness:+.3f}")
        if corr_slowness > 0.7:
            print("  → Strong positive correlation (excellent!)")
        elif corr_slowness > 0.3:
            print("  → Moderate positive correlation (good)")
        else:
            print("  → Weak correlation (may need tuning)")

        print(f"\n✓ Output activity correlates most with: {self.slow_freqs[dominant_freq_idx]:.2f} Hz")
        if dominant_freq_idx == 0:
            print("  → Correctly tracking the SLOWEST signal!")
        else:
            print("  → Warning: Not tracking the slowest signal")

        print(f"\n✓ Weight distribution by frequency:")
        for freq_idx, (freq, w) in enumerate(zip(self.slow_freqs, weight_by_freq)):
            marker = "★" if freq_idx == 0 else " "
            print(f"  {marker} {freq:6.2f} Hz: {w:.2e}")

        if weight_by_freq[0] > weight_by_freq[-1]:
            ratio = weight_by_freq[0] / weight_by_freq[-1]
            print(f"\n✓ Correct ordering: Slow > Fast (ratio: {ratio:.2f}x)")
        else:
            print(f"\n✗ Wrong ordering: Fast > Slow")

        # Final verdict
        success = (corr_slowness > 0.7 and
                  dominant_freq_idx == 0 and
                  weight_by_freq[0] > weight_by_freq[-1])

        print("\n" + "="*70)
        if success:
            print("✓✓ SUCCESS: SFA correctly extracts slow features!")
        else:
            print("⚠ Partial success - may need parameter tuning")
        print("="*70 + "\n")

        # Visualizations
        self._create_plots(spikes_output_torch, z_slow_array, V, weights_final,
                          output_rate_binned, z_slow_binned, dominant_freq_assignment)

    def _create_plots(self, spikes_output, z_slow_array, V, weights_final,
                     output_rate_binned, z_slow_binned, dominant_freq_assignment):
        """Create visualization plots."""

        # Get monitored weights for temporal evolution
        weights_evolution = self.weights.get_variable_tensor(0, "weight").cpu().numpy().squeeze()
        time_weights = np.arange(len(weights_evolution)) * 1e-3
        time_signals = np.arange(len(z_slow_array)) * 1e-3

        # Plot 1: Signals and Output Activity
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        # Top: Multiple signals
        colors = ['blue', 'cyan', 'green', 'orange', 'red']
        for freq_idx, (freq, color) in enumerate(zip(self.slow_freqs, colors)):
            alpha = 0.8 if freq_idx == 0 else 0.5
            lw = 2 if freq_idx == 0 else 1
            ax1.plot(time_signals, z_slow_array[:, freq_idx],
                    color=color, alpha=alpha, lw=lw, label=f'{freq:.2f} Hz')
        ax1.set_ylabel('Signal Amplitude')
        ax1.set_title('Input Signals (slowest = 0.5 Hz in bold)')
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Bottom: Output activity      
        time_rate, rate_out = smooth_spikes(spikes_output[:, 1], to_step=len(self.slow_trace))
        ax2.plot(time_rate, rate_out, 'darkgreen', lw=2, label='Output rate')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Output Rate (Hz)')
        ax2.set_title('Output Activity (should track 0.5 Hz signal)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2_ = ax2.twinx()
        ax2_.plot(time_rate, V, color='C0', label='V', linewidth=1.5)
        ax2_.set_ylabel("Membrane potential (V)", color='C0')
        ax2_.tick_params(axis='y', labelcolor='C0')

        plt.tight_layout()
        plt.show()

        # Plot 2: Weight Analysis
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: Slowness vs Final Weights (scatter)
        ax = axes[0]
        colors_map = {0: 'blue', 1: 'cyan', 2: 'green', 3: 'orange', 4: 'red'}

        for freq_idx in range(self.n_signals):
            mask = dominant_freq_assignment == freq_idx
            if mask.any():
                ax.scatter(self.slowness_score[mask], weights_final[mask],
                         alpha=0.6, s=50, c=colors_map.get(freq_idx, 'gray'),
                         label=f'{self.slow_freqs[freq_idx]:.1f} Hz')

        # Linear fit
        corr = np.corrcoef(self.slowness_score, weights_final)[0, 1]
        z = np.polyfit(self.slowness_score, weights_final, 1)
        p = np.poly1d(z)
        x_fit = np.linspace(self.slowness_score.min(), self.slowness_score.max(), 100)
        ax.plot(x_fit, p(x_fit), 'k--', lw=2, alpha=0.7, label=f'Fit (r={corr:+.3f})')

        ax.set_xlabel('Slowness Score (higher = tuned to slow signals)')
        ax.set_ylabel('Final Synaptic Weight')
        ax.set_title('SFA Selectivity for Slow Features')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Right: Weight Evolution by Frequency Group
        ax = axes[1]
        colors_freq = ['blue', 'cyan', 'green', 'orange', 'red']

        for freq_idx, (freq, color) in enumerate(zip(self.slow_freqs, colors_freq)):
            mask = dominant_freq_assignment == freq_idx
            if mask.any():
                mean_w = weights_evolution[:, mask].mean(axis=1)
                ax.plot(time_weights, mean_w, color=color, lw=2,
                       label=f'{freq:.1f} Hz (n={mask.sum()})', alpha=0.8)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Mean Synaptic Weight')
        ax.set_title('Weight Evolution by Frequency Preference')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    print("="*70)
    print("SFA: Slow Feature Extraction Tutorial")
    print("="*70)
    print("\nThis example demonstrates SFA learning with:")
    print("  • 64 input neurons tuned to 5 signals (0.5 - 23.5 Hz)")
    print("  • Irrational frequency ratios (golden ratio φ²)")
    print("  • HPF learning rule with matched DoE timescales")
    print("\nExpected: Output tracks 0.5 Hz (slowest) signal")
    print("="*70 + "\n")

    exp = SFASlowFeatureExperiment(sim=Simulator(seed=0))
    simulation_time = 100.0  # 100 seconds
    exp.run(steps=int(1000 * simulation_time))