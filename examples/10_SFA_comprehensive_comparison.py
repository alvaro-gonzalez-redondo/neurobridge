"""
SFA Comprehensive Comparison
=============================

Compare multiple SFA configurations side-by-side with the same input stimuli.

This experiment uses multiple output neurons (one per configuration) to ensure
fair comparison under identical conditions.

Configurations tested:
1. Surrogate (tanh) with Oja
2. HPF (no Oja)
3. Surrogate (sigmoid) with Oja
4. HPF with Oja (for comparison)
5. Lipshutz-Voltage (voltage-based mexican-hat STDP)

Key metrics:
- Correlation(slowness_score, weights)
- Correlation(output_activity, slowest_signal)
- Weight ordering (slow > fast)
- Learning speed and stability
"""

from neurobridge import *
import torch
import matplotlib.pyplot as plt
import numpy as np


class SFAComparisonExperiment(Experiment):
    """Compare multiple SFA configurations with same input."""

    n_input: int = 64
    base_rate: float = 20.0
    mod_amp: float = 15.0

    def build_network(self):
        np.random.seed(42)

        # Multiple slow signals with irrational ratios (golden ratio φ²)
        phi = (1 + 5**0.5) / 2
        self.slow_freqs = np.array([0.5 * phi**(2*n) for n in range(5)])
        self.n_signals = len(self.slow_freqs)

        # Each input neuron is tuned to a mixture of slow signals
        self.mod_weights = np.random.dirichlet([1.0] * self.n_signals, self.n_input)

        # Compute slowness score
        slowness_weights = 1.0 / self.slow_freqs
        slowness_weights /= slowness_weights.sum()
        self.slowness_score = self.mod_weights @ slowness_weights

        # Configurations to test
        self.configurations = [
            {
                "name": "Surrogate+Oja",
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
            },
            {
                "name": "HPF",
                "params": {
                    "tau_fast": 1000e-3,    # Optimized for 0.5 Hz discrimination
                    "tau_slow": 2500e-3,
                    "tau_z": 20e-3,
                    "eta": 1e-9,            # Adjusted for new timescales
                    "beta": 1e-6,           # Light Oja regularization
                    "w_min": 0.0,
                    "w_max": 1e-1,
                    "dt": 1e-3,
                    "signal_type": "hpf",
                    "tau_hpf": 1000e-3,     # Match tau_fast
                }
            },
            {
                "name": "Surrogate(sigmoid)+Oja",
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
                    "surrogate_type": "sigmoid",
                    "v_scale": 1.0,
                }
            },
            {
                "name": "HPF+Oja",
                "params": {
                    "tau_fast": 1000e-3,    # Optimized for 0.5 Hz discrimination
                    "tau_slow": 2500e-3,
                    "tau_z": 20e-3,
                    "eta": 1e-9,            # Adjusted for new timescales
                    "beta": 1e-6,           # Stronger Oja than HPF
                    "w_min": 0.0,
                    "w_max": 1e-1,
                    "dt": 1e-3,
                    "signal_type": "hpf",
                    "tau_hpf": 1000e-3,     # Match tau_fast
                }
            },
            {
                "name": "Lipshutz-Voltage",
                "params": {
                    "tau_slow_pre": 1e-3,    # Long timescale for slow features
                    "eta": 1e-5,                # Learning rate for Lipshutz
                    "beta": 0.0,                # No Oja (autoorganized)
                    "w_min": 0.0,
                    "w_max": 1e-1,
                    "dt": 1e-3,
                }
            },
        ]

        self.n_configs = len(self.configurations)

        # Single shared input layer
        with self.sim.autoparent("normal"):
            self.input_neurons = RandomSpikeNeurons(
                n_neurons=self.n_input,
                firing_rate=self.base_rate
            )

        # Multiple output neurons, one per configuration
        self.output_neurons = []
        self.connections = []

        with self.sim.autoparent("normal"):
            for config in self.configurations:
                output = IFNeurons(n_neurons=1)
                self.output_neurons.append(output)

                # Use different plasticity rule for Lipshutz
                rule_name = "lipshutz_voltage" if config["name"] == "Lipshutz-Voltage" else "sfa"

                conn = self.sim.connect(
                    self.input_neurons, output, PlasticDense,
                    pattern="all-to-all",
                    weight=1e-2/self.n_input,
                    delay=1,
                    plasticity={
                        "name": rule_name,
                        "params": config["params"]
                    }
                )
                self.connections.append(conn)

        # Monitors
        with self.sim.autoparent("normal"):
            self.spikes = SpikeMonitor([self.input_neurons] + self.output_neurons)
            self.weights = VariableMonitor(self.connections, ["weight"])

    def pre_step(self):
        """Generate slow latent signals and modulate input firing rates."""
        # Save initial weights on first step
        if self.step == 0:
            self.weights_initial = [conn.weight.detach().cpu().numpy().copy().squeeze()
                                   for conn in self.connections]

        t = self.step * 1e-3

        # Compute all slow signals
        z_slow = np.array([np.sin(2 * np.pi * f * t) for f in self.slow_freqs])

        # Each neuron gets a linear mixture of slow signals
        modulation = self.mod_amp * (self.mod_weights @ z_slow)
        firing_rates = np.clip(self.base_rate + modulation, 1.0, 50.0)

        self.input_neurons.firing_rate = torch.tensor(
            firing_rates, dtype=torch.float32, device=self.input_neurons.device
        )

        if not hasattr(self, "slow_trace"):
            self.slow_trace = []
        self.slow_trace.append(z_slow)

    def on_finish(self):
        """Analyze and compare all configurations."""
        print("\n" + "="*70)
        print("SFA COMPREHENSIVE COMPARISON")
        print("="*70)

        z_slow_array = np.array(self.slow_trace)
        results = []

        # Analyze each configuration
        for config_idx, config in enumerate(self.configurations):
            print(f"\n{'='*70}")
            print(f"Configuration {config_idx+1}: {config['name']}")
            print(f"{'='*70}")

            # Get spikes for this output neuron
            spikes_output = self.spikes.get_spike_tensor(config_idx + 1).cpu().numpy()

            # Compute binned firing rate
            bin_size = 100  # ms
            n_bins = len(self.slow_trace) // bin_size
            output_rate_binned = []
            for i in range(n_bins):
                start_t = i * bin_size
                end_t = (i + 1) * bin_size
                spikes_in_bin = np.sum((spikes_output[:, 1] >= start_t) & (spikes_output[:, 1] < end_t))
                rate = spikes_in_bin / (bin_size * 1e-3)
                output_rate_binned.append(rate)
            output_rate_binned = np.array(output_rate_binned)

            # Average z_slow over bins
            z_slow_binned = []
            for i in range(n_bins):
                start_idx = i * bin_size
                end_idx = (i + 1) * bin_size
                z_avg = np.mean(z_slow_array[start_idx:end_idx], axis=0)
                z_slow_binned.append(z_avg)
            z_slow_binned = np.array(z_slow_binned)

            # Weight analysis
            weights_final = self.connections[config_idx].weight.detach().cpu().numpy().squeeze()
            weights_initial = self.weights_initial[config_idx]
            weights_change = weights_final - weights_initial

            # Output rate
            output_rate = len(spikes_output) / (len(self.slow_trace) * 1e-3)

            # Correlations
            corr_slowness_final = np.corrcoef(self.slowness_score, weights_final)[0, 1]
            corr_slowness_change = np.corrcoef(self.slowness_score, weights_change)[0, 1]

            # Output-signal correlations
            output_signal_corrs = []
            for freq_idx in range(self.n_signals):
                if len(output_rate_binned) > 1:
                    corr = np.corrcoef(output_rate_binned, z_slow_binned[:, freq_idx])[0, 1]
                    output_signal_corrs.append(corr)
                else:
                    output_signal_corrs.append(0.0)

            dominant_freq_idx = np.argmax(np.abs(output_signal_corrs))

            # Weight changes by frequency
            dominant_freq_assignment = np.argmax(self.mod_weights, axis=1)
            weight_by_freq = []
            for freq_idx in range(self.n_signals):
                mask = dominant_freq_assignment == freq_idx
                if mask.any():
                    weight_by_freq.append(weights_final[mask].mean())
                else:
                    weight_by_freq.append(0.0)

            # Determine success
            slowest_idx = 0
            output_follows_slowest = dominant_freq_idx == slowest_idx
            weights_prefer_slowest = (weight_by_freq[0] > weight_by_freq[2] if len(weight_by_freq) > 2 else False)

            # Store results
            result = {
                "name": config["name"],
                "corr_slowness_final": corr_slowness_final,
                "corr_slowness_change": corr_slowness_change,
                "output_rate": output_rate,
                "output_signal_corrs": output_signal_corrs,
                "dominant_freq": self.slow_freqs[dominant_freq_idx],
                "output_follows_slowest": output_follows_slowest,
                "weights_prefer_slowest": weights_prefer_slowest,
                "weight_by_freq": weight_by_freq,
                "weights_change_mean": weights_change.mean(),
                "weights_final": weights_final,
            }
            results.append(result)

            # Print summary for this config
            print(f"\nOutput rate: {output_rate:.1f} Hz")
            print(f"Corr(slowness, weights): {corr_slowness_final:+.3f}")
            print(f"\nOutput correlations with signals:")
            for freq_idx, (freq, corr) in enumerate(zip(self.slow_freqs, output_signal_corrs)):
                marker = "★" if freq_idx == dominant_freq_idx else " "
                print(f"  {marker} {freq:.2f} Hz: {corr:+.3f}")

            print(f"\nWeight distribution by frequency:")
            for freq_idx, (freq, w) in enumerate(zip(self.slow_freqs, weight_by_freq)):
                if w > 0:
                    print(f"  {freq:.2f} Hz: {w:.2e}")

            # Verdict
            if output_follows_slowest and weights_prefer_slowest:
                print(f"\n✓✓ SUCCESS")
            elif output_follows_slowest:
                print(f"\n⚠ PARTIAL: Output correct, weights ambiguous")
            elif weights_prefer_slowest:
                print(f"\n⚠ PARTIAL: Weights correct, output wrong")
            else:
                print(f"\n✗ FAILED")

        # Comparison table
        print(f"\n{'='*70}")
        print("COMPARATIVE SUMMARY")
        print(f"{'='*70}\n")

        print(f"{'Configuration':<25} {'Corr(slow,w)':<15} {'Follows 0.5Hz':<15} {'Status':<10}")
        print("-"*70)

        for r in results:
            corr_str = f"{r['corr_slowness_final']:+.3f}"
            follows = "Yes" if r['output_follows_slowest'] else "No"
            if r['output_follows_slowest'] and r['weights_prefer_slowest']:
                status = "✓✓"
            elif r['output_follows_slowest'] or r['weights_prefer_slowest']:
                status = "⚠"
            else:
                status = "✗"

            print(f"{r['name']:<25} {corr_str:<15} {follows:<15} {status:<10}")

        # Visualization
        self._create_comparison_plots(results)

        print(f"\n{'='*70}")
        print("WINNER")
        print(f"{'='*70}")

        # Find best configuration
        scores = []
        for r in results:
            score = 0
            if r['output_follows_slowest']:
                score += 2
            if r['weights_prefer_slowest']:
                score += 2
            if r['corr_slowness_final'] > 0.7:
                score += 1
            scores.append(score)

        best_idx = np.argmax(scores)
        print(f"\nBest configuration: {results[best_idx]['name']}")
        print(f"  Correlation: {results[best_idx]['corr_slowness_final']:+.3f}")
        print(f"  Output follows slowest: {'Yes' if results[best_idx]['output_follows_slowest'] else 'No'}")
        print(f"  Weights prefer slowest: {'Yes' if results[best_idx]['weights_prefer_slowest'] else 'No'}")
        print(f"{'='*70}\n")

    def _create_comparison_plots(self, results):
        """Create comparative visualization plots."""
        n_configs = len(results)

        # Plot 1: Correlation comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Correlations bar chart
        ax = axes[0]
        names = [r['name'] for r in results]
        corrs = [r['corr_slowness_final'] for r in results]
        colors = ['green' if c > 0.7 else 'orange' if c > 0.3 else 'red' for c in corrs]

        bars = ax.bar(range(n_configs), corrs, color=colors, alpha=0.7, edgecolor='black')
        ax.set_xticks(range(n_configs))
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylabel('Correlation(slowness, weights)')
        ax.set_title('SFA Learning Quality by Configuration')
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax.axhline(0.7, color='g', linestyle='--', alpha=0.3, label='Strong (>0.7)')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()

        # Output correlations heatmap
        ax = axes[1]
        corr_matrix = np.array([r['output_signal_corrs'] for r in results])
        im = ax.imshow(corr_matrix, aspect='auto', cmap='RdYlGn', vmin=-1, vmax=1)

        ax.set_xticks(range(self.n_signals))
        ax.set_xticklabels([f'{f:.1f}' for f in self.slow_freqs], rotation=45)
        ax.set_yticks(range(n_configs))
        ax.set_yticklabels(names)
        ax.set_xlabel('Signal Frequency (Hz)')
        ax.set_title('Output Correlation with Each Signal')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation')

        # Add text annotations
        for i in range(n_configs):
            for j in range(self.n_signals):
                text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)

        plt.tight_layout()
        plt.show()

        # Plot 2: Weight distributions comparison
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for idx, (ax, r) in enumerate(zip(axes, results)):
            if idx < n_configs:
                # Scatter: slowness vs final weights
                dominant_freq_idx = np.argmax(self.mod_weights, axis=1)
                colors_map = {0: 'blue', 1: 'cyan', 2: 'orange', 3: 'red', 4: 'darkred'}

                for freq_idx in range(min(self.n_signals, 5)):
                    mask = dominant_freq_idx == freq_idx
                    if mask.any():
                        ax.scatter(self.slowness_score[mask], r['weights_final'][mask],
                                 alpha=0.6, s=30, c=colors_map.get(freq_idx, 'gray'),
                                 label=f'{self.slow_freqs[freq_idx]:.1f} Hz')

                # Linear fit
                z = np.polyfit(self.slowness_score, r['weights_final'], 1)
                p = np.poly1d(z)
                x_fit = np.linspace(self.slowness_score.min(), self.slowness_score.max(), 100)
                ax.plot(x_fit, p(x_fit), 'k--', lw=2, alpha=0.6)

                ax.set_xlabel('Slowness Score')
                ax.set_ylabel('Final Weight')
                ax.set_title(f"{r['name']}\nr = {r['corr_slowness_final']:+.3f}")
                ax.legend(fontsize=8, loc='best')
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    print("="*70)
    print("SFA Comprehensive Comparison")
    print("="*70)
    print("\nComparing multiple SFA configurations side-by-side:")
    print("  1. Surrogate (tanh) + Oja")
    print("  2. HPF (no Oja)")
    print("  3. Surrogate (sigmoid) + Oja")
    print("  4. HPF + Oja")
    print("\nAll configurations use the same input stimuli for fair comparison.")
    print("="*70 + "\n")

    exp = SFAComparisonExperiment(sim=Simulator(seed=0))
    simulation_time = 100.0
    exp.run(steps=int(1000 * simulation_time))
