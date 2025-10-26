"""
SFA Kernel Test - Frequency Response Analysis

This example demonstrates the fundamental property of the SFA learning rule:
it has a temporal bias towards slow components in the input.

Experimental Setup:
- Single pre→post synapse
- Input: sin(2π·1Hz·t) + sin(2π·10Hz·t) with equal variance
- Measure weight change for each frequency component
- Show empirical Bode plot (weight gain vs frequency)

Expected Result:
- The SFA rule should couple more strongly to the 1 Hz component
- Weight changes track slow variations more than fast variations
- This is the core mechanism that enables slow feature extraction

Reference:
    Sprekeler, H., Michaelis, C., & Wiskott, L. (2007).
    Slowness: An objective for spike-timing-dependent plasticity?
    PLoS computational biology, 3(6), e112.
"""

from neurobridge import *
import torch
import matplotlib.pyplot as plt
import numpy as np


class SFAKernelTest(Experiment):
    """Test SFA kernel response to different frequency components."""

    # Test frequencies (Hz)
    freq_slow: float = 1.0   # Slow component
    freq_fast: float = 10.0  # Fast component
    
    pre_neuron: IFNeurons
    post_neuron: IFNeurons

    def build_network(self):
        with self.sim.autoparent("normal"):
            # Single pre and post neuron
            self.pre_neuron = IFNeurons(n_neurons=1)
            self.post_neuron = IFNeurons(n_neurons=1)

            # Single plastic synapse with SFA rule
            self.synapse = self.sim.connect(
                self.pre_neuron, self.post_neuron, PlasticDense,
                pattern="all-to-all",
                weight=1e-3,  # Initial weight
                delay=1,
                plasticity={
                    "name": "sfa",
                    "params": {
                        "tau_fast": 10e-3,      # 10ms
                        "tau_slow": 100e-3,     # 100ms
                        "tau_z": 20e-3,         # 20ms
                        "tau_hpf": 100e-3,      # 100ms
                        "eta": 1e-6,            # Higher learning rate for visibility
                        "beta": 0e-2,           # Oja normalization
                        "w_min": 0.0,
                        "w_max": 1e-3,
                        "dt": 1e-3,
                    }
                }
            )

        # --- Monitor configuration ---
        with self.sim.autoparent("normal"):
            # Monitor weight evolution
            self.weight_monitor = VariableMonitor(
                [self.synapse,],
                ["weight"]
            )

            # Monitor plasticity signals
            self.plasticity_monitor = VariableMonitor(
                [self.synapse,],
                ["last_eligibility", "last_learning_signal"]
            )

    def pre_step(self):
        """Inject synthetic currents at each timestep."""
        # Time in seconds
        t = self.step * 1e-3

        # Generate composite signal: slow + fast with equal variance
        # Normalize so each component has unit variance
        slow_component = np.sin(2 * np.pi * self.freq_slow * t)
        fast_component = np.sin(2 * np.pi * self.freq_fast * t)

        # Composite signal (equal contribution)
        signal = (slow_component + fast_component) / np.sqrt(2)

        # Scale to appropriate current magnitude
        current_amplitude = 1e-2
        current = current_amplitude * signal
        currents = torch.tensor([current], dtype=torch.float32, device=self.pre_neuron.device)

        # Inject to pre-neuron (which will drive post through plastic synapse)
        self.pre_neuron.inject_currents(currents)

        # Store signal for later analysis
        if not hasattr(self, 'signal_history'):
            self.signal_history = []
            self.slow_history = []
            self.fast_history = []

        self.signal_history.append(signal)
        self.slow_history.append(slow_component)
        self.fast_history.append(fast_component)

    def on_finish(self):
        """Analyze and visualize frequency response."""

        # Convert signal history to numpy arrays
        signal = np.array(self.signal_history)
        slow = np.array(self.slow_history)
        fast = np.array(self.fast_history)
        time = np.arange(len(signal)) * 1e-3  # Time in seconds

        # Get weight evolution
        weights = self.weight_monitor.get_variable_tensor(0, "weight").cpu().numpy()
        weights = weights.squeeze()  # Remove extra dimensions

        # Get plasticity signals
        eligibility = self.plasticity_monitor.get_variable_tensor(0, "last_eligibility").cpu().numpy()
        learning_signal = self.plasticity_monitor.get_variable_tensor(0, "last_learning_signal").cpu().numpy()

        # Compute weight changes
        dw = np.diff(weights, prepend=weights[0])

        # --- Analysis: Correlation with each component ---
        # Use last 80% of simulation (after transient)
        start_idx = int(len(signal) * 0.2)

        corr_dw_slow = np.corrcoef(dw[start_idx:], slow[start_idx:])[0, 1]
        corr_dw_fast = np.corrcoef(dw[start_idx:], fast[start_idx:])[0, 1]

        print("\n" + "="*70)
        print("SFA Kernel Test - Frequency Response Analysis")
        print("="*70)
        print(f"\nInput: sin(2π·{self.freq_slow}Hz·t) + sin(2π·{self.freq_fast}Hz·t)")
        print(f"\nCorrelation (weight change vs component):")
        print(f"  Slow component ({self.freq_slow} Hz): {corr_dw_slow:+.4f}")
        print(f"  Fast component ({self.freq_fast} Hz): {corr_dw_fast:+.4f}")
        print(f"\nRatio (slow/fast): {abs(corr_dw_slow/corr_dw_fast):.2f}x")
        print("\n✅ Expected: SFA couples more strongly to slow component")
        print("="*70 + "\n")

        # --- Visualization ---
        fig, axes = plt.subplots(4, 1, figsize=(14, 12))

        # Plot 1: Input signal components
        ax = axes[0]
        ax.plot(time, slow, 'b-', label=f'Slow ({self.freq_slow} Hz)', linewidth=2, alpha=0.7)
        ax.plot(time, fast, 'r-', label=f'Fast ({self.freq_fast} Hz)', linewidth=2, alpha=0.7)
        ax.plot(time, signal, 'k-', label='Composite', linewidth=1, alpha=0.5)
        ax.set_title("Input Signal Components (equal variance)")
        ax.set_ylabel("Amplitude")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, min(5.0, time[-1])])  # Show first 5 seconds

        # Plot 2: Weight evolution
        ax = axes[1]
        ax.plot(time, weights, 'g-', linewidth=2, label='Weight')
        ax.set_title("Weight Evolution")
        ax.set_ylabel("Weight")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, time[-1]])

        # Plot 3: Weight changes vs components
        ax = axes[2]
        ax.plot(time, dw, 'k-', label='Δw', linewidth=1, alpha=0.6)
        # Overlay slow component (scaled for visibility)
        slow_scaled = slow * np.std(dw[start_idx:]) / np.std(slow[start_idx:]) * 3
        ax.plot(time, slow_scaled, 'b--', label=f'Slow (scaled)', linewidth=2, alpha=0.7)
        ax.set_title(f"Weight Changes (correlation with slow: {corr_dw_slow:+.3f}, fast: {corr_dw_fast:+.3f})")
        ax.set_ylabel("Δw")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, min(10.0, time[-1])])  # Show first 10 seconds

        # Plot 4: Power spectral density of weight changes
        ax = axes[3]

        # Compute PSD using FFT
        from scipy import signal as scipy_signal
        freqs, psd = scipy_signal.welch(dw, fs=1000, nperseg=min(2048, len(dw)//4))

        ax.semilogy(freqs, psd, 'g-', linewidth=2)
        ax.axvline(self.freq_slow, color='blue', linestyle='--', linewidth=2,
                   label=f'{self.freq_slow} Hz', alpha=0.7)
        ax.axvline(self.freq_fast, color='red', linestyle='--', linewidth=2,
                   label=f'{self.freq_fast} Hz', alpha=0.7)
        ax.set_title("Power Spectral Density of Weight Changes (empirical Bode plot)")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power (dB)")
        ax.legend()
        ax.grid(True, alpha=0.3, which='both')
        ax.set_xlim([0, 20])

        plt.tight_layout()
        plt.show()


# --- Execution ---
if __name__ == "__main__":
    print("="*70)
    print("SFA Kernel Test - Frequency Response Analysis")
    print("="*70)
    print("\nThis experiment tests the fundamental property of SFA:")
    print("  → Temporal bias towards slow components")
    print("\nSetup:")
    print("  - Single pre→post synapse with SFA rule")
    print("  - Input: sin(1Hz) + sin(10Hz) with equal variance")
    print("  - Measure: weight change correlation with each component")
    print("\nExpected result:")
    print("  - Weight changes should correlate more with 1 Hz than 10 Hz")
    print("  - PSD of Δw should show peak at lower frequencies")
    print("\nRunning test...\n")

    exp = SFAKernelTest(sim=Simulator(seed=0))
    simulation_time = 3.0  # 30 seconds (enough for ~30 slow cycles)
    exp.run(steps=int(1000 * simulation_time))
