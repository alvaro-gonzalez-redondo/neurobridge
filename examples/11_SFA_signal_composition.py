"""
SFA Signal Composition Example
================================

Demonstrates the new signal composition capabilities for flexible plasticity rules.

This example shows:
1. Manual composition of plasticity rules
2. Using MultiplySignals for reward-modulated learning
3. Using AddSignals for multi-objective learning
4. Dynamically switching between different rules with set_plasticity

Key concepts:
-------------
- **Unsupervised SFA**: Pure temporal learning (no reward)
- **RL-SFA**: Reward-modulated temporal learning
- **Multi-objective**: Combining multiple learning objectives

Architecture:
-------------
- Input: 32 Poisson neurons tuned to 3 signals (0.5, 1.3, 3.4 Hz)
- Output: 3 neurons, each with different plasticity rule:
  1. Unsupervised SFA (Lipshutz voltage only)
  2. RL-SFA (Lipshutz × reward)
  3. Multi-objective (RL-SFA + homeostasis)
"""

from neurobridge import *
import torch
import matplotlib.pyplot as plt
import numpy as np

# Import plasticity components directly for manual composition
from neurobridge.plasticity.rule import PlasticityRule
from neurobridge.plasticity.base import LearningSignalBase
from neurobridge.plasticity.eligibility.lipshutz import EligibilityLipshutzVoltageDense
from neurobridge.plasticity.signals.lipshutz import SignalLipshutzVoltage
from neurobridge.plasticity.signals.constant import SignalConstant
from neurobridge.plasticity.signals.composite import MultiplySignals, AddSignals
from neurobridge.plasticity.updates.oja import UpdateOjaDense


class DummyRewardSignal(LearningSignalBase):
    """Dummy reward signal for demonstration.

    In a real RL scenario, this would come from task performance.
    Here we use a simple sinusoidal reward that correlates with the slowest signal.
    """

    def __init__(self, frequency: float = 0.5, amplitude: float = 1.0):
        self.frequency = frequency
        self.amplitude = amplitude
        self.step_count = 0

    def bind(self, conn):
        self.step_count = 0

    def step(self, conn):
        # Simple sinusoidal reward
        t = self.step_count * 1e-3
        self.step_count += 1

        reward = self.amplitude * (1.0 + np.sin(2 * np.pi * self.frequency * t)) / 2.0
        return torch.full((conn.pos.size,), reward, dtype=torch.float32, device=conn.device)


class SignalCompositionExperiment(Experiment):
    """Compare different signal composition strategies."""

    n_input: int = 32
    n_output_per_rule: int = 1
    base_rate: float = 20.0
    mod_amp: float = 15.0

    def build_network(self):
        np.random.seed(42)

        # Three signals with golden ratio spacing
        phi = (1 + 5**0.5) / 2
        self.slow_freqs = np.array([0.5, 0.5 * phi**2, 0.5 * phi**4])
        # ≈ [0.5, 1.31, 3.43] Hz
        self.n_signals = len(self.slow_freqs)

        # Each input neuron tuned to mixture of signals
        self.mod_weights = np.random.dirichlet([1.0] * self.n_signals, self.n_input)

        # Slowness score for analysis
        slowness_weights = 1.0 / self.slow_freqs
        slowness_weights /= slowness_weights.sum()
        self.slowness_score = self.mod_weights @ slowness_weights

        # Shared input layer
        with self.sim.autoparent("normal"):
            self.input_neurons = RandomSpikeNeurons(
                n_neurons=self.n_input,
                firing_rate=self.base_rate
            )

        # Build three output neurons with different plasticity rules
        self.output_neurons = []
        self.connections = []
        self.rule_names = []

        # Common parameters
        common_params = {
            "tau_slow_pre": 2500e-3,
            "tau_slow_post": 2500e-3,
            "v_rest": -65e-3,
            "v_scale": 30e-3,
            "eta": 1e-5,
            "beta": 0.0,
            "w_min": 0.0,
            "w_max": 1e-1,
        }

        with self.sim.autoparent("normal"):
            # Rule 1: Unsupervised SFA (pure temporal learning)
            rule_unsup = PlasticityRule(
                eligibility = EligibilityLipshutzVoltageDense(
                    tau_slow_pre=common_params["tau_slow_pre"]
                ),
                signal = SignalLipshutzVoltage(
                    tau_slow=common_params["tau_slow_post"],
                    v_rest=common_params["v_rest"],
                    v_scale=common_params["v_scale"]
                ),
                update=UpdateOjaDense(
                    eta=common_params["eta"],
                    beta=common_params["beta"],
                    w_min=common_params["w_min"],
                    w_max=common_params["w_max"]
                )
            )

            output1 = IFNeurons(n_neurons=self.n_output_per_rule)
            self.output_neurons.append(output1)
            conn1 = self.sim.connect(
                self.input_neurons, output1, PlasticDense,
                pattern="all-to-all",
                weight=5e-3/self.n_input,
                delay=1,
                plasticity=rule_unsup
            )
            self.connections.append(conn1)
            self.rule_names.append("Unsupervised SFA")

            # Rule 2: RL-SFA (reward-modulated temporal learning)
            rule_rl = PlasticityRule(
                eligibility = EligibilityLipshutzVoltageDense(
                    tau_slow_pre=common_params["tau_slow_pre"]
                ),
                signal = MultiplySignals([
                    SignalLipshutzVoltage(
                        tau_slow=common_params["tau_slow_post"],
                        v_rest=common_params["v_rest"],
                        v_scale=common_params["v_scale"]
                    ),
                    DummyRewardSignal(frequency=0.5, amplitude=1.0)
                ]),
                update = UpdateOjaDense(
                    eta=common_params["eta"],
                    beta=common_params["beta"],
                    w_min=common_params["w_min"],
                    w_max=common_params["w_max"]
                )
            )

            output2 = IFNeurons(n_neurons=self.n_output_per_rule)
            self.output_neurons.append(output2)
            conn2 = self.sim.connect(
                self.input_neurons, output2, PlasticDense,
                pattern="all-to-all",
                weight=5e-3/self.n_input,
                delay=1,
                plasticity=rule_rl
            )
            self.connections.append(conn2)
            self.rule_names.append("RL-SFA (Temporal × Reward)")

            # Rule 3: Multi-objective (RL-SFA + constant homeostasis term)
            # Note: Using SignalConstant with small weight to simulate homeostasis
            rule_multi = PlasticityRule(
                eligibility = EligibilityLipshutzVoltageDense(
                    tau_slow_pre=common_params["tau_slow_pre"]
                ),
                signal = AddSignals(
                    [
                        MultiplySignals([
                            SignalLipshutzVoltage(
                                tau_slow=common_params["tau_slow_post"],
                                v_rest=common_params["v_rest"],
                                v_scale=common_params["v_scale"]
                            ),
                            DummyRewardSignal(frequency=0.5, amplitude=1.0)
                        ]),
                        SignalConstant()  # Constant homeostatic term
                    ], 
                    weights=[1.0, 0.05] # 95% RL, 5% constant drive,
                ),
                update = UpdateOjaDense(
                    eta=common_params["eta"],
                    beta=common_params["beta"],
                    w_min=common_params["w_min"],
                    w_max=common_params["w_max"]
                )
            )

            output3 = IFNeurons(n_neurons=self.n_output_per_rule)
            self.output_neurons.append(output3)
            conn3 = self.sim.connect(
                self.input_neurons, output3, PlasticDense,
                pattern="all-to-all",
                weight=5e-3/self.n_input,
                delay=1,
                plasticity=rule_multi
            )
            self.connections.append(conn3)
            self.rule_names.append("Multi-objective (RL + Homeostasis)")

        # Monitors
        with self.sim.autoparent("normal"):
            self.spikes = SpikeMonitor(self.output_neurons)
            self.weights = VariableMonitor(self.connections, ["weight"])

        # Store initial weights
        self.weights_initial = []

    def pre_step(self):
        """Generate multi-frequency signals and modulate input firing rates."""
        # Save initial weights on first step
        if self.step == 0:
            for conn in self.connections:
                self.weights_initial.append(
                    conn.weight.detach().cpu().numpy().copy().squeeze()
                )

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
        print("SIGNAL COMPOSITION COMPARISON")
        print("="*70)

        for idx, (conn, name, w_init) in enumerate(
            zip(self.connections, self.rule_names, self.weights_initial)
        ):
            print(f"\n{idx+1}. {name}")
            print("-" * 70)

            # Get final weights
            w_final = conn.weight.detach().cpu().numpy().squeeze()
            w_change = w_final - w_init

            # Correlation with slowness
            corr_slowness = np.corrcoef(self.slowness_score, w_final)[0, 1]

            print(f"Weight change: mean={w_change.mean():.2e}, std={w_change.std():.2e}")
            print(f"Correlation(slowness, weights): {corr_slowness:+.3f}")

            if corr_slowness > 0.5:
                print("  → Strong positive correlation (good slowness extraction)")
            elif corr_slowness > 0.2:
                print("  → Moderate positive correlation")
            else:
                print("  → Weak correlation")

        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print("\nThis example demonstrates:")
        print("  ✓ Manual composition of plasticity rules")
        print("  ✓ MultiplySignals for reward modulation")
        print("  ✓ AddSignals for multi-objective learning")
        print("  ✓ Flexible experimentation without touching factories")
        print("\nKey advantage: Easy transition from unsupervised → RL")
        print("Just change: SignalLipshutzVoltage()")
        print("         to: MultiplySignals([SignalLipshutzVoltage(), SignalReward()])")
        print("="*70 + "\n")

        # Create visualization
        self._create_plots()

    def _create_plots(self):
        """Create comparison plots."""
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        time_signals = np.arange(len(self.slow_trace)) * 1e-3
        z_slow_array = np.array(self.slow_trace)

        # Plot 1: Input signals
        ax = axes[0]
        colors = ['blue', 'cyan', 'green']
        for idx, (freq, color) in enumerate(zip(self.slow_freqs, colors)):
            ax.plot(time_signals, z_slow_array[:, idx],
                   color=color, lw=2, alpha=0.7, label=f'{freq:.2f} Hz')
        ax.set_ylabel('Signal Amplitude')
        ax.set_title('Input Signals (Slowest = 0.5 Hz)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Weight evolution for each rule
        ax = axes[1]
        for idx, (name, conn) in enumerate(zip(self.rule_names, self.connections)):
            weights_evo = self.weights.get_variable_tensor(idx, "weight").cpu().numpy()
            time_weights = np.arange(len(weights_evo)) * 1e-3

            # weights_evo has shape (timesteps, num_weights)
            # Take mean across all weights for each timestep
            if weights_evo.ndim == 3:
                mean_weights = weights_evo.mean(axis=(1, 2))
            else:  # ndim == 2
                mean_weights = weights_evo.mean(axis=1)

            ax.plot(time_weights, mean_weights, lw=2, label=name)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Mean Synaptic Weight')
        ax.set_title('Weight Evolution: Comparing Signal Composition Strategies')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Final weights vs slowness score
        ax = axes[2]
        colors_scatter = ['blue', 'red', 'green']
        for idx, (name, conn, w_init, color) in enumerate(
            zip(self.rule_names, self.connections, self.weights_initial, colors_scatter)
        ):
            w_final = conn.weight.detach().cpu().numpy().squeeze()
            ax.scatter(self.slowness_score, w_final,
                      alpha=0.6, s=50, c=color, label=name)

        ax.set_xlabel('Slowness Score (higher = tuned to slower signals)')
        ax.set_ylabel('Final Synaptic Weight')
        ax.set_title('Final Weights vs Slowness: Signal Composition Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    print("="*70)
    print("SFA Signal Composition Tutorial")
    print("="*70)
    print("\nThis example demonstrates the new signal composition capabilities:")
    print("  1. Unsupervised SFA (pure temporal learning)")
    print("  2. RL-SFA (temporal × reward modulation)")
    print("  3. Multi-objective (RL-SFA + homeostasis)")
    print("\nAll rules are manually composed using:")
    print("  - PlasticityRule(eligibility, signal, update)")
    print("  - MultiplySignals([signal1, signal2, ...])")
    print("  - AddSignals([signal1, signal2], weights=[w1, w2])")
    print("="*70 + "\n")

    exp = SignalCompositionExperiment(sim=Simulator(seed=0))
    simulation_time = 50.0  # 50 seconds
    exp.run(steps=int(1000 * simulation_time))
