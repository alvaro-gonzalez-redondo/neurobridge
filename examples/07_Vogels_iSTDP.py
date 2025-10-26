"""
Vogels Inhibitory STDP (iSTDP)

This example demonstrates Vogels et al. (2011) inhibitory STDP with homeostatic
regulation. The rule maintains balanced network activity by adjusting inhibitory
weights to keep postsynaptic firing rates near a target level.

Key features:
- Plastic inhibitory connections (I→E) using Vogels rule
- Homeostatic regulation: Δw = η · x_pre · (z_post - ρ₀)
- Target firing rate maintenance
- Weight evolution monitoring

Reference:
    Vogels, T. P., Sprekeler, H., Zenke, F., Clopath, C., & Gerstner, W. (2011).
    Inhibitory plasticity balances excitation and inhibition in sensory pathways
    and memory networks. Science, 334(6062), 1569-1573.

Network architecture:
- 100 noise neurons (random spikes at 20Hz)
- 800 excitatory neurons (IF neurons)
- 200 inhibitory neurons (IF neurons)
- Static N→E connections (drive)
- Static E→E connections (recurrent excitation)
- Static E→I connections (activate inhibition)
- Plastic I→E connections (Vogels iSTDP with homeostasis)
"""

from neurobridge import *
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm


class VogelsISTDPExperiment(Experiment):
    n_total_neurons: int = 1_000
    exc_prop: float = 0.8
    conn_prob: float = 0.1
    target_rate: float = 5.0  # Target firing rate in Hz

    def build_network(self):
        n_noise_neurons = 100
        n_excitatory_neurons = int(self.n_total_neurons * self.exc_prop)
        n_inhibitory_neurons = self.n_total_neurons - n_excitatory_neurons

        # --- Network creation ---
        with self.sim.autoparent("normal"):
            noise = RandomSpikeNeurons(n_neurons=n_noise_neurons, firing_rate=20.0)
            exc_neurons = IFNeurons(n_neurons=n_excitatory_neurons)#, tau_refrac=20e-3)
            inh_neurons = IFNeurons(n_neurons=n_inhibitory_neurons)

            # Noise → Excitatory (static drive)
            n2e = self.sim.connect(
                noise, exc_neurons, StaticDense,
                pattern="random", p=self.conn_prob,
                weight=(0, 1e-3),  # Strong drive to maintain activity
                delay=0,
            )

            
            # Excitatory → Inhibitory (static, activate inhibition)
            e2i = self.sim.connect(
                exc_neurons, inh_neurons, StaticDense,
                pattern="random", fanin=10, #p=0.05,
                weight=1e-3, #(0, 1e-3),
                delay=1,
            )


            # Inhibitory → Excitatory (plastic Vogels iSTDP)
            i2e = self.sim.connect(
                inh_neurons, exc_neurons, PlasticDense,
                pattern="random", p=0.2,
                weight=(0, 1e-5),  # Small initial inhibitory weights
                delay=1,
                channel=1,  # GABA (inhibitory)
                plasticity={
                    "name": "vogels",
                    "params": {
                        "tau_pre": 20e-3,       # Presynaptic trace time constant
                        "target_rate": self.target_rate,  # Target firing rate (Hz)
                        "tau_rate": 1.0,        # Rate estimation
                        "eta": 2e-5,            # Learning rate
                        "w_min": 0.0,
                        "w_max": 1e-1,          # Allow stronger inhibition
                        "dt": 1e-3,
                    }
                }
            )

            # Excitatory → Excitatory (static recurrent)
            # AMPA channel (excitatory, fast)
            e2e_ampa = self.sim.connect(
                exc_neurons, exc_neurons, PlasticDense,  # ← Modular framework
                pattern="random", p=self.conn_prob,
                weight=(0, 1e-3),  # Uniform distribution [0, 1e-6]
                delay=5,
                # STDP parameters via modular framework
                # Note: kwargs are passed to ConnectionSpec.params, so we use plasticity={...} directly
                plasticity={
                    "name": "stdp",
                    "params": {
                        "A_plus": 1e-6,       # LTP learning rate
                        "A_minus": -1.2e-6,   # LTD learning rate (20% stronger)
                        "tau_pre": 20e-3,     # Pre-synaptic trace time constant
                        "tau_post": 20e-3,    # Post-synaptic trace time constant
                        "w_min": 0.0,
                        "w_max": 1e-2,
                        "oja_decay": 1e-7,    # ≈ 0.3 × A_plus (balanced normalization)
                    }
                }
            )

            # NMDA channel (excitatory, slow, voltage-dependent)
            e2e_nmda = self.sim.connect(
                exc_neurons, exc_neurons, PlasticDense,  # ← Modular framework
                pattern="random", p=self.conn_prob,
                weight=(0, 1e-4),  # Uniform distribution [0, 1e-6]
                delay=5,
                channel=2,  # Different channel for NMDA
                # STDP parameters via modular framework
                plasticity={
                    "name": "stdp",
                    "params": {
                        "A_plus": 1e-7,       # 4x slower than AMPA
                        "A_minus": -1.2e-7,     # 20% stronger
                        "tau_pre": 20e-3,
                        "tau_post": 20e-3,
                        "w_min": 0.0,
                        "w_max": 1e-2,
                        "oja_decay": 3e-8,  # ≈ 0.3 × A_plus (balanced normalization)
                    }
                }
            )


        # --- Monitor configuration ---
        with self.sim.autoparent("normal"):
            # Monitor spikes from subset of neurons
            self.spike_monitor = SpikeMonitor(
                [
                    noise.where_id(lambda i: i < 50),
                    exc_neurons.where_id(lambda i: i < 100),
                    inh_neurons.where_id(lambda i: i < 50),
                ]
            )

            if False:
                # Monitor membrane potential and currents of first excitatory neuron
                self.state_monitor = VariableMonitor(
                    [exc_neurons.where_id(lambda i: i < 1),],
                    ['V', 'spikes', 'channel_currents@0', 'channel_currents@1', 'channel_currents@2']
                )

                # Monitor weight evolution (E→E AMPA, E→E NMDA, I→E)
                # Set filters for all connections
                e2e_ampa.filter[:, :] = False
                e2e_ampa.filter[:5, :5] = True  # Monitor 5x5 subset

                e2e_nmda.filter[:, :] = False
                e2e_nmda.filter[:5, :5] = True  # Monitor 5x5 subset

                i2e.filter[:, :] = False
                i2e.filter[:5, :5] = True  # Monitor 5x5 subset

                self.weight_monitor = VariableMonitor(
                    [e2e_ampa, e2e_nmda, i2e],
                    ["weight"]
                )

                # Monitor plasticity signals for E→E connections (STDP diagnosis)
                # The neuron indices are automatically deduced from the connection filter
                # (which neurons have True in rows/columns)
                self.stdp_monitor_ampa = VariableMonitor(
                    [e2e_ampa,],
                    ["last_learning_signal", "last_eligibility"]
                )

                self.stdp_monitor_nmda = VariableMonitor(
                    [e2e_nmda,],
                    ["last_learning_signal", "last_eligibility"]
                )

                # Monitor plasticity signals for I→E (Vogels diagnosis)
                self.vogels_monitor = VariableMonitor(
                    [i2e,],
                    ["last_learning_signal", "last_eligibility"]
                )

    def on_finish(self):
        """Visualization after simulation completes."""

        # --- Plot raster plot and firing rates ---
        fig, ax0 = plt.subplots(figsize=(12, 6))
        ax1 = ax0.twinx()
        id_sum = 0

        for idx, label in enumerate(["Noise", "Exc", "Inh"]):
            spikes = self.spike_monitor.get_spike_tensor(idx).cpu()
            spk_steps, spk_neurons = spikes[:, 1], spikes[:, 0]
            spk_times = spk_steps * 1e-3  # Convert to seconds
            ax1.scatter(spk_times, spk_neurons + id_sum, s=1, label=label, c=f"C{idx}")

            n_neurons = int(self.spike_monitor.filters[idx].nonzero(as_tuple=True)[0][-1]) + 1
            id_sum += n_neurons

            # Smooth firing rate
            times, rate = smooth_spikes(spk_steps, n_neurons=n_neurons, to_step=self.step, sigma=0.2)
            ax0.plot(times, rate, c=f"C{idx}", linewidth=2)

        # Add target rate line
        ax0.axhline(y=self.target_rate, color='red', linestyle='--', linewidth=2,
                    label=f'Target rate ({self.target_rate} Hz)', alpha=0.7)

        ax1.legend(loc='upper right')
        ax0.legend(loc='upper left')
        plt.title(f"Vogels iSTDP - Raster plot and firing rates")
        plt.xlabel("Time (s)")
        ax0.set_ylabel("Firing rate (Hz)")
        ax1.set_ylabel("Neuron ID")
        plt.grid(True, alpha=0.3)

        if False:
            # --- Plot membrane potential and currents ---
            fig, ax0 = plt.subplots(figsize=(12, 6))
            ax1 = ax0.twinx()
            ax2 = ax0.twinx()

            V = self.state_monitor.get_variable_tensor(0, 'V')
            ax0.plot(V, color='C0', label='V', linewidth=1.5)
            ax0.set_ylabel("Membrane potential (V)", color='C0')
            ax0.tick_params(axis='y', labelcolor='C0')

            spikes = self.state_monitor.get_variable_tensor(0, 'spikes')
            spike_times = spikes.nonzero(as_tuple=True)[0]
            ax2.vlines(spike_times, ymin=0, ymax=1, color='black', alpha=0.5, linewidth=1)
            ax2.get_yaxis().set_visible(False)

            ampa = self.state_monitor.get_variable_tensor(0, 'channel_currents@0')
            ax1.plot(ampa, color='C1', label='AMPA (exc)', alpha=0.7, linewidth=1.5)
            gaba = self.state_monitor.get_variable_tensor(0, 'channel_currents@1')
            ax1.plot(gaba, color='C2', label='GABA (inh)', alpha=0.7, linewidth=1.5)
            nmda = self.state_monitor.get_variable_tensor(0, 'channel_currents@2')
            ax1.plot(nmda, color='C3', label='NMDA (exc)', alpha=0.7, linewidth=1.5)
            ax1.set_ylabel("Synaptic currents", color='C1')
            ax1.tick_params(axis='y', labelcolor='C1')
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='upper right')

            plt.title("Membrane potential and synaptic currents (first excitatory neuron)")
            plt.xlabel("Time (steps)")
            plt.tight_layout()

            # --- Plot weight evolution (AMPA, NMDA, Inhibitory) ---
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

            # Get weight data
            w_ampa = self.weight_monitor.get_variable_tensor(0, "weight")  # E→E AMPA
            w_nmda = self.weight_monitor.get_variable_tensor(1, "weight")  # E→E NMDA
            w_inh = self.weight_monitor.get_variable_tensor(2, "weight")   # I→E

            steps = torch.arange(len(w_ampa))

            # AMPA weights (E→E)
            mean_ampa = w_ampa.mean(dim=(1, 2))
            std_ampa = w_ampa.std(dim=(1, 2))
            ax1.plot(steps, mean_ampa, 'C1-', linewidth=2, label='AMPA mean')
            ax1.fill_between(steps, mean_ampa - std_ampa, mean_ampa + std_ampa,
                            color='C1', alpha=0.3, label='±1 std')
            ax1.set_title("E→E AMPA weights (STDP)")
            ax1.set_ylabel("Weight")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # NMDA weights (E→E)
            mean_nmda = w_nmda.mean(dim=(1, 2))
            std_nmda = w_nmda.std(dim=(1, 2))
            ax2.plot(steps, mean_nmda, 'C3-', linewidth=2, label='NMDA mean')
            ax2.fill_between(steps, mean_nmda - std_nmda, mean_nmda + std_nmda,
                            color='C3', alpha=0.3, label='±1 std')
            ax2.set_title("E→E NMDA weights (STDP)")
            ax2.set_ylabel("Weight")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Inhibitory weights (I→E)
            mean_inh = w_inh.mean(dim=(1, 2))
            std_inh = w_inh.std(dim=(1, 2))
            ax3.plot(steps, mean_inh, 'C2-', linewidth=2, label='Inhibitory mean')
            ax3.fill_between(steps, mean_inh - std_inh, mean_inh + std_inh,
                            color='C2', alpha=0.3, label='±1 std')
            ax3.set_title("I→E weights (Vogels iSTDP)")
            ax3.set_ylabel("Weight")
            ax3.set_xlabel("Time (steps)")
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # Comparison plot (all three)
            ax4.plot(steps, mean_ampa, 'C1-', linewidth=2, label='AMPA', alpha=0.8)
            ax4.plot(steps, mean_nmda, 'C3-', linewidth=2, label='NMDA', alpha=0.8)
            ax4.plot(steps, mean_inh, 'C2-', linewidth=2, label='Inhibitory', alpha=0.8)
            ax4.set_title("Weight evolution comparison")
            ax4.set_ylabel("Weight")
            ax4.set_xlabel("Time (steps)")
            ax4.legend()
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()

            # --- Plot Vogels plasticity signals (I→E) ---
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

            # Learning signal (homeostatic: z_post - rho_target)
            learning_signal = self.vogels_monitor.get_variable_tensor(0, "last_learning_signal")
            # Note: shape (time, num_post_filtered) - filtered by connection filter columns
            num_post = learning_signal.shape[1]
            for i in range(min(5, num_post)):
                ax1.plot(learning_signal[:, i], alpha=0.7, linewidth=1, label=f'Exc {i}')
            ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            ax1.set_title(f"Vogels Learning Signal L' = z_post - ρ₀ (first 5 of {num_post} monitored)")
            ax1.set_ylabel("Learning Signal")
            ax1.legend(loc='upper right', fontsize=8)
            ax1.grid(True, alpha=0.3)

            # Eligibility (presynaptic traces)
            eligibility = self.vogels_monitor.get_variable_tensor(0, "last_eligibility")
            # Note: shape (time, 1, num_pre_filtered) - filtered by connection filter rows
            eligibility = eligibility[:, 0, :]  # Extract first (and only) component
            num_pre = eligibility.shape[1]
            for i in range(min(5, num_pre)):
                ax2.plot(eligibility[:, i], alpha=0.7, linewidth=1, label=f'Inh {i}')
            ax2.set_title(f"Vogels Eligibility x_pre (first 5 of {num_pre} monitored)")
            ax2.set_ylabel("Eligibility")
            ax2.set_xlabel("Time (steps)")
            ax2.legend(loc='upper right', fontsize=8)
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            # --- Plot STDP balance analysis (E→E connections) ---
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

            # AMPA STDP signals
            L_ampa = self.stdp_monitor_ampa.get_variable_tensor(0, "last_learning_signal")
            e_ampa = self.stdp_monitor_ampa.get_variable_tensor(0, "last_eligibility")

            # Plot AMPA learning signal (postsynaptic spikes)
            num_post_ampa = L_ampa.shape[1]
            for i in range(min(5, num_post_ampa)):
                ax1.plot(L_ampa[:, i], alpha=0.7, linewidth=1, label=f'Neuron {i}')
            ax1.set_title(f"AMPA Learning Signal (first 5 of {num_post_ampa} monitored)")
            ax1.set_ylabel("L' (spike indicator)")
            ax1.legend(loc='upper right', fontsize=8)
            ax1.grid(True, alpha=0.3)

            # Plot AMPA eligibility traces
            # Note: e_ampa has shape (time, 2, num_filtered_neurons)
            num_neurons_ampa = e_ampa.shape[2]
            for i in range(min(3, num_neurons_ampa)):
                ax2.plot(e_ampa[:, 0, i], alpha=0.7, linewidth=1, label=f'x_pre[{i}]')  # Pre trace
            for i in range(min(3, num_neurons_ampa)):
                ax2.plot(e_ampa[:, 1, i], alpha=0.7, linewidth=1, linestyle='--', label=f'x_post[{i}]')  # Post trace
            ax2.set_title(f"AMPA Eligibility (first 3 of {num_neurons_ampa} monitored)")
            ax2.set_ylabel("Trace value")
            ax2.legend(loc='upper right', fontsize=6, ncol=2)
            ax2.grid(True, alpha=0.3)

            # NMDA STDP signals
            L_nmda = self.stdp_monitor_nmda.get_variable_tensor(0, "last_learning_signal")
            e_nmda = self.stdp_monitor_nmda.get_variable_tensor(0, "last_eligibility")

            # Plot NMDA learning signal
            num_post_nmda = L_nmda.shape[1]
            for i in range(min(5, num_post_nmda)):
                ax3.plot(L_nmda[:, i], alpha=0.7, linewidth=1, label=f'Neuron {i}')
            ax3.set_title(f"NMDA Learning Signal (first 5 of {num_post_nmda} monitored)")
            ax3.set_ylabel("L' (spike indicator)")
            ax3.set_xlabel("Time (steps)")
            ax3.legend(loc='upper right', fontsize=8)
            ax3.grid(True, alpha=0.3)

            # Plot NMDA eligibility traces
            # Note: e_nmda has shape (time, 2, num_filtered_neurons)
            num_neurons_nmda = e_nmda.shape[2]
            for i in range(min(3, num_neurons_nmda)):
                ax4.plot(e_nmda[:, 0, i], alpha=0.7, linewidth=1, label=f'x_pre[{i}]')
            for i in range(min(3, num_neurons_nmda)):
                ax4.plot(e_nmda[:, 1, i], alpha=0.7, linewidth=1, linestyle='--', label=f'x_post[{i}]')
            ax4.set_title(f"NMDA Eligibility (first 3 of {num_neurons_nmda} monitored)")
            ax4.set_ylabel("Trace value")
            ax4.set_xlabel("Time (steps)")
            ax4.legend(loc='upper right', fontsize=6, ncol=2)
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()

            # --- STDP/Oja balance diagnostic (print to console) ---
            print("\n" + "="*70)
            print("STDP/Oja Balance Analysis (last 1000 steps)")
            print("="*70)

            # For diagnosis, we need to compute the balance ratio
            # Ideally: |Oja normalization| ≈ 0.3-0.5 × |LTP + LTD|

            # AMPA analysis
            w_ampa_final = w_ampa[-1000:].mean(dim=0)  # Average over last 1000 steps
            L_ampa_mean = L_ampa[-1000:].mean().item()
            e_pre_ampa = e_ampa[-1000:, 0, :].mean().item()   # Extract x_pre component
            e_post_ampa = e_ampa[-1000:, 1, :].mean().item()  # Extract x_post component

            print(f"\nAMPA (E→E):")
            print(f"  Mean weight: {w_ampa_final.mean():.6e}")
            print(f"  Mean L' (post spikes): {L_ampa_mean:.6f}")
            print(f"  Mean x_pre: {e_pre_ampa:.6f}")
            print(f"  Mean x_post: {e_post_ampa:.6f}")
            print(f"  Estimated LTP term: {2e-4 * e_pre_ampa * L_ampa_mean:.6e}")
            print(f"  Estimated Oja term: {6e-5 * (e_post_ampa**2) * w_ampa_final.mean():.6e}")
            oja_ratio_ampa = (6e-5 * (e_post_ampa**2) * w_ampa_final.mean()) / (2e-4 * e_pre_ampa * L_ampa_mean + 1e-12)
            print(f"  Oja/STDP ratio: {oja_ratio_ampa:.3f} (ideal: 0.3-0.5)")

            # NMDA analysis
            w_nmda_final = w_nmda[-1000:].mean(dim=0)
            L_nmda_mean = L_nmda[-1000:].mean().item()
            e_pre_nmda = e_nmda[-1000:, 0, :].mean().item()   # Extract x_pre component
            e_post_nmda = e_nmda[-1000:, 1, :].mean().item()  # Extract x_post component

            print(f"\nNMDA (E→E):")
            print(f"  Mean weight: {w_nmda_final.mean():.6e}")
            print(f"  Mean L' (post spikes): {L_nmda_mean:.6f}")
            print(f"  Mean x_pre: {e_pre_nmda:.6f}")
            print(f"  Mean x_post: {e_post_nmda:.6f}")
            print(f"  Estimated LTP term: {5e-5 * e_pre_nmda * L_nmda_mean:.6e}")
            print(f"  Estimated Oja term: {1.5e-5 * (e_post_nmda**2) * w_nmda_final.mean():.6e}")
            oja_ratio_nmda = (1.5e-5 * (e_post_nmda**2) * w_nmda_final.mean()) / (5e-5 * e_pre_nmda * L_nmda_mean + 1e-12)
            print(f"  Oja/STDP ratio: {oja_ratio_nmda:.3f} (ideal: 0.3-0.5)")

            print("\n" + "="*70)
            print("Interpretation:")
            print("  - Ratio < 0.1: Oja too weak → weights may grow unbounded")
            print("  - Ratio 0.3-0.5: Well balanced (ideal)")
            print("  - Ratio > 1.0: Oja too strong → weights collapse to w_min")
            print("="*70 + "\n")

        plt.show()


# --- Execution ---
if __name__ == "__main__":
    print("="*70)
    print("Vogels Inhibitory STDP (iSTDP)")
    print("="*70)
    print("\nThis example demonstrates homeostatic inhibitory plasticity.")
    print("The Vogels rule adjusts inhibitory weights to maintain")
    print(f"postsynaptic firing rates near the target ({VogelsISTDPExperiment.target_rate} Hz).")
    print("\nKey features:")
    print("  - Plastic I→E connections with Vogels iSTDP")
    print("  - Homeostatic regulation: Δw = η · x_pre · (z_post - ρ₀)")
    print("  - Weight evolution monitoring")
    print("  - Firing rate regulation towards target")
    print("\nRunning simulation...\n")

    exp = VogelsISTDPExperiment(sim=Simulator(seed=0))
    simulation_time = 20.0  # seconds (longer to see homeostasis)
    exp.run(steps=int(1000 * simulation_time))
