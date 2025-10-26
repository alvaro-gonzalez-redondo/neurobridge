"""
Balanced Random Network with Modular STDP

This example demonstrates the modular e-prop plasticity framework by implementing
a balanced random network with STDP learning rules.

Key differences from 04_BRN_STDP.py:
1. **Modular plasticity**: Uses PlasticDense instead of STDPDense (legacy)
   - STDP parameters passed via plasticity={"name": "stdp", "params": {...}}
   - Framework: eligibility + learning_signal + update_policy

2. **Explicit API**: Uses sim.connect() instead of >> operator
   - More readable: sim.connect(pre, post, ConnectionType, ...)
   - Follows standard patterns from NEST, PyNN, Brian2

Network architecture:
- 100 noise neurons (random spikes at 5Hz)
- 800 excitatory neurons (IF neurons)
- 200 inhibitory neurons (IF neurons)
- Plastic E→E connections (AMPA and NMDA channels)
- Static N→E, E→I, I→E connections
"""

from neurobridge import *
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm


class BalancedRandomNetworkModularExperiment(Experiment):
    n_total_neurons: int = 1_000
    exc_prop: float = 0.8
    conn_prob: float = 0.1

    def build_network(self):
        n_noise_neurons = 100
        n_excitatory_neurons = int(self.n_total_neurons * self.exc_prop)
        n_inhibitory_neurons = self.n_total_neurons - n_excitatory_neurons

        # --- Network creation ---
        with self.sim.autoparent("normal"):
            noise = RandomSpikeNeurons(n_neurons=n_noise_neurons, firing_rate=5.0)
            exc_neurons = IFNeurons(n_neurons=n_excitatory_neurons)
            inh_neurons = IFNeurons(n_neurons=n_inhibitory_neurons)

            # Noise → Excitatory (static)
            if True:
                n2e = self.sim.connect(
                    noise, exc_neurons, StaticDense,
                    pattern="random", p=self.conn_prob,
                    weight=(0, 2e-4),  # Uniform distribution [0, 2e-4]
                    delay=0,
                )

            # Excitatory → Excitatory (plastic STDP)
            if True:
                # AMPA channel (excitatory, fast)
                e2e_ampa = self.sim.connect(
                    exc_neurons, exc_neurons, PlasticDense,  # ← Modular framework
                    pattern="random", p=self.conn_prob,
                    weight=(0, 1e-6),  # Uniform distribution [0, 1e-6]
                    delay=5,
                    # STDP parameters via modular framework
                    # Note: kwargs are passed to ConnectionSpec.params, so we use plasticity={...} directly
                    plasticity={
                        "name": "stdp",
                        "params": {
                            "A_plus": 1e-8,       # LTP learning rate
                            "A_minus": -1.2e-8,   # LTD learning rate
                            "tau_pre": 20e-3,     # Pre-synaptic trace time constant
                            "tau_post": 20e-3,    # Post-synaptic trace time constant
                            "w_min": 0.0,
                            "w_max": 1e-6,
                            "oja_decay": 3e-3,    # Homeostatic normalization
                        }
                    }
                )

                # NMDA channel (excitatory, slow, voltage-dependent)
                e2e_nmda = self.sim.connect(
                    exc_neurons, exc_neurons, PlasticDense,  # ← Modular framework
                    pattern="random", p=self.conn_prob,
                    weight=(0, 1e-6),  # Uniform distribution [0, 1e-6]
                    delay=5,
                    channel=2,  # Different channel for NMDA
                    # STDP parameters via modular framework
                    plasticity={
                        "name": "stdp",
                        "params": {
                            "A_plus": 1e-8,
                            "A_minus": -1.2e-8,
                            "tau_pre": 20e-3,
                            "tau_post": 20e-3,
                            "w_min": 0.0,
                            "w_max": 1e-6,
                            "oja_decay": 3e-3,
                        }
                    }
                )

            # Excitatory → Inhibitory (static)
            if True:
                e2i = self.sim.connect(
                    exc_neurons, inh_neurons, StaticDense,
                    pattern="random", p=self.conn_prob,
                    weight=(0, 0.075e-4),  # Uniform distribution [0, 0.075e-4]
                    delay=0,
                )

            # Inhibitory → Excitatory (static, GABA channel)
            if True:
                i2e = self.sim.connect(
                    inh_neurons, exc_neurons, StaticDense,
                    pattern="random", p=self.conn_prob,
                    weight=(0, 1e-4),  # Uniform distribution [0, 1e-4]
                    delay=0,
                    channel=1,  # GABA (inhibitory)
                )

            # Inhibitory → Inhibitory (disabled)
            if False:
                i2i = self.sim.connect(
                    inh_neurons, inh_neurons, StaticDense,
                    pattern="random", p=self.conn_prob,
                    weight=(0, 1e-5),  # Uniform distribution [0, 1e-5]
                    delay=1,
                    channel=1,
                )

        # --- Monitor configuration ---
        with self.sim.autoparent("normal"):
            # Monitor spikes from a subset of neurons in each group
            self.spike_monitor = SpikeMonitor(
                [
                    noise.where_id(lambda i: i < 100),
                    exc_neurons.where_id(lambda i: i < 100),
                    inh_neurons.where_id(lambda i: i < 100),
                ]
            )

            # Monitor membrane potential and currents of first excitatory neuron
            self.state_monitor = VariableMonitor(
                [exc_neurons.where_id(lambda i: i < 1),],
                ['V', 'spikes', 'channel_currents@0', 'channel_currents@1', 'channel_currents@2']
            )

            # Monitor weight evolution of a subset of synapses
            e2e_ampa.filter[:, :] = False
            e2e_ampa.filter[0, :10] = True  # Monitor first 10 connections from neuron 0
            e2e_nmda.filter[:, :] = False
            e2e_nmda.filter[0, :10] = True

            self.weight_monitor = VariableMonitor(
                [
                    e2e_ampa,
                    e2e_nmda,
                ],
                ["weight"]
            )

    def on_finish(self):
        """Visualization after simulation completes."""

        # --- Plot raster plot and firing rates ---
        fig, ax0 = plt.subplots(figsize=(10, 6))
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
            times, rate = smooth_spikes(spk_steps, n_neurons=n_neurons, to_step=self.step, sigma=0.1)
            ax0.plot(times, rate, c=f"C{idx}")

        ax1.legend(loc='lower right')
        plt.title(f"Raster plot and firing rates (Modular STDP)")
        plt.xlabel("Time (s)")
        ax0.set_ylabel("Firing rate (Hz)")
        ax1.set_ylabel("Neuron ID")

        # --- Plot membrane potential and currents ---
        fig, ax0 = plt.subplots(figsize=(10, 6))
        ax1 = ax0.twinx()
        ax2 = ax0.twinx()

        V = self.state_monitor.get_variable_tensor(0, 'V')
        ax0.plot(V, color='C0', label='V')
        ax0.set_ylabel("Membrane potential (V)", color='C0')

        spikes = self.state_monitor.get_variable_tensor(0, 'spikes')
        ax2.vlines(spikes.nonzero(as_tuple=True), ymin=0, ymax=1, color='black')
        ax2.get_yaxis().set_visible(False)

        ampa = self.state_monitor.get_variable_tensor(0, 'channel_currents@0')
        ax1.plot(ampa, color='C1', label='AMPA', alpha=0.7)
        gaba = self.state_monitor.get_variable_tensor(0, 'channel_currents@1')
        ax1.plot(gaba, color='C2', label='GABA', alpha=0.7)
        nmda = self.state_monitor.get_variable_tensor(0, 'channel_currents@2')
        ax1.plot(nmda, color='C3', label='NMDA', alpha=0.7)
        ax1.set_ylabel("Synaptic currents", color='C1')
        ax1.grid()
        ax1.legend(loc='upper right')

        plt.title("Membrane potential and synaptic currents (first excitatory neuron)")
        plt.xlabel("Time (steps)")

        # --- Plot weight evolution ---
        if True:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

            # AMPA weights
            w_ampa = self.weight_monitor.get_variable_tensor(0, "weight")
            for i in range(w_ampa.shape[2]):
                ax1.plot(w_ampa[:, 0, i], alpha=0.7)
            ax1.set_title("AMPA weight evolution (Modular STDP)")
            ax1.set_ylabel("Weight")
            ax1.grid(True)

            # NMDA weights
            w_nmda = self.weight_monitor.get_variable_tensor(1, "weight")
            for i in range(w_nmda.shape[2]):
                ax2.plot(w_nmda[:, 0, i], alpha=0.7)
            ax2.set_title("NMDA weight evolution (Modular STDP)")
            ax2.set_ylabel("Weight")
            ax2.set_xlabel("Time (steps)")
            ax2.grid(True)

            plt.tight_layout()

        plt.show()


# --- Execution ---
if __name__ == "__main__":
    print("="*70)
    print("Balanced Random Network with Modular STDP")
    print("="*70)
    print("\nThis example uses the modular e-prop plasticity framework.")
    print("STDP learning rules are applied to E→E connections (AMPA and NMDA).")
    print("\nKey features demonstrated:")
    print("  - PlasticDense connections with STDP rule")
    print("  - Modular plasticity configuration via params dict")
    print("  - Weight evolution monitoring")
    print("  - Multi-channel synapses (AMPA, GABA, NMDA)")
    print("\nRunning simulation...\n")

    exp = BalancedRandomNetworkModularExperiment(sim=Simulator(seed=0))
    simulation_time = 10.0  # seconds
    exp.run(steps=int(1000 * simulation_time))
