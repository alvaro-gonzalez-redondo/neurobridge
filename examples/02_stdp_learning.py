"""
STDP Learning Example

This example demonstrates spike-timing-dependent plasticity (STDP) in action,
showing how synaptic weights evolve over time based on the relative timing
of pre- and post-synaptic spikes.
"""

from neurobridge import *
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class STDPExample(SimulatorEngine):
    """Simulation demonstrating STDP learning."""

    def build_user_network(self, rank: int, world_size: int):
        """Build a network with STDP synapses for learning.

        Parameters
        ----------
        rank : int
            Current GPU rank (ignored in this single-GPU example).
        world_size : int
            Total number of GPUs (ignored in this single-GPU example).
        """
        # Set up parameters
        n_input = 100  # Number of input neurons
        n_output = 1  # Number of output neurons

        # Setup for STDP demonstration - we'll create input patterns that
        # encourage potentiation for some synapses and depression for others

        with self.autoparent("graph"):
            # Input neurons that we'll manually control with patterns
            self.input_neurons = ParrotNeurons(
                device=self.local_circuit.device, n_neurons=n_input, delay_max=30
            )

            # A single output neuron that will learn through STDP
            self.output_neuron = IFNeurons(
                device=self.local_circuit.device,
                n_neurons=n_output,
                threshold=0.5,  # Lower threshold to encourage activity
                tau=20.0,  # Slower membrane dynamics
                delay_max=30,
            )

            # Connect with STDP synapses - initially weak random weights
            self.synapses = (self.input_neurons >> self.output_neuron)(
                pattern="all-to-all",
                synapse_class=STDPConnection,
                weight=0.01,  # Initial weight
                delay=1,  # 1ms delay
                A_plus=0.01,  # Potentiation rate
                A_minus=0.0105,  # Depression rate (slightly stronger)
                tau_plus=20.0,  # Potentiation time constant
                tau_minus=20.0,  # Depression time constant
                w_min=0.0,  # Minimum weight
                w_max=0.5,  # Maximum weight
            )

        with self.autoparent("normal"):
            # Monitor spikes
            self.spike_monitor = SpikeMonitor(
                [
                    self.input_neurons.where_id(
                        lambda idx: idx < 10
                    ),  # Sample of input neurons
                    self.output_neuron,  # The output neuron
                ]
            )

            # Monitor weights for a subset of synapses
            self.weight_monitor = VariableMonitor(
                [self.synapses.where_id(lambda idx: idx < 100)],  # First 100 synapses
                ["weight"],
            )

    def present_pattern(self, pattern_idx, pattern_width=10):
        """Present an input pattern to create a learning scenario.

        Parameters
        ----------
        pattern_idx : int
            Starting index for the pattern (which subset of neurons to activate).
        pattern_width : int, optional
            Width of the pattern in neurons, by default 10.
        """
        # Create a spike pattern where a block of nearby neurons fire
        spikes = torch.zeros(
            self.input_neurons.size, dtype=torch.bool, device=self.input_neurons.device
        )

        # Activate a group of nearby neurons (creates a "pattern")
        start_idx = pattern_idx % (self.input_neurons.size - pattern_width)
        spikes[start_idx : start_idx + pattern_width] = True

        # Inject the pattern into input neurons
        self.input_neurons.inject_spikes(spikes)

        # We're manually inducing the output neuron to spike with slight delay
        # to create conditions for STDP potentiation of the active synapses
        if self.local_circuit.t % 30 == 5:  # Every 30ms, with 5ms delay after pattern
            self.output_neuron.inject_spikes(
                torch.ones(
                    self.output_neuron.size,
                    dtype=torch.bool,
                    device=self.output_neuron.device,
                )
            )

    def plot_results(self):
        """Plot the simulation results: spikes and weight evolution."""
        # Create a figure with multiple subplots
        fig, (ax1, ax2, ax3) = plt.subplots(
            3, 1, figsize=(10, 10), gridspec_kw={"height_ratios": [2, 1, 2]}
        )

        # Plot spikes for input neurons
        input_spikes = self.spike_monitor.get_spike_tensor(0)
        if input_spikes.shape[0] > 0:
            times, neurons = input_spikes[:, 1], input_spikes[:, 0]
            ax1.scatter(times.cpu(), neurons.cpu(), s=2, c="blue", alpha=0.7)
        ax1.set_ylabel("Input Neuron ID")
        ax1.set_title("Spike Raster Plot")

        # Plot spikes for output neuron
        output_spikes = self.spike_monitor.get_spike_tensor(1)
        if output_spikes.shape[0] > 0:
            times = output_spikes[:, 1].cpu()
            ax2.vlines(times, 0, 1, colors="red")
        ax2.set_ylabel("Output\nSpikes")
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(["", ""])

        # Plot weight evolution
        weight_data = self.weight_monitor.get_variable_tensor(0, "weight")

        # Plot weight evolution over time
        times = np.arange(weight_data.shape[0])
        for i in range(min(10, weight_data.shape[1])):  # Plot first 10 weights only
            ax3.plot(times, weight_data[:, i].cpu(), label=f"Synapse {i}", alpha=0.7)

        # Also plot the final weight distribution as a heatmap below
        final_weights = weight_data[-1, :].reshape(10, 10).cpu()
        ax3.imshow(
            final_weights,
            aspect="auto",
            cmap="viridis",
            extent=[weight_data.shape[0], weight_data.shape[0] * 1.2, 0, 0.2],
            alpha=0.4,
        )

        ax3.set_xlabel("Time (ms)")
        ax3.set_ylabel("Synaptic Weight")
        ax3.set_title("Weight Evolution")

        plt.tight_layout()
        show_or_save_plot("stdp_learning.png", log)

        # Create a separate figure for the final weight distribution
        plt.figure(figsize=(8, 6))
        plt.imshow(weight_data[-1, :].reshape(10, 10).cpu(), cmap="viridis")
        plt.colorbar(label="Weight")
        plt.title("Final Weight Distribution (First 100 synapses)")
        plt.xlabel("Synapse ID (mod 10)")
        plt.ylabel("Synapse ID (div 10)")
        show_or_save_plot("final_weights.png", log)


# Main program
if __name__ == "__main__":
    # Simulation parameters
    simulation_length = 1000  # ms

    # Create and initialize the simulator
    with STDPExample() as sim:
        # Run the simulation with a progress bar
        for t in tqdm(range(simulation_length)):
            # Present different patterns at regular intervals
            if t % 30 == 0:  # Every 30ms
                pattern_idx = (t // 30) * 10 % 90  # Cycle through different patterns
                sim.present_pattern(pattern_idx)
            sim.step()

        # Plot the results
        sim.plot_results()
