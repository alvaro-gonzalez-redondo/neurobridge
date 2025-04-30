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
    use_dense_connections = True

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
        self.n_input = 1001  # Number of input neurons
        self.n_output = 1  # Number of output neurons
        self.initial_weight = 1e-4

        # Setup for STDP demonstration - we'll create input patterns that
        # encourage potentiation for some synapses and depression for others

        with self.autoparent("graph"):
            # Input neurons that we'll manually control with patterns
            self.input_neurons = ParrotNeurons(
                device=self.local_circuit.device, n_neurons=self.n_input, delay_max=30
            )

            # A single output neuron that will learn through STDP
            self.output_neuron = SimpleIFNeurons(
                device=self.local_circuit.device,
                n_neurons=self.n_output,
                threshold=0.5,  # Lower threshold to encourage activity
                tau_membrane=20e-3,  # Slower membrane dynamics
                delay_max=30,
            )

            # Connect with STDP synapses - initially weak random weights
            self.synapses = (self.input_neurons >> self.output_neuron)(
                pattern="all-to-all",
                synapse_class=STDPDenseConnection if self.use_dense_connections else STDPConnection,
                weight=self.initial_weight, #lambda i,j: torch.rand(len(i))*0.2,  # Initial weight
                delay=1,  # 1ms delay
                A_plus=1e-2*self.initial_weight,  # Potentiation rate
                A_minus=-1.2e-2*self.initial_weight,  # Depression rate (slightly stronger)
                tau_plus=20e-3,  # Potentiation time constant
                tau_minus=20e-3,  # Depression time constant
                w_min=0.0,  # Minimum weight
                w_max=0.5,  # Maximum weight
            )

        with self.autoparent("normal"):
            # Monitor spikes
            self.spike_monitor = SpikeMonitor([self.input_neurons, self.output_neuron])

            # Monitor weights for a subset of synapses
            self.weight_monitor = VariableMonitor([self.synapses], ["weight"])

    def present_pattern(self, t: int):
        """Present an input pattern to create a learning scenario.

        Parameters
        ----------
        t : int
            Simulation time.
        """
        # Create a spike pattern where a block of nearby neurons fire
        spikes = torch.zeros(
            self.input_neurons.size, dtype=torch.bool, device=self.input_neurons.device
        )

        # Activate a group of nearby neurons (creates a "pattern")
        idx = t % self.input_neurons.size
        spikes[idx] = True

        # Inject the pattern into input neurons
        self.input_neurons.inject_spikes(spikes)

        # We're manually inducing the output neuron to spike with slight delay
        # to create conditions for STDP potentiation of the active synapses
        if t == (self.input_neurons.size//2): 
            self.output_neuron.inject_spikes(
                torch.ones(
                    self.output_neuron.size,
                    dtype=torch.bool,
                    device=self.output_neuron.device,
                )
            )

    def plot_results(self):
        """Plot the simulation results: spikes and weight evolution."""
        if True:
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
            ax3.plot(times, weight_data.cpu(), alpha=0.7)

            ax3.set_xlabel("Time (ms)")
            ax3.set_ylabel("Synaptic Weight")
            ax3.set_title("Weight Evolution")

            plt.tight_layout()
            show_or_save_plot("stdp_learning.png", log)

        # Create a figure for the final weight distribution
        weight_data = self.weight_monitor.get_variable_tensor(0, "weight")[-1, :].cpu().flip((0,)) - self.initial_weight
        x = np.arange(self.n_input) - self.n_input//2

        plt.figure(figsize=(8, 6))
        plt.plot(x, weight_data)
        plt.grid()
        plt.title("Final Weight Distribution")
        plt.xlabel("$\Delta t$")
        plt.ylabel("$\Delta w$")
        show_or_save_plot("final_weights.png", log)


# Main program
if __name__ == "__main__":
    # Create and initialize the simulator
    with STDPExample() as sim:
        # Simulation parameters
        simulation_length = sim.n_input  # ms

        # Run the simulation with a progress bar
        sim.step()
        for t in tqdm(range(simulation_length)):
            # Present different patterns at regular intervals
            sim.present_pattern(t)
            sim.step()
        sim.step()

        # Plot the results
        sim.plot_results()
