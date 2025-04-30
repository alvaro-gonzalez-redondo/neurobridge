"""
Basic NeuroBridge Simulation Example

This example demonstrates a simple simulation with random spike generators
and integrate-and-fire neurons, illustrating the fundamental concepts
of creating and running a NeuroBridge simulation.
"""

from neurobridge import *
import torch
import matplotlib.pyplot as plt


class BasicExample(SimulatorEngine):
    """A simple example simulation with random and IF neurons."""
    use_dense_connections = False

    def build_user_network(self, rank: int, world_size: int):
        """Build a simple network of random spike generators connected to IF neurons.

        Parameters
        ----------
        rank : int
            Current GPU rank (ignored in this single-GPU example).
        world_size : int
            Total number of GPUs (ignored in this single-GPU example).
        """
        # --- Create neurons within the CUDA graph for optimal performance ---
        with self.autoparent("graph"):
            # Create a random spike generator
            self.source = RandomSpikeNeurons(
                device=self.local_circuit.device,
                n_neurons=1_000,
                firing_rate=5.0,  # Firing rate in Hz
                delay_max=20,  # Maximum delay for spike history
            )

            # Create a group of 50 integrate-and-fire neurons
            self.target = IFNeurons(
                device=self.local_circuit.device,
                n_neurons=20,
            )

            # Connect the source to the target with all-to-all connectivity and random weights
            (self.source >> self.target)(
                pattern="all-to-all",
                synapse_class=StaticDenseConnection if self.use_dense_connections else StaticConnection,
                weight=lambda pre, pos: torch.rand(len(pre)) * (3e-3/self.source.size),
                delay=2,  # 2ms delay
            )


        # --- Create monitors outside the CUDA graph ---
        with self.autoparent("normal"):
            # Monitor spikes from a subset of neurons in each group
            self.spike_monitor = SpikeMonitor(
                [
                    self.source.where_id(lambda idx: idx < 20),  # First 20 source neurons
                    self.target.where_id(lambda idx: idx < 20),  # First 20 target neurons
                ]
            )

            # Monitor membrane potential for a subset of target neurons
            self.voltage_monitor = VariableMonitor(
                [self.target],#.where_id(lambda idx: idx < 5)],  # First 5 target neurons
                ["V"],  # Monitor the membrane potential
            )

    def plot_results(self):
        """Plot the simulation results: spikes and membrane potentials."""
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Plot spikes for source neurons
        source_spikes = self.spike_monitor.get_spike_tensor(0)
        if source_spikes.shape[0] > 0:  # Check if any spikes were recorded
            times, neurons = source_spikes[:, 1], source_spikes[:, 0]
            ax1.scatter(times.cpu(), neurons.cpu(), s=5, c="blue", alpha=0.7)
        ax1.set_ylabel("Source Neuron ID")
        ax1.set_title("Spike Raster Plot")

        # Plot spikes for target neurons
        target_spikes = self.spike_monitor.get_spike_tensor(1)
        if target_spikes.shape[0] > 0:  # Check if any spikes were recorded
            times, neurons = target_spikes[:, 1], target_spikes[:, 0]
            ax1.scatter(times.cpu(), neurons.cpu() + 20, s=5, c="red", alpha=0.7)

        # Plot membrane potentials
        voltage_data = self.voltage_monitor.get_variable_tensor(0, "V")
        for i in range(voltage_data.shape[1]):
            ax2.plot(voltage_data[:, i].cpu(), label=f"Neuron {i}")
        ax2.axhline(y=self.target.threshold.cpu(), linestyle="--", color="k", alpha=0.5)  # Threshold line
        ax2.set_xlabel("Time (ms)")
        ax2.set_ylabel("Membrane Potential")
        ax2.set_title("Membrane Potential Dynamics")
        ax2.legend()

        plt.tight_layout()
        show_or_save_plot("basic_simulation.png", log)


# Main program
if __name__ == "__main__":
    # Simulation parameters
    simulation_length = 1000  # ms

    # Create and initialize the simulator
    with BasicExample() as sim:
        # Run the simulation for the specified number of steps
        for _ in range(simulation_length):
            sim.step()

        # Plot the results
        sim.plot_results()
