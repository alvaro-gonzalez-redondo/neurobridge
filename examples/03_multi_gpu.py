"""
Multi-GPU Spiking Neural Network Example

This example demonstrates how to distribute a spiking neural network
across multiple GPUs, with communication between the GPUs managed by
the BridgeNeuronGroup.

To run this example, use:
torchrun --nproc_per_node=2 03_multi_gpu.py
"""

from neurobridge import *
import matplotlib.pyplot as plt
from tqdm import tqdm


class MultiGPUExample(SimulatorEngine):
    """A distributed simulation running across multiple GPUs."""

    def build_user_network(self, rank: int, world_size: int):
        """Build a network distributed across GPUs.

        Parameters
        ----------
        rank : int
            Current GPU rank (0, 1, ..., world_size-1).
        world_size : int
            Total number of GPUs.
        """
        # Verify we have at least 2 GPUs for this example
        if world_size < 2:
            log(f"This example requires at least 2 GPUs, but only {world_size} found.")
            log("It will still run, but in a single-GPU mode.")

        # Parameters for the network
        n_neurons_per_gpu = 100

        # Create a bridge for inter-GPU communication
        # This allows neurons on different GPUs to be connected
        self.add_default_bridge(n_local_neurons=n_neurons_per_gpu, n_steps=10)
        bridge = self.local_circuit.bridge

        # Different setup for each GPU
        if rank == 0:
            # First GPU: Generate input patterns
            with self.autoparent("graph"):
                # Source neurons that generate random spikes
                self.source_neurons = RandomSpikeNeurons(
                    device=self.local_circuit.device,
                    n_neurons=n_neurons_per_gpu,
                    firing_rate=10.0,  # 10Hz firing rate
                    delay_max=20,
                )

                # Connect source neurons to the bridge, which will transmit to GPU 1
                # Every source neuron connects to its corresponding bridge neuron
                (self.source_neurons >> bridge.where_rank(1))(
                    pattern="one-to-one",  # One-to-one connectivity
                    weight=1.0,  # Full weight to ensure spike transmission
                    delay=0,  # No additional delay
                )

            # Monitoring setup for GPU 0
            with self.autoparent("normal"):
                # Monitor spikes from a sample of source neurons
                self.spike_monitor = SpikeMonitor(
                    [self.source_neurons.where_id(lambda idx: idx < 20)]
                )

        elif rank == 1:
            # Second GPU: Process the input with STDP learning
            with self.autoparent("graph"):
                # Target neurons that receive input from GPU 0
                self.target_neurons = SimpleIFNeurons(
                    device=self.local_circuit.device,
                    n_neurons=n_neurons_per_gpu,
                    threshold=0.9,
                    tau_membrane=15.0,
                    delay_max=20,
                )

                # Connect the bridge (receiving from GPU 0) to the target neurons
                # with STDP synapses for learning
                self.synapses = (bridge.where_rank(0) >> self.target_neurons)(
                    pattern="one-to-one",  # Each input affects one output
                    synapse_class=STDPConnection,  # Use STDP for learning
                    weight=0.5,  # Initial weight
                    delay=1,  # 1ms synaptic delay
                    A_plus=0.01,  # Potentiation rate
                    A_minus=0.0105,  # Depression rate
                    tau_plus=20.0,  # Potentiation time constant
                    tau_minus=20.0,  # Depression time constant
                    w_min=0.0,  # Minimum weight
                    w_max=1.0,  # Maximum weight
                )

            # Monitoring setup for GPU 1
            with self.autoparent("normal"):
                # Monitor spikes and membrane potentials in target neurons
                self.spike_monitor = SpikeMonitor(
                    [self.target_neurons.where_id(lambda idx: idx < 20)]
                )

                # Monitor membrane potentials of a few neurons
                self.voltage_monitor = VariableMonitor(
                    [self.target_neurons.where_id(lambda idx: idx < 5)], ["V"]
                )

                # Monitor weights of a sample of synapses
                self.weight_monitor = VariableMonitor(
                    [self.synapses.where_id(lambda idx: idx < 20)], ["weight"]
                )

    def plot_results(self):
        """Plot the simulation results based on the GPU rank."""
        if self.rank == 0:
            # Plot for GPU 0: Input spike patterns
            plt.figure(figsize=(10, 5))

            # Get spikes from input neurons
            spikes = self.spike_monitor.get_spike_tensor(0)
            if spikes.shape[0] > 0:
                times, neurons = spikes[:, 1], spikes[:, 0]
                plt.scatter(times.cpu(), neurons.cpu(), s=5, c="blue", alpha=0.7)

            plt.title(f"GPU {self.rank}: Input Neuron Spikes")
            plt.xlabel("Time (ms)")
            plt.ylabel("Neuron ID")
            plt.ylim(-1, 20)  # Show only the monitored neurons

            # Save the figure
            show_or_save_plot(f"gpu{self.rank}_spikes.png", log)

        elif self.rank == 1:
            # Plot for GPU 1: Output spikes, membrane potentials, and weights

            # Plot 1: Output spikes
            plt.figure(figsize=(10, 5))

            spikes = self.spike_monitor.get_spike_tensor(0)
            if spikes.shape[0] > 0:
                times, neurons = spikes[:, 1], spikes[:, 0]
                plt.scatter(times.cpu(), neurons.cpu(), s=5, c="red", alpha=0.7)

            plt.title(f"GPU {self.rank}: Target Neuron Spikes")
            plt.xlabel("Time (ms)")
            plt.ylabel("Neuron ID")
            plt.ylim(-1, 20)  # Show only the monitored neurons

            # Save the figure
            show_or_save_plot(f"gpu{self.rank}_spikes.png", log)

            # Plot 2: Membrane potentials
            plt.figure(figsize=(10, 5))

            v_data = self.voltage_monitor.get_variable_tensor(0, "V")
            for i in range(v_data.shape[1]):
                plt.plot(v_data[:, i].cpu(), label=f"Neuron {i}")

            plt.axhline(y=0.9, linestyle="--", color="k", alpha=0.5)  # Threshold
            plt.title(f"GPU {self.rank}: Membrane Potentials")
            plt.xlabel("Time (ms)")
            plt.ylabel("Membrane Potential")
            plt.legend()

            # Save the figure
            show_or_save_plot(f"gpu{self.rank}_voltages.png", log)

            # Plot 3: Synaptic weights over time
            plt.figure(figsize=(10, 5))

            w_data = self.weight_monitor.get_variable_tensor(0, "weight")
            for i in range(min(5, w_data.shape[1])):  # Plot first 5 weights
                plt.plot(w_data[:, i].cpu(), label=f"Synapse {i}")

            plt.title(f"GPU {self.rank}: Synaptic Weight Evolution")
            plt.xlabel("Time (ms)")
            plt.ylabel("Weight")
            plt.legend()

            # Save the figure
            show_or_save_plot(f"gpu{self.rank}_weights.png", log)


# Main program
if __name__ == "__main__":
    # Simulation parameters
    simulation_length = 1000  # ms

    # Create and initialize the simulator
    with MultiGPUExample() as sim:
        # Run the simulation
        # We only show progress on rank 0 to avoid clutter
        for _ in tqdm(range(simulation_length), disable=(sim.rank != 0)):
            sim.step()

        # Plot the results
        sim.plot_results()
