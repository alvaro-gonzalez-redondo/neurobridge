"""
Ring Network Example

This example demonstrates a ring network distributed across multiple GPUs,
where spiking activity propagates around the ring. This showcases
the communication capabilities of NeuroBridge.

To run this example, use:
torchrun --nproc_per_node=N 04_ring_network.py
where N is the number of GPUs to use (2 or more).
"""

from neurobridge import *
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm


class RingNetworkExample(SimulatorEngine):
    """A ring network distributed across multiple GPUs."""

    def build_user_network(self, rank: int, world_size: int):
        """Build a segment of a ring network on each GPU.

        Parameters
        ----------
        rank : int
            Current GPU rank (0, 1, ..., world_size-1).
        world_size : int
            Total number of GPUs.
        """
        # Ensure we have at least 2 GPUs
        if world_size < 2:
            log("This example is designed for 2 or more GPUs.")
            log(
                "It will run on a single GPU, but as a linear chain rather than a ring."
            )

        # Configuration parameters
        n_neurons_per_gpu = 20

        # Create a bridge for inter-GPU communication
        self.add_default_bridge(n_local_neurons=n_neurons_per_gpu, n_steps=10)
        bridge = self.local_circuit.bridge

        with self.autoparent("graph"):
            # Create a group of neurons for this segment of the ring
            self.neurons = ParrotNeurons(
                device=self.local_circuit.device,
                n_neurons=n_neurons_per_gpu,
                delay_max=20,
            )

            # Connect to the next GPU in the ring
            # (or to the first GPU if this is the last one)
            next_rank = (rank + 1) % world_size

            (self.neurons >> bridge.where_rank(next_rank))(
                pattern="one-to-one", weight=1.0, delay=0
            )

            # Connect from the previous GPU in the ring
            # (or from the last GPU if this is the first one)
            prev_rank = (rank - 1) % world_size

            (bridge.where_rank(prev_rank) >> self.neurons)(
                pattern="one-to-one", weight=1.0, delay=0
            )

        with self.autoparent("normal"):
            # Monitor spikes for visualization
            self.spike_monitor = SpikeMonitor([self.neurons])

    def inject_initial_spike(self, step: int):
        """Inject an initial spike to start the activity propagation.

        Parameters
        ----------
        step : int
            Current simulation step.
        """
        # Only on GPU 0, and only at the beginning
        if self.rank == 0 and step == 10:
            # Inject a spike to the first neuron
            spikes = torch.zeros(
                self.neurons.size, dtype=torch.bool, device=self.neurons.device
            )
            spikes[0] = True
            self.neurons.inject_spikes(spikes)

            # Log the action
            log("Injected initial spike to neuron 0 on GPU 0")

    def plot_results(self):
        """Plot the spike raster for each GPU."""
        plt.figure(figsize=(12, 6))

        # Get spikes from the neurons
        spikes = self.spike_monitor.get_spike_tensor(0)

        if spikes.shape[0] > 0:
            # Adjust neuron IDs to show the position in the overall ring
            # by adding an offset based on the GPU rank
            times = spikes[:, 1].cpu()
            neurons = spikes[:, 0].cpu() + (self.rank * self.neurons.size)

            plt.scatter(
                times,
                neurons,
                s=10,
                c=f"C{self.rank}",
                alpha=0.8,
                label=f"GPU {self.rank}",
            )

        plt.title(f"Spike Propagation in Ring Network - GPU {self.rank}")
        plt.xlabel("Time (ms)")
        plt.ylabel("Neuron ID (global)")
        plt.legend()

        # Save the figure for each GPU
        show_or_save_plot(f"ring_network_gpu{self.rank}.png", log)


# Main program
if __name__ == "__main__":
    # Simulation parameters
    simulation_length = 200  # ms

    # Create and initialize the simulator
    with RingNetworkExample() as sim:
        # Run the simulation with a progress bar
        for step in tqdm(range(simulation_length), disable=(sim.rank != 0)):
            # Inject initial spike to start the activity
            sim.inject_initial_spike(step)

            # Advance the simulation
            sim.step()

            # Log spiking activity for visualization
            if step % 20 == 0:
                buffer = sim.neurons.get_spike_buffer()
                t = sim.local_circuit.t.item() - 1
                phase = t % sim.neurons.delay_max
                spks = buffer[:, phase].squeeze().tolist()
                spks_str = "".join(["|" if spk else "_" for spk in spks])
                log(f"GPU {sim.rank}, t={step}: {spks_str}")

        # Plot the results
        sim.plot_results()
