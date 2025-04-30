"""
Pattern Recognition Example

This example demonstrates a simple pattern recognition task using
STDP learning. The network is trained to recognize specific input patterns
and respond selectively to them.
"""

from neurobridge import *
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class PatternRecognitionExample(SimulatorEngine):
    """A simulation demonstrating pattern recognition with STDP learning."""

    def build_user_network(self, rank: int, world_size: int):
        """Build a pattern recognition network.

        Parameters
        ----------
        rank : int
            Current GPU rank (ignored in this single-GPU example).
        world_size : int
            Total number of GPUs (ignored in this single-GPU example).
        """
        # Network parameters
        input_size = 100  # Size of the input layer (10x10 grid)
        output_size = 4  # Number of output neurons (one per pattern)

        with self.autoparent("graph"):
            # Input layer: neurons that we'll stimulate with patterns
            self.input_layer = ParrotNeurons(
                device=self.local_circuit.device, n_neurons=input_size, delay_max=30
            )

            # Output layer: neurons that will learn to recognize patterns
            self.output_layer = SimpleIFNeurons(
                device=self.local_circuit.device,
                n_neurons=output_size,
                threshold=0.7,  # Threshold for spiking
                tau_membrane=20.0,  # Membrane time constant
                delay_max=30,
            )

            # Connect input to output with STDP synapses
            self.synapses = (self.input_layer >> self.output_layer)(
                pattern="all-to-all",
                synapse_class=STDPConnection,
                weight=lambda pre, post: torch.rand(len(pre))
                * 0.1,  # Random initial weights
                delay=1,
                A_plus=0.02,  # Potentiation rate
                A_minus=0.021,  # Depression rate
                tau_plus=20.0,
                tau_minus=20.0,
                w_min=0.0,
                w_max=1.0,
            )

            # Lateral inhibition within output layer
            # (when one output neuron fires, it inhibits the others)
            self.lateral_inhibition = (self.output_layer >> self.output_layer)(
                pattern="all-to-all",
                synapse_class=StaticConnection,
                weight=lambda pre, post: torch.where(
                    pre.unsqueeze(1) == post.unsqueeze(0),  # Diagonal mask
                    torch.zeros(len(pre) * len(post), device=self.local_circuit.device),
                    torch.ones(len(pre) * len(post), device=self.local_circuit.device) * -0.5,
                ),
                delay=1,
            )

        with self.autoparent("normal"):
            # Monitor spikes
            self.spike_monitor = SpikeMonitor(
                [
                    self.input_layer.where_id(
                        lambda idx: idx < 100
                    ),  # All input neurons
                    self.output_layer,  # All output neurons
                ]
            )

            # Monitor weights for visualization
            self.weight_monitor = VariableMonitor([self.synapses], ["weight"])

            # Create patterns for training and testing
            self.patterns = self._create_patterns(input_size, n_patterns=4)

    def _create_patterns(self, input_size, n_patterns=4):
        """Create input patterns for recognition.

        Parameters
        ----------
        input_size : int
            Size of the input layer.
        n_patterns : int, optional
            Number of patterns to create, by default 4.

        Returns
        -------
        list of torch.Tensor
            List of binary patterns as tensors.
        """
        patterns = []
        grid_size = int(np.sqrt(input_size))  # Assume square grid

        # Pattern 1: Horizontal line in the middle
        p1 = torch.zeros(input_size, dtype=torch.bool, device=self.local_circuit.device)
        p1[(grid_size // 2) * grid_size : (grid_size // 2 + 1) * grid_size] = True
        patterns.append(p1)

        # Pattern 2: Vertical line in the middle
        p2 = torch.zeros(input_size, dtype=torch.bool, device=self.local_circuit.device)
        p2[grid_size // 2 :: grid_size] = True
        patterns.append(p2)

        # Pattern 3: Diagonal from top-left to bottom-right
        p3 = torch.zeros(input_size, dtype=torch.bool, device=self.local_circuit.device)
        for i in range(grid_size):
            p3[i * grid_size + i] = True
        patterns.append(p3)

        # Pattern 4: X pattern (both diagonals)
        p4 = torch.zeros(input_size, dtype=torch.bool, device=self.local_circuit.device)
        for i in range(grid_size):
            p4[i * grid_size + i] = True  # Main diagonal
            p4[i * grid_size + (grid_size - 1 - i)] = True  # Anti-diagonal
        patterns.append(p4)

        return patterns[:n_patterns]

    def present_pattern(self, pattern_idx, with_supervision=False):
        """Present a pattern to the network.

        Parameters
        ----------
        pattern_idx : int
            Index of the pattern to present.
        with_supervision : bool, optional
            Whether to provide supervisory signal (force the corresponding
            output neuron to spike), by default False.
        """
        if pattern_idx >= len(self.patterns):
            raise ValueError(f"Pattern index {pattern_idx} out of range.")

        # Present the pattern to the input layer
        self.input_layer.inject_spikes(self.patterns[pattern_idx])

        # During training, provide supervisory signal to the correct output neuron
        if with_supervision:
            # Create a spike pattern for the output layer where only the neuron
            # corresponding to the current pattern fires
            output_spikes = torch.zeros(
                self.output_layer.size,
                dtype=torch.bool,
                device=self.output_layer.device,
            )
            output_spikes[pattern_idx] = True

            # Inject the supervisory spike with a slight delay
            if self.local_circuit.t % 50 == 5:  # 5ms after pattern presentation
                self.output_layer.inject_spikes(output_spikes)

    def run_training(self, n_presentations=100):
        """Run the training phase.

        Parameters
        ----------
        n_presentations : int, optional
            Number of pattern presentations, by default 100.
        """
        log("Starting training phase...")

        for i in tqdm(range(n_presentations)):
            # Present each pattern in sequence with supervision
            pattern_idx = i % len(self.patterns)

            # Present the pattern
            self.present_pattern(pattern_idx, with_supervision=True)

            # Run the simulation for a few steps to allow STDP to occur
            for _ in range(20):  # 20ms per pattern presentation
                self.step()

            # Add some time between patterns
            for _ in range(30):
                self.step()

    def run_testing(self, n_presentations=20):
        """Run the testing phase.

        Parameters
        ----------
        n_presentations : int, optional
            Number of pattern presentations, by default 20.
        """
        log("Starting testing phase...")

        # Reset spike recording to only keep test phase spikes
        self.spike_monitor = SpikeMonitor(
            [self.input_layer.where_id(lambda idx: idx < 100), self.output_layer]
        )

        for i in tqdm(range(n_presentations)):
            # Present each pattern in sequence without supervision
            pattern_idx = i % len(self.patterns)

            # Present the pattern
            self.present_pattern(pattern_idx, with_supervision=False)

            # Run the simulation for a few steps
            for _ in range(20):
                self.step()

            # Add some time between patterns
            for _ in range(30):
                self.step()

    def plot_results(self):
        """Plot the simulation results."""
        # 1. Plot the weight matrices (from input to each output neuron)
        plt.figure(figsize=(15, 5))

        # Get the final weights
        weights = self.weight_monitor.get_variable_tensor(0, "weight")[-1].cpu().numpy()

        # Reshape to have one matrix per output neuron
        grid_size = int(np.sqrt(self.input_layer.size))
        n_outputs = self.output_layer.size

        for i in range(n_outputs):
            plt.subplot(1, n_outputs, i + 1)

            # Get weights for this output neuron and reshape to grid
            w = weights[i * self.input_layer.size : (i + 1) * self.input_layer.size]
            w = w.reshape(grid_size, grid_size)

            plt.imshow(w, cmap="viridis")
            plt.title(f"Output Neuron {i}")
            plt.colorbar()

        plt.tight_layout()
        show_or_save_plot("learned_weight_patterns.png", log)

        # 2. Plot the spike responses during testing
        plt.figure(figsize=(15, 8))

        # Input spikes
        plt.subplot(2, 1, 1)
        input_spikes = self.spike_monitor.get_spike_tensor(0)
        if input_spikes.shape[0] > 0:
            times, neurons = input_spikes[:, 1], input_spikes[:, 0]
            plt.scatter(times.cpu(), neurons.cpu(), s=2)
        plt.title("Input Neuron Spikes")
        plt.ylabel("Neuron ID")

        # Output spikes
        plt.subplot(2, 1, 2)
        output_spikes = self.spike_monitor.get_spike_tensor(1)
        if output_spikes.shape[0] > 0:
            times, neurons = output_spikes[:, 1], output_spikes[:, 0]
            plt.scatter(times.cpu(), neurons.cpu(), s=10)
        plt.title("Output Neuron Spikes")
        plt.xlabel("Time (ms)")
        plt.ylabel("Neuron ID")
        plt.yticks(range(self.output_layer.size))

        # Add pattern presentation indicators
        for i in range(len(self.patterns)):
            for t in range(0, 1000, 50):  # Assume pattern presentations every 50ms
                if i == (t // 50) % len(self.patterns):
                    plt.axvspan(t, t + 20, alpha=0.2, color=f"C{i}")

        plt.tight_layout()
        show_or_save_plot("pattern_recognition_spikes.png", log)


# Main program
if __name__ == "__main__":
    # Create and initialize the simulator
    with PatternRecognitionExample() as sim:
        # Training phase
        sim.run_training(n_presentations=200)

        # Testing phase
        sim.run_testing(n_presentations=20)

        # Plot results
        sim.plot_results()
