# NeuroBridge: Multi-GPU Spiking Neural Networks

NeuroBridge is a high-performance library for simulating Spiking Neural Networks (SNNs) across multiple GPUs. Built on PyTorch, it provides an intuitive API for constructing, simulating, and analyzing complex neural circuits.

## Features

- **Multi-GPU Scaling**: Efficiently distribute neural simulations across multiple GPUs
- **Optimized Performance**: Utilizes CUDA graphs for maximizing throughput
- **Flexible Network Construction**: Intuitive object-oriented API with support for various neuron models and connection patterns
- **Biologically-Inspired Learning**: Built-in support for spike-timing-dependent plasticity (STDP)
- **Comprehensive Monitoring**: Track spikes, membrane potentials, and synaptic weights during simulation
- **Hierarchical Design**: Organize complex networks with hierarchical structure

## Installation

```bash
pip install neurobridge
```

## Requirements

- Python 3.7+
- PyTorch 1.9+
- CUDA-compatible GPU

## Quick Start

Here's a simple example of a random spike generator connected to integrate-and-fire neurons:

```python
from neurobridge.all import *
import torch
import matplotlib.pyplot as plt

class SimpleDemo(SimulatorEngine):
    def build_user_network(self, rank, world_size):
        # Create components within the CUDA graph for performance
        with self.autoparent("graph"):
            # Random spike source (50 neurons at 5 Hz)
            source = RandomSpikeGenerator(
                device=self.local_circuit.device,
                n_neurons=50,
                firing_rate=5.0
            )
            
            # Target Integrate-and-Fire neurons (20 neurons)
            target = IFNeuronGroup(
                device=self.local_circuit.device,
                n_neurons=20,
                threshold=1.0,
                tau=0.5
            )
            
            # Connect with random weights
            (source >> target)(
                pattern="all-to-all",
                weight=lambda pre, post: torch.rand(len(pre)) * 0.05,
                delay=1
            )
        
        # Monitoring components outside the CUDA graph
        with self.autoparent("normal"):
            self.spike_monitor = SpikeMonitor([
                source.where_id(lambda ids: ids < 10),
                target.where_id(lambda ids: ids < 10)
            ])

# Run the simulation
with SimpleDemo() as sim:
    # Run for 1000 steps
    for _ in range(1000):
        sim.step()
    
    # Visualize results
    for i, name in enumerate(["Source", "Target"]):
        spikes = sim.spike_monitor.get_spike_tensor(i)
        times, neurons = spikes[:,1], spikes[:,0]
        
        plt.figure(figsize=(10, 4))
        plt.scatter(times, neurons, s=5)
        plt.title(f"{name} Neurons")
        plt.xlabel("Time (steps)")
        plt.ylabel("Neuron ID")
    
    plt.show()
```

## Multi-GPU Example

NeuroBridge makes it easy to distribute simulations across multiple GPUs:

```python
from neurobridge.all import *
from tqdm import tqdm

class DistributedDemo(SimulatorEngine):
    def build_user_network(self, rank, world_size):
        # Create a bridge for inter-GPU communication
        self.add_default_bridge(n_local_neurons=100, n_steps=10)
        bridge = self.local_circuit.bridge
        
        if rank == 0:
            # First GPU: Generate spikes
            with self.autoparent("graph"):
                source = RandomSpikeGenerator(
                    device=self.local_circuit.device,
                    n_neurons=100,
                    firing_rate=10.0
                )
                
                # Send spikes to second GPU
                (source >> bridge.where_rank(1))(
                    pattern="one-to-one",
                    weight=1.0
                )
                
            with self.autoparent("normal"):
                self.spike_monitor = SpikeMonitor([source.where_id(lambda ids: ids < 20)])
                
        elif rank == 1:
            # Second GPU: Process the spikes
            with self.autoparent("graph"):
                target = IFNeuronGroup(
                    device=self.local_circuit.device,
                    n_neurons=100
                )
                
                # Receive spikes from first GPU
                (bridge.where_rank(0) >> target)(
                    pattern="one-to-one",
                    weight=0.5
                )
                
            with self.autoparent("normal"):
                self.spike_monitor = SpikeMonitor([target.where_id(lambda ids: ids < 20)])

# Run with torchrun to use multiple GPUs
# torchrun --nproc_per_node=2 distributed_example.py
```

## Core Components

- **NeuronGroup**: Base class for neuron populations
  - **IFNeuronGroup**: Integrate-and-Fire neurons
  - **ParrotGroup**: Simple relay neurons
  - **RandomSpikeGenerator**: Generate random spikes according to a Poisson process
  
- **SynapticGroup**: Base class for synaptic connections
  - **StaticSynapse**: Fixed-weight synapses
  - **STDPSynapse**: Spike-Timing-Dependent Plasticity synapses
  
- **Monitors**: Tools for recording simulation data
  - **SpikeMonitor**: Record spike events
  - **VariableMonitor**: Track variables over time (e.g., membrane potentials, weights)
  
- **BridgeNeuronGroup**: Enables communication between GPUs

## Documentation

For full documentation, see [https://neurobridge.readthedocs.io](https://neurobridge.readthedocs.io)

## Contributing

Contributions are welcome! Please check out our [contributing guidelines](CONTRIBUTING.md).

## License

NeuroBridge is released under the MIT License. See the [LICENSE](LICENSE) file for details.

## Citation

If you use NeuroBridge in your research, please cite:

```
@software{neurobridge,
  author = {Álvaro González-Redondo},
  title = {NeuroBridge: Multi-GPU Spiking Neural Networks},
  url = {https://github.com/alvaro-gonzalez-redondo/neurobridge},
  year = {2023},
}
```