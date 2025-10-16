# NeuroBridge Connection API Specification (v0.1.9)

**Last Updated:** 2025-10-06
**Status:** Current implementation specification

This document describes the actual working API of NeuroBridge v0.1.9, not future plans.

---

## 1. Core Connection Operator (`>>`)

The primary mechanism for creating connections uses the `>>` operator:

```python
conn = (source_pop >> target_pop)(pattern='random', p=0.1, weight=0.5, delay=0)
```

### 1.1 Basic Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pattern` | str | required | Connection pattern (`'all-to-all'`, `'random'`, `'one-to-one'`, `'distance'`) |
| `weight` | scalar/tensor/callable | 1.0 | Synaptic weights |
| `delay` | scalar/tensor/callable | 0 | Synaptic delays (time steps) |
| `synapse_class` | class | `StaticSparse` | Synapse type (`StaticSparse`, `STDPSparse`, `StaticDense`, `STDPDense`) |
| `channel` | int | 0 | Target synaptic channel (for multi-channel neurons) |

**Additional parameters** are passed through `**kwargs` to the synapse class (e.g., STDP parameters).

### 1.2 Pattern-Specific Parameters

#### Random Connectivity

```python
# Connection probability
conn = (pop1 >> pop2)(pattern='random', p=0.1)

# Fixed fan-in (each target receives exactly N inputs)
conn = (pop1 >> pop2)(pattern='random', fanin=10)

# Fixed fan-out (each source connects to exactly N targets)
conn = (pop1 >> pop2)(pattern='random', fanout=20)

# Both fanin and fanout (balanced)
conn = (pop1 >> pop2)(pattern='random', fanin=10, fanout=10)
```

**Parameters:**
- `p`: Connection probability (0.0-1.0)
- `fanin`: Fixed inputs per target
- `fanout`: Fixed outputs per source
- Only one of `p`, `fanin`, `fanout`, or both `fanin+fanout` can be specified

#### Distance-Based Connectivity

```python
# Gaussian probability
conn = (layer1 >> layer2)(
    pattern='distance',
    sigma=2.0,
    p_max=1.0,
    max_distance=5.0
)

# Fixed fan-in with distance constraints
conn = (layer1 >> layer2)(
    pattern='distance',
    fanin=10,
    max_distance=10.0
)

# Custom probability function
conn = (layer1 >> layer2)(
    pattern='distance',
    prob_func=lambda src_pos, tgt_pos, dists: torch.exp(-dists/sigma)
)
```

**Parameters:**
- `max_distance`: Maximum connection distance (optional)
- `p_max`: Maximum connection probability at distance=0 (default: 1.0)
- `sigma`: Gaussian width for probability decay
- `fanin`/`fanout`: Select closest neighbors
- `prob_func`: Custom probability function `(src_block, tgt_all, dists_block) -> probs`

**Implementation:** Uses `block_distance_connect()` for memory efficiency

#### All-to-All Connectivity

```python
conn = (pop1 >> pop2)(pattern='all-to-all', weight=0.1)
```

Connects every source neuron to every target neuron (N_src × N_tgt connections).

#### One-to-One Connectivity

```python
conn = (pop1 >> pop2)(pattern='one-to-one', weight=1.0)
```

**Requirement:** `pop1.size == pop2.size`

Connects neuron i to neuron i for all i.

### 1.3 Function-Based Parameters

Parameters (`weight`, `delay`) can be:
1. **Scalar**: Same value for all connections
2. **Tensor**: Explicit per-connection values (must match number of connections)
3. **Callable**: Function that generates values

**Function signature:**
```python
def param_func(src_idx, tgt_idx, src_pos, tgt_pos):
    """
    Parameters:
    - src_idx: torch.LongTensor of source neuron indices [M]
    - tgt_idx: torch.LongTensor of target neuron indices [M]
    - src_pos: torch.Tensor of source positions [M, spatial_dims] (or None)
    - tgt_pos: torch.Tensor of target positions [M, spatial_dims] (or None)

    Returns:
    - torch.Tensor of parameter values [M]
    """
    return ...
```

**Examples:**

```python
# Random weights
conn = (pop1 >> pop2)(
    pattern='all-to-all',
    weight=lambda src, tgt, sp, tp: torch.rand(src.numel()) * 0.1
)

# Distance-dependent weights
conn = (layer1 >> layer2)(
    pattern='distance',
    weight=lambda src, tgt, src_pos, tgt_pos:
        torch.exp(-torch.norm(src_pos - tgt_pos, dim=1) / 2.0)
)

# Delay proportional to distance
conn = (layer1 >> layer2)(
    pattern='distance',
    delay=lambda src, tgt, src_pos, tgt_pos:
        (torch.norm(src_pos - tgt_pos, dim=1) / 0.1).long().clamp(0, 10)
)
```

### 1.4 Synapse Type Selection

```python
# Static sparse (default)
conn = (pop1 >> pop2)(pattern='random', p=0.1, weight=0.5)

# STDP sparse
conn = (pop1 >> pop2)(
    pattern='random', p=0.1,
    synapse_class=STDPSparse,
    weight=0.5,
    A_plus=1e-4,
    A_minus=-1.2e-4,
    tau_plus=20e-3,
    tau_minus=20e-3,
    w_min=0.0,
    w_max=1.0,
    oja_decay=1e-5
)

# Dense connections (use for high connectivity)
conn = (exc >> exc)(
    pattern='random', p=0.8,
    synapse_class=STDPDense,
    weight=1e-6,
    w_max=1e-5
)
```

**Available synapse classes:**
- `StaticSparse` - Fixed weights, sparse representation
- `STDPSparse` - Plastic weights with STDP, sparse
- `StaticDense` - Fixed weights, dense matrix
- `STDPDense` - Plastic weights with STDP, dense matrix

**STDP Parameters:**
- `A_plus`: Potentiation rate (positive, e.g., 1e-4)
- `A_minus`: Depression rate (negative, e.g., -1.2e-4)
- `tau_plus`: Pre-synaptic trace time constant (seconds)
- `tau_minus`: Post-synaptic trace time constant (seconds)
- `w_min`: Minimum weight
- `w_max`: Maximum weight
- `oja_decay`: Homeostatic normalization rate

---

## 2. Population Filtering and Masking

### 2.1 Index-Based Filtering

```python
# Filter by neuron index
conn = (pop1.where_id(lambda i: i < 100) >> pop2)(pattern='all-to-all')
conn = (source.where_id(lambda ids: ids % 2 == 0) >> target)(...)
```

### 2.2 Position-Based Filtering

```python
# Filter by position (for SpatialGroups)
conn = (layer1.where_pos(lambda pos: pos[:, 0]**2 + pos[:, 1]**2 < 25) >> layer2)(
    pattern='distance'
)

# Select neurons in specific region
conn = (layer1.where_pos(lambda pos: pos[:, 0] > 0.0) >> layer2)(...)
```

### 2.3 Rank-Based Filtering (Multi-GPU)

```python
# Connect only to neurons on specific GPU
conn = (pop1 >> bridge.where_rank(1))(pattern='one-to-one')
```

### 2.4 Combined Filtering

```python
# Chain multiple filters
conn = (pop1.where_id(lambda i: i < 100)
             .where_pos(lambda pos: pos[:, 0] > 0) >> pop2)(...)
```

**Important:** Filters create clones - original groups are not modified.

---

## 3. Neuron Models

### 3.1 Available Models

```python
from neurobridge import (
    ParrotNeurons,      # Simple relay neurons
    SimpleIFNeurons,    # Basic integrate-and-fire
    RandomSpikeNeurons, # Poisson spike generators
    IFNeurons           # Multi-channel IF with conductances
)
```

### 3.2 ParrotNeurons

```python
neurons = ParrotNeurons(
    n_neurons=100,
    spatial_dimensions=2,
    delay_max=20,
    device='cuda:0'
)
```

Simply relay input spikes/currents without dynamics.

### 3.3 SimpleIFNeurons

```python
neurons = SimpleIFNeurons(
    n_neurons=100,
    spatial_dimensions=2,
    delay_max=20,
    threshold=1.0,        # Spike threshold
    tau_membrane=0.01,    # Membrane time constant (seconds)
    device='cuda:0'
)
```

Basic IF model with exponential decay and reset.

### 3.4 RandomSpikeNeurons

```python
neurons = RandomSpikeNeurons(
    n_neurons=100,
    firing_rate=10.0,     # Firing rate in Hz
    spatial_dimensions=2,
    delay_max=20,
    device='cuda:0'
)
```

Poisson spike generators, ignores input.

### 3.5 IFNeurons (Multi-Channel)

```python
neurons = IFNeurons(
    n_neurons=100,
    spatial_dimensions=2,
    delay_max=20,
    n_channels=3,         # AMPA, GABA, NMDA by default
    channel_time_constants=[
        (0.001, 0.005),   # AMPA: rise 1ms, decay 5ms
        (0.001, 0.010),   # GABA: rise 1ms, decay 10ms
        (0.002, 0.100),   # NMDA: rise 2ms, decay 100ms
    ],
    channel_reversal_potentials=[
        0.0,              # AMPA: 0 mV
        -0.070,           # GABA: -70 mV
        0.0,              # NMDA: 0 mV
    ],
    threshold=-0.050,     # -50 mV
    tau_membrane=0.010,   # 10 ms
    E_rest=-0.065,        # -65 mV
    device='cuda:0'
)
```

Realistic IF with conductance-based synapses, bi-exponential dynamics, and reversal potentials.

**Connecting to specific channels:**
```python
# AMPA input
(src >> tgt)(pattern='random', p=0.1, weight=0.001, channel=0)

# GABA input (inhibition)
(inh >> exc)(pattern='random', p=0.2, weight=0.001, channel=1)

# NMDA input
(src >> tgt)(pattern='random', p=0.05, weight=0.0005, channel=2)
```

---

## 4. Monitoring

### 4.1 SpikeMonitor

```python
monitor = SpikeMonitor([
    source.where_id(lambda i: i < 10),
    target.where_id(lambda i: i < 10)
])

# After simulation:
spikes = monitor.get_spike_tensor(group_index=0)  # [N_spikes, 2] (neuron_id, time)
```

### 4.2 VariableMonitor

```python
monitor = VariableMonitor(
    [neurons.where_id(lambda i: i < 5)],
    ['V', 'spikes', 'channel_currents@0']  # @ for indexing extra dimensions
)

# After simulation:
voltages = monitor.get_variable_tensor(0, 'V')  # [T, N]
ampa_currents = monitor.get_variable_tensor(0, 'channel_currents@0')  # [T, N]
```

### 4.3 RingBufferSpikeMonitor

```python
monitor = RingBufferSpikeMonitor(
    [neurons],
    max_spikes=1_000_000  # Pre-allocated buffer size
)

# Can retrieve during or after simulation:
spikes = monitor.get_spike_tensor(0, to_cpu=True)
```

CUDA-graph compatible, efficient for continuous monitoring.

---

## 5. Experiment Framework

```python
from neurobridge import Experiment, Simulator

class MyExperiment(Experiment):
    def build_network(self):
        # Create neurons and connections
        with self.sim.autoparent("graph"):  # Inside CUDA graph
            self.neurons = IFNeurons(n_neurons=100, device=self.local_device)
            self.source = RandomSpikeNeurons(n_neurons=50, firing_rate=10.0)
            (self.source >> self.neurons)(pattern='random', p=0.1, weight=0.05)

        with self.sim.autoparent("normal"):  # Outside graph (monitors)
            self.monitor = SpikeMonitor([self.neurons])

    def on_start(self):
        print("Starting simulation...")

    def pre_step(self):
        pass  # Called before each step

    def pos_step(self):
        pass  # Called after each step

    def on_finish(self):
        spikes = self.monitor.get_spike_tensor(0)
        print(f"Recorded {len(spikes)} spikes")

# Run:
exp = MyExperiment(sim=Simulator(seed=42))
exp.run(steps=10_000)
```

### Useful properties:
- `self.step`: Current simulation step
- `self.time`: Current time in milliseconds
- `self.local_rank`: GPU rank (multi-GPU)
- `self.world_size`: Number of GPUs
- `self.local_device`: This GPU's device

---

## 6. Multi-GPU Setup

### Single GPU:
```bash
python my_experiment.py
```

### Multiple GPUs:
```bash
torchrun --nproc_per_node=2 my_experiment.py
```

### Using Bridge for Inter-GPU Communication:

```python
class MultiGPUExperiment(Experiment):
    def build_network(self):
        # Add bridge
        self.add_default_bridge(n_local_neurons=100, n_steps=10)
        bridge = self.sim.local_circuit.bridge

        if self.local_rank == 0:
            with self.sim.autoparent("graph"):
                source = RandomSpikeNeurons(n_neurons=100, firing_rate=10.0)
                # Send to GPU 1
                (source >> bridge.where_rank(1))(pattern='one-to-one', weight=1.0)

        elif self.local_rank == 1:
            with self.sim.autoparent("graph"):
                target = IFNeurons(n_neurons=100)
                # Receive from GPU 0
                (bridge.where_rank(0) >> target)(pattern='one-to-one', weight=0.5)
```

**Bridge parameters:**
- `n_local_neurons`: Neurons per GPU in the bridge
- `n_steps`: Aggregation window (latency = n_steps + 1)

---

## 7. Current Limitations

1. **No state save/load** - Checkpointing not implemented
2. **Dense connections always allocate full matrix** - No automatic sparse conversion
3. **Bridge only for single-node multi-GPU** - No inter-node distribution
4. **No connection manipulation** - Can't prune or modify connections after creation
5. **Random positions by default** - No built-in grid/topology generators

---

## 8. Performance Tips

1. **Use CUDA Graphs:** Put critical code in `autoparent("graph")`
2. **Choose sparse vs dense wisely:** Sparse for <10% connectivity, dense otherwise
3. **Minimize monitors:** Only monitor what you need
4. **Tune bridge latency:** Lower `n_steps` = lower latency but more overhead
5. **Batch operations:** Create multiple connections together rather than one-by-one

---

## 9. Quick Reference

### Import Statement:
```python
from neurobridge import (
    Experiment, Simulator,
    IFNeurons, SimpleIFNeurons, RandomSpikeNeurons, ParrotNeurons,
    StaticSparse, STDPSparse, StaticDense, STDPDense,
    SpikeMonitor, VariableMonitor,
    log, show_or_save_plot
)
```

### Basic Workflow:
```python
class MyExp(Experiment):
    def build_network(self):
        # 1. Create neurons
        # 2. Connect them
        # 3. Add monitors

exp = MyExp(sim=Simulator())
exp.run(steps=10_000)
```

---

**For more details, see:**
- Working examples in `examples/` directory
- Architectural overview in `docs/Architectural overview.md`
- Project status in root `PROJECT_STATUS.md`
