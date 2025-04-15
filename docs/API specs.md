# NeuroBridge Connection API Specification

This document outlines the specifications for the enhanced connection API in the NeuroBridge framework. The core design goals are:

1. Provide an intuitive, Pythonic API for creating neural connections
2. Support various connection patterns with minimal code
3. Enable memory-efficient representations for large-scale simulations
4. Allow flexible parameter specification, including functional parameters
5. Support both topological and non-topological populations

## 1. Core Connection Operator (`>>`)

The primary mechanism for creating connections is the `>>` operator:

```python
(source_pop >> target_pop)(pattern='random', p=0.1, weight=0.5, delay=1)
```

### 1.1 Basic Parameters

- `pattern`: Connection pattern type (`'all_to_all'`, `'random'`, `'one_to_one'`, `'distance'`, etc.)
- `weight`: Synaptic weight (scalar, tensor, or function)
- `delay`: Synaptic delay (scalar, tensor, or function)
- `synapse`: Synapse class (e.g., `StaticSynapse`, `STDPSynapse`, or custom class)
- `sparse`: Format selection (`True`, `False`)

### 1.2 Pattern-Specific Parameters

#### 1.2.1 Random Connectivity

```python
# Connection by probability
conn = (pop1 >> pop2)(pattern='random', p=0.1)

# Connection by fixed fan-in (each target receives exactly 10 inputs randomly)
conn = (pop1 >> pop2)(pattern='random', fanin=10)

# Connection by fixed fan-out (each source connects to exactly 20 targets randomly)
conn = (pop1 >> pop2)(pattern='random', fanout=20)
```
- `p`: Connection probability (0.0-1.0)
- `fanin`: Fixed number of inputs per target neuron
- `fanout`: Fixed number of outputs per source neuron
- `allow_self_connections`: For recurrent connections (default: False)
- `seed`: For reproducibility

#### 1.2.2 Distance-Based Connectivity

```python
conn = (layer1 >> layer2)(
    pattern='distance', 
    max_distance=5.0,
    p_max=1.0,
    sigma=2.0
)

# Distance-based with fixed fan-in
conn = (layer1 >> layer2)(
    pattern='distance',
    fanin=10,  # Each target connects to 10 close sources randomly
    sigma=2.0  # Normal distribution of probability
)
```
- `max_distance`: Maximum connection distance
- `p_max`: Maximum connection probability (at distance=0)
- `sigma`: Width parameter for probability decay
- `fanin`: Fixed number of inputs per target neuron (selects closest sources)
- `fanout`: Fixed number of outputs per source neuron (selects closest targets)
- `prob_func`: Custom probability function (optional)

#### 1.2.3 Specific Connectivity

```python
conn = (pop1 >> pop2)(pattern='specific', sources=[0, 1, 2], targets=[5, 6, 7])
```
- `sources`: List of source neuron indices
- `targets`: List of target neuron indices

### 1.3 Function-Based Parameters

Parameters can be functions of relevant properties. Depending on the number of parameters accepted by the lambda function, distance (1 param) or absolute coordinates (2 params) can be used to calculate the values.

```python
# Weight as function of distance in 2D
conn = (layer1 >> layer2)(
    pattern='distance',
    max_distance=10.0,
    weight=lambda src, tgt: torch.sqrt(((src[:2] - tgt[:2])**2).sum()) #Two params, so absolute coordinates with src and tgt are given
)

# Weight using custom distance function with absolute coordinate access
conn = (ayer1 >> layer2)(
    pattern='distance',
    prob_func=lambda src, tgt: torch.sqrt(((src[:2] - tgt[:2])**2).sum()), #Two params, abs coord
    weight=lambda d: 0.5 * torch.exp(-d/10) #One param, so distances are given
)

# Delay proportional to distance
(layer1 >> layer2)(
    pattern='distance',
    max_distance=10.0,
    delay=lambda d: 1 + d.floor() #One param, so distance
)
```

### 1.4 Synapse Type Selection

```python
# Default static synapse
conn = (pop1 >> pop2)(weight=1.0)

# STDP synapse with custom parameters
conn = (pop1 >> pop2)(
    synapse=STDPSynapse,
    weight=0.5,
    stdp_params={
        'A_plus': 0.01,
        'A_minus': 0.0105,
        'tau_plus': 20.0
    }
)

# Custom synapse class
conn = (pop1 >> pop2)(
    synapse=MyCustomSynapse,
    weight=0.5,
    my_param=42
)
```

## 2. Population Filtering and Masking

### 2.1 Index-Based Filtering

```python
# Filter by neuron index
conn = (pop1.where_id(lambda i: i < 100) >> pop2)(pattern='all_to_all')
```

### 2.2 Position-Based Filtering (for topological populations)

```python
# Filter by neuron position
conn = (layer1.where_pos(lambda x, y: x**2 + y**2 < 25) >> layer2)(pattern='distance', max_distance=5)

# Using predefined regions
conn = (layer1.where_region('center', radius=3) >> layer2)(pattern='all_to_all')
conn = (layer1.where_region('quadrant', which=1) >> layer2)(pattern='all_to_all')
```

### 2.3 Combined Filtering

```python
# Combine multiple filters
conn = (pop1.where_id(lambda i: i % 2 == 0).where_pos(lambda x, y: x > 0) >> pop2)(pattern='random', p=0.1)
```

## 3. Connection Objects

The `>>` operator returns a connection object with various methods:

```python
conn = (pop1 >> pop2)(pattern='random', p=0.1)
```

### 3.1 Connection Properties

```python
# Access connection information
print(f"Created {len(conn.indices)} connections")
print(f"Average weight: {conn.weights.mean()}")
# Maybe more info?
```

### 3.2 Connection Manipulation

```python
# Scale all weights
conn.scale_weights(factor=0.5)

# Prune connections
conn.prune_weakest(fraction=0.2)
conn.prune_by_weight(threshold=0.1)

# Apply mask to existing connections
conn.filter(lambda w, d: w > 0.1 and d < 3) #This requires more work to be correctly defined
```

## 4. Network and Group Operations

### 4.1 Network Definition

```python
# Create a network
network = Network()
network.add(pop1, name="input")
network.add(pop2, name="hidden")
network.add(pop3, name="output")
```

### 4.2 Network-Level Connections

```python
# Connect entire network to external population
conn = (network >> external_pop)(pattern='random', p=0.05)

# Connect specific populations within network
conn = (network["input"] >> network["hidden"])(pattern='all_to_all')
conn = (network["input", "hidden"] >> network["output"])(pattern='random', p=0.1)
```

### 4.3 Network Filtering

```python
# Apply filters to network populations
conn = (network["input"].where_id(lambda i: i < 50) >> network["output"])(pattern='all_to_all')

# For topological networks
conn = (network.where_region('center', radius=5) >> external_pop)(pattern='distance', max_distance=3)
```

## 5. Topological Populations

### 5.1 Creating Topological Populations

```python
# 1D topology
layer_1d = population(100, neuron_type=IFNeuron, topology='1d', length=1.0)

# 2D grid topology with periodic boundaries
layer_2d = population(400, neuron_type=LIFNeuron, topology='2d', 
                      shape=(20, 20), extent=(1.0, 1.0), periodic=True)

# 2D with custom positions
positions = torch.rand(100, 2)  # Custom 2D positions
layer_custom = population(100, neuron_type=AdExNeuron, topology='custom', positions=positions)
```

### 5.2 Accessing Topology Information

```python
# Get neuron positions
positions = layer_2d.positions  # Returns tensor of shape [n_neurons, dimensions]

# Get distances between neurons
distances = layer_2d.distance_matrix()  # Full distance matrix between all neurons
```

## 6. Connection Templates

### 6.1 Defining Templates

```python
# Define connection template
template = ConnectionTemplate()
template.add("E->E", pattern="random", p=0.1, weight=0.2)
template.add("E->I", pattern="random", p=0.4, weight=0.3)
template.add("I->E", pattern="random", p=0.4, weight=-0.5)
template.add("I->I", pattern="random", p=0.1, weight=-0.2)
```

### 6.2 Applying Templates

```python
# Apply template to specific populations
template.apply(
    E=exc_pop,
    I=inh_pop
)

# Override parameters
template.apply(
    E=exc_pop,
    I=inh_pop,
    modifiers={"E->E": {"weight": 0.3}}
)
```

### 6.3 Predefined Templates

```python
# Use built-in templates
CorticalMicrocircuit().apply(L4E=l4e, L4I=l4i, L2E=l2e, L2I=l2i)
```
