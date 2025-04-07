# NeuroBridge Connection API Implementation Roadmap

This document outlines the implementation roadmap for the enhanced NeuroBridge connection API, organized by priority and dependency.

## Phase 1: Core Connection Operator

**Target: Establish the fundamental connection mechanism with the `>>` operator.**

### Step 1.1: Base Connection Classes (HIGH PRIORITY)

1. Create a `ConnectionSpec` class that handles connection creation parameters
2. Implement the `__rshift__` operator (`>>`) for neuron populations
3. Create a `Connection` class to represent the result of a connection operation
4. Implement basic connection parameter validation

```python
# Example target implementation:
pop1 >> pop2(pattern='all_to_all', weight=1.0, delay=1)
```

### Step 1.2: Basic Connection Patterns (HIGH PRIORITY)

1. Implement the following connection patterns:
   - `all_to_all`: Connect every source to every target
   - `one_to_one`: Connect matching indices
   - `random`: Connect with probability p or fixed fanin/fanout
2. Add support for custom connection patterns through explicit indices

```python
# Random connectivity by probability
pop1 >> pop2(pattern='random', p=0.1)

# Random connectivity by fanin/fanout
pop1 >> pop2(pattern='random', fanin=10)
pop1 >> pop2(pattern='random', fanout=20)

# Custom indices
idx_pre = torch.tensor([0, 1, 2])
idx_pos = torch.tensor([5, 6, 7])
pop1 >> pop2(pattern=(idx_pre, idx_pos), weight=1.0)
```

### Step 1.3: Connection Object Methods (HIGH PRIORITY)

1. Create methods to inspect connection properties:
   - `get_indices()`
   - `get_weights()`
   - `get_delays()`
   - `get_density()`
2. Implement basic connection manipulation methods:
   - `scale_weights(factor)`
   - `set_weights(new_weights)`

## Phase 2: Advanced Connectivity Features

**Target: Expand connectivity options with functional parameters and synapse type selection.**

### Step 2.1: Function-Based Parameters (HIGH PRIORITY)

1. Extend `ConnectionSpec` to handle callable parameters
2. Implement parameter evaluation based on relevant properties
3. Add support for different parameter function signatures

```python
# Weight as function of index
pop1 >> pop2(weight=lambda i, j: 1.0 if i < j else 0.5)

# Weight as function of multi-dimensional coordinates
layer1 >> layer2(
    pattern='distance',
    weight=lambda x, y: 0.5 * torch.exp(-(x**2 + y**2)/10)
)

# Custom distance function
layer1 >> layer2(
    pattern='distance',
    distance_func=lambda src, tgt: torch.sqrt(((src - tgt)**2).sum()),
    weight=lambda d: 0.5 * torch.exp(-d/10)
)
```

### Step 2.2: Synapse Type Selection (MEDIUM PRIORITY)

1. Extend `ConnectionSpec` to handle synapse class selection
2. Create a mechanism to instantiate the specified synapse class
3. Implement parameter forwarding to specific synapse types

```python
# Using synapse class
pop1 >> pop2(
    synapse=STDPSynapse,
    weight=0.5,
    stdp_params={'A_plus': 0.01}
)
```

### Step 2.3: Enhanced Connection Manipulation (MEDIUM PRIORITY)

1. Add more sophisticated connection manipulation methods:
   - `prune_weakest(fraction)`
   - `prune_by_weight(threshold)`
   - `filter(condition_func)`
2. Implement connection statistics and analysis functions

## Phase 3: Topological Populations and Distance-Based Connectivity

**Target: Add support for spatial arrangements of neurons and distance-based connectivity.**

### Step 3.1: Topological Population Base (HIGH PRIORITY)

1. Create a `TopologyMixin` class to be inherited by neuron populations
2. Implement different topology types (1D, 2D, 3D, custom)
3. Add position generation and access methods
4. Implement periodic boundary option as a property of the topology

```python
# Create 2D grid of neurons with periodic boundaries
layer = population(400, neuron_type=IFNeuron, topology='2d', 
                  shape=(20, 20), periodic=True)

# Access positions
positions = layer.positions  # Returns tensor of shape [n_neurons, 2]
```

### Step 3.2: Distance-Based Connectivity (HIGH PRIORITY)

1. Implement distance-based connection pattern
2. Add support for different distance metrics
3. Create efficient distance computation methods
4. Implement fanin/fanout options for distance-based connections

```python
# Connect neurons based on distance with probability
layer1 >> layer2(pattern='distance', max_distance=5.0, p_max=1.0)

# Connect each target to its 10 closest sources
layer1 >> layer2(pattern='distance', fanin=10)
```

### Step 3.3: Custom Distance Functions (MEDIUM PRIORITY)

1. Add support for user-defined distance functions
2. Implement efficient application of custom distance functions
3. Create common distance function presets

```python
# Custom distance function
layer1 >> layer2(
    pattern='distance',
    distance_func=lambda src, tgt: torch.sqrt(((src[:2] - tgt[:2])**2).sum()),
    weight=lambda d: 0.5 * torch.exp(-d/10)
)
```

## Phase 4: Population Filtering and Masking

**Target: Add ability to select subsets of neurons for connections.**

### Step 4.1: Index-Based Filtering (HIGH PRIORITY)

1. Implement `where_id` method for neuron populations
2. Create filter application mechanism
3. Ensure filters pass through the connection operation

```python
# Connect only specific neurons based on index
pop1.where_id(lambda i: i < 100) >> pop2(pattern='all_to_all')
```

### Step 4.2: Position-Based Filtering (MEDIUM PRIORITY)

1. Implement `where_pos` method for topological populations
2. Create predefined region selectors (center, quadrant, etc.)
3. Add error handling for incompatible operations

```python
# Filter by position
layer1.where_pos(lambda x, y: x**2 + y**2 < 25) >> layer2(pattern='distance')

# Use predefined regions
layer1.where_region('center', radius=3) >> layer2(pattern='all_to_all')
```

### Step 4.3: Combined Filtering (LOW PRIORITY)

1. Enable chaining of different filter types
2. Optimize filter application for performance
3. Add filter visualization tools

```python
# Combine filters
pop1.where_id(lambda i: i % 2 == 0).where_pos(lambda x, y: x > 0) >> pop2()
```

## Phase 5: Network and Group Operations

**Target: Support operations on groups of populations.**

### Step 5.1: Basic Network Class (MEDIUM PRIORITY)

1. Create a `Network` class to contain multiple populations
2. Implement population addition and retrieval
3. Add network-level connection operations

```python
# Create a network
network = Network()
network.add(pop1, name="input")
network.add(pop2, name="hidden")

# Connect entire network
network >> external_pop(pattern='random', p=0.05)
```

### Step 5.2: Network Component Selection (MEDIUM PRIORITY)

1. Implement indexing for network populations
2. Enable connections between network components
3. Add support for multiple population selection

```python
# Connect specific populations
network["input"] >> network["hidden"](pattern='all_to_all')
```

### Step 5.3: Network Filtering (LOW PRIORITY)

1. Extend filtering mechanisms to network level
2. Implement specialized network-level filters
3. Add support for cross-population filters

```python
# Filter at network level
network.where_region('center', radius=5) >> external_pop()
```

## Phase 6: Advanced Features and Utilities

**Target: Add productivity features and tools for complex networks.**

### Step 6.1: Connection Templates (MEDIUM PRIORITY)

1. Create a `ConnectionTemplate` class
2. Implement template definition and application
3. Add parameter override capabilities

```python
# Define and apply template
template = ConnectionTemplate()
template.add("E→E", pattern="random", p=0.1, weight=0.2)
template.add("E→I", pattern="random", p=0.4, weight=0.3)

template.apply(E=exc_pop, I=inh_pop)
```

### Step 6.2: Predefined Templates (LOW PRIORITY)

1. Implement common network motifs as predefined templates
2. Create scaling mechanisms for templates
3. Add validation for template application

```python
# Use built-in template
CorticalMicrocircuit().apply(L4E=l4e, L4I=l4i, L2E=l2e, L2I=l2i)
```

### Step 6.3: Visualization Utilities (LOW PRIORITY)

1. Create standalone visualization functions
2. Implement connectivity matrix visualization
3. Add weight distribution and spatial projection visualizations

```python
# Visualize connections
visualize_connectivity(conn)
visualize_weight_distribution(conn)
```

## Phase 7: Documentation and Examples

**Target: Ensure users can effectively utilize the API.**

### Step 7.1: API Documentation (HIGH PRIORITY)

1. Create comprehensive docstrings for all public methods
2. Implement type annotations for better IDE support
3. Add examples to docstrings

### Step 7.2: Jupyter Notebook Tutorials (MEDIUM PRIORITY)

1. Create basic usage tutorials
2. Add advanced pattern examples
3. Implement complete network examples

### Step 7.3: Performance Guide (LOW PRIORITY)

1. Document performance characteristics of different patterns
2. Create guidelines for efficient large-scale network creation
3. Add benchmarks for comparison

## Implementation Strategy

### Iterative Approach

1. Implement and test each phase independently
2. Release incrementally rather than waiting for the full implementation
3. Gather feedback on early phases to refine later implementations

### Backwards Compatibility

1. Maintain compatibility with existing code where possible
2. Provide migration guides for breaking changes
3. Consider a parallel API strategy during transition

### Testing Strategy

1. Create unit tests for each API component
2. Implement integration tests for complex scenarios
3. Add performance benchmarks to track efficiency

### Documentation Update Process

1. Update documentation with each phase release
2. Include migration guides for existing users
3. Create example code that demonstrates new features
