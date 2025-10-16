# Conceptual Guide / White Paper / Architectural Overview

## 1. Purpose, Audience, and Niche

The central purpose of this library is to simulate **Spiking Neural Networks (SNNs)** on a single computer equipped with multiple GPUs. Its design targets two main objectives:
1. **Fully leverage GPU parallel computing power** to run simulations with large numbers of neurons and synapses.  
2. **Minimize inter-GPU communication latency**, allowing results to be used in fast-response contexts such as **real-time robotics** or interactive experiments.

The intended audience is not absolute beginners. It is aimed at:
- Researchers and developers with an **intermediate to advanced level of PyTorch**, familiar with CUDA, `torch.distributed`, and the NCCL backend.  
- Users with **some knowledge of computational neuroscience**, capable of understanding neuron models such as _Integrate-and-Fire_ (IF) and plasticity rules such as STDP (_Spike-Timing-Dependent Plasticity_).  

The library’s niche differs from well-known alternatives:
- Compared to general frameworks such as **NEST** or **Brian2**, this library specializes in **multi-GPU operation within a single node**, avoiding multinode distributed overhead.  
- Compared to projects such as **BindsNET**, its focus is less on user-friendly interfaces and more on **compact and efficient execution**, featuring **CUDA Graphs** and **inter-GPU spike bridges**.  

In summary, it is positioned as a tool for those seeking a **balance between expressiveness and performance**, especially when the priority is to run real-time SNN simulations on locally available GPU hardware.

## 2. Execution Model and Scene (Node Tree)

The library organizes the simulation through a **hierarchical node structure**. Each model element — whether a neuron group, a synaptic connection, or a monitor — is a **`Node`** that may contain child nodes.

Variants of this abstraction include:
- **`Node`**: a generic base with no device assignment.  
- **`GPUNode`**: a node bound to a specific GPU, with a `device` attribute indicating where its tensors and operations run.  
- **`ParentStack`**: an internal mechanism that automatically manages the node tree. When a new group or connection is declared within a context, it becomes attached to the correct parent node.  

The execution cycle relies on **two key methods** each node may implement:
- **`_ready()`**: pre-simulation initialization. The node tree is recursively traversed to prepare buffers, parameters, and dependencies.  
- **`_process()`**: executes a single simulation step. It is also called recursively so that each node updates its state in sequence.  

A major feature is the separation between:
- The **“capturable” subtree** (instance of `CUDAGraphSubTree`), whose operations are recorded into a **CUDA Graph** for efficient replay each cycle.  
- The **remaining nodes**, which handle auxiliary tasks such as monitoring or logging and are excluded from capture.  

Temporal evolution is managed by the **`LocalCircuit.t`** counter, a GPU-resident tensor representing the current simulation step. This counter increments in unit steps and combines with a **circular indexing** scheme of size `delay_max` for spike buffers. The scheme allows modeling synaptic delays up to `delay_max – 1` steps, optimizing memory and access.

Together, this hierarchical model ensures that:
- Each simulation component integrates cleanly into the global cycle.  
- The compute-critical part is contained within a CUDA Graph, minimizing per-kernel call overhead.  
- Time and delay handling remain coherent and efficient on GPU.

## 3. Simulation Engine (`Simulator` and `LocalCircuit`)

The library’s operational core consists of **`Simulator`** and **`LocalCircuit`**, which initialize, coordinate, and execute simulation steps.

**Initialization**
- The number of available GPUs is detected automatically.  
- If the script runs under `torchrun` with environment variables (`WORLD_SIZE`, `RANK`), the distributed mode with **NCCL** backend is activated. Each process receives its `rank` and binds to the corresponding GPU.  
- `LocalCircuit` selects the `device` and sets a random seed, ensuring reproducibility per GPU/rank.  

**Execution Graph Preparation**
- The simulator distinguishes nodes belonging to the **capturable subtree** (`CUDAGraphSubTree`) from others.  
- A multi-step **warm-up** initializes kernels and buffers.  
- Optionally, **`torch.compile`** is used for function optimization.  
- A **CUDA Graph** is then captured, encapsulating the `_process()` cycle of all capturable nodes.  

**Execution**
- Each call to `step()` advances the simulation one time step.  
- If a captured graph exists, it is executed directly via `graph.replay()`, avoiding repeated kernel launch overhead.  
- The time counter `t` increments and synchronizes with circular spike buffers.  

**Finalization**
- On termination, distributed groups are safely released (`torch.distributed.destroy_process_group()`).
- The `Simulator` object cleans up resources and maintains inter-process consistency.  

In essence, the engine handles all technical infrastructure so that the user can focus on neuronal and synaptic modeling, delegating GPU management, process communication, and execution capture to the central system.

## 4. Groups and Filtering (Selections)

Neuronal populations and other entity sets are organized via the **`Group`** class and its derivatives. It manages collections (e.g., neurons or connections) in a vectorized and flexible manner.

**Key Features**
- Each group has a `size` attribute specifying the number of elements.  
- A **filter vector** (`filter`) marks which subset is active for each operation.  
- Selection methods (`where_*`) return copies of the group with adjusted filters, leaving the original intact.

**Selection Types**
- **`where_id(indices)`**: activates elements by absolute identifier.  
- **`where_pos(condition)`** (in `SpatialGroup`): selects by spatial coordinates (position in cube/grid).  
- **`where_rank(r)`**: filters elements associated with a given distributed execution rank.

**Practical Use**
- Filters are **composable**: they can be applied sequentially for refinement.  
- Groups can be **cloned** to define independent filters while preserving the original as a global reference.  
- Useful, for example, to select excitatory or inhibitory neuron subsets or define specific connection targets.

**Conceptual Advantage**  
This mechanism lets large populations be treated as global GPU tensors while enabling dynamic subset operations without physically copying data.

## 5. Base Neuron Model and Variants

Neuronal behavior is defined through the **`NeuronGroup`** class, a common template for all neuron models.

**Internal Structure**
- Each neuron keeps a **circular spike buffer** with dimensions `[N, delay_max]`, where `N` is group size.  
- The active index corresponds to `t % delay_max`.  
- External or internal currents can be injected into the current step via `add_current_in_step`.  

**Spike Reading**
- **`get_spikes()`** returns the spike vector at the current instant.  
- **`get_spikes_at(delay)`** accesses delayed states, useful for modeling transmission delays.  

**Implemented Models**
- **`ParrotNeurons`**: have no intrinsic dynamics; simply relay incoming spikes/currents.  
- **`SimpleIFNeurons`**: basic _Integrate-and-Fire_ model with threshold, reset, and exponential decay.  
- **`RandomSpikeNeurons`**: probabilistic spike generators following an independent Poisson rate.  
- **`IFNeurons`** (extended): include multiple conductance channels with bi-exponential decay and reversal potentials, enabling realistic excitatory/inhibitory inputs.  

**Key Points**
- The unified API ensures any neuron model connects identically to synapses and monitors.  
- The circular buffer synchronized with `delay_max` enables access to past spikes without storing full histories.

## 6. Synaptic Connections: Sparse and Dense

Connections between neuron groups are built via the **`>>`** operator, returning a **`ConnectionOperator`** that combines:
1. A **connectivity pattern** (`all-to-all`, `one-to-one`, `specific`).  
2. A **synapse class** defining transmission and plasticity dynamics.  

**Sparse Connections (`ConnectionGroup`)**
- Represented as flat index lists (`idx_pre`, `idx_pos`).  
- Each connection stores its parameters: weight, delay, and channel.  
- Propagation uses vectorized `index_add_`, summing presynaptic currents onto postsynaptic targets.  
- Presynaptic spike access uses the corresponding circular buffer and connection-specific delay.  

**Dense Connections (`ConnectionDense`)**
- Represented as weight matrices `[N_pre, N_pos]`.  
- Propagation via matrix multiplication (`matmul`), efficient for dense connectivity.  
- Plastic variants exist (`STDPDenseConnection`).  

**Parameter Handling**  
Parameter initialization is handled by `_compute_parameter`, supporting:
- Scalars (uniform values).  
- Tensors (explicit per-connection values).  
- Lists or functions (procedural generation).  

**Structural Constraints**
- Connection delay must be **less than `delay_max`** of the presynaptic group.  
- The specified channel must exist in the postsynaptic group.

This sparse/dense duality allows modeling both biologically irregular networks and high-density experimental setups, balancing flexibility and computational efficiency.

## 7. Inter-GPU Communication: `BridgeNeuronGroup`

When simulations span multiple GPUs, neuron populations must exchange spikes across devices. This is handled by **`BridgeNeuronGroup`**, a specialized node acting as a gateway.

**Main Function**
- Collects spikes generated on one GPU and transmits them to others.  
- Inserts received spikes into destination buffers, time-shifted to compensate for communication latency.  

**Operation**
1. During `n_bridge_steps`, local spikes accumulate in a **temporary buffer (`_write_buffer`)**.  
2. After the block, spikes are packed (bool→uint8) to reduce size.  
3. A collective **`dist.all_gather(async_op=True)`** exchanges data among all processes (one GPU per rank).  
4. Results are unpacked and written into the **future spike buffer (`_spike_buffer`)** with a time offset ≥ `n_bridge_steps + 1`.  

**Validity Conditions**
- Requires **`n_bridge_steps < delay_max`** so transmitted spikes remain within the valid time ring.  
- Effective communication latency is at least `n_bridge_steps`, relevant for real-time control.  

**Subpopulation Selection**
- **`where_rank(r)`** defines neuron subsets visible only to a specific rank, allowing fine-grained distributed architectures.  

**Non-distributed Mode**
- In single-GPU runs, the bridge performs direct GPU copies preserving latency semantics.

Thus, `BridgeNeuronGroup` transforms a local simulation into a **multi-GPU distributed simulation**, maintaining synchronization and spike-exchange efficiency.

## 8. Monitors and Data Extraction

The library includes **monitors** to record neuronal activity and internal variables without interrupting simulation.

**Main Types**

- **`SpikeMonitor`**  
    - Reads spikes from circular buffers.  
    - Extracts data only when `t % delay_max == delay_max – 1` (end of time ring cycle).  
    - Returns `(neuron_id, time)` pairs for reconstructing spike trains.  
- **`VariableMonitor`**  
    - Records arbitrary tensor values each step.  
    - Filters by neuron subsets defined via `where`.  
    - Accumulates data in CPU arrays for later analysis.  
- **`RingBufferSpikeMonitor`**  
    - Designed for continuous GPU-side collection.  
    - Maintains a **circular GPU buffer** storing recent spikes.  
    - Periodically copies blocks to **pinned CPU memory** via asynchronous transfers (`non_blocking=True`).  
    - Suitable for near-real-time readout without degrading performance.

**Common Features**
- Monitors are excluded from CUDA Graph capture, avoiding overhead in the critical loop.  
- They execute safely at controlled cycle points (`pos_step`, `on_finish`).  
- Provide a balance between **performance** and **data accessibility**, keeping compute on GPU and deferring CPU transfer.  

Overall, monitors form the standard interface for observing and analyzing network dynamics without compromising simulation efficiency.

## 9. Experiment Lifecycle (`Experiment`)

The **`Experiment`** class organizes full simulation logic, providing a standard framework to build, run, and control complete experiments.

**Main Methods**
- **`build_network()`**: explicitly constructs the node graph (neuronal groups, synapses, monitors). For multi-GPU runs, also adds the default bridge (`add_default_bridge`).  
- **`on_start()`**: executed once before simulation, useful for initializing variables or preparing logs.  
- **`pre_step()`** / **`pos_step()`**: hooks called before and after each simulation step, allowing controlled interventions (e.g., stimulus injection, periodic logging).  
- **`on_finish()`**: executed after simulation completion for saving results or releasing resources.  

**Execution**
- **`run(steps)`** executes the specified number of steps:  
    - Initializes the simulator and capture subtree if not done.  
    - Calls `on_start()`.  
    - Iterates through steps invoking `pre_step()`, graph replay, and `pos_step()`.  
    - Finally executes `on_finish()`.  
- Error handling ensures distributed groups are cleaned up even on interruption.  

**Autoparenting Contexts**
- **`autoparent("graph")`**: declarations inside are included in the CUDA-capturable subtree.  
- **`autoparent("normal")`**: declarations outside the critical graph (monitors, utilities).  

**Structural Advantage**  
`Experiment` clearly separates **model definition** from **controlled execution**, standardizing workflow and enhancing reproducibility.

## 10. Utilities and Logging

Beyond the main engine, the library provides utilities that facilitate experiment instrumentation.

**Logging**
- Implements rank-differentiated output in distributed runs.  
- Each process can print color-coded messages to identify origin.  
- Logging is used across core (`Simulator`, `Experiment`) and examples to provide traceability without flooding standard output.  

**Visualization and Analysis**
- Includes simple **plotting** functions for raster plots or time series.  
- Automatically switches between **display** and **file-save** modes depending on execution context (interactive or batch).  
- Provides a `smooth_spikes` utility applying convolution windows to binary spike data, useful for estimating firing rates.  

**Environment Detection**
- Helpers like **`is_distributed`** check whether multiple processes (`torch.distributed`) are active.  
- **`can_use_torch_compile`** tests availability of `torch.compile` and enables it safely.  

**Pedagogical Role**  
These utilities are not part of the simulation core but are essential for turning a simulation into a **reproducible, analyzable experiment**, reducing user effort on routine tasks.

## 11. Included Examples (Quick Map)

The library ships with example scripts demonstrating how to build and run various simulations. They serve both educational and validation purposes as minimal templates.

**Main Examples**

- **`lib_example_01_multipleGPUs.py`**  
    - Demonstrates spike communication between GPUs via `BridgeNeuronGroup`.  
    - Implements a simple “ping-pong” activity exchange between GPUs.  
- **`lib_example_02_oneGPU.py`**  
    - Single-GPU simulation example.  
    - Includes sparse STDP synapses and monitors for recording activity.  
- **`lib_example_03_twoGPUs.py`**  
    - Builds a network distributed across two GPUs.  
    - A source on GPU0 sends spikes through the bridge to a target on GPU1 with STDP.  
    - Shows how to combine multi-GPU and plasticity.  
- **`lib_example_04_BRN_STDP.py`** and **`lib_example_04_BRN_STDP_multipleGPUs.py`**  
    - Implement a larger excitatory/inhibitory network with dense STDP plasticity.  
    - One version runs on a single GPU; the other distributes across several.  

**Practical Utility**  
These examples illustrate real-world usage patterns:
- How to initialize an experiment and build the graph in `build_network`.  
- How to add monitors and collect data.  
- How to configure dense or sparse connections.  
- How to run locally or in distributed mode with `torchrun`.

Together, they form a **minimal reference manual** complementing documentation and clarifying component semantics.

## 12. Performance and Real-Time Operation

The library is designed for **efficiency** and **low latency**, prerequisites for near-real-time spike simulation.

**CUDA Graph Optimization**
- The simulation loop is captured in a **CUDA Graph** after a warm-up phase.  
- Each step then runs via a single `graph.replay()` call, eliminating CPU-side kernel launch overhead.  

**CPU-GPU Transfer Minimization**
- Critical data (spikes, states, weights) remain on GPU.  
- Monitors transfer to CPU in controlled blocks to reduce performance impact.  

**Spike Compression for Communication**
- Spikes are packed (bool→uint8) before inter-GPU transfer, achieving 8:1 compression.  
- This reduces collective-operation data volume, speeding communication with `torch.distributed`.  

**Critical Latency Parameters**
- **`n_bridge_steps`**: number of steps aggregated before sending spikes through the bridge. Higher values improve throughput but increase effective delay.  
- **`delay_max`**: circular buffer length; must exceed `n_bridge_steps` for temporal consistency.  
- **Connection density**: dense representations are memory-heavy; sparse are scalable but index-overhead-bound.  
- **Monitor buffer size**: trades off read frequency versus memory load.  

**Practical Recommendations**
- Include as much computation as possible inside the capturable subtree (`graph`) to benefit from CUDA Graphs.  
- Avoid dynamic allocations or CPU calls within `_process()`.  
- Tune `n_bridge_steps` according to acceptable delay (robotic control vs. offline simulation).  

These design choices allow the library to scale across multiple GPUs on a single host while maintaining performance suitable for strict-latency applications.

## 13. Limitations and Validation Points

Though powerful within its niche, the library has constraints and aspects that must be checked per use case.

**Structural Limitations**
- Distributed execution is limited to **multi-GPU within one host**; inter-node communication is not supported.  
- **State save/restore (`save_state`)** is unimplemented—each simulation starts fresh.  
- Random connectivity mask generation in examples occurs explicitly on CPU, potentially **high memory cost** for very large networks.  

**STDP Plasticity**
- In sparse STDP connections, the **depression term** adds `A_minus` rather than subtracting it. If `A_minus > 0`, both potentiation and “depression” increase the weight.  
- This must be validated (e.g., internal sign convention vs. user adjustment).  
- [ ] #task Verify that `A_minus` is correctly implemented in Neurobridge.  

**Monitor Timing**
- Spike monitors extract data only when a `delay_max` cycle completes, i.e., in block granularity rather than per step.  
- For continuous readout (e.g., closed-loop control), use `RingBufferSpikeMonitor`.  

**Other Considerations**
- Effective bridge latency depends on `n_bridge_steps`. For real-time applications, compute delay offset relative to `delay_max`.  
- With dense connections, GPU memory may become a limiting factor before compute does.  

In summary, the library offers a solid framework but requires users to **explicitly validate** plasticity, timing, and memory configurations to ensure intended behavior.

## 14. Extensibility

The library is designed to be **modular**, allowing users to add new models or tools without modifying the core.

**New Neuron Models**
- Inherit from **`NeuronGroup`**.  
- Must implement membrane dynamics and spike buffer handling.  
- Must comply with the current-injection (`add_current_in_step`) and spike-read (`get_spikes`, `get_spikes_at`) API.  
- Example: define an adaptive model with a slow adaptation current while maintaining compatibility with connections and monitors.  

**New Synapse Types**
- Derived from **`ConnectionGroup`** (sparse) or **`ConnectionDense`** (dense).  
- Implement two key methods:  
    - **`_init_connection`**: initializes parameters (weight, delay, channel).  
    - **`_update`**: defines propagation/plasticity rule per step.  
- Enables easy addition of new STDP variants, Hebbian rules, or triplet-based models.

**New Monitors**
- Based on **`Node`**, collecting data in `_process` or hooks (`pos_step`, `on_finish`).  
- Intensive CPU operations must occur **outside the capturable subtree** to preserve CUDA Graph efficiency.  

**General Flexibility**
- The strict separation between capturable and normal nodes allows integrating extra code without breaking the critical simulation path.  
- The hierarchical node architecture ensures extensions naturally fit into the lifecycle (`_ready`, `_process`).  

Thus, extensibility turns the library into an open platform for exploring new neuron models, synaptic rules, or observation mechanisms without touching the core engine.

## 15. Setup and Distributed Execution

The library can run on **a single GPU** or **multiple GPUs within one host**.

**Local Mode (Single GPU)**
- Simply launch the experiment script with normal Python.  
- The simulator detects the `device` and assigns tensors to that GPU.  
- No distributed communication is initialized.  

**Distributed Mode (Multi-GPU)**
- Uses **`torchrun`** as launcher, creating one process per GPU.  
- Typical command:  
    ```bash
    torchrun --nproc_per_node=K script.py
    ```
    where `K` is the number of GPUs on the node.  
- Each process automatically receives `WORLD_SIZE` and `RANK`.  
- Communication backend: **NCCL**, optimized for NVIDIA hardware.  
- Each process selects its GPU via `cuda:rank`.  

**Automatic Initialization**
- The simulator detects `WORLD_SIZE`/`RANK` variables and enters distributed mode.  
- Creates and safely destroys the distributed process group at termination.  
- No manual `torch.distributed.init_process_group` calls required.  

**Bridge Compatibility (`BridgeNeuronGroup`)**
- In distributed mode, bridges handle collective spike communication.  
- In local mode, they still operate using direct GPU copies to preserve latency semantics.  

**Summary**  
Setup is straightforward:
- **Single GPU** → run directly with Python.  
- **Multi-GPU (single node)** → run with `torchrun`.  

This design enables scaling a simulation without changing code—only the launch mode.