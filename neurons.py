from __future__ import annotations
from typing import Union, Optional, Final
import torch
import numpy as np

from . import globals
from .utils import RandomDistribution
from .core import ConnectionOperator
from .group import SpatialGroup

# Define clear types for documentation purposes
NeuronParam = Union[float, torch.Tensor, RandomDistribution]
"""Neuronal parameter: can be a scalar (homogeneous), a tensor (heterogeneous),
or a random distribution."""

def resolve_neuron_param(
    value: NeuronParam,
    n: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Resolve a neuronal parameter into a flat tensor of shape (n,).

    Ensures that any parameter (scalar, distribution, or tensor)
    is converted into a state vector ready for simulation.
    """
    if isinstance(value, RandomDistribution):
        return value.sample(n, device).to(dtype)
    elif isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.expand(n).to(device=device, dtype=dtype)
        assert value.shape[0] == n, f"Shape {value.shape} does not match n={n}"
        return value.to(device=device, dtype=dtype)
    else:
        return torch.full((n,), float(value), device=device, dtype=dtype)


class NeuronGroup(SpatialGroup):
    """
    Base class for neuronal populations with delayed spike propagation.

    Computational Representation:
    ------------------------------
    Manages the discrete and continuous state of a population of N neurons.
    Implements a circular buffer (`_spike_buffer`) to handle axonal/dendritic
    delays in a vectorized manner without dynamic memory reallocation.

    General Cycle Dynamics:
    -----------------------
    1. **Input**: Accumulation of synaptic currents (`inject_currents`) and
       forced spikes (`inject_spikes`).
    2. **Process**: Integration of differential equations (implemented in subclasses).
    3. **Output**: Spike generation and writing to the historical buffer.

    Attributes:
    -----------
    Defines the temporal topology and memory capacity of the population.
    """

    # --- Static Configuration (Immutable after initialization) ---
    dt: Final[float]
    """Simulation time integration step [seconds]."""

    n_channels: Final[int]
    """Number of synaptic input channels (e.g. 1=Current, 3=AMPA/NMDA/GABA)."""

    delay_max: Final[torch.Tensor]
    """Maximum delay supported by the buffer [time steps] (scalar tensor)."""

    delay_max_int: Final[int]
    """Maximum delay as a primitive integer for indexing operations."""

    # --- Dynamic State (Mutable during simulation) ---
    _spike_buffer: torch.Tensor
    """Circular buffer of spike history. Shape: [n_neurons, delay_max] (bool)."""

    _input_currents: torch.Tensor
    """Accumulator for synaptic currents for step t+1. Shape: [n_neurons, n_channels] (float32)."""

    _input_spikes: torch.Tensor
    """Accumulator for forced (clamped) spikes for step t+1. Shape: [n_neurons] (bool)."""

    spikes: torch.Tensor
    """Vector of spikes emitted at the current time step t. Shape: [n_neurons] (bool)."""

    dYdt: torch.Tensor
    """Auxiliary variable to store derivatives (e.g. for surrogate-gradient learning). Shape: [n_neurons] (float32)."""

    def __init__(
        self,
        n_neurons: int,
        spatial_dimensions: int = 2,
        delay_max: int = 20,
        n_channels: int = 1,
        dt: float = 1e-3,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ):
        """
        Initialize the neuronal group and allocate memory on the device (CPU/GPU).

        Parameters
        ----------
        n_neurons : int
            Population size.
        spatial_dimensions : int
            Dimensions for spatial placement (e.g. 2 for a 2D grid).
        delay_max : int
            Size of the history buffer (in time steps).
            Defines the maximum connection delay.
        n_channels : int
            Number of independent conductance or current channels.
        dt : float
            Temporal resolution of the simulation in seconds.
        device : str | torch.device, optional
            Compute device.
        """
        super().__init__(n_neurons, spatial_dimensions, device, **kwargs)

        # Assign configuration constants
        self.dt = float(dt)
        self.n_channels = int(n_channels)
        self.delay_max = torch.tensor([delay_max], dtype=torch.long, device=self.device)
        self.delay_max_int = int(delay_max)

        # Allocate state memory (initialized to rest/silence)
        self._spike_buffer = torch.zeros(
            (n_neurons, self.delay_max_int), dtype=torch.bool, device=self.device
        )
        self._input_currents = torch.zeros(
            (n_neurons, self.n_channels), dtype=torch.float32, device=self.device
        )
        self._input_spikes = torch.zeros(n_neurons, dtype=torch.bool, device=self.device)

        self.spikes = torch.zeros(n_neurons, dtype=torch.bool, device=self.device)
        self.dYdt = torch.zeros(n_neurons, dtype=torch.float32, device=self.device)

    def get_spike_buffer(self) -> torch.Tensor:
        """Return the full spike history tensor [n_neurons, delay_max]."""
        return self._spike_buffer

    def inject_currents(self, I: torch.Tensor, chn: int = 0) -> None:
        """
        Accumulate input current for the next processing step.

        Parameters
        ----------
        I : torch.Tensor
            Vector of currents to inject. Shape: [n_neurons].
        chn : int, optional
            Target synaptic channel index (default 0).
        """
        assert I.shape[0] == self.size, \
            f"Dimension mismatch in current injection: Input {I.shape} vs Group {self.size}"
        self._input_currents[:, chn].add_(I)

    def inject_spikes(self, spikes: torch.Tensor) -> None:
        """
        Force the specified neurons to spike at the next step.

        This operation is a 'direct somatic injection' and generally
        bypasses sub-threshold membrane potential dynamics.

        Parameters
        ----------
        spikes : torch.Tensor
            Boolean mask or tensor convertible to bool.
            Shape: [n_neurons].
        """
        assert spikes.shape[0] == self.size, \
            f"Dimension mismatch in spike injection: Input {spikes.shape} vs Group {self.size}"
        self._input_spikes.logical_or_(spikes.bool())

    def get_spikes(self) -> torch.Tensor:
        """
        Retrieve spikes generated at the *current* time step.

        Returns
        -------
        torch.Tensor
            Boolean spike vector [n_neurons].
        """
        # Compute the current phase in the circular buffer
        phase = globals.simulator.local_circuit.current_step % self.delay_max_int
        return self._spike_buffer[:, phase]

    def get_spikes_at(
        self, delays: Union[int, torch.Tensor], indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Retrieve spikes that occurred in the past (t - delay).

        Used by synaptic projections to apply conduction delays,
        or by learning rules (e.g. STDP) to query temporal traces.

        Parameters
        ----------
        delays : int | torch.Tensor
            Delay in time steps. Can be a scalar (uniform delay)
            or a tensor [n_connections] (heterogeneous delays).
        indices : torch.Tensor
            Indices of neurons to query. Shape: [n_connections].

        Returns
        -------
        torch.Tensor
            Spike state of the requested neurons at time t-delay.
        """
        # phase = (current_time - delay) % buffer_capacity
        phase = (globals.simulator.local_circuit.current_step - delays) % self.delay_max_int
        return self._spike_buffer[indices, phase]

    def _write_spikes_to_buffer(self, t_idx: Union[int, torch.Tensor]) -> None:
        """
        Write current spikes in the circular buffer safely for tensors.
        Uses index_copy_ to avoid silent copy problems with advanced indexing.
        """
        # Aseguramos que el índice sea un tensor 1D [1]
        if isinstance(t_idx, int):
            idx = torch.tensor([t_idx], dtype=torch.long, device=self.device)
        else:
            idx = t_idx.view(1).long()

        # index_copy_ espera que la fuente tenga la misma dimensionalidad que el destino en la dim operada
        # Buffer: [N, Delay]. Operamos en dim 1.
        # Spikes: [N]. Necesitamos [N, 1].
        self._spike_buffer.index_copy_(1, idx, self.spikes.unsqueeze(1))
    
    def __rshift__(self, other: NeuronGroup) -> ConnectionOperator:
        """
        Syntactic '>>' operator to define outgoing connections.

        Example: `layer1 >> layer2` creates a connection operator.
        """
        return ConnectionOperator(self, other)


class ParrotNeurons(NeuronGroup):
    r"""
    Parrot neurons (Transparency / Relay Layer).

    Mathematical Model:
    -------------------
    Implements an instantaneous identity transfer function with no
    membrane dynamics or temporal integration.

    The spiking condition is deterministic and immediate:

    $$ S_{out}(t) = S_{forced}(t) \\lor
       \\left( \\sum_{k=1}^{N_{ch}} I_{in, k}(t) > 0 \\right) $$

    Where:
    - $S_{forced}$ are directly injected (clamped) spikes.
    - $I_{in, k}$ is the input current on channel $k$.

    Applications:
    -------------
    - **Input Layers:** Interface between static data (images, tensors)
      and the network spike graph.
    - **Relay Nodes:** To add pure propagation delays or multiplex signals
      without altering their dynamics.
    """

    def _process(self) -> None:
        """
        Execute the processing cycle of the transparent layer.

        Steps:
        1. **Cleanup:** Clear old buffer entries at the current time phase $t$.
        2. **Detection:** Check for positive input current or forced spikes.
        3. **Write:** Record new spikes in the buffer and in `self.spikes`.
        4. **Reset:** Clear input accumulators for $t+1$.
        """
        super(NeuronGroup, self)._process() # We call Group._process to skip unnecesary logic in case there is any
        
        phase = globals.simulator.local_circuit.current_step % self.delay_max_int

        # 1. Preventive cleaning
        self._spike_buffer[:, phase].fill_(False)

        # 2. Detection
        has_input_current = self._input_currents.sum(dim=1) > 0
        self.spikes.copy_(has_input_current)
        self.spikes.logical_or_(self._input_spikes)

        # 3. Write
        self._spike_buffer[:, phase].copy_(self.spikes)

        # 4. Clearning
        self._input_spikes.fill_(False)
        self._input_currents.fill_(0.0)


class SimpleIFNeurons(NeuronGroup):
    r"""
    Simplified Leaky Integrate-and-Fire (LIF) model (Current-Based).

    Mathematical Model:
    -------------------
    Governed by the linear differential equation for a leaky capacitor:
    
    $$ \tau_m \frac{dV(t)}{dt} = - (V(t) - V_{rest}) + R \cdot I_{in}(t) $$
    
    Discretized Dynamics (Exponential Euler):
    -----------------------------------------
    $$ V[t+1] = V[t] \cdot \alpha + \sum I_{in}[t] $$
    
    Where $\alpha = e^{-\Delta t / \tau_m}$ is the decay factor.
    
    Simplifications:
    ----------------
    - **Current-Based (CUBA):** Inputs are treated as direct current injection, 
      not conductance changes.
    - **Hard Reset:** Membrane potential resets to 0 immediately after spiking.
    - **No Refractory Period:** Neurons can theoretically spike every time step 
      if input is strong enough.
    - **Dimensionless Units:** Resistance $R$ is assumed to be 1.
    """

    # --- Static Configuration (Immutable parameters) ---
    threshold: Final[torch.Tensor]
    """Firing threshold. If $V > \theta$, a spike is emitted. Shape: [n_neurons]."""

    decay: Final[torch.Tensor]
    """Multiplicative decay factor per time step ($\alpha$). Shape: [n_neurons]."""

    # --- Dynamic State (Mutable) ---
    V: torch.Tensor
    """Current membrane potential. Resets to 0 upon spiking. Shape: [n_neurons]."""

    def __init__(
        self,
        n_neurons: int,
        threshold: NeuronParam = 1.0,
        tau_membrane: NeuronParam = 0.1,  # [seconds]
        dt: float = 1e-3,                 # [seconds]
        **kwargs,
    ):
        """
        Initialize the LIF group.

        Parameters
        ----------
        n_neurons : int
            Number of neurons in the group.
        threshold : float | torch.Tensor | RandomDistribution
            Membrane potential value at which a spike is generated.
        tau_membrane : float | torch.Tensor | RandomDistribution
            Membrane time constant in seconds. Determines the decay rate.
        dt : float
            Simulation time step in seconds.
        """
        # Initialize base group (topology, memory)
        super().__init__(
            n_neurons=n_neurons,
            n_channels=1,
            dt=dt,
            **kwargs
        )
        
        # Initialize State
        self.V = torch.zeros(n_neurons, dtype=torch.float32, device=self.device)
        
        # Initialize Parameters
        self.threshold = resolve_neuron_param(threshold, n_neurons, self.device)
        
        # Calculate discrete decay factor: alpha = exp(-dt / tau)
        tau = resolve_neuron_param(tau_membrane, n_neurons, self.device)
        self.decay = torch.exp(-self.dt / tau)

    def _process(self) -> None:
        """
        Execute the simulation step: Integration -> Spike -> Reset.
        """
        super(NeuronGroup, self)._process()
        
        # We get the phase
        phase = globals.simulator.local_circuit.current_step % self.delay_max_int

        # 1. Dynamics
        self.V.mul_(self.decay)
        self.V.add_(self._input_currents.sum(dim=1))
        self._input_currents.fill_(0.0)

        # 2. Fire
        self.spikes.copy_(self.V >= self.threshold)
        self.spikes.logical_or_(self._input_spikes)
        
        # 3. Write spikes
        self._spike_buffer[:, phase].copy_(self.spikes)
        
        # 4. Reset
        self.V.masked_fill_(self.spikes, 0.0)
        self._input_spikes.fill_(False)


class RandomSpikeNeurons(NeuronGroup):
    r"""
    Stochastic Spike Generator (Poisson Process).

    Mathematical Model:
    -------------------
    Models spike generation as an independent Poisson process for each neuron.
    
    The probability of firing in a small time interval $\Delta t$ is given by:
    
    $$ P(\text{spike}) = r(t) \cdot \Delta t $$
    
    Where $r(t)$ is the instantaneous firing rate in Hz.
    
    Bernoulli Approximation:
    ------------------------
    Since the simulation is time-stepped, this is implemented as a Bernoulli trial 
    at every step $t$:
    
    $$ S[t] \sim \text{Bernoulli}(r[t] \cdot \Delta t) $$

    Key Features:
    -------------
    - **No Dynamics:** Does not integrate inputs or maintain membrane potential.
    - **Differentiability:** Calculates a symbolic derivative `dYdt` representing
      the rate of change of the firing rate, useful for some learning rules.
    """

    # --- Configuration & Parameters (Mutable/Immutable depending on plasticity) ---
    firing_rate: torch.Tensor
    """Target firing frequency in Hz. Can be dynamic. Shape: [n_neurons]."""

    previous_fr: torch.Tensor
    """Firing rate at the previous time step, used to compute `dYdt`. Shape: [n_neurons]."""

    # --- Internal State ---
    probabilities: torch.Tensor
    r"""Buffer for random uniform samples $U \sim [0, 1]$. Shape: [n_neurons]."""

    def __init__(
        self,
        n_neurons: int,
        firing_rate: NeuronParam = 10.0,  # [Hz]
        dt: float = 1e-3,                 # [seconds]
        **kwargs,
    ):
        """
        Initialize the Poisson generator.

        Parameters
        ----------
        n_neurons : int
            Number of neurons.
        firing_rate : float | torch.Tensor | RandomDistribution
            Target firing rate in Hz (spikes per second).
        dt : float
            Simulation time step in seconds.
        """
        super().__init__(
            n_neurons=n_neurons,
            dt=dt,
            **kwargs
        )
        
        # Initialize Rate Parameters
        self.firing_rate = resolve_neuron_param(firing_rate, n_neurons, self.device)
        self.previous_fr = self.firing_rate.clone()
        
        # Pre-allocate randomness buffer
        self.probabilities = torch.zeros(n_neurons, dtype=torch.float32, device=self.device)

    def _process(self) -> None:
        """
        Execute the stochastic generation step.
        """
        super(NeuronGroup, self)._process()

        # Current phase in the circular buffer
        phase = globals.simulator.local_circuit.current_step % self.delay_max_int

        # 1. Calculate Spike Probability
        # p = rate [Hz] * dt [s] -> dimensionless probability
        # Assumption: rate * dt << 1 (Poisson limit)
        p_spike = self.firing_rate * self.dt
        
        # 2. Stochastic Sampling (Bernoulli Trial)
        # Generate uniform random numbers U ~ [0, 1)
        self.probabilities.uniform_(0, 1)

        # Spike if U < p_spike
        self.spikes.copy_(self.probabilities < p_spike)
        
        # Write to history buffer
        self._spike_buffer[:, phase].copy_(self.spikes)

        # 3. Rate Derivative Calculation
        # Compute discrete d(rate)/dt for potential gradient-based learning rules
        # that might tune the firing rate.
        self.dYdt = (self.firing_rate - self.previous_fr) / self.dt
        
        # Update state for next step
        self.previous_fr.copy_(self.firing_rate)


class ConductanceNeuronBase(NeuronGroup):
    r"""
    Base class for Conductance-Based neuron models.

    Physics & Modeling:
    -------------------
    Implements the analytic integration of the membrane equation under the assumption
    of constant conductance over a single time step $\Delta t$.
    
    This model supports multiple synaptic channels (e.g., AMPA, NMDA, GABA), each
    with its own reversal potential and bi-exponential temporal dynamics.

    Differential Equations:
    -----------------------
    1. **Membrane Dynamics:**
       $$ C_m \frac{dV}{dt} = -g_{leak}(V - E_{leak}) - \sum_{ch} g_{ch}(t)(V - E_{rev, ch}) $$
    
    2. **Synaptic Dynamics (Bi-exponential):**
       $$ g_{ch}(t) = S_{ch}(t) * (e^{-t/\tau_{d}} - e^{-t/\tau_{r}}) $$
       Implemented as a linear combination of two decaying state variables per channel.

    Analytic Solution (Time Step):
    ------------------------------
    Since $g_{total}$ is assumed constant during $\Delta t$, the voltage relaxes exponentially
    towards an instantaneous steady-state target $E_{eff}$:
    
    $$ V[t+\Delta t] = E_{eff} + (V[t] - E_{eff}) \cdot e^{-\frac{g_{total} \Delta t}{C_m}} $$

    Where:
    - $g_{total} = g_{leak} + \sum g_{ch}$
    - $E_{eff} = \frac{g_{leak} E_{leak} + \sum g_{ch} E_{rev}}{g_{total}}$
    """

    # --- Physical Constants (Immutable) ---
    Cm: Final[torch.Tensor]
    """Membrane capacitance [Farads] (or time constant if conductance is normalized). Shape: [n_neurons]."""

    g_leak: Final[float]
    """Leak conductance. Normalized to 1.0 for simplicity in this implementation."""

    dt_over_Cm: Final[torch.Tensor]
    r"""Pre-calculated constant $\Delta t / C_m$ for optimization. Shape: [n_neurons]."""

    E_rev: Final[torch.Tensor]
    """Reversal potentials for each synaptic channel. Shape: [n_channels]."""

    E_rev_row: Final[torch.Tensor]
    """Broadcastable view of reversal potentials. Shape: [1, n_channels]."""

    # --- Synaptic Kernel Parameters ---
    decay_factors: Final[torch.Tensor]
    """Exponential decay factors for rise and decay states: $[\alpha_{rise}, \alpha_{decay}]$. Shape: [n_channels, 2]."""

    norm_factors: Final[torch.Tensor]
    """Normalization factors to ensure consistent peak amplitude/charge for spikes. Shape: [n_channels]."""

    # --- Dynamic State (Mutable) ---
    V: torch.Tensor
    """Membrane potential. Shape: [n_neurons]."""

    E_rest: torch.Tensor
    """Resting potential (Leak reversal potential). Shape: [n_neurons]."""

    channel_states: torch.Tensor
    """State variables for bi-exponential synapses (rise, decay). Shape: [n_neurons, n_channels, 2]."""

    channel_currents: torch.Tensor
    """Instantaneous current per channel (monitoring only). Shape: [n_neurons, n_channels]."""

    def __init__(
        self,
        n_neurons: int,
        spatial_dimensions: int,
        delay_max: int,
        n_channels: int,
        channel_time_constants: list[tuple[float, float]],
        channel_reversal_potentials: list[float],
        tau_membrane: NeuronParam,
        E_rest: NeuronParam,
        dt: float = 1e-3,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs
    ):
        """
        Initialize the conductance-based model.

        Parameters
        ----------
        channel_time_constants : list[tuple[float, float]]
            List of (tau_rise, tau_decay) in seconds for each channel.
        channel_reversal_potentials : list[float]
            List of reversal potentials (Volts) for each channel.
        tau_membrane : float | torch.Tensor
            Membrane time constant (or Capacitance if g_leak=1).
        E_rest : float | torch.Tensor
            Resting potential (Leak reversal potential).
        """
        super().__init__(
            n_neurons=n_neurons,
            spatial_dimensions=spatial_dimensions,
            delay_max=delay_max,
            n_channels=n_channels,
            dt=dt,
            device=device,
            **kwargs
        )

        # Validation
        assert len(channel_time_constants) == n_channels, "Time constants must match n_channels"
        assert len(channel_reversal_potentials) == n_channels, "Reversal potentials must match n_channels"

        # 1. Physical Parameters
        self.g_leak = 1.0  # Normalized conductance
        self.Cm = resolve_neuron_param(tau_membrane, n_neurons, self.device)
        self.dt_over_Cm = self.dt / self.Cm

        # 2. Membrane State Initialization
        self.E_rest = resolve_neuron_param(E_rest, n_neurons, self.device)
        self.V = torch.zeros(n_neurons, dtype=torch.float32, device=self.device)
        self.V.copy_(self.E_rest) # Start at resting potential

        # 3. Synaptic Channel Configuration
        # Extract taus
        tau_rise = torch.tensor([tc[0] for tc in channel_time_constants], device=self.device)
        tau_decay = torch.tensor([tc[1] for tc in channel_time_constants], device=self.device)

        # Pre-calculate exponential steps: alpha = exp(-dt/tau)
        # Stacked as [rise_factor, decay_factor] for vectorized ops
        self.decay_factors = torch.stack([
            torch.exp(-self.dt / tau_rise), 
            torch.exp(-self.dt / tau_decay)
        ], dim=1) # Shape: (n_channels, 2)

        # Normalization factor (Charge conservation logic)
        # Factor = (tau_decay - tau_rise) / (tau_decay * tau_rise)
        # Ensures total integrated conductance is proportional to weight
        self.norm_factors = (tau_decay - tau_rise) / (tau_decay * tau_rise)

        self.E_rev = torch.tensor(channel_reversal_potentials, dtype=torch.float32, device=self.device)
        self.E_rev_row = self.E_rev.unsqueeze(0) # For broadcasting

        # 4. Synaptic State Initialization
        self.channel_states = torch.zeros(n_neurons, n_channels, 2, dtype=torch.float32, device=self.device)
        self.channel_currents = torch.zeros(n_neurons, n_channels, dtype=torch.float32, device=self.device)

    def _update_channels(self) -> None:
        """
        Update synaptic state variables (Rise & Decay) using input currents.
        
        This step consumes `_input_currents`, treating them as incoming spikes/weights.
        """
        # Normalize inputs based on channel dynamics
        normalized_input = self._input_currents.mul(self.norm_factors.unsqueeze(0))
        
        # Clear inputs after consumption
        self._input_currents.zero_()

        # Apply exponential decay to previous state
        # channel_states shape: [n_neurons, n_channels, 2]
        # decay_factors shape: [n_channels, 2] -> broadcast over neurons
        self.channel_states.mul_(self.decay_factors)
        
        # Add new inputs (jump condition for exponentials)
        # Input adds to BOTH rise and decay terms to form the bi-exponential difference
        self.channel_states[:, :, 0].add_(normalized_input)
        self.channel_states[:, :, 1].add_(normalized_input)

    def _integrate_membrane(self) -> None:
        """
        Analytically update membrane potential $V$.
        
        Solves: dV/dt = -g_eff/Cm * (V - E_eff)
        Solution: V(t+dt) = V(t) -> E_eff exponentially
        """
        # 1. Calculate Instantaneous Conductances (Difference of Exponentials)
        # g(t) = state_decay - state_rise
        # ReLU ensures numerical stability (conductance >= 0)
        g_channels = torch.relu(self.channel_states[:, :, 1] - self.channel_states[:, :, 0])
        
        # Monitor currents (optional, useful for debugging)
        # I_ch = g_ch * (E_rev - V)
        self.channel_currents.copy_(g_channels * (self.E_rev_row - self.V.unsqueeze(1)))

        # 2. Calculate Effective Steady-State Targets
        # Total conductance: g_tot = g_leak + sum(g_channels)
        g_syn = g_channels.sum(dim=1)
        g_eff = g_syn + self.g_leak

        # Effective Potential (Weighted average of reversal potentials)
        # E_eff = (g_leak*E_leak + sum(g_ch*E_ch)) / g_tot
        num = self.g_leak * self.E_rest + (g_channels * self.E_rev_row).sum(dim=1)
        E_eff = num / g_eff

        # 3. Calculate Derivative (for learning rules)
        # dV/dt = (g_eff/Cm) * (E_eff - V)
        self.dYdt.copy_(g_eff * (E_eff - self.V) * (1.0 / self.Cm))

        # 4. Analytic Update
        # alpha_eff = exp(-g_eff * dt / Cm)
        # V_new = V_old * alpha + E_eff * (1 - alpha)
        exp_term = torch.exp(-g_eff * self.dt_over_Cm)
        self.V.mul_(exp_term).add_(E_eff * (1.0 - exp_term))

    def _integrate_step(self) -> None:
        """Helper to run the full integration cycle (Synapses + Membrane)."""
        self._update_channels()
        self._integrate_membrane()
        
    def _process(self) -> None:
        """
        Parent process method. 
        Note: Subclasses must invoke `_integrate_step` and spike generation logic explicitly.
        """
        super()._process()


class IFDeterministicBase(ConductanceNeuronBase):
    r"""
    Base class for Deterministic Integrate-and-Fire neurons with Refractory Period.

    Behavior:
    ---------
    1. **Integration:** Inherits conductance-based dynamics from `ConductanceNeuronBase`.
    2. **Threshold:** Spikes if $V(t) \ge \theta$.
    3. **Reset:** If a spike occurs, $V \to E_{rest}$ immediately.
    4. **Refractory Period:** After a spike, the neuron ignores dynamics and remains
       silent for $t_{refrac}$ seconds.
    5. **Adaptation Hooks:** Provides methods (`_update_adaptation_state`, `_update_threshold`)
       for subclasses (like ALIF) to modify dynamics.

    State Variables:
    ----------------
    - `refrac_counter` (int): Steps remaining in refractory period.
    """

    # --- Configuration ---
    threshold: torch.Tensor
    """Spike threshold potential [Volts]. Shape: [n_neurons]."""

    threshold_base: Optional[torch.Tensor]
    """Base threshold value, used if threshold is adaptive. Shape: [n_neurons]."""

    refrac_steps: Final[torch.Tensor]
    """Refractory period duration in discrete time steps. Shape: [n_neurons]."""

    # --- Dynamic State ---
    refrac_counter: torch.Tensor
    """Countdown timer for refractory state. 0 means active. Shape: [n_neurons]."""

    def __init__(
        self,
        threshold: NeuronParam,
        tau_refrac: NeuronParam,
        **kwargs
    ):
        """
        Initialize the deterministic IF base.

        Parameters
        ----------
        tau_refrac : float | torch.Tensor
            Refractory period in seconds.
        **kwargs : dict
            Parameters passed to ConductanceNeuronBase.
        """
        # Initialize the physics engine
        super().__init__(**kwargs)

        # 1. Threshold Configuration
        # 'threshold' is mandatory in kwargs for this base class
        self.threshold = resolve_neuron_param(threshold, self.size, self.device)
        self.threshold_base = None # Subclasses like ALIF will initialize this if needed

        # 2. Refractory Period Configuration
        tau_refrac_vec = resolve_neuron_param(tau_refrac, self.size, self.device)
        
        # Convert seconds to integer steps: ceil(tau / dt)
        self.refrac_steps = torch.ceil(tau_refrac_vec / self.dt).long()
        
        # Initialize counter state (0 = not refractory)
        self.refrac_counter = torch.zeros(self.size, dtype=torch.long, device=self.device)

    # --- Extension Hooks (Override in subclasses) ---
    
    def _update_adaptation_state(self) -> None:
        """Hook to update internal adaptation variables (e.g., calcium trace)."""
        pass

    def _update_threshold(self) -> None:
        """Hook to update dynamic threshold based on adaptation state."""
        pass

    def _on_spikes(self, mask: torch.Tensor) -> None:
        """Hook called when neurons spike (e.g., to increment adaptation trace)."""
        pass

    # --- Main Cycle ---

    def _process(self) -> None:
        """
        Execute the deterministic IF cycle:
        Integrate -> Adapt -> Check Refractory -> Spike -> Reset -> Write.
        """
        # Call NeuronGroup._process() directly to handle overhead
        # We skip ConductanceNeuronBase._process() because we orchestrate the integration here manually
        super(NeuronGroup, self)._process() 
        
        t_idx = globals.simulator.local_circuit.current_step % self.delay_max_int

        # 1. Physical Integration (Synapses + Membrane)
        self._integrate_step()

        # 2. Adaptation Dynamics (ALIF/Adaptive Thresholds)
        self._update_adaptation_state()
        self._update_threshold()

        # 3. Refractory Management
        # Decrement counter, clamp to 0
        self.refrac_counter.sub_(1).clamp_min_(0)
        is_refractory = self.refrac_counter > 0

        # 4. Spike Detection
        # Condition: Voltage >= Threshold AND Not Refractory
        self.spikes.copy_(self.V >= self.threshold)
        self.spikes.logical_and_(~is_refractory)
        
        # Apply Forced Spikes (Clamping)
        # Forced spikes override refractory period and threshold
        self.spikes.logical_or_(self._input_spikes)

        # 5. Post-Spike Effects
        # A. Enter Refractory Period
        mask = self.spikes  # bool [n]
        new_vals = self.refrac_steps  # int/float [n]
        self.refrac_counter = torch.where(mask, new_vals, self.refrac_counter)
        
        # B. Hard Reset of Membrane Potential
        # V -> E_rest
        self.V = torch.where(self.spikes, self.E_rest, self.V)
        
        # C. Trigger Adaptation Hooks
        self._on_spikes(self.spikes)

        # 6. Output Registration
        self._write_spikes_to_buffer(t_idx)
        
        # Clear inputs
        self._input_spikes.fill_(False)


class ContinuousNoResetBase(ConductanceNeuronBase):
    """
    Base class for Conductance-based neurons without Hard Reset.

    Behavior:
    ---------
    1. **Continuous Integration:** Membrane potential $V$ evolves continuously 
       via conductance dynamics (inherited).
    2. **Soft/No Reset:** Unlike standard LIF, $V$ is NOT automatically reset 
       to $E_{rest}$ after a spike.
    3. **Custom Spiking:** Subclasses must implement `_generate_spikes` to define 
       the condition for event generation (e.g., stochastic probability, phase crossing).

    Use Cases:
    ----------
    - **Stochastic Models:** Where spiking is a probabilistic function of $V$ (e.g., sigmoid).
    - **Phase Models:** Where neurons represent oscillators.
    - **Rate-Based Approximations:** Where "spikes" are actually discretized rate events.
    """

    def _generate_spikes(self, t_idx: int) -> None:
        """
        Abstract method to determine which neurons spike.
        
        Responsibilities:
        1. Write boolean mask to `self.spikes`.
        2. Handle forced spikes (`self._input_spikes`).
        3. Call `self._write_spikes_to_buffer(t_idx)`.
        
        Parameters
        ----------
        t_idx : int
            Current time index in the circular buffer.
        """
        raise NotImplementedError("Subclasses must implement spike generation logic.")

    def _process(self) -> None:
        """
        Execute the continuous cycle: Integrate -> Custom Spike Generation.
        """
        # Call grandparent process (NeuronGroup) to handle basic administrative tasks
        super(NeuronGroup, self)._process()
        
        t_idx = globals.simulator.local_circuit.current_step % self.delay_max_int

        # 1. Physical Integration (Synapses + Membrane)
        # V evolves according to dV/dt, ignoring any previous spike events
        self._integrate_step()

        # 2. Spike Generation
        # Delegate to specific model implementation (Stochastic, Phase, etc.)
        self._generate_spikes(t_idx)

        # Clear inputs
        self._input_spikes.fill_(False)


class LIFNeurons(IFDeterministicBase):
    """
    Standard Leaky Integrate-and-Fire (LIF) with Conductance-Based Synapses.

    Configuration:
    --------------
    A specialized version of `IFDeterministicBase` with default biological parameters
    for a standard cortical neuron (Excitatory/Inhibitory).

    Dynamics:
    ---------
    - **Integration:** Conductance-based (AMPA/GABA/NMDA support).
    - **Adaptation:** None. Threshold is fixed.
    - **Refractory Period:** Hard constraint ($V$ clamped to $E_{rest}$).
    
    Default Channels:
    -----------------
    1. **AMPA** (Excitatory, fast): $\tau_{rise}=1ms, \tau_{decay}=5ms, E_{rev}=0mV$
    2. **GABA** (Inhibitory, fast): $\tau_{rise}=1ms, \tau_{decay}=10ms, E_{rev}=-70mV$
    3. **NMDA** (Excitatory, slow): $\tau_{rise}=2ms, \tau_{decay}=100ms, E_{rev}=0mV$
    """

    def __init__(
        self,
        n_neurons: int,
        spatial_dimensions: int = 2,
        delay_max: int = 20,
        n_channels: int = 3,
        channel_time_constants: list[tuple[float, float]] = (
            (0.001, 0.005),  # Channel 0: AMPA
            (0.001, 0.010),  # Channel 1: GABA
            (0.002, 0.100),  # Channel 2: NMDA
        ),
        channel_reversal_potentials: list[float] = (
            0.0,     # AMPA
            -0.070,  # GABA
            0.0,     # NMDA
        ),
        threshold: NeuronParam = -0.050,  # [Volts]
        tau_membrane: NeuronParam = 0.010, # [seconds]
        E_rest: NeuronParam = -0.065,      # [Volts]
        tau_refrac: NeuronParam = 0.002,   # [seconds]
        dt: float = 1e-3,                  # [seconds]
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ):
        """
        Initialize standard LIF neurons.

        Parameters
        ----------
        n_neurons : int
            Population size.
        n_channels : int
            Number of synaptic receptor types.
        channel_time_constants : list
            (Rise, Decay) times for each channel in seconds.
        channel_reversal_potentials : list
            Reversal potential for each channel in Volts.
        """
        super().__init__(
            n_neurons=n_neurons,
            spatial_dimensions=spatial_dimensions,
            delay_max=delay_max,
            n_channels=n_channels,
            channel_time_constants=channel_time_constants,
            channel_reversal_potentials=channel_reversal_potentials,
            threshold=threshold,
            tau_membrane=tau_membrane,
            E_rest=E_rest,
            tau_refrac=tau_refrac,
            dt=dt,
            device=device,
            **kwargs
        )

    # Note: No need to override _update_adaptation_state, _update_threshold, or _on_spikes.
    # The base class implementation (no-op) provides the correct standard LIF behavior.


class ALIFNeurons(IFDeterministicBase):
    r"""
    Adaptive Leaky Integrate-and-Fire (ALIF).

    Model Dynamics:
    ---------------
    Extends the standard LIF model with a dynamic threshold adaptation mechanism
    to model spike-frequency adaptation (SFA).

    1. **Membrane Dynamics:** Same as LIF (Conductance-based).
    2. **Threshold Dynamics:**
       $$ \theta(t) = \theta_{base} + \beta \cdot A(t) $$
    3. **Adaptation Trace:**
       $$ \tau_{adapt} \frac{dA}{dt} = -A(t) $$
       $$ A(t) \leftarrow A(t) + 1 \quad \text{(on spike)} $$

    Variables:
    ----------
    - $\theta_{base}$: Static baseline threshold.
    - $\beta$: Adaptation strength (threshold increase per spike).
    - $\tau_{adapt}$: Time constant of the adaptation decay.
    """

    # --- Configuration (ALIF specific) ---
    tau_adapt: Final[torch.Tensor]
    """Adaptation time constant [seconds]. Shape: [n_neurons]."""

    beta: Final[torch.Tensor]
    """Adaptation strength [Volts]. Threshold jump per spike. Shape: [n_neurons]."""

    alpha_adapt: Final[torch.Tensor]
    r"""Pre-calculated multiplicative decay factor per step: $e^{-\Delta t / \tau_{adapt}}$. Shape: [n_neurons]."""

    # --- Dynamic State ---
    A: torch.Tensor
    """Adaptation trace (unitless/count-based). Decays over time, jumps on spike. Shape: [n_neurons]."""

    def __init__(
        self,
        n_neurons: int,
        spatial_dimensions: int = 2,
        delay_max: int = 20,
        n_channels: int = 3,
        channel_time_constants: list[tuple[float, float]] = (
            (0.001, 0.005),
            (0.001, 0.010),
            (0.002, 0.100),
        ),
        channel_reversal_potentials: list[float] = (
            0.0,
            -0.070,
            0.0,
        ),
        threshold: NeuronParam = -0.050,
        tau_membrane: NeuronParam = 0.010,
        E_rest: NeuronParam = -0.065,
        tau_refrac: NeuronParam = 0.002,
        tau_adapt: NeuronParam = 0.200,   # [seconds] (200ms)
        beta: NeuronParam = 0.0017,       # [Volts] (1.7mV)
        dt: float = 1e-3,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ):
        # Initialize Base (IFDeterministicBase handles Threshold, Refrac, Membrane)
        super().__init__(
            n_neurons=n_neurons,
            spatial_dimensions=spatial_dimensions,
            delay_max=delay_max,
            n_channels=n_channels,
            channel_time_constants=channel_time_constants,
            channel_reversal_potentials=channel_reversal_potentials,
            threshold=threshold,
            tau_membrane=tau_membrane,
            E_rest=E_rest,
            tau_refrac=tau_refrac,
            dt=dt,
            device=device,
            **kwargs
        )

        # Initialize ALIF Parameters
        self.tau_adapt = resolve_neuron_param(tau_adapt, n_neurons, self.device)
        self.beta = resolve_neuron_param(beta, n_neurons, self.device)
        
        # Pre-calculate decay factor
        self.alpha_adapt = torch.exp(-self.dt / self.tau_adapt)
        
        # Initialize State
        self.A = torch.zeros(n_neurons, dtype=torch.float32, device=self.device)

        # IMPORTANT: IFDeterministicBase initializes self.threshold. 
        # For ALIF, that value acts as the *baseline* threshold.
        # We need to copy it to ensure we can modify self.threshold dynamically 
        # while remembering the baseline.
        self.threshold_base = self.threshold.clone()

    def _update_adaptation_state(self) -> None:
        r"""
        Decay the adaptation trace $A$ for the current time step.
        $$ A[t+1] = A[t] \cdot \alpha_{adapt} $$
        """
        self.A.mul_(self.alpha_adapt)

    def _update_threshold(self) -> None:
        r"""
        Update the effective firing threshold based on the adaptation state.
        $$ \theta[t] = \theta_{base} + \beta \cdot A[t] $$
        """
        self.threshold.copy_(self.threshold_base + self.beta * self.A)

    def _on_spikes(self, spike_mask: torch.Tensor) -> None:
        r"""
        Increment adaptation trace for neurons that spiked.
        $$ A \leftarrow A + 1 $$
        """
        self.A[spike_mask] += 1.0


class PowerLawALIFNeurons(IFDeterministicBase):
    r"""
    ALIF with Power-Law Adaptation dynamics.

    Mathematical Model:
    -------------------
    Unlike standard ALIF (exponential adaptation), this model approximates a 
    Power-Law decay for the adaptation kernel: $K(t) \sim t^{-\gamma}$.
    This allows the neuron to maintain memory over multiple time scales.

    Implementation (Sum of Exponentials):
    -------------------------------------
    The power-law kernel is approximated by a weighted sum of $N_{basis}$ 
    exponential filters with geometrically spaced time constants:
    
    $$ A_{eff}(t) = \sum_{k=1}^{N_{basis}} w_k \cdot A_k(t) $$
    
    $$ \theta(t) = \theta_{base} + \beta \cdot A_{eff}(t) $$

    Homeostatic Plasticity:
    -----------------------
    Includes a slow homeostatic adjustment of the baseline threshold $\theta_{base}$
    to maintain a target firing rate $r_{target}$:
    
    $$ \Delta \theta_{base} \propto (r_{estimated} - r_{target}) $$

    State Variables:
    ----------------
    - `A_`: Bank of adaptation traces. Shape: [n_neurons, n_basis].
    - `A`: Effective scalar adaptation (for monitoring). Shape: [n_neurons].
    """

    # --- Configuration (Power-Law specific) ---
    n_basis: Final[int]
    """Number of exponential basis functions used for approximation."""

    taus: Final[torch.Tensor]
    """Time constants for the basis functions (log-spaced). Shape: [n_basis]."""

    alpha_adapt: Final[torch.Tensor]
    """Decay factors for each basis function. Shape: [n_basis]."""

    basis_weights: Final[torch.Tensor]
    """Mixing weights $w_k$ to approximate the power law. Shape: [n_basis]."""

    beta: Final[torch.Tensor]
    """Global adaptation gain (scaling factor). Shape: [n_neurons]."""

    r_target_hz: Final[float]
    """Target firing rate for homeostasis [Hz]."""

    eta_theta: Final[float]
    """Learning rate for intrinsic threshold plasticity."""

    rate_norm: Final[float]
    """Normalization factor to convert adaptation state to Hz estimate."""

    # --- Dynamic State ---
    A_: torch.Tensor
    """Internal state: Bank of adaptation traces. Shape: [n_neurons, n_basis]."""

    A: torch.Tensor
    """Effective adaptation value (scalar summary). Shape: [n_neurons]."""

    r_hat: torch.Tensor
    """Estimated instantaneous firing rate [Hz]. Shape: [n_neurons]."""

    def __init__(
        self,
        n_neurons: int,
        # Power-Law Params
        n_basis: int = 10,
        tau_min: float = 0.010,       # [seconds]
        tau_max: float = 5.0,         # [seconds]
        power_law_gamma: float = 0.5, # Exponent (0 < gamma < 1)
        
        # Homeostasis Params
        r_target_hz: float = 10.0,    # [Hz]
        eta_theta: float = 1e-6,      # Learning rate
        
        # Standard ALIF Params
        beta: NeuronParam = 0.0017,   # [Volts]
        
        # Base Params
        spatial_dimensions: int = 2,
        delay_max: int = 20,
        n_channels: int = 3,
        channel_time_constants: list[tuple[float, float]] = (
            (0.001, 0.005),
            (0.001, 0.010),
            (0.002, 0.100),
        ),
        channel_reversal_potentials: list[float] = (0.0, -0.070, 0.0),
        threshold: NeuronParam = -0.050,
        tau_membrane: NeuronParam = 0.010,
        E_rest: NeuronParam = -0.065,
        tau_refrac: NeuronParam = 0.002,
        dt: float = 1e-3,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ):
        # We CANNOT use `locals()` here easily because of the complex initialization logic.
        # Explicit initialization is safer and clearer.
        super().__init__(
            n_neurons=n_neurons,
            spatial_dimensions=spatial_dimensions,
            delay_max=delay_max,
            n_channels=n_channels,
            channel_time_constants=channel_time_constants,
            channel_reversal_potentials=channel_reversal_potentials,
            tau_membrane=tau_membrane,
            E_rest=E_rest,
            threshold=threshold,
            tau_refrac=tau_refrac,
            dt=dt,
            device=device,
            **kwargs,
        )

        # 1. Gain Configuration
        self.beta = resolve_neuron_param(beta, n_neurons, self.device)

        # 2. Basis Functions Configuration
        self.n_basis = int(n_basis)
        
        # Generate log-spaced time constants
        log_taus = torch.linspace(np.log(tau_min), np.log(tau_max), n_basis, device=self.device)
        self.taus = torch.exp(log_taus)
        
        # Pre-calculate decay factors: alpha = exp(-dt/tau)
        self.alpha_adapt = torch.exp(-self.dt / self.taus) # Shape: [n_basis]

        # 3. Power-Law Approximation (Curve Fitting)
        # Calculate optimal weights w_k such that sum(w_k * exp(-t/tau_k)) ~ t^-gamma
        raw_weights = self._fit_power_law_weights(self.taus, tau_min, tau_max, power_law_gamma)
        
        # Normalize weights so sum(weights) = 1.0 (Unit gain for a step input)
        weight_sum = raw_weights.sum()
        if weight_sum != 0:
            self.basis_weights = raw_weights / weight_sum
        else:
            self.basis_weights = raw_weights
        
        # 4. Homeostasis Config
        self.r_target_hz = float(r_target_hz)
        self.eta_theta = float(eta_theta)
        # Normalization factor for rate estimation (area under the kernel)
        self.rate_norm = float(torch.sum(self.basis_weights * self.taus))

        # 5. State Initialization
        # A_: [neurons, basis] -> Each neuron maintains 'n_basis' trace variables
        self.A_ = torch.zeros((n_neurons, n_basis), dtype=torch.float32, device=self.device)
        self.A = torch.zeros(n_neurons, dtype=torch.float32, device=self.device)
        self.r_hat = torch.zeros(n_neurons, dtype=torch.float32, device=self.device)

        # Base threshold needs to be mutable for homeostasis
        self.threshold_base = self.threshold.clone()

    def _fit_power_law_weights(
        self, taus: torch.Tensor, t_min: float, t_max: float, gamma: float, n_points: int = 1000
    ) -> torch.Tensor:
        r"""
        Solve Least Squares (Ax=b) to find weights that approximate the Power Law.
        Target: $t^{-\gamma} \approx \sum w_k e^{-t/\tau_k}$
        """
        # Time points for evaluation (log-spaced)
        t_eval = torch.logspace(np.log10(t_min), np.log10(t_max), n_points, device=self.device)
        
        # Design Matrix M: M[i, k] = exp(-t_i / tau_k)
        # Shape: [n_points, n_basis]
        M = torch.exp(-t_eval.unsqueeze(1) / taus.unsqueeze(0))
        
        # Target Vector y: y[i] = t_i^(-gamma)
        y = t_eval.pow(-gamma)
        
        # Solve linear system
        # Returns a named tuple in newer PyTorch versions
        result = torch.linalg.lstsq(M, y)
        return result.solution.to(dtype=torch.float32)

    def _update_adaptation_state(self) -> None:
        r"""
        Decay all basis functions independently.
        $$ A\_[i, k] \leftarrow A\_[i, k] \cdot \alpha_{adapt}[k] $$
        """
        # Broadcasting: [N, basis] * [basis]
        self.A_.mul_(self.alpha_adapt)

    def _update_threshold(self) -> None:
        """
        1. Compute effective adaptation (weighted sum).
        2. Update dynamic threshold.
        3. Apply homeostatic plasticity to baseline.
        """
        # 1. Collapsing the basis bank: A_eff = dot(A_, weights)
        # Result Shape: [n_neurons]
        A_eff = torch.sum(self.A_ * self.basis_weights, dim=1)
        self.A.copy_(A_eff) # Store for monitoring
        
        # 2. Threshold Modulation
        self.threshold.copy_(self.threshold_base + self.beta * A_eff)

        # 3. Homeostasis (Intrinsic Plasticity)
        # Estimate rate from the adaptation trace integral
        self.r_hat.copy_(A_eff / self.rate_norm)
        
        # Simple delta rule: dTheta_base = eta * (r_est - r_target)
        # If rate is too high -> Increase threshold
        error = self.r_hat - self.r_target_hz
        self.threshold_base.add_(error, alpha=self.eta_theta)

    def _on_spikes(self, spike_mask: torch.Tensor) -> None:
        r"""
        Kick all basis functions for spiking neurons.
        $$ A\_[k] \leftarrow A\_[k] + 1 $$
        """
        # Add 1.0 to all basis traces for active neurons
        self.A_[spike_mask, :] += 1.0


class StochasticIFNeurons(ContinuousNoResetBase):
    r"""
    Stochastic Integrate-and-Fire (Soft-Threshold / GLM-like).

    Mathematical Model:
    -------------------
    1. **Membrane Potential:** Evolves via standard conductance-based dynamics 
       (inherited), but is **NOT** reset after spiking.
       
    2. **Firing Probability:** The instantaneous firing rate $\rho(t)$ is a 
       non-linear function of the membrane potential (sigmoid/logistic):
       
       $$ \rho(t) = r_{max} \cdot \sigma\left( \beta_{gain} (V(t) - V_{threshold}) \right) $$
       
       Where $\sigma(x) = \frac{1}{1 + e^{-x}}$.

    3. **Spike Generation:** Bernoulli process at each step:
       $$ P(\text{spike}) = \rho(t) \cdot \Delta t $$

    Use Cases:
    ----------
    - Modeling population codes where precise spike timing is noisy.
    - Implementing "Sampling" based neural networks.
    - Rate-based approximations compatible with SNN hardware.
    """

    # --- Configuration ---
    beta: Final[torch.Tensor]
    """Gain (slope) of the sigmoid activation function. Higher = closer to deterministic step. Shape: [n_neurons]."""

    target_rate: Final[torch.Tensor]
    r"""Maximum firing rate ($r_{max}$) achievable when $V \gg \theta$. Shape: [n_neurons]."""

    threshold: Final[torch.Tensor]
    r"""Soft-threshold inflection point ($\theta$). At $V=\theta$, rate is $0.5 \cdot r_{max}$. Shape: [n_neurons]."""

    def __init__(
        self,
        n_neurons: int,
        spatial_dimensions: int = 2,
        delay_max: int = 20,
        n_channels: int = 3,
        channel_time_constants: list[tuple[float, float]] = (
            (0.001, 0.005),
            (0.001, 0.010),
            (0.002, 0.100),
        ),
        channel_reversal_potentials: list[float] = (
            0.0,
            -0.070,
            0.0,
        ),
        threshold: NeuronParam = -0.050,  # [Volts] Inflection point
        tau_membrane: NeuronParam = 0.010,
        E_rest: NeuronParam = -0.065,
        beta: NeuronParam = 200.0,        # [1/Volts] Gain
        target_rate: NeuronParam = 20.0,  # [Hz] Max Rate
        dt: float = 1e-3,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ):
        """
        Initialize the Stochastic IF group.
        """
        super().__init__(
            n_neurons=n_neurons,
            spatial_dimensions=spatial_dimensions,
            delay_max=delay_max,
            n_channels=n_channels,
            channel_time_constants=channel_time_constants,
            channel_reversal_potentials=channel_reversal_potentials,
            tau_membrane=tau_membrane,
            E_rest=E_rest,
            dt=dt,
            device=device,
            **kwargs
        )

        self.beta = resolve_neuron_param(beta, n_neurons, self.device)
        self.target_rate = resolve_neuron_param(target_rate, n_neurons, self.device)
        self.threshold = resolve_neuron_param(threshold, n_neurons, self.device)

    def _generate_spikes(self, t_idx: int) -> None:
        """
        Generate spikes probabilistically based on current V.
        """
        # 1. Calculate Instantaneous Rate [Hz]
        # r = r_max * sigmoid(beta * (V - theta))
        activation = torch.sigmoid(self.beta * (self.V - self.threshold))
        r_inst = self.target_rate * activation

        # 2. Calculate Probability for this dt
        # p = rate * dt (dimensionless)
        p_spike = r_inst * self.dt

        # 3. Bernoulli Sampling
        # Generate random mask and compare
        random_draw = torch.rand_like(self.V) # U ~ [0, 1]
        self.spikes.copy_(random_draw < p_spike)
        
        # Apply forced spikes (always override noise)
        self.spikes.logical_or_(self._input_spikes)

        # 4. Record Output
        self._write_spikes_to_buffer(t_idx)


class PhaseIFNeurons(ContinuousNoResetBase):
    r"""
    Phase-Integrate-and-Fire Neurons.

    Mathematical Model:
    -------------------
    Combines conductance-based membrane dynamics with a continuous "Phase" integration 
    mechanism. This decouples the signal integration (V) from the event generation ($\phi$).

    1. **Signal Integration (V):** Standard conductance dynamics (inherited).
    
    2. **Rate Conversion:** Membrane potential drives an instantaneous frequency:
       $$ r(t) = r_{max} \cdot \sigma(\beta (V(t) - \theta)) $$
    
    3. **Phase Integration:**
       $$ \frac{d\phi}{dt} = -\frac{\phi}{\tau_{\phi}} + r(t) + \xi(t) $$
       Where $\xi(t)$ is Gaussian phase noise (jitter).

    4. **Spiking Condition:** A spike is emitted when the phase completes a full cycle ($\phi \ge 1$).
       After spiking, the integer part is drained: $\phi \leftarrow \phi - 1$.
       The membrane potential $V$ is NOT reset, but may undergo a soft After-Hyperpolarization (AHP).

    Variables:
    ----------
    - $\phi$: Accumulated phase (turns).
    - $\theta$: Soft threshold inflection point.
    - $jitter$: Standard deviation of phase noise.
    - $AHP$: Voltage drop after spike (optional).
    """

    # --- Configuration ---
    beta: Final[torch.Tensor]
    """Gain of the V-to-Rate sigmoid. Shape: [n_neurons]."""

    r_max: Final[torch.Tensor]
    """Maximum driving frequency [Hz]. Shape: [n_neurons]."""

    theta: Final[torch.Tensor]
    """Inflection point for the V-to-Rate curve. Shape: [n_neurons]."""

    phi_decay: Final[float]
    r"""Multiplicative phase decay factor: $e^{-\Delta t / \tau_{\phi}}$. Scalar."""

    jitter_std: Final[float]
    """Standard deviation of phase noise per step. Scalar."""

    ahp_drop: Final[float]
    """Voltage drop (subtraction) applied to V after a spike. Scalar [Volts]."""

    # --- Dynamic State ---
    phase: torch.Tensor
    r"""Current accumulated phase. Spikes when $\ge 1.0$. Shape: [n_neurons]."""

    r: torch.Tensor
    """Instantaneous driving rate [Hz] (monitoring). Shape: [n_neurons]."""

    def __init__(
        self,
        n_neurons: int,
        spatial_dimensions: int = 2,
        delay_max: int = 20,
        n_channels: int = 3,
        channel_time_constants: list[tuple[float, float]] = (
            (0.001, 0.005),
            (0.001, 0.010),
            (0.002, 0.100),
        ),
        channel_reversal_potentials: list[float] = (
            0.0,
            -0.070,
            0.0,
        ),
        tau_membrane: NeuronParam = 0.010,
        E_rest: NeuronParam = -0.065,
        
        # PhaseIF Specifics
        threshold: NeuronParam = -0.050,   # Used as default for theta if theta is None
        beta: NeuronParam = 200.0,         # Sigmoid Gain
        r_max: NeuronParam = 300.0,        # Max Rate [Hz]
        theta: Optional[NeuronParam] = None, # Inflection point (defaults to threshold)
        tau_phi: NeuronParam = 0.050,      # Phase leak time constant [s]
        jitter_std: float = 0.0,           # Phase noise
        ahp_drop: float = 0.0,             # V drop after spike
        
        dt: float = 1e-3,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ):
        # Initialize base
        super().__init__(
            n_neurons=n_neurons,
            spatial_dimensions=spatial_dimensions,
            delay_max=delay_max,
            n_channels=n_channels,
            channel_time_constants=channel_time_constants,
            channel_reversal_potentials=channel_reversal_potentials,
            tau_membrane=tau_membrane,
            E_rest=E_rest,
            dt=dt,
            device=device,
            **kwargs,
        )

        # 1. Rate Function Config
        self.beta = resolve_neuron_param(beta, n_neurons, self.device)
        self.r_max = resolve_neuron_param(r_max, n_neurons, self.device)
        
        theta_val = threshold if theta is None else theta
        self.theta = resolve_neuron_param(theta_val, n_neurons, self.device)

        # 2. Phase Dynamics Config
        self.jitter_std = float(jitter_std)
        self.ahp_drop = float(ahp_drop)
        
        tau_phi_val = resolve_neuron_param(tau_phi, 1, self.device).item()
        self.phi_decay = float(np.exp(-self.dt / tau_phi_val))

        # 3. State Init
        self.phase = torch.zeros(n_neurons, dtype=torch.float32, device=self.device)
        self.r = torch.zeros(n_neurons, dtype=torch.float32, device=self.device)

    def _generate_spikes(self, t_idx: int) -> None:
        """
        Drive phase dynamics and check for cycle completion.
        """
        # 1. Calculate Driving Rate [Hz]
        # r = r_max * sigmoid(beta * (V - theta))
        self.r = self.r_max * torch.sigmoid(self.beta * (self.V - self.theta))

        # 2. Integrate Phase
        # phi = phi * decay + r * dt
        self.phase.mul_(self.phi_decay).add_(self.r * self.dt)

        # 3. Add Phase Jitter (Optional)
        # Adds Wiener process noise: dPhi += N(0, std * sqrt(dt))
        if self.jitter_std > 0.0:
            noise = torch.randn_like(self.phase) * (self.jitter_std * (self.dt ** 0.5))
            self.phase.add_(noise)

        # 4. Cycle Detection (Spike)
        self.phase.clamp_min_(0.0) # Phase cannot be negative
        
        # Determine number of full cycles completed (usually 0 or 1)
        n_spikes = self.phase.floor()
        
        # Spike condition: at least one full cycle
        self.spikes.copy_(n_spikes >= 1.0)
        self.spikes.logical_or_(self._input_spikes)

        # 5. Phase Wrap-Around (Soft Reset)
        # Drain the integer part, keep the fractional remainder
        # phi <- phi - floor(phi)
        self.phase.sub_(n_spikes)

        # 6. After-Hyperpolarization (Optional)
        # Soft subtraction from V instead of hard reset
        if self.ahp_drop > 0.0:
            self.V.sub_(self.spikes.float() * self.ahp_drop)

        # Record
        self._write_spikes_to_buffer(t_idx)


class AdExNeurons(IFDeterministicBase):
    r"""
    Adaptive Exponential Integrate-and-Fire (AdEx) Model.

    Mathematical Model:
    -------------------
    A biologically realistic model that captures spike frequency adaptation and 
    resonance properties using two coupled differential equations.

    1. **Membrane Potential $V(t)$:**
       $$ C_m \frac{dV}{dt} = -g_L(V - E_L) + g_L \Delta_T \exp\left(\frac{V - V_T}{\Delta_T}\right) + I_{syn}(t) - w(t) $$
       
       Includes an exponential term that models the rapid sodium channel activation 
       near the threshold (sharp spike initiation).

    2. **Adaptation Current $w(t)$:**
       $$ \tau_w \frac{dw}{dt} = a(V - E_L) - w $$
       
       Models slow adaptation currents (e.g., K-channels).

    Reset Dynamics:
    ---------------
    When $V(t) > V_{peak}$ (numerical threshold):
    - $V \leftarrow V_{reset}$
    - $w \leftarrow w + b$ (Spike-triggered adaptation jump)

    Integration:
    ------------
    Unlike the standard LIF classes, AdEx requires **Euler integration** because 
    the exponential term prevents a closed-form analytic solution.
    """

    # --- AdEx Specific Configuration ---
    E_leak: Final[torch.Tensor]
    """Leak reversal potential ($E_L$). Distinct from the reset potential. Shape: [n_neurons]."""

    v_rheobase: Final[torch.Tensor]
    """Rheobase threshold ($V_T$). Voltage where the exponential term activates. Shape: [n_neurons]."""

    delta_T: Final[torch.Tensor]
    r"""Slope factor ($\Delta_T$). Controls the sharpness of spike initiation. Shape: scalar/tensor."""

    a: Final[torch.Tensor]
    """Sub-threshold adaptation conductance ($a$). Shape: [n_neurons]."""

    b: Final[torch.Tensor]
    """Spike-triggered adaptation current ($b$). Shape: [n_neurons]."""

    dt_over_tau_w: Final[torch.Tensor]
    r"""Pre-calculated factor $\Delta t / \tau_w$ for w update. Shape: [n_neurons]."""

    # --- Dynamic State ---
    w: torch.Tensor
    """Adaptation current variable. Units: [Amperes] (or normalized). Shape: [n_neurons]."""

    def __init__(
        self,
        n_neurons: int,
        spatial_dimensions: int = 2,
        delay_max: int = 20,
        n_channels: int = 3,
        channel_time_constants: list[tuple[float, float]] = (
            (0.001, 0.005), 
            (0.001, 0.010), 
            (0.002, 0.100),
        ),
        channel_reversal_potentials: list[float] = (0.0, -0.070, 0.0),
        
        # AdEx Physics Parameters
        tau_membrane: NeuronParam = 0.020, # Acts as Cm/gL
        E_leak: NeuronParam = -0.070,      # $E_L$ (Physical resting potential)
        v_rheobase: NeuronParam = -0.050,  # $V_T$ (Exponential threshold)
        v_reset: float = -0.060,           # $V_{reset}$ (Post-spike voltage)
        threshold: float = -0.030,         # $V_{peak}$ (Numerical cut-off for spike detection)
        delta_T: float = 0.002,            # $\Delta_T$ (Slope factor)
        
        # Adaptation Parameters
        a: NeuronParam = 0.0,              # Sub-threshold adaptation (nS)
        b: NeuronParam = 0.010,            # Spike-triggered adaptation (pA)
        tau_w: NeuronParam = 0.100,        # Time constant for w
        
        dt: float = 1e-3,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ):
        """
        Initialize AdEx Neurons.
        
        Note:
        - `threshold` argument maps to $V_{peak}$ (spike detection).
        - `E_rest` (base class arg) maps to $V_{reset}$.
        """
        # Pass V_peak as 'threshold' for base detection logic.
        # Pass V_reset as 'E_rest' for base reset logic.
        # AdEx implies no hard refractory period (refractory is modeled by reset + w), so tau_refrac=0.
        super().__init__(
            n_neurons=n_neurons,
            spatial_dimensions=spatial_dimensions,
            delay_max=delay_max,
            n_channels=n_channels,
            channel_time_constants=channel_time_constants,
            channel_reversal_potentials=channel_reversal_potentials,
            tau_membrane=tau_membrane,
            E_rest=v_reset,     # Base class uses this for reset
            threshold=threshold, # Base class uses this for spike detection
            tau_refrac=0.0,      
            dt=dt,
            device=device,
            **kwargs
        )

        # 1. Voltage Dynamics Params
        self.E_leak = resolve_neuron_param(E_leak, n_neurons, self.device)
        self.v_rheobase = resolve_neuron_param(v_rheobase, n_neurons, self.device)
        self.delta_T = torch.tensor(delta_T, device=self.device)
        
        # 2. Adaptation Dynamics Params
        self.a = resolve_neuron_param(a, n_neurons, self.device)
        self.b = resolve_neuron_param(b, n_neurons, self.device)
        
        tau_w_vec = resolve_neuron_param(tau_w, n_neurons, self.device)
        self.dt_over_tau_w = self.dt / tau_w_vec

        # 3. State Initialization
        self.w = torch.zeros(n_neurons, dtype=torch.float32, device=self.device)

    def _integrate_membrane(self) -> None:
        """
        Numerical Integration (Euler Forward) for the AdEx system.
        Overrides the analytic integration of `ConductanceNeuronBase`.
        """
        # 1. Calculate Synaptic Current ($I_{syn}$)
        # Note: _update_channels() was already called by parent, consuming inputs.
        # g_channels = Decay - Rise (Bi-exponential state)
        g_channels = torch.relu(self.channel_states[:, :, 1] - self.channel_states[:, :, 0])
        
        # Sum of currents: sum(g_i * (E_rev_i - V))
        I_syn = (g_channels * (self.E_rev_row - self.V.unsqueeze(1))).sum(dim=1)

        # 2. Compute Equation Terms
        # Leak Term: -gL * (V - E_L) (assuming gL=1 normalized)
        diff_V_EL = self.V - self.E_leak
        
        # Exponential Term: gL * delta_T * exp((V - V_T) / delta_T)
        # Clamp exponent to prevent overflow during upswing
        arg_exp = (self.V - self.v_rheobase) / self.delta_T
        arg_exp = torch.clamp(arg_exp, max=20.0) 
        I_exp = self.delta_T * torch.exp(arg_exp)

        # Total Current (normalized by gL=1)
        # I_total = -Leak + Exp + Syn - w
        I_total = -diff_V_EL + I_exp + I_syn - self.w
        
        # 3. Update V (Forward Euler)
        # dV = (I_total / Cm) * dt
        dV = (I_total / self.Cm) * self.dt
        
        # Store derivative for optional gradient estimation
        self.dYdt.copy_(I_total / self.Cm)
        
        self.V.add_(dV)

        # 4. Update w (Forward Euler)
        # dw = (a(V - E_L) - w) * dt / tau_w
        dw = self.dt_over_tau_w * (self.a * diff_V_EL - self.w)
        self.w.add_(dw)

    def _on_spikes(self, mask: torch.Tensor) -> None:
        r"""
        Apply spike-triggered adaptation jump.
        $$ w \leftarrow w + b $$
        Note: Voltage reset ($V \leftarrow V_{reset}$) is handled by `IFDeterministicBase`.
        """
        if self.b is not None: # check if 'b' is effectively zero (scalar) or tensor
             # In tensor case this check is implicit by values, but explicit adding is fine
             self.w[mask] += self.b[mask]