from __future__ import annotations

from typing import Union

from . import globals

from .core import ConnectionOperator
from .group import SpatialGroup

import torch


class NeuronGroup(SpatialGroup):
    """Base class for groups of neurons with spike propagation capabilities.

    Extends _SpatialGroup to provide basic functionality for handling spikes,
    delays, and input currents common to all neuron models.

    Attributes
    ----------
    delay_max : torch.Tensor
        Maximum delay in time steps for spike propagation.
    _spike_buffer : torch.Tensor
        Boolean tensor of shape (n_neurons, delay_max) that stores spike history.
    _input_currents : torch.Tensor
        Float tensor of shape (n_neurons, n_channels) for accumulating input currents.
    _input_spikes : torch.Tensor
        Boolean tensor of shape (n_neurons,) for injected spikes.
    """

    delay_max: torch.Tensor #[1]
    _spike_buffer: torch.Tensor #[neuron, delay]
    _input_currents: torch.Tensor #[neuron, channel]
    n_channels: int
    _input_spikes: torch.Tensor #[neuron]
    spikes: torch.Tensor #[neuron] Output spikes in this time step.

    def __init__(
        self,
        n_neurons: int,
        spatial_dimensions: int = 2,
        delay_max: int = 20,
        n_channels: int = 1,
        device: torch.device = None,
    ):
        """Initialize a group of neurons.

        Parameters
        ----------
        device : str
            String representation of the GPU device (e.g., 'cuda:0').
        n_neurons : int
            Number of neurons in the group.
        spatial_dimensions : int, optional
            Number of spatial dimensions, by default 2.
        delay_max : int, optional
            Maximum delay in time steps for spike propagation, by default 20.
        """
        super().__init__(n_neurons, spatial_dimensions, device)
        self.delay_max = torch.tensor([delay_max], dtype=torch.long, device=self.device)
        self._spike_buffer = torch.zeros(
            (n_neurons, delay_max), dtype=torch.bool, device=self.device
        )
        self._input_currents = torch.zeros(
            (n_neurons, n_channels), dtype=torch.float32, device=self.device
        )
        self.n_channels = n_channels
        self._input_spikes = torch.zeros(
            n_neurons, dtype=torch.bool, device=self.device
        )
        self.spikes = torch.zeros(n_neurons, dtype=torch.bool, device=self.device)

    def get_spike_buffer(self):
        """Get the internal spike buffer.

        Returns
        -------
        torch.Tensor
            Boolean tensor of shape (n_neurons, delay_max) containing spike history.
        """
        return self._spike_buffer

    def inject_currents(self, I: torch.Tensor, chn: int=0) -> None:
        """Inject input currents into the neurons.

        The input currents are accumulated and processed during the next call
        to _process().

        Parameters
        ----------
        I : torch.Tensor
            Float tensor of shape (n_neurons,) containing input currents.

        Raises
        ------
        AssertionError
            If the shape of I doesn't match the number of neurons.
        """
        assert I.shape[0] == self.size
        self._input_currents[:,chn].add_(I)

    def inject_spikes(self, spikes: torch.Tensor) -> None:
        """Force neurons to spike, independently of their weights or state.

        The injected spikes are accumulated and processed during the next call
        to _process().

        Parameters
        ----------
        spikes : torch.Tensor
            Boolean or convertible-to-boolean tensor of shape (n_neurons,)
            indicating which neurons should spike.

        Raises
        ------
        AssertionError
            If the shape of spikes doesn't match the number of neurons.
        """
        assert spikes.shape[0] == self.size
        self._input_spikes |= spikes.bool()

    def get_spikes(self) -> torch.Tensor:
        """Get the current spikes.

        Returns
        -------
        torch.Tensor
            Boolean tensor of shape (M,) with the spike status for each neuron.

        """
        phase = (globals.simulator.local_circuit.current_step - 1) % self.delay_max
        return self._spike_buffer[:, phase].squeeze_(1)
    
    def get_spikes_at(
        self, delays: Union[int, torch.Tensor], indices: torch.Tensor
    ) -> torch.Tensor:
        """Get the spikes for specific neurons at specific delays.

        Parameters
        ----------
        delays : int or torch.Tensor
            - If int: scalar delay applied to all indices.
            - If Tensor: integer tensor of shape (M,) with per-connection delays.
        indices : torch.Tensor
            Integer tensor of shape (M,) with neuron indices.

        Returns
        -------
        torch.Tensor
            Boolean tensor of shape (M,) with the spike status for each
            (neuron, delay) pair.

        Raises
        ------
        AssertionError
            If delays is a tensor and its shape does not match indices.

        Notes
        -----
        This method is used primarily by synaptic connections to retrieve
        pre-synaptic or post-synaptic spikes with appropriate delays.
        """
        if isinstance(delays, int):
            # Escalar: aplicar mismo delay a todas las conexiones
            phase = (globals.simulator.local_circuit.current_step - delays) % self.delay_max
            return self._spike_buffer[indices, phase]

        # Caso tensorial
        assert delays.shape == indices.shape, "Delays and indices must match in shape"
        phase = (globals.simulator.local_circuit.current_step - delays) % self.delay_max
        return self._spike_buffer[indices, phase]

    def __rshift__(self, other) -> ConnectionOperator:
        """Implement the >> operator for creating connections between neuron groups.

        This operator provides a concise syntax for defining connections:
        (source_group >> target_group)([params])

        Parameters
        ----------
        other : NeuronGroup
            Target neuron group for the connection.

        Returns
        -------
        ConnectionOperator
            An operator object that can be called to specify connection parameters.

        Examples
        --------
        >>> # Create all-to-all connections with weight 0.1
        >>> (source_group >> target_group)(pattern='all-to-all', weight=0.1)
        """
        return ConnectionOperator(self, other)


class ParrotNeurons(NeuronGroup):
    """Neuron group that simply repeats input spikes or currents.

    This neuron model acts as a simple repeater - any input current or spike
    directly generates an output spike without any dynamics or threshold.
    Useful for relay operations or input layers.
    """

    def _process(self) -> None:
        """Process inputs and generate outputs for the current time step.

        Implements the parrot neuron behavior: any positive input current or
        injected spike causes the neuron to emit a spike at the current time step.
        """
        super()._process()

        # Clear any remaining spikes
        phase = globals.simulator.local_circuit.current_step % self.delay_max
        self._spike_buffer.index_fill_(1, phase, 0)

        # Process any injected spikes
        # Store spikes in the buffer at current t
        self._spike_buffer.index_copy_(
            1,
            phase,
            (
                self._spike_buffer.index_select(1, phase)
                | self._input_spikes.unsqueeze(1)
            ),
        )

        # Clear injected spikes
        self._input_spikes.fill_(False)

        # Process input currents
        # Generate spikes for neurons receiving any positive current
        spikes = self._input_currents.squeeze() > 0
        self._spike_buffer.index_copy_(
            1, phase, (self._spike_buffer.index_select(1, phase) | spikes.unsqueeze(1))
        )
        # Clear input currents
        self._input_currents.fill_(0.0)

        # Save spikes
        self.spikes.copy_(self._spike_buffer[:, phase].squeeze(1))


class SimpleIFNeurons(NeuronGroup):
    """Integrate-and-Fire neuron model.

    A simple Integrate-and-Fire model where the membrane potential integrates
    input current with decay, and spikes when a threshold is reached.

    Attributes
    ----------
    V : torch.Tensor
        Membrane potential for each neuron.
    threshold : torch.Tensor
        Spike threshold value.
    decay : torch.Tensor
        Membrane potential decay factor (per time step).
    """

    V: torch.Tensor
    threshold: torch.Tensor
    decay: torch.Tensor

    def __init__(
        self,
        n_neurons: int,
        spatial_dimensions: int = 2,
        delay_max: int = 20,
        threshold: float = 1.0,
        tau_membrane: float = 0.1,
        device: str = None,
    ):
        """Initialize an Integrate-and-Fire neuron group.

        Parameters
        ----------
        device : str
            String representation of the GPU device (e.g., 'cuda:0').
        n_neurons : int
            Number of neurons in the group.
        spatial_dimensions : int, optional
            Number of spatial dimensions, by default 2.
        delay_max : int, optional
            Maximum delay in time steps for spike propagation, by default 20.
        threshold : float, optional
            Membrane potential threshold for spiking, by default 1.0.
        tau : float, optional
            Membrane time constant in seconds, by default 0.1.
            Determines the decay rate of the membrane potential.
        """
        super().__init__(n_neurons=n_neurons, spatial_dimensions=spatial_dimensions, delay_max=delay_max, n_channels=1, device=device)
        self.V = torch.zeros(n_neurons, dtype=torch.float32, device=self.device)
        self.threshold = torch.tensor([threshold], dtype=torch.float32, device=self.device)
        self.decay = torch.exp(
            torch.tensor(-1e-3 / tau_membrane, dtype=torch.float32, device=self.device)
        )

    def _process(self):
        """Update membrane potentials and generate spikes.

        For each neuron, updates the membrane potential with decay and input current,
        checks if the threshold is reached, and generates spikes accordingly.
        After spiking, the membrane potential is reset to zero.
        """
        super()._process()
        phase = globals.simulator.local_circuit.current_step % self.delay_max

        # Update potential with decay and input
        self.V *= self.decay
        self.V += self._input_currents.squeeze()
        self._input_currents.fill_(0.0)

        # Determine which neurons spike
        self.spikes.copy_(self.V >= self.threshold)
        self.spikes.logical_or_(self._input_spikes)
        self._spike_buffer.index_copy_(1, phase, self.spikes.unsqueeze(1))
        self.V[self.spikes] = 0.0  # Reset membrane potential
        self._input_spikes.fill_(False)


class RandomSpikeNeurons(NeuronGroup):
    """Generates random spikes according to a Poisson process.

    This neuron model does not integrate inputs, but rather generates random
    spikes based on a specified firing rate.

    Attributes
    ----------
    firing_rate : torch.Tensor
        Firing rate in kHz (spikes per millisecond).
    probabilities : torch.Tensor
        Temporary storage for random values.
    """

    firing_rate: torch.Tensor  # In Hz
    probabilities: torch.Tensor

    def __init__(
        self,
        n_neurons: int,
        firing_rate: float = 10.0,
        spatial_dimensions: int = 2,
        delay_max: int = 20,
        device: str = None,
    ):
        """Initialize a random spike generator neuron group.

        Parameters
        ----------
        device : str
            String representation of the GPU device (e.g., 'cuda:0').
        n_neurons : int
            Number of neurons in the group.
        firing_rate : float, optional
            Firing rate in Hz, by default 10.0.
        spatial_dimensions : int, optional
            Number of spatial dimensions, by default 2.
        delay_max : int, optional
            Maximum delay in time steps for spike propagation, by default 20.
        """
        super().__init__(
            n_neurons = n_neurons,
            spatial_dimensions = spatial_dimensions,
            delay_max = delay_max,
            device = device,
        )
        self.firing_rate = torch.tensor(
            firing_rate * 1e-3, dtype=torch.float32, device=self.device
        )
        self.probabilities = torch.zeros(n_neurons, dtype=torch.float32, device=self.device)

    def _process(self):
        """Generate random spikes based on the firing rate.

        Each neuron has a probability of firing equal to the firing rate
        times the time step (in milliseconds).
        """
        super()._process()
        phase = globals.simulator.local_circuit.current_step % self.delay_max

        self.probabilities.uniform_()
        self.spikes.copy_(self.probabilities < self.firing_rate)
        self._spike_buffer.index_copy_(1, phase, self.spikes.unsqueeze(1))


class IFNeurons(NeuronGroup):
    """Integrate-and-Fire neurons with multi-channel conductance-based dynamics.

    Uses analytical integration for membrane potential to avoid overshoots and
    ensure unconditional stability. Each channel has bi-exponential conductance
    dynamics (rise/decay) and a reversal potential.

    The membrane potential evolves according to:
        Cm·dV/dt = -g_leak·(V - E_rest) - Σᵢ gᵢ·(V - Eᵢ)

    which is integrated exactly using:
        V(t+dt) = E_eff + (V - E_eff)·exp(-g_eff·dt/Cm)
    where g_eff = g_leak + Σᵢ gᵢ and E_eff is the weighted reversal potential.
    """

    V: torch.Tensor
    threshold: torch.Tensor
    E_rest: torch.Tensor
    E_channels: torch.Tensor
    channel_states: torch.Tensor  # (n_neurons, n_channels, 2)
    channel_currents: torch.Tensor  # (n_neurons, n_channels) - for monitoring only
    channel_decay_factors: torch.Tensor  # (n_channels, 2)
    channel_normalization: torch.Tensor  # (n_channels,)
    input_channels: torch.Tensor  # (n_neurons, n_channels)
    _V_reset_buffer: torch.Tensor
    refrac_counter: torch.Tensor  # Absolute refractory period counter
    refrac_steps: int  # Number of steps for refractory period

    # Physical parameters for analytical integration
    dt: float  # Timestep in seconds
    Cm: float  # Membrane capacitance (normalized to 1.0)
    g_leak: float  # Leak conductance

    def __init__(
        self,
        n_neurons: int,
        spatial_dimensions: int = 2,
        delay_max: int = 20,
        n_channels: int = 3,
        channel_time_constants: list[tuple[float, float]] = (
            (0.001, 0.005),  # AMPA: subida 1ms, caída 5ms
            (0.001, 0.010),  # GABA: subida 1ms, caída 10ms
            (0.002, 0.100),  # NMDA: subida 2ms, caída 100ms
        ),
        channel_reversal_potentials: list[float] = (
            0.0,     # AMPA: 0 mV
            -0.070,  # GABA: -70 mV
            0.0,     # NMDA: 0 mV
        ),
        threshold: float = -0.050,   # -50 mV
        tau_membrane: float = 0.010, # 10 ms
        E_rest: float = -0.065,      # -65 mV
        tau_refrac: float = 0.002,   # 2 ms absolute refractory period
        device: str = None,
    ):
        super().__init__(
            n_neurons = n_neurons,
            spatial_dimensions = spatial_dimensions,
            delay_max = delay_max,
            n_channels = n_channels,
            device = device)

        assert len(channel_time_constants) == n_channels
        assert len(channel_reversal_potentials) == n_channels

        # Physical parameters for analytical integration
        # Scale Cm and g_leak together to maintain tau_membrane while keeping
        # g_leak comparable to synaptic conductances (which are in weight units)
        self.dt = 1e-3          # 1 ms per timestep
        self.Cm = tau_membrane  # Capacitance scaled to match tau (e.g., 0.01 for 10ms)
        self.g_leak = 1.0       # Normalized leak conductance (comparable to weight scales)

        self.V = torch.full((n_neurons,), E_rest, dtype=torch.float32, device=self.device)
        self._V_reset_buffer = torch.empty_like(self.V)
        self.threshold = torch.tensor([threshold], dtype=torch.float32, device=self.device)

        tau_rise = torch.tensor([tc[0] for tc in channel_time_constants], dtype=torch.float32, device=self.device)
        tau_decay = torch.tensor([tc[1] for tc in channel_time_constants], dtype=torch.float32, device=self.device)

        self.channel_decay_factors = torch.stack([
            torch.exp(-self.dt / tau_rise),
            torch.exp(-self.dt / tau_decay)
        ], dim=1)  # (n_channels, 2)

        self.channel_normalization = (tau_decay - tau_rise) / (tau_decay * tau_rise)

        self.channel_states = torch.zeros(n_neurons, n_channels, 2, dtype=torch.float32, device=self.device)
        self.input_channels = torch.zeros(n_neurons, n_channels, dtype=torch.float32, device=self.device)
        self.channel_currents = torch.zeros(n_neurons, n_channels, dtype=torch.float32, device=self.device)

        self.E_rest = torch.tensor([E_rest], dtype=torch.float32, device=self.device)
        self.E_channels = torch.tensor(channel_reversal_potentials, dtype=torch.float32, device=self.device)

        # Absolute refractory period (prevents unrealistic firing rates)
        self.refrac_steps = int(tau_refrac / self.dt)  # Convert to timesteps
        self.refrac_counter = torch.zeros(n_neurons, dtype=torch.int32, device=self.device)

    def _process(self) -> None:
        """Updates internal states, integrates dynamics and generates spikes.

        Uses analytical integration for unconditional stability and to avoid
        voltage overshoots. The membrane potential converges exponentially
        towards the weighted reversal potential E_eff.
        """
        super()._process()
        phase = globals.simulator.local_circuit.current_step % self.delay_max

        # --- Bi-exponenciales: diferencia de dos LPs con la MISMA entrada ---
        normalized_input = self._input_currents * self.channel_normalization.unsqueeze(0)  # (n_neurons, n_channels)
        self._input_currents.zero_()

        # Decaimiento por componente
        self.channel_states[:, :, 0].mul_(self.channel_decay_factors[:, 0])  # rise
        self.channel_states[:, :, 1].mul_(self.channel_decay_factors[:, 1])  # decay

        # La entrada debe sumarse a AMBAS componentes
        self.channel_states[:, :, 0].add_(normalized_input)
        self.channel_states[:, :, 1].add_(normalized_input)

        # Conductancia de canal: g_i = relu(decay - rise) >= 0
        g_channels = torch.relu(self.channel_states[:, :, 1] - self.channel_states[:, :, 0])  # (n_neurons, n_channels)

        # Corriente de canal (g_i * (E_i - V)) - stored for monitoring only
        channel_drive = self.E_channels.unsqueeze(0) - self.V.unsqueeze(1)  # (n_neurons, n_channels)
        self.channel_currents.copy_(g_channels * channel_drive)

        # --- ANALYTICAL INTEGRATION (conductance-based, unconditionally stable) ---
        # Total synaptic conductance
        g_syn = g_channels.sum(dim=1)  # (n_neurons,)

        # Effective conductance: leak + synaptic
        g_eff = g_syn + self.g_leak  # (n_neurons,)

        # Effective reversal potential: weighted by conductances
        # E_eff = (g_leak*E_rest + sum_i g_i*E_i) / g_eff
        E_eff_num = self.g_leak * self.E_rest + (g_channels * self.E_channels).sum(dim=1)
        E_eff = E_eff_num / g_eff

        # Exact solution of linear ODE: V(t+dt) = E_eff + (V - E_eff) * exp(-g_eff * dt / Cm)
        exp_term = torch.exp(-g_eff * (self.dt / self.Cm))
        self.V.copy_(E_eff + (self.V - E_eff) * exp_term)

        # Decrementar contador refractario (clamp a 0)
        self.refrac_counter.sub_(1).clamp_(min=0)

        # Generación de spikes: solo si V >= threshold Y no está en periodo refractario
        spike_candidates = self.V >= self.threshold
        not_refractory = self.refrac_counter == 0
        self.spikes.copy_(spike_candidates & not_refractory)
        self.spikes.logical_or_(self._input_spikes)

        # Resetear contador refractario para neuronas que dispararon
        self.refrac_counter.masked_fill_(self.spikes, self.refrac_steps)

        # Registrar spikes y aplicar reset a E_rest tras el disparo
        self._spike_buffer.index_copy_(1, phase, self.spikes.unsqueeze(1))
        self._V_reset_buffer.copy_(self.V)
        torch.where(self.spikes, self.E_rest.expand_as(self.V), self._V_reset_buffer, out=self.V)

        # Limpiar spikes inyectados
        self._input_spikes.fill_(False)


class StochasticIFNeurons(IFNeurons):
    """Integrate-and-Fire neurons without reset, with stochastic spike generation.

    The membrane potential decays continuously; spikes are emitted
    probabilistically as a smooth function of V relative to threshold.
    """

    def __init__(self, *args, beta: float = 200.0, **kwargs):
        """
        beta : float
            Steepness of the sigmoid that converts voltage into spike probability.
            Larger beta -> more deterministic firing.
        """
        super().__init__(*args, **kwargs)
        self.beta = beta
        # Eliminamos el uso del contador refractario (ya no se necesita un reset duro)
        self.refrac_counter.zero_()

    def _process(self) -> None:
        super(NeuronGroup, self)._process()  # ⚠️ saltar llamada a IFNeurons._process
        phase = globals.simulator.local_circuit.current_step % self.delay_max

        # --- Dinámica sináptica idéntica a tu versión ---
        normalized_input = self._input_currents * self.channel_normalization.unsqueeze(0)
        self._input_currents.zero_()

        self.channel_states[:, :, 0].mul_(self.channel_decay_factors[:, 0])
        self.channel_states[:, :, 1].mul_(self.channel_decay_factors[:, 1])
        self.channel_states[:, :, 0].add_(normalized_input)
        self.channel_states[:, :, 1].add_(normalized_input)

        g_channels = torch.relu(self.channel_states[:, :, 1] - self.channel_states[:, :, 0])
        channel_drive = self.E_channels.unsqueeze(0) - self.V.unsqueeze(1)
        self.channel_currents.copy_(g_channels * channel_drive)

        g_syn = g_channels.sum(dim=1)
        g_eff = g_syn + self.g_leak
        E_eff_num = self.g_leak * self.E_rest + (g_channels * self.E_channels).sum(dim=1)
        E_eff = E_eff_num / g_eff

        exp_term = torch.exp(-g_eff * (self.dt / self.Cm))
        self.V.copy_(E_eff + (self.V - E_eff) * exp_term)

        # --- 🔸 NUEVA PARTE: generación de spikes estocástica ---
        # Probabilidad instantánea de spike: sigmoide del voltaje
        #   p = σ(β(V - V_th)) * dt_scale
        # dt_scale ajusta la probabilidad a segundos (asumiendo dt en segundos)
        target_firing_rate = 20.0
        dt_scale = 1e-3  # o self.dt si quieres interpretar p por segundo
        p_spike = torch.sigmoid(self.beta * (self.V - self.threshold)) * dt_scale * target_firing_rate

        # Generar spikes Bernoulli(p)
        rand_vals = torch.rand_like(p_spike)
        self.spikes.copy_(rand_vals < p_spike)
        self._spike_buffer.index_copy_(1, phase, self.spikes.unsqueeze(1))

        # --- 🔸 Sin reset: el voltaje sigue su dinámica continua ---
        # (opcional: ligera caída tras un spike, para evitar runaway)
        #self.V.sub_(self.spikes.float() * 0.002)  # caída suave opcional de 2 mV

        # Limpiar spikes inyectados
        self._input_spikes.fill_(False)


class PhaseIFNeurons(IFNeurons):
    """IF sin reset con disparo determinista vía integración de fase.
    
    La tasa instantánea r(V) se obtiene con un umbral suave (sigmoide) y
    se integra como dphi/dt = r(V). Se emite spike cuando phi >= 1.
    No hay reset del potencial: V sigue su dinámica continua.
    """

    phase: torch.Tensor
    beta: float
    r_max: float
    theta: torch.Tensor
    jitter_std: float
    ahp_drop: float

    def __init__(
        self,
        *args,
        beta: float = 200.0,         # “dureza” del umbral suave
        r_max: float = 300.0,        # Hz, tasa máxima (seguridad/clip)
        theta: float = None,         # umbral para la sigmoide; por defecto usa self.threshold
        jitter_std: float = 0.0,     # desviación típica del jitter de fase (en unidades de fase/sqrt(s))
        ahp_drop: float = 0.0,       # caída suave tras spike (ej. 0.002 => ~2 mV)
        tau_phi: float = 0.05,       # 50 ms de constante de tiempo típica
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.beta = float(beta)
        self.r_max = float(r_max)
        self.theta = self.threshold.clone() if theta is None else torch.tensor([theta], dtype=torch.float32, device=self.device)
        self.jitter_std = float(jitter_std)
        self.ahp_drop = float(ahp_drop)
        self.phi_decay = torch.exp(-self.dt / tau_phi)
        # Fase inicial
        self.phase = torch.zeros_like(self.V)
        # El refractario “duro” ya no se usa; lo dejamos en cero
        if hasattr(self, "refrac_counter"):
            self.refrac_counter.zero_()

    @torch.no_grad()
    def _process(self) -> None:
        # ⚠️ No llamar a IFNeurons._process (haría reset). Llamamos al de la clase base de la jerarquía.
        super(NeuronGroup, self)._process()
        phase = globals.simulator.local_circuit.current_step % self.delay_max

        # === 1) Dinámica sináptica: igual que en IFNeurons ===
        normalized_input = self._input_currents * self.channel_normalization.unsqueeze(0)  # (n_neurons, n_channels)
        self._input_currents.zero_()

        # Decaimiento bi-exponencial
        self.channel_states[:, :, 0].mul_(self.channel_decay_factors[:, 0])  # rise
        self.channel_states[:, :, 1].mul_(self.channel_decay_factors[:, 1])  # decay
        # Inyección a ambas ramas
        self.channel_states[:, :, 0].add_(normalized_input)
        self.channel_states[:, :, 1].add_(normalized_input)

        # g_i >= 0
        g_channels = torch.relu(self.channel_states[:, :, 1] - self.channel_states[:, :, 0])  # (n_neurons, n_channels)

        # Solo para monitorización (como en tu clase)
        channel_drive = self.E_channels.unsqueeze(0) - self.V.unsqueeze(1)
        self.channel_currents.copy_(g_channels * channel_drive)

        # === 2) Integración analítica del voltaje (sin reset) ===
        g_syn = g_channels.sum(dim=1)                      # (n,)
        g_eff = g_syn + self.g_leak                        # (n,)
        E_eff_num = self.g_leak * self.E_rest + (g_channels * self.E_channels).sum(dim=1)
        E_eff = E_eff_num / g_eff
        exp_term = torch.exp(-g_eff * (self.dt / self.Cm))
        self.V.copy_(E_eff + (self.V - E_eff) * exp_term)

        # === 3) Tasa instantánea y avance de fase ===
        # r(V) = r_max * sigmoid(beta*(V - theta))
        # Nota: r se interpreta en Hz, dt en segundos -> incremento de fase = r * dt
        r = self.r_max * torch.sigmoid(self.beta * (self.V - self.theta))  # (n,)
        self.phase.add_(r * self.dt)

        # Avance de fase con decaimiento
        self.phase.mul_(self.phi_decay)
        self.phase.add_(r * self.dt)

        # Jitter opcional en fase (evita sincronías demasiado rígidas)
        if self.jitter_std > 0.0:
            # Escala ~ sqrt(dt) para ruido blanco en tiempo continuo
            self.phase.add_(torch.randn_like(self.phase) * (self.jitter_std * (self.dt ** 0.5)))

        # === 4) Detección de spikes por sobrepaso de 1 ciclo de fase ===
        # Contabilizamos potencialmente múltiples cruces en un dt (r*dt > 1),
        # pero el buffer es booleano por paso, así que solo marcamos True si hubo >=1.
        n_spikes_float = torch.floor(torch.clamp_min(self.phase, 0.0))
        spike_candidates = n_spikes_float >= 1.0

        # Drenamos fase en bloque (equivalente a while phi>=1: phi-=1)
        self.phase.sub_(n_spikes_float)

        # AHP suave opcional (baja un poco V hacia E_rest tras disparo)
        if self.ahp_drop > 0.0:
            # Aplicamos caída solo donde hubo spike
            # (V <- V - ahp_drop) o, más fisiológico: V <- V - ahp_drop*(V - E_rest)
            self.V.sub_(spike_candidates.float() * self.ahp_drop)

        # === 5) Inyección de spikes externos y escritura del buffer ===
        # Unimos candidatos con inyecciones externas de este paso
        self.spikes.copy_(spike_candidates | self._input_spikes)
        # Registrar en el buffer (bool por paso)
        self._spike_buffer.index_copy_(1, phase, self.spikes.unsqueeze(1))
        # Limpiar inyección para el siguiente paso
        self._input_spikes.fill_(False)
