from __future__ import annotations

from typing import Union

from . import globals
from .utils import RandomDistribution

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
    delay_max_int: int
    _spike_buffer: torch.Tensor #[neuron, delay]
    _input_currents: torch.Tensor #[neuron, channel]
    n_channels: int
    _input_spikes: torch.Tensor #[neuron]
    spikes: torch.Tensor #[neuron] Output spikes in this time step.
    dYdt: torch.Tensor       # (n_neurons,) Added for learning rules


    def __init__(
        self,
        n_neurons: int,
        spatial_dimensions: int = 2,
        delay_max: int = 20,
        n_channels: int = 1,
        device: torch.device = None,
        **kwargs,
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
        super().__init__(n_neurons, spatial_dimensions, device, **kwargs)
        self.delay_max = torch.tensor([delay_max], dtype=torch.long, device=self.device)
        self.delay_max_int = delay_max
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
        self.dYdt = torch.zeros(n_neurons, dtype=torch.float32, device=self.device)

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
        phase = (globals.simulator.local_circuit.current_step - 1) % self.delay_max_int
        return self._spike_buffer[:, phase].squeeze(1)
    
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
            phase = (globals.simulator.local_circuit.current_step - delays) % self.delay_max_int
            return self._spike_buffer[indices, phase]

        # Caso tensorial
        assert delays.shape == indices.shape, "Delays and indices must match in shape"
        phase = (globals.simulator.local_circuit.current_step - delays) % self.delay_max_int
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
        phase = globals.simulator.local_circuit.current_step % self.delay_max_int
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
        phase = globals.simulator.local_circuit.current_step % self.delay_max_int

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
    previous_fr: torch.Tensor
    probabilities: torch.Tensor

    def __init__(
        self,
        n_neurons: int,
        firing_rate: Union[float, torch.Tensor] = 10.0,
        spatial_dimensions: int = 2,
        delay_max: int = 20,
        device: str = None,
        **kwargs,
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
            **kwargs,
        )
        
        if isinstance(firing_rate, torch.Tensor):
            fr = firing_rate.to(device=self.device, dtype=torch.float32)
        else:
            fr = torch.tensor(firing_rate, device=self.device, dtype=torch.float32)
        self.firing_rate = torch.full((n_neurons,), fr, device=self.device, dtype=torch.float32)
        self.previous_fr = torch.full((n_neurons,), fr, device=self.device, dtype=torch.float32)

        self.probabilities = torch.zeros(n_neurons, dtype=torch.float32, device=self.device)

    def _process(self):
        """Generate random spikes based on the firing rate.

        Each neuron has a probability of firing equal to the firing rate
        times the time step (in milliseconds).
        """
        super()._process()
        
        phase = globals.simulator.local_circuit.current_step % self.delay_max_int
        self.probabilities.uniform_()
        self.spikes.copy_(self.probabilities < (self.firing_rate*1e-3))
        self._spike_buffer.index_copy_(1, phase, self.spikes.unsqueeze(1))

        self.dYdt = (self.firing_rate - self.previous_fr) / 1e-3
        self.previous_fr.copy_(self.firing_rate)


class ConductanceNeuronBase(NeuronGroup):
    """Base para neuronas con dinámica conductance-based multicanal.

    Implementa:
    - Canales sinápticos bi-exponenciales (rise/decay) por canal.
    - Integración analítica del potencial de membrana:
        Cm dV/dt = -g_leak (V - E_rest) - Σ g_i (V - E_rev_i)

      ⇒ V(t+dt) = E_eff + (V - E_eff) * exp(-g_eff * dt / Cm)

    NO define:
    - cómo se generan los spikes,
    - refractario,
    - reset de membrana.
    """

    V: torch.Tensor          # (n_neurons,)
    E_rest: torch.Tensor     # (n_neurons,)
    E_rev: torch.Tensor      # (n_channels,)
    E_rev_row: torch.Tensor  # (1, n_channels) para broadcast

    channel_states: torch.Tensor      # (n_neurons, n_channels, 2) [rise, decay]
    channel_currents: torch.Tensor    # (n_neurons, n_channels)
    decay_factors: torch.Tensor       # (n_channels, 2) [rise, decay]
    norm_factors: torch.Tensor        # (n_channels,)

    dt: float
    Cm: float
    g_leak: float
    dt_over_Cm: float

    def __init__(
        self,
        n_neurons: int,
        spatial_dimensions: int,
        delay_max: int,
        n_channels: int,
        channel_time_constants: list[tuple[float, float]],
        channel_reversal_potentials: list[float],
        tau_membrane: float,
        E_rest: float,
        dt: float = 1e-3,
        device: str | torch.device | None = None,
        **kwargs
    ):
        super().__init__(
            n_neurons=n_neurons,
            spatial_dimensions=spatial_dimensions,
            delay_max=delay_max,
            n_channels=n_channels,
            device=device,
            **kwargs
        )

        assert len(channel_time_constants) == n_channels
        assert len(channel_reversal_potentials) == n_channels

        # Parámetros físicos
        self.dt = float(dt)
        self.Cm = float(tau_membrane)
        self.g_leak = 1.0
        self.dt_over_Cm = self.dt / self.Cm

        # Estado de membrana: E_rest vectorial por si quieres homeostasis por neurona
        self.V = torch.full(
            (n_neurons,),
            E_rest,
            dtype=torch.float32,
            device=self.device,
        )
    
        self.E_rest = torch.full(
            (n_neurons,),
            E_rest,
            dtype=torch.float32,
            device=self.device,
        )

        # Canales sinápticos
        tau_rise = torch.tensor(
            [tc[0] for tc in channel_time_constants],
            dtype=torch.float32,
            device=self.device,
        )
        tau_decay = torch.tensor(
            [tc[1] for tc in channel_time_constants],
            dtype=torch.float32,
            device=self.device,
        )

        self.decay_factors = torch.stack(
            [
                torch.exp(-self.dt / tau_rise),
                torch.exp(-self.dt / tau_decay),
            ],
            dim=1,
        )  # (n_channels, 2) [rise, decay]

        self.norm_factors = (tau_decay - tau_rise) / (tau_decay * tau_rise)

        self.channel_states = torch.zeros(
            n_neurons,
            n_channels,
            2,
            dtype=torch.float32,
            device=self.device,
        )
        self.channel_currents = torch.zeros(
            n_neurons,
            n_channels,
            dtype=torch.float32,
            device=self.device,
        )

        self.E_rev = torch.tensor(
            channel_reversal_potentials,
            dtype=torch.float32,
            device=self.device,
        )
        self.E_rev_row = self.E_rev.unsqueeze(0)  # (1, n_channels)

    # ----- Dinámica común -----

    def _update_channels(self) -> None:
        """Actualiza estados bi-exponenciales de los canales."""
        # Entrada normalizada
        normalized_input = self._input_currents.mul(
            self.norm_factors.unsqueeze(0)
        )  # (n_neurons, n_channels)
        self._input_currents.zero_()

        # Decaimiento rise/decay
        self.channel_states[:, :, 0].mul_(self.decay_factors[:, 0])  # rise
        self.channel_states[:, :, 1].mul_(self.decay_factors[:, 1])  # decay

        # Inyección de entrada en ambas ramas
        self.channel_states[:, :, 0].add_(normalized_input)
        self.channel_states[:, :, 1].add_(normalized_input)

    def _integrate_membrane(self) -> None:
        """Calcula conductancias y actualiza V analíticamente."""
        # g_i >= 0
        g_channels = torch.relu_(self.channel_states[:, :, 1] - self.channel_states[:, :, 0])
        # Corrientes por canal (solo monitorización)
        self.channel_currents[:] = g_channels * (self.E_rev_row - self.V.unsqueeze(1))

        # Integración analítica
        g_syn = g_channels.sum(dim=1)
        g_eff = g_syn.add(self.g_leak)  # (n_neurons,)

        E_eff_num = self.g_leak * self.E_rest + (g_channels * self.E_rev_row).sum(dim=1)
        E_eff = E_eff_num / g_eff

        # --- Derivada analítica dV/dt (antes de actualizar V)
        # dV/dt = -(g_eff/Cm) * (V - E_eff)
        # Y como self.dt_over_Cm = dt/Cm → (g_eff/Cm) = g_eff * (dt_over_Cm/dt)
        self.dYdt[:] = g_eff * (E_eff - self.V) * (self.dt_over_Cm / self.dt)

        exp_term = torch.exp(-g_eff * self.dt_over_Cm)
        # V = V * exp + E_eff * (1 - exp)
        self.V.mul_(exp_term).add_(E_eff * (1.0 - exp_term))

    def _integrate_step(self) -> None:
        """Paso completo de integración de canales + membrana."""
        self._update_channels()
        self._integrate_membrane()

    def _write_spikes_to_buffer(self, t_idx: int) -> None:
        self._spike_buffer.index_copy_(1, t_idx, self.spikes.unsqueeze(1))

    def _process(self) -> None:
        """Solo despacha el _process del NeuronGroup."""
        super()._process()
        # Cada subclase decide cuándo llamar a _integrate_step() y cómo generar spikes.


class IFDeterministicBase(ConductanceNeuronBase):
    """Base para neuronas IF deterministas con:
    - threshold por neurona,
    - refractario duro,
    - reset a E_rest,
    - hooks de adaptación (ALIF, etc.).
    """

    threshold: torch.Tensor          # (n_neurons,)
    threshold_base: torch.Tensor | None
    refrac_counter: torch.Tensor
    refrac_steps: int

    def __init__(
        self,
        n_neurons: int,
        spatial_dimensions: int,
        delay_max: int,
        n_channels: int,
        channel_time_constants: list[tuple[float, float]],
        channel_reversal_potentials: list[float],
        tau_membrane: float,
        E_rest: float,
        threshold: float,
        tau_refrac: float,
        dt: float = 1e-3,
        device: str | torch.device | None = None,
        **kwargs
    ):
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

        # Threshold por neurona (aunque inicialmente sea constante)
        self.threshold = torch.full(
            (n_neurons,),
            float(threshold),
            dtype=torch.float32,
            device=self.device,
        )
        self.threshold_base = None  # las subclases (ALIF) pueden usarlo

        # Refractario
        self.refrac_steps = int(tau_refrac / self.dt)
        self.refrac_counter = torch.zeros(
            n_neurons,
            dtype=torch.int32,
            device=self.device,
        )

        self._V_reset_buffer = torch.empty_like(self.V)

    # Hooks de adaptación

    def _update_adaptation_state(self) -> None:
        """Por defecto no hay adaptación."""
        pass

    def _update_threshold(self) -> None:
        """Por defecto threshold fijo."""
        pass

    def _on_spikes(self, spike_mask: torch.Tensor) -> None:
        """Hook tras spike (ALIF lo usa para incrementar adaptación)."""
        pass

    def _process(self) -> None:
        super()._process()
        t_idx = globals.simulator.local_circuit.current_step % self.delay_max_int

        # 1) Integración canales + membrana
        self._integrate_step()

        # 2) Adaptación + threshold
        self._update_adaptation_state()
        self._update_threshold()

        # 3) Refractario + detección de spikes
        self.refrac_counter.sub_(1).clamp_(min=0)
        not_refractory = self.refrac_counter.eq(0)

        self.spikes[:] = self.V >= self.threshold
        self.spikes.logical_and_(not_refractory)
        self.spikes.logical_or_(self._input_spikes)

        self.refrac_counter.masked_fill_(self.spikes, self.refrac_steps)

        # 4) Buffer de spikes
        self._write_spikes_to_buffer(t_idx)

        # 5) Reset + hook
        spk = self.spikes
        if spk.any():
            self.V[spk] = self.E_rest[spk]
            self._on_spikes(spk)

        self._input_spikes.fill_(False)


class ContinuousNoResetBase(ConductanceNeuronBase):
    """Base para neuronas sin reset ni refractario.

    - Integran membrana continuamente (conductance-based).
    - Cada subclase define cómo generar self.spikes.
    - No se altera V tras los spikes (salvo que la subclase lo haga explícitamente).
    """

    def _generate_spikes(self, t_idx: int) -> None:
        """Debe escribir self.spikes y llamar a _write_spikes_to_buffer(t_idx)."""
        raise NotImplementedError

    def _process(self) -> None:
        super()._process()
        t_idx = globals.simulator.local_circuit.current_step % self.delay_max_int

        # Integración multicanal + membrana
        self._integrate_step()

        # Generación de spikes según el modelo concreto
        self._generate_spikes(t_idx)

        self._input_spikes.fill_(False)


class LIFNeurons(IFDeterministicBase):
    """LIF clásico multicanal con integración analítica y refractario duro."""

    def __init__(
        self,
        n_neurons: int,
        spatial_dimensions: int = 2,
        delay_max: int = 20,
        n_channels: int = 3,
        channel_time_constants: list[tuple[float, float]] = (
            (0.001, 0.005),  # AMPA
            (0.001, 0.010),  # GABA
            (0.002, 0.100),  # NMDA
        ),
        channel_reversal_potentials: list[float] = (
            0.0,     # AMPA
            -0.070,  # GABA
            0.0,     # NMDA
        ),
        threshold: float = -0.050,
        tau_membrane: float = 0.010,
        E_rest: float = -0.065,
        tau_refrac: float = 0.002,
        dt: float = 1e-3,
        device: str | torch.device | None = None,
        **kwargs,
    ):
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
            **kwargs
        )

    def _update_adaptation_state(self) -> None:
        pass

    def _update_threshold(self) -> None:
        # threshold ya es fijo, no cambiamos nada
        pass

    def _on_spikes(self, spike_mask: torch.Tensor) -> None:
        pass


class ALIFNeurons(IFDeterministicBase):
    """ALIF multicanal:
    - Misma dinámica conductance-based que LIF.
    - Adaptación de umbral: threshold_i = threshold_base + beta_i * A_i.
    """

    A: torch.Tensor
    tau_adapt: torch.Tensor
    beta: torch.Tensor
    alpha_adapt: torch.Tensor

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
        threshold: float = -0.050,
        tau_membrane: float = 0.010,
        E_rest: float = -0.065,
        tau_refrac: float = 0.002,
        tau_adapt=200e-3,  # float, tensor (n,) o RandomDistribution
        beta=1.7e-3,        # float, tensor (n,) o RandomDistribution
        dt: float = 1e-3,
        device: str | torch.device | None = None,
    ):
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
        )

        # tau_adapt
        if isinstance(tau_adapt, RandomDistribution):
            self.tau_adapt = tau_adapt.sample(n_neurons, self.device).to(
                dtype=torch.float32
            )
        elif isinstance(tau_adapt, torch.Tensor):
            if tau_adapt.shape != (n_neurons,):
                raise ValueError(
                    f"tau_adapt tensor must have shape ({n_neurons},), got {tau_adapt.shape}"
                )
            self.tau_adapt = tau_adapt.to(self.device, dtype=torch.float32)
        else:
            self.tau_adapt = torch.full(
                (n_neurons,),
                float(tau_adapt),
                dtype=torch.float32,
                device=self.device,
            )

        # beta
        if isinstance(beta, RandomDistribution):
            self.beta = beta.sample(n_neurons, self.device).to(dtype=torch.float32)
        elif isinstance(beta, torch.Tensor):
            if beta.shape != (n_neurons,):
                raise ValueError(
                    f"beta tensor must have shape ({n_neurons},), got {beta.shape}"
                )
            self.beta = beta.to(self.device, dtype=torch.float32)
        else:
            self.beta = torch.full(
                (n_neurons,),
                float(beta),
                dtype=torch.float32,
                device=self.device,
            )

        self.alpha_adapt = torch.exp(-self.dt / self.tau_adapt)
        self.A = torch.zeros(n_neurons, dtype=torch.float32, device=self.device)

        # Guardamos threshold_base (vector) y usamos threshold como vector mutable
        self.threshold_base = self.threshold.clone()

    def _update_adaptation_state(self) -> None:
        self.A.mul_(self.alpha_adapt)

    def _update_threshold(self) -> None:
        self.threshold[:] = self.threshold_base + self.beta * self.A

    def _on_spikes(self, spike_mask: torch.Tensor) -> None:
        if spike_mask.any():
            self.A[spike_mask] += 1.0


class StochasticIFNeurons(ContinuousNoResetBase):
    """Neurona IF sin reset con disparo estocástico Bernoulli por paso.

    r(V) = target_rate * σ(beta * (V - threshold))
    p_spike = r(V) * dt
    """

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
        threshold: float = -0.050,
        tau_membrane: float = 0.010,
        E_rest: float = -0.065,
        beta: float = 200.0,
        target_rate: float = 20.0,  # Hz
        dt: float = 1e-3,
        device: str | torch.device | None = None,
    ):
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
        )

        self.beta = float(beta)
        self.target_rate = float(target_rate)

        self.threshold = torch.full(
            (n_neurons,),
            float(threshold),
            dtype=torch.float32,
            device=self.device,
        )

    def _generate_spikes(self, t_idx: int) -> None:
        # r(V) en Hz y p = r * dt
        r = self.target_rate * torch.sigmoid(self.beta * (self.V - self.threshold))
        p = r * self.dt

        self.spikes[:] = torch.rand(self.spikes.shape, device=self.device) < p
        self._write_spikes_to_buffer(t_idx)


class PhaseIFNeurons(ContinuousNoResetBase):
    """Neurona IF sin reset con disparo determinista vía integración de fase.

    r(V) = r_max * σ(beta * (V - theta))
    phi ← phi * phi_decay + r(V) * dt + ruido
    spike si phi >= 1 (se drena la parte entera).
    """

    phase: torch.Tensor
    beta: float
    r: torch.Tensor
    r_max: float
    theta: torch.Tensor
    jitter_std: float
    ahp_drop: float
    phi_decay: float

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
        threshold: float = -0.050,   # referencia base
        tau_membrane: float = 0.010,
        E_rest: float = -0.065,
        beta: float = 200.0,
        r_max: float = 300.0,
        theta: float | None = None,  # si None, usar threshold
        jitter_std: float = 0.0,
        ahp_drop: float = 0.0,
        tau_phi: float = 0.05,
        dt: float = 1e-3,
        device: str | torch.device | None = None,
    ):
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
        )

        self.beta = float(beta)
        self.r_max = float(r_max)
        self.jitter_std = float(jitter_std)
        self.ahp_drop = float(ahp_drop)
        self.phi_decay = float(torch.exp(torch.tensor(-self.dt / tau_phi)))

        # Threshold base para referencia (no se usa directamente en spikes)
        self.threshold = torch.full(
            (n_neurons,),
            float(threshold),
            dtype=torch.float32,
            device=self.device,
        )

        theta_val = threshold if theta is None else theta
        self.theta = torch.full(
            (n_neurons,),
            float(theta_val),
            dtype=torch.float32,
            device=self.device,
        )

        self.phase = torch.zeros_like(self.V)
        self.r = torch.zeros_like(self.V)

    def _generate_spikes(self, t_idx: int) -> None:
        # r(V) en Hz
        self.r = self.r_max * torch.sigmoid(self.beta * (self.V - self.theta))

        # φ ← φ * decay + r * dt
        self.phase.mul_(self.phi_decay).add_(self.r * self.dt)

        # Ruido opcional en la fase (para romper sincronías rígidas)
        if self.jitter_std > 0.0:
            self.phase.add_(
                torch.randn_like(self.phase) * (self.jitter_std * (self.dt ** 0.5))
            )

        # Número de vueltas completas de fase
        self.phase.clamp_min_(0.0)
        n_spikes = self.phase.floor()
        self.spikes[:] = n_spikes >= 1.0

        # Drenar fase entera
        self.phase.sub_(n_spikes)

        # AHP suave opcional
        if self.ahp_drop > 0.0:
            self.V.sub_(self.spikes.float() * self.ahp_drop)

        self._write_spikes_to_buffer(t_idx)