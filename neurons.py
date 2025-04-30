from __future__ import annotations

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
        Float tensor of shape (n_neurons,) for accumulating input currents.
    _input_spikes : torch.Tensor
        Boolean tensor of shape (n_neurons,) for injected spikes.
    """

    delay_max: torch.Tensor
    _spike_buffer: torch.Tensor
    _input_currents: torch.Tensor
    _input_spikes: torch.Tensor

    def __init__(
        self,
        device: str,
        n_neurons: int,
        spatial_dimensions: int = 2,
        delay_max: int = 20,
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
        super().__init__(device, n_neurons, spatial_dimensions)
        self.delay_max = torch.tensor([delay_max], dtype=torch.int, device=self.device)
        self._spike_buffer = torch.zeros(
            (n_neurons, delay_max), dtype=torch.bool, device=self.device
        )
        self._input_currents = torch.zeros(
            n_neurons, dtype=torch.float32, device=self.device
        )
        self._input_spikes = torch.zeros(
            n_neurons, dtype=torch.bool, device=self.device
        )

    def get_spike_buffer(self):
        """Get the internal spike buffer.

        Returns
        -------
        torch.Tensor
            Boolean tensor of shape (n_neurons, delay_max) containing spike history.
        """
        return self._spike_buffer

    def inject_currents(self, I: torch.Tensor) -> None:
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
        self._input_currents += I

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
        t_indices = (globals.engine.local_circuit.t - 1) % self.delay_max
        return self._spike_buffer[:, t_indices].squeeze_(1)
    
    def get_spikes_at(
        self, delays: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        """Get the spikes for specific neurons at specific delays.

        Parameters
        ----------
        delays : torch.Tensor
            Integer tensor of shape (M,) with delay values.
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
            If delays and indices don't have the same shape.

        Notes
        -----
        This method is used primarily by synaptic connections to retrieve
        pre-synaptic spikes with appropriate delays.
        """
        assert delays.shape == indices.shape, "Delays and indices must match in shape"

        t_indices = (globals.engine.local_circuit.t - delays) % self.delay_max
        return self._spike_buffer[indices, t_indices]

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
        t_idx = globals.engine.local_circuit.t % self.delay_max
        self._spike_buffer.index_fill_(1, t_idx, 0)

        # Process any injected spikes
        # Store spikes in the buffer at current t
        self._spike_buffer.index_copy_(
            1,
            t_idx,
            (
                self._spike_buffer.index_select(1, t_idx)
                | self._input_spikes.unsqueeze(1)
            ),
        )
        # Clear injected spikes
        self._input_spikes.fill_(False)

        # Process input currents
        # Generate spikes for neurons receiving any positive current
        spikes = self._input_currents > 0
        self._spike_buffer.index_copy_(
            1, t_idx, (self._spike_buffer.index_select(1, t_idx) | spikes.unsqueeze(1))
        )
        # Clear input currents
        self._input_currents.fill_(0.0)


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
        device: str,
        n_neurons: int,
        spatial_dimensions: int = 2,
        delay_max: int = 20,
        threshold: float = 1.0,
        tau_membrane: float = 0.1,
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
        super().__init__(device, n_neurons, spatial_dimensions, delay_max)
        self.V = torch.zeros(n_neurons, dtype=torch.float32, device=self.device)
        self.threshold = torch.tensor([threshold], dtype=torch.float32, device=device)
        self.decay = torch.exp(
            torch.tensor(-1e-3 / tau_membrane, dtype=torch.float32, device=device)
        )

    def _process(self):
        """Update membrane potentials and generate spikes.

        For each neuron, updates the membrane potential with decay and input current,
        checks if the threshold is reached, and generates spikes accordingly.
        After spiking, the membrane potential is reset to zero.
        """
        super()._process()
        t_idx = globals.engine.local_circuit.t % self.delay_max

        # Update potential with decay and input
        self.V *= self.decay
        self.V += self._input_currents
        self._input_currents.fill_(0.0)

        # Determine which neurons spike
        spikes = (self.V >= self.threshold) | self._input_spikes
        self._spike_buffer.index_copy_(1, t_idx, spikes.unsqueeze(1))
        self.V[spikes] = 0.0  # Reset membrane potential
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
        device: str,
        n_neurons: int,
        firing_rate: float = 10.0,
        spatial_dimensions: int = 2,
        delay_max: int = 20,
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
            device,
            n_neurons,
            spatial_dimensions=spatial_dimensions,
            delay_max=delay_max,
        )
        self.firing_rate = torch.tensor(
            firing_rate * 1e-3, dtype=torch.float32, device=device
        )
        self.probabilities = torch.zeros(n_neurons, dtype=torch.float32, device=device)

    def _process(self):
        """Generate random spikes based on the firing rate.

        Each neuron has a probability of firing equal to the firing rate
        times the time step (in milliseconds).
        """
        super()._process()
        t_idx = globals.engine.local_circuit.t % self.delay_max

        self.probabilities.uniform_()
        spikes = self.probabilities < self.firing_rate
        self._spike_buffer.index_copy_(1, t_idx, spikes.unsqueeze(1))


class IFNeurons(NeuronGroup):
    """Neuronas Integrate-and-Fire multicanal con potenciales de reversión y bi-exponenciales."""

    V: torch.Tensor
    threshold: torch.Tensor
    decay: torch.Tensor
    E_rest: torch.Tensor
    E_channels: torch.Tensor
    channel_states: torch.Tensor  # (n_channels, n_neurons, 2)
    channel_decay_factors: torch.Tensor  # (n_channels, 2)
    channel_normalization: torch.Tensor  # (n_channels,)
    input_channels: torch.Tensor  # (n_channels, n_neurons)

    def __init__(
        self,
        device: str,
        n_neurons: int,
        n_channels: int = 3,
        channel_time_constants: list[tuple[float, float]] = (
            (0.001, 0.005),  # AMPA: subida 1ms, caída 5ms
            (0.001, 0.010),  # GABA: subida 1ms, caída 10ms
            (0.002, 0.100),  # NMDA: subida 2ms, caída 100ms
        ),
        channel_reversal_potentials: list[float] = (
            0.0,    # AMPA: 0 mV
            -0.070,  # GABA: -70 mV
            0.0,    # NMDA: 0 mV
        ),
        spatial_dimensions: int = 2,
        delay_max: int = 20,
        threshold: float = -0.050,  # -50 mV
        tau_membrane: float = 0.010,  # 10 ms
        E_rest: float = -0.065,  # -65 mV
    ):
        super().__init__(device, n_neurons, spatial_dimensions, delay_max)

        assert len(channel_time_constants) == n_channels
        assert len(channel_reversal_potentials) == n_channels

        self.V = torch.full((n_neurons,), E_rest, dtype=torch.float32, device=self.device)
        self._V_reset_buffer = torch.empty_like(self.V)
        self.threshold = torch.tensor([threshold], dtype=torch.float32, device=self.device)
        self.decay = torch.exp(torch.tensor(-1e-3 / tau_membrane, dtype=torch.float32, device=self.device))

        tau_rise = torch.tensor([tc[0] for tc in channel_time_constants], dtype=torch.float32, device=self.device)
        tau_decay = torch.tensor([tc[1] for tc in channel_time_constants], dtype=torch.float32, device=self.device)

        self.channel_decay_factors = torch.stack([
            torch.exp(-1e-3 / tau_rise),
            torch.exp(-1e-3 / tau_decay)
        ], dim=1)  # (n_channels, 2)

        self.channel_normalization = (tau_decay - tau_rise) / (tau_decay * tau_rise)

        self.channel_states = torch.zeros(n_channels, n_neurons, 2, dtype=torch.float32, device=self.device)
        self.input_channels = torch.zeros(n_channels, n_neurons, dtype=torch.float32, device=self.device)

        self.E_rest = torch.tensor([E_rest], dtype=torch.float32, device=self.device)
        self.E_channels = torch.tensor(channel_reversal_potentials, dtype=torch.float32, device=self.device)

    def inject_currents(self, I: torch.Tensor, chn: int=0) -> None:
        """Inyecta corriente en un único canal, normalizada por el área."""
        assert I.shape[0] == self.size
        self.input_channels[chn, :].add_(I * self.channel_normalization[chn])

    def _process(self) -> None:
        """Actualiza los estados internos, integra la dinámica y genera spikes."""
        super()._process()
        t_idx = globals.engine.local_circuit.t % self.delay_max

        # Actualización de bi-exponenciales
        self.channel_states.mul_(self.channel_decay_factors.unsqueeze(1)).add_(
            self.input_channels.unsqueeze(-1)
        )
        self.input_channels.zero_()

        # Corrientes inducidas por canales considerando el potencial de reversión
        channel_drive = self.E_channels.unsqueeze(1) - self.V.unsqueeze(0)  # (n_channels, n_neurons)
        channel_currents = (self.channel_states[:, :, 1] - self.channel_states[:, :, 0]) * channel_drive
        total_current = channel_currents.sum(dim=0)

        # Decaimiento hacia E_rest + integración de corrientes
        self.V.mul_(self.decay).add_(self.E_rest * (1.0 - self.decay)).add_(total_current)

        # Generación de spikes
        spikes = (self.V >= self.threshold) | self._input_spikes
        self._spike_buffer.index_copy_(1, t_idx, spikes.unsqueeze(1))
        torch.where(spikes, self.E_rest.expand_as(self.V), self.V, out=self._V_reset_buffer)
        self.V.copy_(self._V_reset_buffer)
        self._input_spikes.fill_(False)
