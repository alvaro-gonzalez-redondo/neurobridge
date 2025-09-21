from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .neurons import NeuronGroup
from .group import Group
from .utils import _compute_parameter

from typing import Any

import torch


class ConnectionGroup(Group):
    """Base class for groups of synaptic connections.

    Represents a collection of synaptic connections between pre-synaptic and
    post-synaptic neuron groups, with associated weights, delays, and dynamics.

    Attributes
    ----------
    pre : NeuronGroup
        Pre-synaptic (source) neuron group.
    pos : NeuronGroup
        Post-synaptic (target) neuron group.
    idx_pre : torch.Tensor
        Indices of pre-synaptic neurons for each connection.
    idx_pos : torch.Tensor
        Indices of post-synaptic neurons for each connection.
    weight : torch.Tensor
        Synaptic weights for each connection.
    delay : torch.Tensor
        Synaptic delays in time steps for each connection.
    _current_buffer : torch.Tensor
        Buffer for accumulating post-synaptic currents.
    """

    pre: NeuronGroup
    pos: NeuronGroup
    idx_pre: torch.Tensor
    idx_pos: torch.Tensor
    weight: torch.Tensor
    delay: torch.Tensor
    channel: int
    _current_buffer: torch.Tensor

    def __init__(
        self,
        pre: NeuronGroup,
        pos: NeuronGroup,
    ):
        """Initialize a synaptic connection group.

        Parameters
        ----------
        pre : NeuronGroup
            Pre-synaptic (source) neuron group.
        pos : NeuronGroup
            Post-synaptic (target) neuron group.

        Raises
        ------
        RuntimeError
            If pre-synaptic and post-synaptic groups are on different devices.
        """
        if pre.device != pos.device:
            raise RuntimeError("Connected populations must be from the same device.")
        self.pre = pre
        self.pos = pos

    def _establish_connection(self, pattern: str, **kwargs: Any):
        """Establish connections according to the specified pattern.
        
        This can be implemented by subclasses to define specific connection methods.
        """
        source_indices, target_indices = None, None

        if pattern == "all-to-all":
            source_indices, target_indices = self._connect_all_to_all(**kwargs)

        elif pattern == "specific":
            source_indices, target_indices = self._connect_specific(**kwargs)

        elif pattern == "one-to-one":
            source_indices, target_indices = self._connect_one_to_one(**kwargs)

        else:
            raise NotImplementedError(
                f"Connection pattern '{pattern}' is not implemented."
            )

        # Shared parameters for all synapses
        weight = _compute_parameter(
            kwargs.get("weight", 0.0), source_indices, target_indices, self.pre.device
        )
        delay = _compute_parameter(
            kwargs.get("delay", 0), source_indices, target_indices, self.pre.device
        )
        channel = kwargs.get("channel", 0)

        assert torch.all(
            delay < self.pre.delay_max
        ), f"Connection delay ({torch.max(delay)}) must be less than the `delay_max` parameter of the presynaptic population ({self.pre.delay_max})."

        # Check final connections
        if source_indices.numel() != target_indices.numel():
            raise RuntimeError(
                f"The number of sources ({source_indices.numel()}) and targets ({target_indices.numel()}) do not match."
            )
        size = source_indices.numel()

        super().__init__(size=size, device=self.pre.device)

        self.idx_pre = source_indices
        self.idx_pos = target_indices
        self.weight = weight.to(device=self.pre.device, dtype=torch.float32)
        self.delay = delay.to(device=self.pre.device, dtype=torch.long)
        self.channel = channel

        self._current_buffer = torch.zeros(
            self.pos.size, dtype=torch.float32, device=self.pre.device
        )
    
    def _init_connection(self, **kwargs):
        raise NotImplementedError(f"Connection classes must implement their own initialization code.")

    def _connect_all_to_all(self, **kwargs):
        """Establish all-to-all connections."""
        #raise NotImplementedError(f"{self.__class__.__name__} does not support all-to-all connections.")

        # Filtered subsets
        valid_pre = self.pre.filter.nonzero(as_tuple=True)[0]
        valid_pos = self.pos.filter.nonzero(as_tuple=True)[0]

        grid_pre, grid_pos = torch.meshgrid(valid_pre, valid_pos, indexing="ij")
        source_indices = grid_pre.flatten()
        target_indices = grid_pos.flatten()

        return source_indices, target_indices
    
    def _connect_one_to_one(self, **kwargs):
        """Establish one-to-one connections."""
        #raise NotImplementedError(f"{self.__class__.__name__} does not support one-to-one connections.")

        # Filtered subsets
        valid_pre = self.pre.filter.nonzero(as_tuple=True)[0]
        valid_pos = self.pos.filter.nonzero(as_tuple=True)[0]

        assert valid_pre.numel() == valid_pos.numel()
        source_indices = valid_pre.clone()
        target_indices = valid_pos.clone()

        return source_indices, target_indices
    
    def _connect_specific(self, **kwargs):
        """Establish connections with specific indices."""
        #raise NotImplementedError(f"{self.__class__.__name__} does not support specific connections.")

        try:
            source_indices = _compute_parameter(
                kwargs["idx_pre"], kwargs["idx_pre"], kwargs["idx_pre"], self.pre.device
            )
            target_indices = _compute_parameter(
                kwargs["idx_pos"], kwargs["idx_pos"], kwargs["idx_pos"], self.pre.device
            )
        except KeyError:
            raise RuntimeError(
                "`specific`pattern requires both 'idx_pre' and 'idx_pos' parameters."
            )
        
        return source_indices, target_indices

    def _process(self):
        """Process the synaptic group for the current time step.

        This method propagates spikes from pre-synaptic to post-synaptic neurons
        and updates synaptic weights according to the learning rule.
        """
        super()._process()
        self._propagate()
        self._update()

    def _propagate(self):
        """Propagate spikes from pre-synaptic to post-synaptic neurons.

        Retrieves pre-synaptic spikes with appropriate delays, multiplies by weights,
        and injects the resulting currents into post-synaptic neurons.
        """
        spikes_mask = self.pre.get_spikes_at(self.delay, self.idx_pre)
        mask_f = spikes_mask.to(self.weight.dtype)
        contrib = self.weight * mask_f
        self._current_buffer.zero_()
        self._current_buffer.index_add_(0, self.idx_pos, contrib)
        self.pos.inject_currents(self._current_buffer, self.channel)

    def _update(self) -> None:
        """Update synaptic weights according to the learning rule.

        This method should be implemented by subclasses to define specific
        plasticity mechanisms.

        Raises
        ------
        NotImplementedError
            If not implemented by a subclass.
        """
        raise NotImplementedError(
            "`update` method in `SynapseGroup` must be implemented."
        )


class StaticConnection(ConnectionGroup):
    """Static (non-plastic) synaptic connections.

    A simple synaptic model with fixed weights that do not change over time.
    """

    def _init_connection(self, **kwargs):
        pass

    def _update(self) -> None:
        """Update synaptic weights (no-op for static synapses).

        Static synapses have fixed weights, so this method does nothing.
        """
        pass


class STDPConnection(ConnectionGroup):
    """Spike-Timing-Dependent Plasticity (STDP) synaptic connections.

    Implements STDP, a biologically-inspired learning rule where synaptic
    weights are modified based on the relative timing of pre- and post-synaptic spikes.

    Attributes
    ----------
    A_plus : torch.Tensor
        Learning rate for potentiation (when pre-synaptic spike precedes post-synaptic).
    A_minus : torch.Tensor
        Learning rate for depression (when post-synaptic spike precedes pre-synaptic).
    tau_plus : torch.Tensor
        Time constant for pre-synaptic trace decay.
    tau_minus : torch.Tensor
        Time constant for post-synaptic trace decay.
    w_min : torch.Tensor
        Minimum allowed weight value.
    w_max : torch.Tensor
        Maximum allowed weight value.
    x_pre : torch.Tensor
        Pre-synaptic spike traces for each connection.
    x_pos : torch.Tensor
        Post-synaptic spike traces for each connection.
    alpha_pre : torch.Tensor
        Decay factor for pre-synaptic traces.
    alpha_pos : torch.Tensor
        Decay factor for post-synaptic traces.
    _delay_1 : torch.Tensor
        Constant tensor of ones for accessing post-synaptic spikes.
    """

    A_plus: torch.Tensor
    A_minus: torch.Tensor
    tau_plus: torch.Tensor
    tau_minus: torch.Tensor
    w_min: torch.Tensor
    w_max: torch.Tensor
    x_pre: torch.Tensor
    x_pos: torch.Tensor
    alpha_pre: torch.Tensor
    alpha_pos: torch.Tensor
    _delay_1: torch.Tensor

    def _init_connection(self, **kwargs):
        self.A_plus = torch.tensor(kwargs.get("A_plus", 1e-2), device=self.device)
        self.A_minus = torch.tensor(kwargs.get("A_minus", 1.2e-2), device=self.device)
        self.tau_plus = torch.tensor(kwargs.get("tau_plus", 20e-3), device=self.device)
        self.tau_minus = torch.tensor(kwargs.get("tau_minus", 20e-3), device=self.device)
        self.w_min = torch.tensor(kwargs.get("w_min", 0.0), device=self.device)
        self.w_max = torch.tensor(kwargs.get("w_max", 1.0), device=self.device)

        self.x_pre = torch.zeros(len(self.idx_pre), dtype=torch.float32, device=self.device)
        self.x_pos = torch.zeros(len(self.idx_pos), dtype=torch.float32, device=self.device)

        self.alpha_pre = torch.exp(-1e-3 / self.tau_plus)
        self.alpha_pos = torch.exp(-1e-3 / self.tau_minus)

        self._delay_1 = torch.ones_like(self.idx_pos, device=self.device)
        
    def _update(self) -> None:
        """Update synaptic weights according to the STDP rule.

        Implements the STDP learning rule:
        1. Decay pre- and post-synaptic traces
        2. Update traces based on current spikes
        3. Potentiate weights when pre-synaptic spikes arrive at post-synaptic neurons
        4. Depress weights when post-synaptic neurons spike
        5. Clamp weights to the allowed range
        """
        self.x_pre *= self.alpha_pre
        self.x_pos *= self.alpha_pos

        # Selecting relevant spikes with the right delays
        pre_spikes = self.pre.get_spikes_at(self.delay, self.idx_pre)
        pos_spikes = self.pos.get_spikes_at(self._delay_1, self.idx_pos)

        # Updating traces
        self.x_pre += pre_spikes.to(torch.float32)
        self.x_pos += pos_spikes.to(torch.float32)

        # STDP - pre before post
        dw = self.A_plus * self.x_pre * pos_spikes
        self.weight += dw

        # STDP - post before pre
        dw = self.A_minus * self.x_pos * pre_spikes
        self.weight += dw

        self.weight.clamp_(self.w_min, self.w_max)
