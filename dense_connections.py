from __future__ import annotations

from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from .neurons import NeuronGroup
from . import globals
from .dense import DenseNode
from .utils import _compute_parameter

import torch


class ConnectionDense(DenseNode):
    """Dense connection between two neuron groups.
    
    This class represents a synaptic connection between pre-synaptic and post-synaptic
    neuron groups, handling spike propagation and synaptic weight updates.
    
    Attributes
    ----------
    pre : NeuronGroup
        Pre-synaptic (source) neuron group.
    pos : NeuronGroup
        Post-synaptic (target) neuron group.
    mask : torch.Tensor
        Boolean mask indicating existing connections between neurons.
    weight : torch.Tensor
        Synaptic weights for each connection.
    delay : Int
        Synaptic delays for all connection, in time steps.
    """
    pre: NeuronGroup
    pos: NeuronGroup
    mask: torch.Tensor
    weight: torch.Tensor
    delay: torch.Tensor
    channel: int

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
        
        Parameters
        ----------
        pattern : str
            Connection pattern type. Supported patterns are:
            - "all-to-all": Connect each pre-synaptic neuron to all post-synaptic neurons.
            - "one-to-one": Connect pre-synaptic neurons to post-synaptic neurons in a one-to-one fashion.
            - "specific": Connect using a specific connection mask provided in kwargs.
        
        **kwargs : Any
            Additional parameters for connection establishment:
            - weight: Tensor of synaptic weights with shape (pre.size, pos.size).
            - delay: Scalar for synaptic delay
            - mask: For "specific" pattern, a boolean mask indicating connections.
            
        Raises
        ------
        NotImplementedError
            If the requested connection pattern is not implemented.
        ValueError
            If weight or delay shapes don't match the connection shape.
        """
        shape = (self.pre.size, self.pos.size)
        super().__init__(shape, self.pre.device)

        if pattern == "all-to-all":
            self.mask = self._connect_all_to_all(**kwargs)

        elif pattern == "specific":
            self.mask = self._connect_specific(**kwargs)

        elif pattern == "one-to-one":
            self.mask = self._connect_one_to_one(**kwargs)

        else:
            raise NotImplementedError(
                f"Connection pattern '{pattern}' is not implemented."
            )
        if self.mask is None:
            raise RuntimeError("Connection mask was not set by the selected connection pattern.")

        # Generate dense indices
        idx_pre, idx_pos = torch.meshgrid(
            torch.arange(self.pre.size, device=self.pre.device),
            torch.arange(self.pos.size, device=self.pre.device),
            indexing="ij"
        )
        idx_pre = idx_pre.flatten()
        idx_pos = idx_pos.flatten()

        # Shared parameters for all synapses
        self.weight = _compute_parameter(
            kwargs.get("weight", 0.0), idx_pre, idx_pos, self.pre.device
        ).to(dtype=torch.float32).view(self.pre.size, self.pos.size)
        
        self.delay = kwargs.get("delay", 0)

        self.channel = kwargs.get("channel", 0)

        assert torch.all(
            self.delay < self.pre.delay_max
        ), f"Connection delay ({torch.max(self.delay)}) must be less than the `delay_max` parameter of the presynaptic population ({self.pre.delay_max})."

        assert self.channel < self.pos.n_channels, f"Channel {self.channel} does not exist in post-synaptic neuron with {self.pos.n_channels} channels."

    def _init_connection(self, **kwargs):
        pass

    def _connect_all_to_all(self, **kwargs):
        """Establish all-to-all connections.
        
        Creates connections from each pre-synaptic neuron to every post-synaptic neuron,
        respecting the filter masks of both neuron groups.
        
        Returns
        -------
        torch.Tensor
            Boolean mask indicating connections between neurons.
        """
        mask = torch.ones(self.shape, device=self.device)
        mask[~self.pre.filter, :] = False
        mask[:, ~self.pos.filter] = False

        return mask
    
    def _connect_one_to_one(self, **kwargs):
        """Establish one-to-one connections.
        
        Creates diagonal connections where pre-synaptic neuron i connects only 
        to post-synaptic neuron i, respecting the filter masks of both neuron groups.
        
        Returns
        -------
        torch.Tensor
            Boolean mask indicating connections between neurons.
        """
        mask = torch.zeros(self.shape, device=self.device)
        js = torch.arange(min(*self.shape))
        mask[js,js] = True
        mask[~self.pre.filter, :] = False
        mask[:, ~self.pos.filter] = False

        return mask
    
    def _connect_specific(self, **kwargs):
        """Establish connections with specific indices.
        
        Uses a provided mask to determine which connections should exist.
        
        Parameters
        ----------
        **kwargs : Any
            Must contain a "mask" key with a boolean tensor indicating desired connections.
            
        Returns
        -------
        torch.Tensor
            Boolean mask indicating connections between neurons.
            
        Raises
        ------
        AssertionError
            If the provided mask shape doesn't match the connection shape.
        """
        mask = kwargs["mask"].to(torch.bool)
        assert mask.shape == self.shape, "Mask shape must be the same as the connection shape."

        return mask

    def _process(self):
        """Process the synaptic dense connections for the current time step.

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
        t_indices = (globals.engine.local_circuit.t - self.delay) % self.pre.delay_max
        spikes_mask = self.pre._spike_buffer[:, t_indices]
        mask_f = spikes_mask.to(self.weight.dtype).squeeze_()
        contrib = torch.matmul(mask_f, self.weight*self.mask)
        self.pos.inject_currents(contrib, self.channel)

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

class StaticDenseConnection(ConnectionDense):
    """Static (non-plastic) synaptic connections.

    A simple synaptic model with fixed weights that do not change over time.
    """

    def _update(self) -> None:
        """Update synaptic weights (no-op for static synapses).

        Static synapses have fixed weights, so this method does nothing.
        """
        pass


class STDPDenseConnection(ConnectionDense):
    """Spike-Timing-Dependent Plasticity (STDP) synaptic connections for dense networks.

    Implements STDP, a biologically-inspired learning rule where synaptic
    weights are modified based on the relative timing of pre- and post-synaptic spikes.
    This version is optimized for dense connectivity patterns using matrix operations.

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
        Pre-synaptic spike traces for each pre-synaptic neuron.
    x_pos : torch.Tensor
        Post-synaptic spike traces for each post-synaptic neuron.
    alpha_pre : torch.Tensor
        Decay factor for pre-synaptic traces.
    alpha_pos : torch.Tensor
        Decay factor for post-synaptic traces.
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

    def __init__(
        self,
        pre: NeuronGroup,
        pos: NeuronGroup,
    ):
        """Initialize a STDP connection group.

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
        super().__init__(pre, pos)
        
    def _establish_connection(self, pattern: str, **kwargs: Any):
        """Establish connections according to the specified pattern and initialize STDP parameters.
        
        Parameters
        ----------
        pattern : str
            Connection pattern type. Supported patterns are:
            - "all-to-all": Connect each pre-synaptic neuron to all post-synaptic neurons.
            - "one-to-one": Connect pre-synaptic neurons to post-synaptic neurons in a one-to-one fashion.
            - "specific": Connect using a specific connection mask provided in kwargs.
        
        **kwargs : Any
            Additional parameters for connection establishment and STDP:
            - weight: Tensor of synaptic weights with shape (pre.size, pos.size).
            - delay: Tensor of synaptic delays with shape (pre.size, pos.size).
            - mask: For "specific" pattern, a boolean mask indicating connections.
            - A_plus: Learning rate for potentiation (default: 0.01).
            - A_minus: Learning rate for depression (default: 0.012).
            - tau_plus: Time constant for pre-synaptic trace decay (default: 20.0).
            - tau_minus: Time constant for post-synaptic trace decay (default: 20.0).
            - w_min: Minimum allowed weight value (default: 0.0).
            - w_max: Maximum allowed weight value (default: 1.0).
            
        Raises
        ------
        NotImplementedError
            If the requested connection pattern is not implemented.
        ValueError
            If weight or delay shapes don't match the connection shape.
        """
        # First establish basic connections using parent method
        super()._establish_connection(pattern, **kwargs)
        
    
    def _init_connection(self, **kwargs: Any):
        """Initialize STDP-specific parameters.
        
        Parameters
        ----------
        **kwargs : Any
            Parameters for STDP initialization:
            - A_plus: Learning rate for potentiation (default: 1e-2).
            - A_minus: Learning rate for depression (default: 1.2e-2).
            - tau_plus: Time constant for pre-synaptic trace decay (default: 20e-3).
            - tau_minus: Time constant for post-synaptic trace decay (default: 20e-3).
            - w_min: Minimum allowed weight value (default: 0.0).
            - w_max: Maximum allowed weight value (default: 1.0).
        """
        device = self.device
        
        # Initialize STDP parameters
        self.A_plus = torch.tensor(kwargs.get("A_plus", 1e-2), device=device)
        self.A_minus = torch.tensor(kwargs.get("A_minus", 1.2e-2), device=device)
        self.tau_plus = torch.tensor(kwargs.get("tau_plus", 20e-3), device=device)
        self.tau_minus = torch.tensor(kwargs.get("tau_minus", 20e-3), device=device)
        self.w_min = torch.tensor(kwargs.get("w_min", 0.0), device=device)
        self.w_max = torch.tensor(kwargs.get("w_max", 1.0), device=device)

        # Initialize traces for pre- and post-synaptic neurons
        self.x_pre = torch.zeros(self.pre.size, dtype=torch.float32, device=device)
        self.x_pos = torch.zeros(self.pos.size, dtype=torch.float32, device=device)

        # Compute decay factors
        self.alpha_pre = torch.exp(-1e-3 / self.tau_plus)
        self.alpha_pos = torch.exp(-1e-3 / self.tau_minus)
        
    def _update(self) -> None:
        """Update synaptic weights according to the STDP rule.

        Implements the STDP learning rule:
        1. Decay pre- and post-synaptic traces
        2. Update traces based on current spikes
        3. Potentiate weights when pre-synaptic spikes arrive at post-synaptic neurons
        4. Depress weights when post-synaptic neurons spike
        5. Clamp weights to the allowed range
        """
        # Decay traces
        self.x_pre *= self.alpha_pre
        self.x_pos *= self.alpha_pos

        # Get current spikes with appropriate delays
        # For pre-synaptic neurons, we consider delayed spikes
        t_indices_pre = (globals.engine.local_circuit.t - self.delay) % self.pre.delay_max
        pre_spikes = self.pre._spike_buffer[:, t_indices_pre].squeeze_()
        
        # For post-synaptic neurons, we consider current spikes (no delay)
        pos_spikes = self.pos.get_spikes()
        
        # Update traces
        self.x_pre += pre_spikes.float()
        self.x_pos += pos_spikes.float()
        
        # Create weight update matrix
        # Pre -> Post potentiation: outer product of pre spikes with post traces
        potentiation = torch.outer(self.x_pre, pos_spikes.float()) * self.A_plus
        
        # Post -> Pre depression: outer product of pre traces with post spikes
        depression = torch.outer(pre_spikes.float(), self.x_pos) * self.A_minus
        
        # Combine potentiation and depression
        weight_update = potentiation + depression
        
        # Only apply updates to existing connections (using the mask)
        weight_update = weight_update * self.mask
        
        # Update weights
        self.weight += weight_update
        
        # Clamp weights to allowed range
        self.weight.clamp_(self.w_min, self.w_max)