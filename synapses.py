from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .neurons import NeuronGroup
from .groups import _Group

from typing import Optional, Any, Type, Union

import torch


class _ConnectionOperator:
    """Handles the creation of synaptic connections between neuron groups.

    This class is created when using the >> operator between neuron groups
    and provides methods to specify connection parameters and patterns.

    Attributes
    ----------
    pre : NeuronGroup
        The pre-synaptic (source) neuron group.
    pos : NeuronGroup
        The post-synaptic (target) neuron group.
    device : torch.device
        The GPU device shared by both neuron groups.
    pattern : Optional[str]
        The connection pattern to use (e.g., 'all-to-all', 'one-to-one').
    kwargs : dict
        Additional connection parameters.
    """

    pre: NeuronGroup
    pos: NeuronGroup
    pattern: Optional[str]
    kwargs: dict[str, Any]

    def __init__(self, pre: NeuronGroup, pos: NeuronGroup) -> None:
        """Initialize a connection operator between two neuron groups.

        Parameters
        ----------
        pre : NeuronGroup
            The pre-synaptic (source) neuron group.
        pos : NeuronGroup
            The post-synaptic (target) neuron group.

        Raises
        ------
        RuntimeError
            If the pre-synaptic and post-synaptic groups are on different devices.
        """
        if pre.device != pos.device:
            raise RuntimeError(
                "It is not possible to directly connect two populations in different GPUs."
            )
        self.pre = pre
        self.pos = pos
        self.device = self.pre.device
        self.pattern = None
        self.kwargs = {}

    def __call__(
        self,
        pattern: str = "all-to-all",
        synapse_class: Optional[Type[SynapticGroup]] = None,
        **kwargs: Any,
    ) -> SynapticGroup:
        """Create a synaptic connection between the neuron groups.

        Parameters
        ----------
        pattern : str, optional
            The connection pattern to use, by default 'all-to-all'.
            Supported patterns:
                - 'all-to-all': Connect every pre-synaptic neuron to every
                  post-synaptic neuron.
                - 'one-to-one': Connect pre-synaptic neurons to post-synaptic
                  neurons one-to-one (requires equal number of neurons).
                - 'specific': Connect using provided indices (requires 'idx_pre'
                  and 'idx_pos' in kwargs).
        synapse_class : Optional[Type[SynapticGroup]], optional
            The class to use for the synaptic connections, by default None.
            If None, StaticSynapse is used.
        **kwargs : Any
            Additional connection parameters, including:
                - weight: Synaptic weights (scalar, tensor, or function).
                - delay: Synaptic delays in time steps (scalar, tensor, or function).
                - Additional parameters specific to the synapse class.

        Returns
        -------
        SynapticGroup
            The created synaptic connection group.

        Raises
        ------
        NotImplementedError
            If an unsupported connection pattern is specified.
        RuntimeError
            If required parameters for a specific pattern are missing.

        Notes
        -----
        After the connection is created, the filters of both pre-synaptic and
        post-synaptic groups are reset.

        Examples
        --------
        >>> # All-to-all connection with fixed weight and delay
        >>> (pre_group >> post_group)(pattern='all-to-all', weight=0.1, delay=1)
        >>>
        >>> # One-to-one connection with plastic synapses
        >>> (pre_group >> post_group)(
        ...     pattern='one-to-one',
        ...     synapse_class=STDPSynapse,
        ...     weight=0.5,
        ...     A_plus=0.01
        ... )
        """

        self.pattern = pattern
        self.kwargs = kwargs

        # Generar subconjuntos filtrados (o completos)
        valid_pre = self.pre.filter.nonzero(as_tuple=True)[0]
        valid_pos = self.pos.filter.nonzero(as_tuple=True)[0]

        if pattern == "all-to-all":
            grid_pre, grid_pos = torch.meshgrid(valid_pre, valid_pos, indexing="ij")
            source_indices = grid_pre.flatten()
            target_indices = grid_pos.flatten()

        elif pattern == "specific":
            try:
                source_indices = self._compute_parameter(
                    kwargs["idx_pre"], kwargs["idx_pre"], kwargs["idx_pre"]
                )
                target_indices = self._compute_parameter(
                    kwargs["idx_pos"], kwargs["idx_pos"], kwargs["idx_pos"]
                )
            except KeyError:
                raise RuntimeError(
                    "Faltan 'idx_pre' o 'idx_pos' en los parámetros para el patrón 'specific'."
                )

        elif pattern == "one-to-one":
            assert valid_pre.numel() == valid_pos.numel()
            source_indices = valid_pre.clone()
            target_indices = valid_pos.clone()

        else:
            raise NotImplementedError(
                f"Patrón de conexión '{pattern}' no implementado."
            )

        # Parámetros comunes a todas las sinapsis
        delay = self._compute_parameter(
            kwargs.get("delay", 0), source_indices, target_indices
        ).to(torch.long)
        weight = self._compute_parameter(
            kwargs.get("weight", 0.0), source_indices, target_indices
        )

        assert torch.all(
            delay < self.pre.delay_max
        ), f"Connection delay ({torch.max(delay)}) must be less than the `delay_max` parameter of the presynaptic population ({self.pre.delay_max})."

        # Crear objeto sináptico
        if synapse_class is None or synapse_class is StaticSynapse:
            connection = StaticSynapse(
                pre=self.pre,
                pos=self.pos,
                idx_pre=source_indices,
                idx_pos=target_indices,
                delay=delay,
                weight=weight,
            )

        elif synapse_class is STDPSynapse:
            connection = STDPSynapse(
                pre=self.pre,
                pos=self.pos,
                idx_pre=source_indices,
                idx_pos=target_indices,
                delay=delay,
                weight=weight,
                A_plus=kwargs.get("A_plus", 0.01),
                A_minus=kwargs.get("A_minus", 0.012),
                tau_plus=kwargs.get("tau_plus", 20.0),
                tau_minus=kwargs.get("tau_minus", 20.0),
                dt=kwargs.get("dt", 1.0),
                w_min=kwargs.get("w_min", 0.0),
                w_max=kwargs.get("w_max", 1.0),
            )

        else:
            raise NotImplementedError(
                f"Clase de sinapsis '{synapse_class}' no soportada."
            )

        # Limpiar filtros tras conectar
        self.pre.reset_filter()
        self.pos.reset_filter()

        return connection

    def _compute_parameter(
        self, param: Any, idx_pre: torch.Tensor, idx_post: torch.Tensor
    ) -> torch.Tensor:
        """Compute per-connection parameter values based on various input types.

        Parameters
        ----------
        param : Any
            The parameter specification, which can be:
                - A scalar: Used for all connections.
                - A tensor: Used directly if it matches the number of connections.
                - A list: Converted to a tensor.
                - A function: Called with (idx_pre, idx_post) to compute values.
        idx_pre : torch.Tensor
            Indices of pre-synaptic neurons for each connection.
        idx_post : torch.Tensor
            Indices of post-synaptic neurons for each connection.

        Returns
        -------
        torch.Tensor
            Tensor of parameter values for each connection.

        Raises
        ------
        ValueError
            If the tensor dimensions don't match the number of connections.
        TypeError
            If a function parameter doesn't return a tensor.
        """
        n = len(idx_pre)

        if callable(param):
            values = param(idx_pre, idx_post)
            if not isinstance(values, torch.Tensor):
                raise TypeError("Las funciones deben devolver un tensor.")
            if values.shape[0] != n:
                raise ValueError(
                    f"El tensor devuelto debe tener tamaño {n}, pero tiene {values.shape[0]}."
                )
            return values.to(device=self.device)

        elif isinstance(param, torch.Tensor):
            if param.numel() == 1:
                return torch.full((n,), param.item(), device=self.device)
            if param.numel() != n:
                raise ValueError(
                    f"Expected a tensor of length {n}, got {param.numel()}."
                )
            return param.to(device=self.device)

        elif isinstance(param, list):
            param = torch.tensor(param, device=self.device)
            return self._compute_parameter(param, idx_pre, idx_post)

        else:  # Escalar
            return torch.full((n,), float(param), device=self.device)


class SynapticGroup(_Group):
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
    _current_buffer: torch.Tensor

    def __init__(
        self,
        pre: NeuronGroup,
        pos: NeuronGroup,
        idx_pre: torch.Tensor,
        idx_pos: torch.Tensor,
        weight: torch.Tensor,
        delay: torch.Tensor,
    ):
        """Initialize a synaptic connection group.

        Parameters
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

        Raises
        ------
        RuntimeError
            If pre-synaptic and post-synaptic groups are on different devices.
            If the number of source and target indices don't match.
        """
        if pre.device != pos.device:
            raise RuntimeError("Connected populations must be from the same device.")
        device = pre.device

        if idx_pre.numel() != idx_pos.numel():
            raise RuntimeError(
                f"The number of sources ({idx_pre.numel()}) and targets ({idx_pos.numel()}) do not match."
            )
        size = idx_pre.numel()

        super().__init__(device, size)

        self.pre = pre
        self.pos = pos
        self.idx_pre = idx_pre
        self.idx_pos = idx_pos
        self.weight = weight.to(device=pre.device, dtype=torch.float32)
        self.delay = delay

        self._current_buffer = torch.zeros(
            self.pos.size, dtype=torch.float32, device=self.device
        )

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
        self.pos.inject_currents(self._current_buffer)

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
            "`update` method in `SynapticGroup` must be implemented."
        )


class StaticSynapse(SynapticGroup):
    """Static (non-plastic) synaptic connections.

    A simple synaptic model with fixed weights that do not change over time.
    """

    def __init__(self, pre, pos, idx_pre, idx_pos, weight, delay):
        """Initialize static synaptic connections.

        Parameters
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
        """
        super().__init__(pre, pos, idx_pre, idx_pos, weight, delay)

    def _update(self) -> None:
        """Update synaptic weights (no-op for static synapses).

        Static synapses have fixed weights, so this method does nothing.
        """
        pass


# synapses.py
class STDPSynapse(SynapticGroup):
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

    def __init__(
        self,
        pre: NeuronGroup,
        pos: NeuronGroup,
        idx_pre: torch.Tensor,
        idx_pos: torch.Tensor,
        delay: torch.Tensor,
        weight: Union[float, torch.Tensor],
        A_plus: float = 1e-3,
        A_minus: float = 2e-3,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
        dt: float = 1.0,
        w_min: float = 0.0,
        w_max: float = 1.0,
    ):
        """Initialize STDP synaptic connections.

        Parameters
        ----------
        pre : NeuronGroup
            Pre-synaptic (source) neuron group.
        pos : NeuronGroup
            Post-synaptic (target) neuron group.
        idx_pre : torch.Tensor
            Indices of pre-synaptic neurons for each connection.
        idx_pos : torch.Tensor
            Indices of post-synaptic neurons for each connection.
        delay : torch.Tensor
            Synaptic delays in time steps for each connection.
        weight : Union[float, torch.Tensor]
            Initial synaptic weights (scalar or tensor).
        A_plus : float, optional
            Learning rate for potentiation, by default 1e-3.
        A_minus : float, optional
            Learning rate for depression, by default 2e-3.
        tau_plus : float, optional
            Time constant for pre-synaptic trace decay, by default 20.0.
        tau_minus : float, optional
            Time constant for post-synaptic trace decay, by default 20.0.
        dt : float, optional
            Time step in milliseconds, by default 1.0.
        w_min : float, optional
            Minimum allowed weight value, by default 0.0.
        w_max : float, optional
            Maximum allowed weight value, by default 1.0.
        """
        super().__init__(pre, pos, idx_pre, idx_pos, weight, delay)
        self.weight = (
            torch.full(
                (len(idx_pre),), float(weight), dtype=torch.float32, device=self.device
            )
            if isinstance(weight, (int, float))
            else weight.to(device=self.device)
        )
        # Optimization attempts to reduce cache misses
        # sorted_indices = torch.argsort(self.idx_pre)
        # self.idx_pre = self.idx_pre[sorted_indices]
        # self.idx_pos = self.idx_pos[sorted_indices]
        # self.weight = self.weight[sorted_indices]

        self.A_plus = torch.tensor(A_plus, device=self.device)
        self.A_minus = torch.tensor(A_minus, device=self.device)
        self.tau_plus = torch.tensor(tau_plus, device=self.device)
        self.tau_minus = torch.tensor(tau_minus, device=self.device)
        self.w_min = torch.tensor(w_min, device=self.device)
        self.w_max = torch.tensor(w_max, device=self.device)

        self.x_pre = torch.zeros(len(idx_pre), dtype=torch.float32, device=self.device)
        self.x_pos = torch.zeros(len(idx_pos), dtype=torch.float32, device=self.device)

        self.alpha_pre = torch.exp(torch.tensor(-dt / tau_plus, device=self.device))
        self.alpha_pos = torch.exp(torch.tensor(-dt / tau_minus, device=self.device))

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

        # Spikes relevantes con los delays correctos
        pre_spikes = self.pre.get_spikes_at(self.delay, self.idx_pre)
        pos_spikes = self.pos.get_spikes_at(self._delay_1, self.idx_pos)

        # Actualización de trazas
        self.x_pre += pre_spikes.to(torch.float32)
        self.x_pos += pos_spikes.to(torch.float32)

        # STDP - pre dispara antes que post
        dw = self.A_plus * self.x_pos * pre_spikes
        self.weight += dw

        # STDP - post dispara después que pre
        dw = self.A_minus * self.x_pre * pos_spikes
        self.weight += dw

        self.weight.clamp_(self.w_min, self.w_max)
