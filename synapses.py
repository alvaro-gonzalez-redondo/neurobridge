from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .neurons import NeuronGroup
from .groups import _Group

from typing import Optional, Any, Type, Union

import torch


class _ConnectionOperator:
    pre: NeuronGroup
    pos: NeuronGroup
    pattern: Optional[str]
    kwargs: dict[str, Any]


    def __init__(self, pre: NeuronGroup, pos: NeuronGroup) -> None:
        if pre.device != pos.device:
            raise RuntimeError("No se pueden conectar poblaciones en diferentes GPUs directamente.")
        self.pre = pre
        self.pos = pos
        self.device = self.pre.device
        self.pattern = None
        self.kwargs = {}
    

    def __call__(
        self,
        pattern: str = 'all-to-all',
        synapse_class: Optional[Type[SynapticGroup]] = None,
        **kwargs: Any
    ) -> SynapticGroup:

        self.pattern = pattern
        self.kwargs = kwargs

        # Generar subconjuntos filtrados (o completos)
        valid_pre = self.pre.filter.nonzero(as_tuple=True)[0]
        valid_pos = self.pos.filter.nonzero(as_tuple=True)[0]

        if pattern == 'all-to-all':
            grid_pre, grid_pos = torch.meshgrid(valid_pre, valid_pos, indexing='ij')
            source_indices = grid_pre.flatten()
            target_indices = grid_pos.flatten()

        elif pattern == 'specific':
            try:
                source_indices = self._compute_parameter(kwargs['idx_pre'], kwargs['idx_pre'], kwargs['idx_pre'])
                target_indices = self._compute_parameter(kwargs['idx_pos'], kwargs['idx_pos'], kwargs['idx_pos'])
            except KeyError:
                raise RuntimeError("Faltan 'idx_pre' o 'idx_pos' en los parámetros para el patrón 'specific'.")
        
        elif pattern == 'one-to-one':
            assert valid_pre.numel() == valid_pos.numel()
            source_indices = valid_pre.clone()
            target_indices = valid_pos.clone()

        else:
            raise NotImplementedError(f"Patrón de conexión '{pattern}' no implementado.")

        # Parámetros comunes a todas las sinapsis
        delay = self._compute_parameter(kwargs.get('delay', 0), source_indices, target_indices).to(torch.long)
        weight = self._compute_parameter(kwargs.get('weight', 0.0), source_indices, target_indices)

        assert torch.all(delay<self.pre.delay_max), f"Connection delay ({torch.max(delay)}) must be less than the `delay_max` parameter of the presynaptic population ({self.pre.delay_max})."

        # Crear objeto sináptico
        if synapse_class is None or synapse_class is StaticSynapse:
            connection = StaticSynapse(
                pre=self.pre,
                pos=self.pos,
                idx_pre=source_indices,
                idx_pos=target_indices,
                delay=delay,
                weight=weight
            )

        elif synapse_class is STDPSynapse:
            connection = STDPSynapse(
                pre=self.pre,
                pos=self.pos,
                idx_pre=source_indices,
                idx_pos=target_indices,
                delay=delay,
                weight=weight,
                A_plus=kwargs.get('A_plus', 0.01),
                A_minus=kwargs.get('A_minus', 0.012),
                tau_plus=kwargs.get('tau_plus', 20.0),
                tau_minus=kwargs.get('tau_minus', 20.0),
                dt=kwargs.get('dt', 1.0),
                w_min=kwargs.get('w_min', 0.0),
                w_max=kwargs.get('w_max', 1.0)
            )

        else:
            raise NotImplementedError(f"Clase de sinapsis '{synapse_class}' no soportada.")

        # Limpiar filtros tras conectar
        self.pre.reset_filter()
        self.pos.reset_filter()

        return connection


    def _compute_parameter(
        self,
        param: Any,
        idx_pre: torch.Tensor,
        idx_post: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcula un tensor de parámetros por conexión a partir de:
        - Escalar: se replica.
        - Tensor: se usa directamente o se expande.
        - Lista: se convierte a tensor.
        - Función: se aplica vectorizadamente a idx_pre, idx_post (ambos tensores).
        """
        n = len(idx_pre)
    
        if callable(param):
            values = param(idx_pre, idx_post)
            if not isinstance(values, torch.Tensor):
                raise TypeError("Las funciones deben devolver un tensor.")
            if values.shape[0] != n:
                raise ValueError(f"El tensor devuelto debe tener tamaño {n}, pero tiene {values.shape[0]}.")
            return values.to(device=self.device)
    
        elif isinstance(param, torch.Tensor):
            if param.numel() == 1:
                return torch.full((n,), param.item(), device=self.device)
            if param.numel() != n:
                raise ValueError(f"Expected a tensor of length {n}, got {param.numel()}.")
            return param.to(device=self.device)
    
        elif isinstance(param, list):
            param = torch.tensor(param, device=self.device)
            return self._compute_parameter(param, idx_pre, idx_post)
    
        else:  # Escalar
            return torch.full((n,), float(param), device=self.device)


# synapses.py
class SynapticGroup(_Group):
    pre: NeuronGroup
    pos: NeuronGroup
    idx_pre: torch.Tensor
    idx_pos: torch.Tensor
    weight: torch.Tensor
    delay: torch.Tensor
    _current_buffer: torch.Tensor


    def __init__(self, pre: NeuronGroup, pos: NeuronGroup, idx_pre: torch.Tensor, idx_pos: torch.Tensor, weight: torch.Tensor, delay: torch.Tensor):
        if pre.device != pos.device:
            raise RuntimeError("Connected populations must be from the same device.")
        device = pre.device
        
        if idx_pre.numel() != idx_pos.numel():
            raise RuntimeError(f"The number of sources ({idx_pre.numel()}) and targets ({idx_pos.numel()}) do not match.")
        size = idx_pre.numel()

        super().__init__(device, size)

        self.pre = pre
        self.pos = pos
        self.idx_pre = idx_pre
        self.idx_pos = idx_pos
        self.weight = weight.to(device=pre.device, dtype=torch.float32)
        self.delay = delay

        self._current_buffer = torch.zeros(self.pos.size, dtype=torch.float32, device=self.device)


    def _process(self):
        super()._process()
        self._propagate()
        self._update()


    def _propagate(self):
        spikes_mask = self.pre.get_spikes_at(self.delay, self.idx_pre)
        mask_f = spikes_mask.to(self.weight.dtype)
        contrib = self.weight * mask_f
        self._current_buffer.zero_()
        self._current_buffer.index_add_(0, self.idx_pos, contrib)
        self.pos.inject_currents(self._current_buffer)


    def _update(self) -> None:
        """
        Time-driven learning rules are implemented here
        """
        raise NotImplementedError("`update` method in `SynapticGroup` must be implemented.")


# synapses.py
class StaticSynapse(SynapticGroup):

    def __init__(self, pre, pos, idx_pre, idx_pos, weight, delay):
        super().__init__(pre, pos, idx_pre, idx_pos, weight, delay)

    def _update(self) -> None:
        pass
        

# synapses.py
class STDPSynapse(SynapticGroup):
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
        w_max: float = 1.0
    ):
        super().__init__(pre, pos, idx_pre, idx_pos, weight, delay)
        self.weight = (
            torch.full((len(idx_pre),), float(weight), dtype=torch.float32, device=self.device)
            if isinstance(weight, (int, float))
            else weight.to(device=self.device)
        )
        # Optimization attempts to reduce cache misses
        #sorted_indices = torch.argsort(self.idx_pre)
        #self.idx_pre = self.idx_pre[sorted_indices]
        #self.idx_pos = self.idx_pos[sorted_indices]
        #self.weight = self.weight[sorted_indices]
        
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

        self._delay_1 = torch.full_like(self.idx_pos, 1, device=self.device)


    def _update(self) -> None:
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
        dw = -self.A_minus * self.x_pre * pos_spikes
        self.weight += dw

        self.weight.clamp_(self.w_min, self.w_max)
