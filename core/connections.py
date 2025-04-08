import torch
from typing import Optional, Type, Any, Callable
from neurobridge.core.synaptic_group import SynapticGroup
from neurobridge.synaptic_groups.static_synapse import StaticSynapse
from neurobridge.core.neuron_group import NeuronGroup


class ConnectionOperator:
    """
    Implementa el operador >> para crear conexiones entre poblaciones de neuronas.
    """
    pre: NeuronGroup
    pos: NeuronGroup
    device: torch.device
    pattern: Optional[str]
    kwargs: dict[str, Any]

    def __init__(self, pre: NeuronGroup, pos: NeuronGroup) -> None:
        if pre.device != pos.device:
            raise RuntimeError("No se pueden conectar poblaciones en diferentes GPUs directamente, utiliza AxonalBridge.")
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
        valid_pre = torch.arange(self.pre.size, device=self.device) if self.pre.filter is None else self.pre.filter.nonzero(as_tuple=True)[0]
        valid_pos = torch.arange(self.pos.size, device=self.device) if self.pos.filter is None else self.pos.filter.nonzero(as_tuple=True)[0]

        if pattern == 'all-to-all':
            grid_pre, grid_pos = torch.meshgrid(valid_pre, valid_pos, indexing='ij')
            source_indices = grid_pre.flatten()
            target_indices = grid_pos.flatten()

        elif pattern == 'specific':
            try:
                source_indices = self._compute_parameter(kwargs['idx_pre'])
                target_indices = self._compute_parameter(kwargs['idx_pos'])
            except KeyError:
                raise RuntimeError("Faltan 'idx_pre' o 'idx_pos' en los parámetros para el patrón 'specific'.")

        else:
            raise NotImplementedError(f"Patrón de conexión '{pattern}' no implementado.")

        # Crear objeto sináptico
        if synapse_class is StaticSynapse:
            try:
                delay = self._compute_parameter(kwargs['delay'], source_indices, target_indices)
                weight = self._compute_parameter(kwargs['weight'], source_indices, target_indices)
            except KeyError:
                raise RuntimeError("Faltan 'delay' o 'weight' en los parámetros para StaticSynapse.")
            
            connection = StaticSynapse(
                pre=self.pre,
                pos=self.pos,
                idx_pre=source_indices,
                idx_pos=target_indices,
                delay=delay,
                weight=weight
            )

        else:
            raise NotImplementedError(f"Clase de sinapsis '{synapse_class}' no implementada o no especificada.")

        # Limpiar filtros tras conectar
        self.pre.filter = None
        self.pos.filter = None

        return connection


    def _compute_parameter(
        self,
        param: Any,
        idx_pre: Optional[torch.Tensor] = None,
        idx_post: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calcula un tensor de parámetros por conexión a partir de:
        - Un valor escalar: se replica.
        - Un tensor: se usa directamente o se expande.
        - Una función: se evalúa por conexión usando (idx_pre, idx_post).
        """
        if callable(param):
            if idx_pre is None or idx_post is None:
                raise ValueError("Se requieren 'idx_pre' y 'idx_post' si el parámetro es una función.")
            n = len(idx_pre)
            values = torch.zeros(n, device=self.device)
            for i in range(n):
                values[i] = param(idx_pre[i].item(), idx_post[i].item())
        
        elif isinstance(param, torch.Tensor):
            n = len(idx_pre) if idx_pre is not None else param.shape[0]
            if param.numel() == 1:
                values = torch.full((n,), param.item(), device=self.device)
            else:
                if param.numel() != n:
                    raise ValueError(f"Esperado tensor de longitud {n}, pero tiene {param.numel()}.")
                values = param.to(device=self.device)
        
        else:
            n = len(idx_pre) if idx_pre is not None else 1
            values = torch.full((n,), float(param), device=self.device)

        return values
