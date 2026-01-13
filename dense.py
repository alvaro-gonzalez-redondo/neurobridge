from __future__ import annotations

from .core import GPUNode
from .neurons import NeuronGroup

import copy
import torch
from typing import Callable, Iterable, Tuple, Union


DenseIndexCondition = Union[
    Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    Tuple[int, int],
    Iterable[Tuple[int, int]],
    torch.Tensor,  # shape (K, 2)
    dict,          # {'rows': ..., 'cols': ...}
]


class Dense(GPUNode):
    """Base class for dense matrix representations between two groups.

    Attributes
    ----------
    pre : Group
        Source group (pre-synaptic).
    pos : Group
        Target group (post-synaptic).
    shape : tuple
        Shape of the connection matrix (N_pre, N_pos).
    mask : torch.Tensor
        (Topology) Boolean tensor indicating which connections physically exist.
        Used for spike propagation.
    filter : torch.Tensor
        (Selection) Boolean tensor indicating which connections are currently selected.
        Used for analysis, visualization, or targeted operations.
    """
    pre: NeuronGroup
    pos: NeuronGroup
    shape: tuple
    mask: torch.Tensor
    filter: torch.Tensor

    def __init__(self, pre: NeuronGroup, pos: NeuronGroup, device: torch.device = None, **kwargs):
        # Heredar dispositivo de los grupos si no se especifica
        device = device or pre.device
        super().__init__(device, **kwargs)
        
        self.pre = pre
        self.pos = pos
        self.shape = (pre.size, pos.size)
        
        # MASK: Topología física (se definirá en subclases o specs, aquí por defecto todo conectado)
        self.mask = torch.ones(self.shape, dtype=torch.bool, device=self.device)
        
        # FILTER: Selección actual (inicialmente todo seleccionado)
        self.filter = torch.ones(self.shape, dtype=torch.bool, device=self.device)

    def _clone_with_new_filter(self) -> 'Dense':
        """Creates a view of the connection with a modified filter.
        
        Note: Shallow copy. 'mask' and 'weight' (in subclasses) are shared references.
        Only 'filter' is cloned and independent.
        """
        clone = copy.copy(self)
        clone.filter = self.filter.clone()
        return clone

    # --- Funciones de Filtrado (Lambda) ---

    def where_idx(self, condition: DenseIndexCondition) -> "Dense":
        """
        Filter dense connections based on indices (i, j).
        Updates clone.filter = clone.filter & new_mask

        Use examples:
        # 1. Callable (clásico)
        dense.where_idx(lambda i, j: i == j)          # diagonal
        dense.where_idx(lambda i, j: (i + j) % 2 == 0)

        # 2. Índice único
        dense.where_idx((3, 7))

        # 3. Índices explícitos
        dense.where_idx([(0, 1), (4, 2), (9, 9)])

        # 4. Filtrado estructural
        dense.where_idx({'rows': [0, 1, 2], 'cols': [5, 6]})
        """

        clone = self._clone_with_new_filter()
        device = self.device
        N, M = self.shape

        # ---- Case 1: callable (i, j) -> mask ----
        if callable(condition):
            rows = torch.arange(N, device=device)
            cols = torch.arange(M, device=device)
            grid_i, grid_j = torch.meshgrid(rows, cols, indexing="ij")

            mask = condition(grid_i, grid_j)

            if not isinstance(mask, torch.Tensor):
                raise TypeError("Callable must return a torch.Tensor")

            if mask.shape != (N, M) or mask.dtype != torch.bool:
                raise ValueError(
                    f"Callable must return bool mask of shape ({N}, {M}), "
                    f"got {mask.shape} with dtype {mask.dtype}"
                )

        # ---- Case 2: single (i, j) index ----
        elif isinstance(condition, tuple) and len(condition) == 2:
            i, j = condition
            if not (0 <= i < N and 0 <= j < M):
                raise IndexError(f"Index {(i, j)} out of bounds for shape {self.shape}")

            mask = torch.zeros((N, M), dtype=torch.bool, device=device)
            mask[i, j] = True

        # ---- Case 3: list / tensor of (i, j) ----
        elif isinstance(condition, (list, tuple, torch.Tensor)):
            indices = torch.as_tensor(condition, device=device)

            if indices.ndim != 2 or indices.shape[1] != 2:
                raise ValueError("Index list must have shape (K, 2)")

            if indices.dtype not in (torch.int32, torch.int64):
                raise TypeError("Indices must be integers")

            if torch.any(indices[:, 0] < 0) or torch.any(indices[:, 0] >= N) \
            or torch.any(indices[:, 1] < 0) or torch.any(indices[:, 1] >= M):
                raise IndexError("Some indices are out of bounds")

            mask = torch.zeros((N, M), dtype=torch.bool, device=device)
            mask[indices[:, 0], indices[:, 1]] = True

        # ---- Case 4: rows / cols dict (optional but powerful) ----
        elif isinstance(condition, dict):
            rows = condition.get("rows", None)
            cols = condition.get("cols", None)

            mask = torch.ones((N, M), dtype=torch.bool, device=device)

            if rows is not None:
                rows = torch.as_tensor(rows, device=device)
                row_mask = torch.zeros(N, dtype=torch.bool, device=device)
                row_mask[rows] = True
                mask &= row_mask[:, None]

            if cols is not None:
                cols = torch.as_tensor(cols, device=device)
                col_mask = torch.zeros(M, dtype=torch.bool, device=device)
                col_mask[cols] = True
                mask &= col_mask[None, :]

        else:
            raise TypeError(f"Unsupported condition type: {type(condition)}")

        # Combine with existing filter
        clone.filter &= mask
        return clone

    def where_pos(self, condition: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> 'Dense':
        """Filter connections based on spatial positions. updates self.filter"""
        if not (hasattr(self.pre, 'positions') and hasattr(self.pos, 'positions')):
             raise RuntimeError("Groups must have 'positions' for spatial filtering.")

        clone = self._clone_with_new_filter()
        
        # Broadcasting: (N, 1, D) vs (1, M, D)
        p_pre = self.pre.positions.unsqueeze(1)
        p_pos = self.pos.positions.unsqueeze(0)
        
        selection_mask = condition(p_pre, p_pos)
        
        if selection_mask.shape != self.shape:
             raise ValueError(f"Lambda returned shape {selection_mask.shape}, expected {self.shape}")
             
        # Actualizamos FILTER, no mask
        clone.filter &= selection_mask
        return clone

    # --- Utilidades de Interacción entre Mask y Filter ---

    def get_selected_mask(self) -> torch.Tensor:
        """Returns the intersection of physical connections and selected filter.
        
        Useful for operations that should only affect existing connections that
        are also currently selected.
        """
        return self.mask & self.filter

    def prune_filtered(self) -> None:
        """Modifies the physical topology (mask) by removing currently filtered elements.
        
        This makes the current 'filter' permanent in the 'mask'.
        """
        # Desactivamos físicamente lo que no esté en el filtro
        self.mask &= self.filter
        # Reset del filtro (opcional, dependiendo de tu lógica de UX)
        self.reset_filter()

    def reset_filter(self) -> None:
        """Resets the selection to include all elements."""
        self.filter.fill_(True)
