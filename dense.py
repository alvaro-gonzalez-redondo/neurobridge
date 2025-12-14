from __future__ import annotations

from .core import GPUNode
from .neurons import NeuronGroup

import copy
import torch
from typing import Callable, Optional

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

    def where_idx(self, condition: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> 'Dense':
        """Filter connections based on indices (i, j). update self.filter"""
        clone = self._clone_with_new_filter()
        
        rows = torch.arange(self.shape[0], device=self.device)
        cols = torch.arange(self.shape[1], device=self.device)
        
        # Meshgrid para pasar (i, j) a la lambda
        grid_rows, grid_cols = torch.meshgrid(rows, cols, indexing='ij')
        
        selection_mask = condition(grid_rows, grid_cols)
        
        if selection_mask.shape != self.shape:
             raise ValueError(f"Lambda must return mask of shape {self.shape}")

        # Actualizamos FILTER, no mask
        clone.filter &= selection_mask
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
