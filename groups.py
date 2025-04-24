from __future__ import annotations

from .core import _GPUNode

from typing import Callable

import copy
import torch


class _Group(_GPUNode):
    size: int
    filter: torch.Tensor

    def __init__(self, device:str, size:int):
        super().__init__(device)
        self.size = size
        self.filter = torch.ones(self.size, dtype=torch.bool, device=self.device)


    def _clone_with_new_filter(self) -> _Group:
        clone = copy.copy(self)
        clone.filter = clone.filter.clone()
        return clone


    def where_id(self, condition: Callable[[torch.Tensor], torch.Tensor]) -> _Group:
        """
        Aplica un filtro basado en los índices (vectorizado).
    
        Args:
            condition: Función que recibe un tensor de índices y devuelve una máscara booleana.
    
        Returns:
            A sí mismo con el filtro actualizado.
        """
        clone = self._clone_with_new_filter()
        idx = torch.arange(clone.size, device=clone.device)
        mask = condition(idx)
        if mask.shape != (clone.size,) or mask.dtype != torch.bool:
            raise ValueError("La función debe devolver una máscara booleana del mismo tamaño que el grupo.")
        clone.filter &= mask
        return clone
        

    def reset_filter(self) -> None:
        self.filter.fill_(True)


class _SpatialGroup(_Group):
    spatial_dimensions: torch.Tensor
    positions: torch.Tensor


    def __init__(self, device:str, size:int, spatial_dimensions:int=2):
        super().__init__(device, size)
        self.spatial_dimensions = torch.tensor(spatial_dimensions, dtype=torch.int32, device=self.device)
        self.positions = torch.randn((self.size, self.spatial_dimensions), device=self.device)


    def where_pos(self, condition: Callable[[torch.Tensor], torch.Tensor]) -> _SpatialGroup:
        """
        Aplica un filtro basado en las posiciones (vectorizado).
    
        Args:
            condition: Función que toma un tensor de posiciones (n x d) y devuelve una máscara booleana.
    
        Returns:
            A sí mismo con el filtro actualizado.
        """
        clone = self._clone_with_new_filter()

        if clone.positions is None:
            raise RuntimeError("Este grupo no tiene posiciones definidas.")
        
        mask = condition(clone.positions)
        if mask.shape != (clone.size,) or mask.dtype != torch.bool:
            raise ValueError("La función debe devolver una máscara booleana del mismo tamaño que el grupo.")
        
        clone.filter &= mask
        return clone