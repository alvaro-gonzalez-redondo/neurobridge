from __future__ import annotations

from .core import GPUNode

import copy
import torch


class DenseNode(GPUNode):
    """Base class for nodes with dense matrix representations."""
    shape: tuple
    filter: torch.Tensor

    def __init__(self, shape: tuple, device: str):
        super().__init__(device)
        self.shape = shape
        self.filter = torch.ones(shape, dtype=torch.bool)
    
    def _clone_with_new_filter(self):
        clone = copy.copy(self)
        clone.filter = self.filter.clone()
        return clone

    def __getitem__(self, key):
        new_filter = torch.zeros(self.shape, dtype=torch.bool)
        new_filter[key] = True  # Mark selected indices
        clone = self._clone_with_new_filter()
        clone.filter &= new_filter  # Filters intersection
        return clone

    def reset_filter(self) -> None:
        """Reset the filter to select all elements.

        After calling this method, all elements in the group will be selected.
        """
        self.filter.fill_(True)

