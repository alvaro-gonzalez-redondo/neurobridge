from __future__ import annotations

from .core import GPUNode

from typing import Callable

import copy
import torch


class Group(GPUNode):
    """Base class for groups of elements with filtering capabilities.

    A Group represents a collection of elements that can be filtered to select
    specific subsets. It serves as a base class for specialized groups like
    neuron populations or synaptic connections.

    Attributes
    ----------
    size : int
        Number of elements in the group.
    filter : torch.Tensor
        Boolean tensor of shape (size,) indicating which elements are selected.
        Initially all elements are selected (all True).
    device : torch.device
        The GPU device this group is associated with (inherited from GPUNode).
    """

    size: int
    filter: torch.Tensor

    def __init__(self, device: str, size: int):
        """Initialize a new Group.

        Parameters
        ----------
        device : str
            String representation of the GPU device (e.g., 'cuda:0').
        size : int
            Number of elements in the group.
        """
        super().__init__(device)
        self.size = size
        self.filter = torch.ones(self.size, dtype=torch.bool, device=self.device)

    def _clone_with_new_filter(self) -> Group:
        """Create a copy of this group with a cloned filter tensor.

        Used internally for filter operations to avoid modifying the original group.

        Returns
        -------
        Group
            A shallow copy of the group with an independent filter tensor.
        """
        clone = copy.copy(self)
        clone.filter = clone.filter.clone()
        return clone

    def where_id(self, condition: Callable[[torch.Tensor], torch.Tensor]) -> Group:
        """Filter the group based on element indices.

        Applies a vectorized filtering operation based on element indices.
        The condition function should accept a tensor of indices and return
        a boolean mask of the same size.

        Parameters
        ----------
        condition : Callable[[torch.Tensor], torch.Tensor]
            Function that takes a tensor of indices and returns a boolean mask.

        Returns
        -------
        Group
            A new group with the updated filter.

        Raises
        ------
        ValueError
            If the condition function returns a tensor with incorrect shape or type.

        Examples
        --------
        >>> # Select even-indexed elements
        >>> filtered = group.where_id(lambda ids: ids % 2 == 0)
        >>> # Select the first 10 elements
        >>> filtered = group.where_id(lambda ids: ids < 10)
        """
        clone = self._clone_with_new_filter()
        idx = torch.arange(clone.size, device=clone.device)
        mask = condition(idx)
        if mask.shape != (clone.size,) or mask.dtype != torch.bool:
            raise ValueError(
                "Function must return a boolean mask of the same size of the group."
            )
        clone.filter &= mask
        return clone

    def reset_filter(self) -> None:
        """Reset the filter to select all elements.

        After calling this method, all elements in the group will be selected.
        """
        self.filter.fill_(True)


class SpatialGroup(Group):
    """Group with spatial positions for each element.

    Extends the base Group class by adding spatial coordinates for each element,
    enabling filtering based on spatial properties.

    Attributes
    ----------
    spatial_dimensions : torch.Tensor
        Scalar tensor containing the number of spatial dimensions.
    positions : torch.Tensor
        Tensor of shape (size, spatial_dimensions) containing the spatial
        coordinates for each element.
    """

    spatial_dimensions: torch.Tensor
    positions: torch.Tensor

    def __init__(self, device: str, size: int, spatial_dimensions: int = 2):
        """Initialize a new SpatialGroup.

        Parameters
        ----------
        device : str
            String representation of the GPU device (e.g., 'cuda:0').
        size : int
            Number of elements in the group.
        spatial_dimensions : int, optional
            Number of spatial dimensions, by default 2.
        """
        super().__init__(device, size)
        self.spatial_dimensions = torch.tensor(
            spatial_dimensions, dtype=torch.int32, device=self.device
        )
        self.positions = torch.randn(
            (self.size, self.spatial_dimensions), device=self.device
        )

    def where_pos(
        self, condition: Callable[[torch.Tensor], torch.Tensor]
    ) -> SpatialGroup:
        """Filter the group based on element positions.

        Applies a vectorized filtering operation based on element positions.
        The condition function should accept a tensor of positions and return
        a boolean mask of the same size.

        Parameters
        ----------
        condition : Callable[[torch.Tensor], torch.Tensor]
            Function that takes a tensor of positions (shape [size, spatial_dimensions])
            and returns a boolean mask (shape [size]).

        Returns
        -------
        SpatialGroup
            A new spatial group with the updated filter.

        Raises
        ------
        RuntimeError
            If the group does not have positions defined.
        ValueError
            If the condition function returns a tensor with incorrect shape or type.

        Examples
        --------
        >>> # Select elements in the upper half of the space
        >>> filtered = group.where_pos(lambda pos: pos[:, 1] > 0)
        >>> # Select elements within a certain radius of the origin
        >>> filtered = group.where_pos(lambda pos: torch.norm(pos, dim=1) < 1.0)
        """
        clone = self._clone_with_new_filter()

        if clone.positions is None:
            raise RuntimeError("Este grupo no tiene posiciones definidas.")

        mask = condition(clone.positions)
        if mask.shape != (clone.size,) or mask.dtype != torch.bool:
            raise ValueError(
                "La función debe devolver una máscara booleana del mismo tamaño que el grupo."
            )

        clone.filter &= mask
        return clone
