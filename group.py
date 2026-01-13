from __future__ import annotations
from typing import Callable, Final, Optional, Union, Iterable
import copy
import torch
from .core import GPUNode
from . import globals


IndexCondition = Union[
    Callable[[torch.Tensor], torch.Tensor],
    int,
    Iterable[int],
    torch.Tensor,
]


class Group(GPUNode):
    r"""
    Base class for collections of computational elements with masking capabilities.

    Computational Model:
    --------------------
    Represents a discrete set of $N$ elements (neurons, synapses, etc.) allocated 
    on a specific compute device.
    
    Implements a **Masking Paradigm**:
    Operations on the group can be restricted to a subset of elements via a boolean 
    `filter` mask. Filtering is:
    1. **Accumulative:** `group.where_A().where_B()` results in $A \land B$.
    2. **Non-destructive:** Returns a shallow copy with a new mask; underlying data 
       (like states or positions) is shared, not duplicated.

    Attributes:
    -----------
    size : int
        Total cardinality of the set ($N$).
    filter : torch.Tensor
        Active element mask. True = Selected. Shape: [size].
    """

    # --- Configuration ---
    size: Final[int]
    """Number of elements in the group."""

    # --- State ---
    filter: torch.Tensor
    """Boolean selection mask. Operations should only affect elements where filter is True. Shape: [size]."""

    def __init__(self, size: int, device: Optional[Union[str, torch.device]] = None, **kwargs):
        """
        Initialize a generic group.

        Parameters
        ----------
        size : int
            Number of elements.
        device : str | torch.device
            Target GPU/CPU device.
        """
        super().__init__(device, **kwargs)
        self.size = int(size)
        
        # Initialize filter: All elements active by default
        self.filter = torch.ones(self.size, dtype=torch.bool, device=self.device)

    def _clone_with_new_filter(self) -> Group:
        """
        Internal: Create a shallow copy of the group with an independent filter tensor.
        
        Note:
        - The `filter` tensor is cloned (deep copy).
        - Other attributes (like `positions` or `V`) remain shared (shallow copy) 
          to save memory.
        """
        clone = copy.copy(self)
        clone.filter = clone.filter.clone()
        return clone

    def where_idx(self, condition: IndexCondition) -> Group:
        r"""
        Filter the group based on element indices.

        Parameters
        ----------
        condition :
            One of:
            - Callable: f(indices) -> bool mask
            - int: single index
            - iterable[int] or Tensor[int]: list of indices

        Returns
        -------
        Group
            A view of the group with the updated filter
            (OldFilter ∧ NewCondition).
        """
        clone = self._clone_with_new_filter()

        N = clone.size
        device = clone.device

        # Generate index vector
        idx = torch.arange(N, device=device)

        # ---- Case 1: callable ----
        if callable(condition):
            mask = condition(idx)

            if not isinstance(mask, torch.Tensor):
                raise TypeError("Callable condition must return a torch.Tensor")

            if mask.shape != (N,) or mask.dtype != torch.bool:
                raise ValueError(
                    f"Callable must return bool mask of shape ({N},), "
                    f"got {mask.shape} with dtype {mask.dtype}"
                )

        # ---- Case 2: single integer ----
        elif isinstance(condition, int):
            if condition < 0 or condition >= N:
                raise IndexError(f"Index {condition} out of bounds for size {N}")

            mask = torch.zeros(N, dtype=torch.bool, device=device)
            mask[condition] = True

        # ---- Case 3: list / iterable / tensor of indices ----
        else:
            indices = torch.as_tensor(condition, device=device)

            if indices.dtype not in (torch.int32, torch.int64):
                raise TypeError("Index list must contain integers")

            if torch.any(indices < 0) or torch.any(indices >= N):
                raise IndexError("Some indices are out of bounds")

            mask = torch.zeros(N, dtype=torch.bool, device=device)
            mask[indices] = True

        # Combine with existing filter
        clone.filter.logical_and_(mask)
        return clone

    def where_rank(self, rank: int) -> Group:
        """
        Filter elements based on the MPI/GPU Rank (Distributed Computing).

        Useful for defining connections that span multiple GPUs via a Bridge.
        If `rank` matches the local simulator rank, the group remains active. 
        Otherwise, all elements are filtered out locally.

        Parameters
        ----------
        rank : int
            Target GPU index.

        Returns
        -------
        Group
            The group (fully active or fully inactive depending on rank match).
        """
        clone = self._clone_with_new_filter()
        
        # Determine if this group belongs to the requested rank
        is_target_rank = (rank == globals.simulator.local_circuit.rank)
        
        if is_target_rank:
            # Keep current filter
            pass
        else:
            # Mask out everything
            clone.filter.fill_(False)
        
        return clone

    def reset_filter(self) -> None:
        """Reset the filter to select all elements (Mask = True)."""
        self.filter.fill_(True)


class SpatialGroup(Group):
    """
    Group where elements possess Euclidean spatial coordinates.

    Extends `Group` to allow topological filtering (e.g., "select neurons in 
    region X" or "select neighbors within radius R").

    Attributes:
    -----------
    spatial_dimensions : int
        Dimensionality of the space (e.g., 1, 2, 3).
    positions : torch.Tensor
        Coordinate vector for each element. Shape: [size, spatial_dimensions].
    """

    # --- Configuration ---
    spatial_dimensions: Final[int]
    """Number of spatial dimensions (D)."""

    # --- State ---
    positions: torch.Tensor
    """Spatial coordinates matrix. Shape: [size, D]."""

    def __init__(
        self, 
        size: int, 
        spatial_dimensions: int = 2, 
        device: Optional[Union[str, torch.device]] = None, 
        **kwargs
    ):
        """
        Initialize a spatial group.

        Parameters
        ----------
        size : int
            Number of elements.
        spatial_dimensions : int
            1 for Line, 2 for Grid/Plane, 3 for Volume.
        """
        super().__init__(size=size, device=device, **kwargs)
        
        self.spatial_dimensions = int(spatial_dimensions)
        
        # Initialize positions (Default: Standard Normal Distribution)
        # Note: Users are expected to overwrite this with specific topologies
        # (e.g., Grid, Sphere) after initialization if needed.
        self.positions = torch.randn(
            (self.size, self.spatial_dimensions), dtype=torch.float32, device=self.device
        )

    def where_pos(
        self, condition: Callable[[torch.Tensor], torch.Tensor]
    ) -> SpatialGroup:
        """
        Filter the group based on spatial coordinates.

        Parameters
        ----------
        condition : Callable
            A function `f(positions) -> bool_mask`.
            Input: Tensor of shape $[N, D]$.
            Output: Boolean Tensor of shape $[N]$.

        Returns
        -------
        SpatialGroup
            A view of the group with the updated filter.

        Example
        -------
        >>> # Select neurons in the upper-right quadrant
        >>> subset = group.where_pos(lambda p: (p[:,0] > 0) & (p[:,1] > 0))
        """
        clone = self._clone_with_new_filter()

        if clone.positions is None:
            raise RuntimeError("SpatialGroup has no positions defined.")

        mask = condition(clone.positions)
        
        if mask.shape != (clone.size,) or mask.dtype != torch.bool:
            raise ValueError(
                f"Condition must return a boolean mask of shape ({clone.size},), "
                f"got {mask.shape}."
            )

        clone.filter.logical_and_(mask)
        return clone