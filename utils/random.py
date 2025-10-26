"""
Random distribution utilities for connection parameters.

Provides convenient ways to specify random weights and delays without verbose lambdas.

Examples
--------
Simple tuple syntax (uniform distribution):
>>> self.sim.connect(pre, post, StaticDense, weight=(0, 1e-4), delay=(1, 5))

Explicit distribution classes:
>>> self.sim.connect(pre, post, StaticDense,
...                  weight=Uniform(0, 1e-4),
...                  delay=Normal(mean=5, std=1))
"""

from __future__ import annotations
import torch


class RandomDistribution:
    """Base class for random distributions in connection parameters.

    Subclasses must implement the `sample` method to generate random values.
    """

    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        """Generate n random samples on the specified device.

        Parameters
        ----------
        n : int
            Number of samples to generate.
        device : torch.device
            Device to place the tensor on.

        Returns
        -------
        torch.Tensor
            Tensor of shape (n,) with random values.
        """
        raise NotImplementedError


class Uniform(RandomDistribution):
    """Uniform distribution over [low, high).

    Parameters
    ----------
    low : float
        Lower bound (inclusive).
    high : float
        Upper bound (exclusive).

    Examples
    --------
    >>> weight = Uniform(0, 1e-4)
    >>> weight = Uniform(low=0.5, high=1.5)
    """

    def __init__(self, low: float = 0.0, high: float = 1.0):
        self.low = low
        self.high = high

    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        return torch.rand(n, device=device) * (self.high - self.low) + self.low

    def __repr__(self):
        return f"Uniform({self.low}, {self.high})"


class Normal(RandomDistribution):
    """Normal (Gaussian) distribution.

    Parameters
    ----------
    mean : float
        Mean of the distribution.
    std : float
        Standard deviation of the distribution.

    Examples
    --------
    >>> weight = Normal(mean=0.5, std=0.1)
    >>> weight = Normal(0.5, 0.1)
    """

    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = mean
        self.std = std

    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        return torch.randn(n, device=device) * self.std + self.mean

    def __repr__(self):
        return f"Normal(mean={self.mean}, std={self.std})"


class UniformInt(RandomDistribution):
    """Uniform distribution over integers [low, high] (both inclusive).

    Useful for random delays.

    Parameters
    ----------
    low : int
        Lower bound (inclusive).
    high : int
        Upper bound (inclusive).

    Examples
    --------
    >>> delay = UniformInt(1, 5)  # Random delays between 1 and 5 (inclusive)
    >>> delay = UniformInt(low=2, high=10)
    """

    def __init__(self, low: int, high: int):
        self.low = low
        self.high = high

    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        return torch.randint(self.low, self.high + 1, (n,), device=device)

    def __repr__(self):
        return f"UniformInt({self.low}, {self.high})"


class LogNormal(RandomDistribution):
    """Log-normal distribution.

    Useful for modeling biological parameters with multiplicative variability.

    Parameters
    ----------
    mean : float
        Mean of the underlying normal distribution.
    std : float
        Standard deviation of the underlying normal distribution.

    Examples
    --------
    >>> weight = LogNormal(mean=-3, std=0.5)
    """

    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = mean
        self.std = std

    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        normal_samples = torch.randn(n, device=device) * self.std + self.mean
        return torch.exp(normal_samples)

    def __repr__(self):
        return f"LogNormal(mean={self.mean}, std={self.std})"


class Constant(RandomDistribution):
    """Constant value (not actually random, but fits the interface).

    Useful for consistency when mixing random and fixed parameters.

    Parameters
    ----------
    value : float
        Constant value.

    Examples
    --------
    >>> weight = Constant(1e-4)
    """

    def __init__(self, value: float):
        self.value = value

    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        return torch.full((n,), self.value, device=device, dtype=torch.float32)

    def __repr__(self):
        return f"Constant({self.value})"
