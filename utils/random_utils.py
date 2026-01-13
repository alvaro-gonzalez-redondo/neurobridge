"""
Utilities for random parameter distribution in Neural Networks.

This module provides a declarative interface to specify connection parameters
(weights, delays) using standard arithmetic operations. It supports lazy evaluation,
PyTorch integration, and closed-form statistical analysis where possible.
"""

from __future__ import annotations

import math
import warnings
from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple, TypeVar, Callable

import torch

# Type alias for scalar values that can operate with distributions
Scalar = Union[int, float]
T = TypeVar("T", bound="RandomDistribution")


class RandomDistribution(ABC):
    """
    Abstract base class for random distributions.

    Subclasses must implement ``sample``, ``mean``, and ``var``.
    """

    @abstractmethod
    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        """
        Generate random samples.

        Parameters
        ----------
        n : int
            Number of samples to generate.
        device : torch.device
            Target device for the tensor.

        Returns
        -------
        torch.Tensor
            Tensor of shape (n,).
        """
        pass

    @abstractmethod
    def mean(self, **kwargs) -> float:
        r"""
        Calculate the expected value :math:`\mathbb{E}[X]`.
        """
        pass

    @abstractmethod
    def var(self, **kwargs) -> float:
        r"""
        Calculate the variance :math:`\text{Var}(X)`.
        """
        pass

    def std(self, **kwargs) -> float:
        r"""
        Calculate the standard deviation :math:`\sigma_X = \sqrt{\text{Var}(X)}`.

        Ensures numerical stability by clamping variance to 0.
        """
        v = self.var(**kwargs)
        # Numerical stability: variance cannot be negative, but floating point errors might occur
        return math.sqrt(max(0.0, v))

    def estimate_stats(self, n: int = 10000) -> Tuple[float, float]:
        """
        Estimate mean and variance via Monte Carlo sampling on CPU.

        Useful when analytical forms are undefined or complex (e.g., after clamping).

        Returns
        -------
        mean, var : Tuple[float, float]
        """
        # We use CPU to avoid memory fragmentation on GPU for simple stat checks
        samples = self.sample(n, torch.device("cpu")).float()
        return samples.mean().item(), samples.var().item()

    # ---------- fluent interface ----------

    def clip(self, pmin: float = 0.01, pmax: float = 0.99) -> Clipped:
        """Return a distribution clipped at percentiles."""
        return Clipped(self, pmin, pmax)

    def clamp(self, min: Optional[float] = None, max: Optional[float] = None) -> Clamped:
        """Return a distribution clamped between min and max values."""
        return Clamped(self, min, max)

    # ---------- Operator Overloading ----------

    def __neg__(self) -> Scaled:
        return Scaled(self, -1.0)

    def __mul__(self, other: Union[Scalar, RandomDistribution]) -> RandomDistribution:
        if isinstance(other, (int, float)):
            return Scaled(self, float(other))
        if isinstance(other, RandomDistribution):
            return BinaryOp(self, other, torch.mul, "*")
        return NotImplemented

    def __rmul__(self, other: Scalar) -> RandomDistribution:
        if isinstance(other, (int, float)):
            return Scaled(self, float(other))
        return NotImplemented

    def __truediv__(self, other: Union[Scalar, RandomDistribution]) -> RandomDistribution:
        if isinstance(other, (int, float)):
            return Scaled(self, 1.0 / float(other))
        if isinstance(other, RandomDistribution):
            return BinaryOp(self, other, torch.div, "/")
        return NotImplemented

    def __add__(self, other: Union[Scalar, RandomDistribution]) -> RandomDistribution:
        if isinstance(other, (int, float)):
            return Shifted(self, float(other))
        if isinstance(other, RandomDistribution):
            return BinaryOp(self, other, torch.add, "+")
        return NotImplemented

    def __radd__(self, other: Scalar) -> RandomDistribution:
        if isinstance(other, (int, float)):
            return Shifted(self, float(other))
        return NotImplemented

    def __sub__(self, other: Union[Scalar, RandomDistribution]) -> RandomDistribution:
        if isinstance(other, (int, float)):
            return Shifted(self, -float(other))
        if isinstance(other, RandomDistribution):
            return BinaryOp(self, other, torch.sub, "-")
        return NotImplemented


# ==============================================================================
# Modifiers (Decorators)
# ==============================================================================

class Scaled(RandomDistribution):
    r"""Distribution scaled by a constant factor :math:`Y = k \cdot X`."""

    def __init__(self, base: RandomDistribution, factor: float):
        self.base = base
        self.factor = factor

    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        return self.base.sample(n, device) * self.factor

    def mean(self, **kwargs) -> float:
        return self.factor * self.base.mean(**kwargs)

    def var(self, **kwargs) -> float:
        return (self.factor ** 2) * self.base.var(**kwargs)

    def __repr__(self) -> str:
        return f"({self.base} * {self.factor})"


class Shifted(RandomDistribution):
    """Distribution shifted by a constant offset :math:`Y = X + c`."""

    def __init__(self, base: RandomDistribution, offset: float):
        self.base = base
        self.offset = offset

    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        return self.base.sample(n, device) + self.offset

    def mean(self, **kwargs) -> float:
        return self.base.mean(**kwargs) + self.offset

    def var(self, **kwargs) -> float:
        return self.base.var(**kwargs)

    def __repr__(self) -> str:
        return f"({self.base} + {self.offset})"


class Clamped(RandomDistribution):
    """
    Distribution clamped to [min, max].

    .. warning::
       Analytic mean and variance are not preserved after clamping.
       Accessing ``mean()`` or ``var()`` will raise an error unless
       ``force_estimate=True`` is passed.
    """

    def __init__(self, base: RandomDistribution, min_val: Optional[float] = None, max_val: Optional[float] = None):
        self.base = base
        self.min_val = min_val
        self.max_val = max_val

    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        x = self.base.sample(n, device)
        return torch.clamp(x, self.min_val, self.max_val)

    def _check_estimate(self, force_estimate: bool, n_samples: int) -> Tuple[float, float]:
        if not force_estimate:
            raise NotImplementedError(
                f"Analytic statistics for {self.__class__.__name__} are not available due to non-linearity. "
                "Use .mean(force_estimate=True) to estimate via Monte Carlo."
            )
        warnings.warn(
            f"Estimating statistics for {self.__class__.__name__} via Monte Carlo ({n_samples} samples). "
            "This is an approximation.",
            UserWarning,
            stacklevel=3
        )
        return self.estimate_stats(n_samples)

    def mean(self, force_estimate: bool = False, n_samples: int = 10000, **kwargs) -> float:
        m, _ = self._check_estimate(force_estimate, n_samples)
        return m

    def var(self, force_estimate: bool = False, n_samples: int = 10000, **kwargs) -> float:
        _, v = self._check_estimate(force_estimate, n_samples)
        return v

    def __repr__(self) -> str:
        return f"Clamp({self.base}, {self.min_val}, {self.max_val})"


class Clipped(Clamped):
    """
    Distribution clipped at empirical percentiles (pmin, pmax).

    This operation requires computing quantiles on the sampled data.
    """

    def __init__(self, base: RandomDistribution, pmin: float = 0.01, pmax: float = 0.99):
        super().__init__(base, None, None)  # min/max determined at runtime
        self.pmin = pmin
        self.pmax = pmax

    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        x = self.base.sample(n, device)
        # Note: quantile requires float
        x_f = x.float()
        lo = torch.quantile(x_f, self.pmin)
        hi = torch.quantile(x_f, self.pmax)
        return torch.clamp(x, lo, hi)

    def __repr__(self) -> str:
        return f"Clip({self.base}, p={self.pmin}..{self.pmax})"


# ==============================================================================
# Concrete Distributions
# ==============================================================================

class Uniform(RandomDistribution):
    """
    Continuous Uniform distribution :math:`U(a, b)`.

    Parameters
    ----------
    low : float
        Lower bound (inclusive).
    high : float
        Upper bound (exclusive).
    """

    def __init__(self, low: float = 0.0, high: float = 1.0):
        self.low = low
        self.high = high

    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        return torch.rand(n, device=device) * (self.high - self.low) + self.low

    def mean(self, **kwargs) -> float:
        return (self.low + self.high) / 2.0

    def var(self, **kwargs) -> float:
        return ((self.high - self.low) ** 2) / 12.0

    # --- Optimization: Closed form arithmetic ---
    def __mul__(self, k: Union[Scalar, RandomDistribution]) -> RandomDistribution:
        if isinstance(k, (int, float)):
            return Uniform(self.low * k, self.high * k)
        return super().__mul__(k)

    def __add__(self, c: Union[Scalar, RandomDistribution]) -> RandomDistribution:
        if isinstance(c, (int, float)):
            return Uniform(self.low + c, self.high + c)
        return super().__add__(c)

    def __repr__(self) -> str:
        return f"Uniform({self.low}, {self.high})"


class Normal(RandomDistribution):
    r"""
    Normal (Gaussian) distribution :math:`N(\mu, \sigma^2)`.
    """

    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self._mean = mean
        self._std = std

    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        return torch.randn(n, device=device) * self._std + self._mean

    def mean(self, **kwargs) -> float:
        return self._mean

    def var(self, **kwargs) -> float:
        return self._std ** 2

    # --- Optimization: Closed form arithmetic ---
    def __mul__(self, k: Union[Scalar, RandomDistribution]) -> RandomDistribution:
        if isinstance(k, (int, float)):
            return Normal(self._mean * k, abs(k) * self._std)
        return super().__mul__(k)

    def __add__(self, c: Union[Scalar, RandomDistribution]) -> RandomDistribution:
        if isinstance(c, (int, float)):
            return Normal(self._mean + c, self._std)
        return super().__add__(c)

    def __repr__(self) -> str:
        return f"Normal(μ={self._mean}, σ={self._std})"


class UniformInt(RandomDistribution):
    r"""
    Discrete Uniform distribution over integers :math:`U\{a, b\}`.

    Note: The sample method returns a LongTensor (integers), but mean/var return floats.
    """

    def __init__(self, low: int, high: int):
        self.low = low
        self.high = high

    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        # high is inclusive in pytorch's randint(low, high+1)? No, randint is [low, high)
        # Wait, user doc said "inclusive". To make high inclusive in torch.randint, we need high + 1
        return torch.randint(self.low, self.high + 1, (n,), device=device)

    def mean(self, **kwargs) -> float:
        return (self.low + self.high) / 2.0

    def var(self, **kwargs) -> float:
        # Variance of discrete uniform [a, b] is ((b - a + 1)^2 - 1) / 12
        n = self.high - self.low + 1
        return (n ** 2 - 1) / 12.0

    def __repr__(self) -> str:
        return f"UniformInt({self.low}, {self.high})"


class LogNormal(RandomDistribution):
    r"""
    Log-Normal distribution.

    If :math:`Y \sim N(\mu, \sigma)`, then :math:`X = e^Y \sim \text{LogNormal}(\mu, \sigma)`.

    Parameters
    ----------
    mean : float
        Mean (:math:`\mu`) of the underlying normal distribution.
    std : float
        Standard deviation (:math:`\sigma`) of the underlying normal distribution.
    """

    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self._mean = mean
        self._std = std

    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        normal_samples = torch.randn(n, device=device) * self._std + self._mean
        return torch.exp(normal_samples)

    def mean(self, **kwargs) -> float:
        return math.exp(self._mean + 0.5 * self._std ** 2)

    def var(self, **kwargs) -> float:
        sigma2 = self._std ** 2
        return (math.exp(sigma2) - 1) * math.exp(2 * self._mean + sigma2)

    def __mul__(self, k: Union[Scalar, RandomDistribution]) -> RandomDistribution:
        # Optimization: if X ~ LogNormal(mu, sigma) and k > 0,
        # then kX ~ LogNormal(mu + ln(k), sigma)
        if isinstance(k, (int, float)) and k > 0:
            return LogNormal(self._mean + math.log(k), self._std)
        return super().__mul__(k)

    def __repr__(self) -> str:
        return f"LogNormal(μ={self._mean}, σ={self._std})"


class Constant(RandomDistribution):
    """Deterministic distribution (Dirac delta)."""

    def __init__(self, value: float):
        self.value = value

    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        return torch.full((n,), self.value, device=device, dtype=torch.float32)

    def mean(self, **kwargs) -> float:
        return self.value

    def var(self, **kwargs) -> float:
        return 0.0

    def __repr__(self) -> str:
        return f"Constant({self.value})"


class BinaryOp(RandomDistribution):
    """
    Combines two distributions via a binary operator.

    .. assumption::
       Assumes the two distributions are statistically independent.
    """

    def __init__(self, left: RandomDistribution, right: RandomDistribution,
                 op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], symbol: str):
        self.left = left
        self.right = right
        self.op = op
        self.symbol = symbol

    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        return self.op(
            self.left.sample(n, device),
            self.right.sample(n, device),
        )

    def mean(self, **kwargs) -> float:
        if self.symbol in ("/",):
            # E[X/Y] != E[X]/E[Y]. Requires Taylor expansion or numerical integration.
            raise NotImplementedError(
                f"Analytic mean is not defined for division. Use .sample().mean() or approximate."
            )

        ml = self.left.mean(**kwargs)
        mr = self.right.mean(**kwargs)

        if self.symbol == "+":
            return ml + mr
        elif self.symbol == "-":
            return ml - mr
        elif self.symbol == "*":
            # Independence assumption: E[XY] = E[X]E[Y]
            return ml * mr
        else:
            raise NotImplementedError(f"Mean not implemented for {self.symbol}")

    def var(self, **kwargs) -> float:
        if self.symbol in ("/",):
            raise NotImplementedError("Analytic variance not defined for division.")

        vl = self.left.var(**kwargs)
        vr = self.right.var(**kwargs)

        if self.symbol in ("+", "-"):
            # Var(X ± Y) = Var(X) + Var(Y) (assuming independence)
            return vl + vr

        elif self.symbol == "*":
            # Var(XY) = E[X²Y²] - (E[X]E[Y])²
            # Assuming independence:
            # Var(XY) = (Var(X) + E[X]²)(Var(Y) + E[Y]²) - E[X]²E[Y]²
            #         = Var(X)Var(Y) + Var(X)E[Y]² + Var(Y)E[X]²
            ml = self.left.mean(**kwargs)
            mr = self.right.mean(**kwargs)
            return vl * vr + vl * (mr ** 2) + vr * (ml ** 2)

        else:
            raise NotImplementedError(f"Variance not implemented for {self.symbol}")

    def __repr__(self) -> str:
        return f"({self.left} {self.symbol} {self.right})"