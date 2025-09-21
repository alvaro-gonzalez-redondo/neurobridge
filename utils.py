from __future__ import annotations

from typing import Any

from . import globals

import logging
import sys
import os

import torch
import torch.distributed as dist

import numpy as np

import matplotlib
from matplotlib import pyplot as plt

from scipy.ndimage import gaussian_filter1d


def _compute_parameter(
    param: Any, idx_pre: torch.Tensor, idx_post: torch.Tensor, device: str
) -> torch.Tensor:
    """Compute per-connection parameter values based on various input types.

    Parameters
    ----------
    param : Any
        The parameter specification, which can be:
            - A scalar: Used for all connections.
            - A tensor: Used directly if it matches the number of connections.
            - A list: Converted to a tensor.
            - A function: Called with (idx_pre, idx_post) to compute values.
    idx_pre : torch.Tensor
        Indices of pre-synaptic neurons for each connection.
    idx_post : torch.Tensor
        Indices of post-synaptic neurons for each connection.

    Returns
    -------
    torch.Tensor
        Tensor of parameter values for each connection.

    Raises
    ------
    ValueError
        If the tensor dimensions don't match the number of connections.
    TypeError
        If a function parameter doesn't return a tensor.
    """
    n = len(idx_pre)

    if callable(param):
        values = param(idx_pre, idx_post)
        if not isinstance(values, torch.Tensor):
            raise TypeError("Functions must return a tensor.")
        if values.shape[0] != n:
            raise ValueError(
                f"Returned tensor must have size {n}, but it has size {values.shape[0]}."
            )
        return values.to(device=device)

    elif isinstance(param, torch.Tensor):
        if param.numel() == 1:
            return torch.full((n,), param.item(), device=device)
        if param.numel() != n:
            raise ValueError(
                f"Expected a tensor of length {n}, got {param.numel()}."
            )
        return param.to(device=device)

    elif isinstance(param, list):
        param = torch.tensor(param, device=device)
        return _compute_parameter(param, idx_pre, idx_post, device)

    else:  # Scalar
        return torch.full((n,), float(param), device=device)


def is_distributed() -> bool:
    """Check if the simulation is running in distributed mode.

    Returns
    -------
    bool
        True if PyTorch distributed is available and initialized.
    """
    return dist.is_available() and dist.is_initialized()


def can_use_torch_compile() -> bool:
    # Si no hay CUDA, descartamos
    if not torch.cuda.is_available():
        return False

    # Verificar capability mínima (Ampere o superior recomendado)
    major, minor = torch.cuda.get_device_capability()
    if major < 7:
        return False

    # Intentar importar triton
    try:
        import triton
        import triton.language as tl  # forzar comprobación
    except Exception:
        return False

    # Si todo bien, podemos usar torch.compile
    return True


def _rgb_escape(r, g, b):
    """Generate ANSI 24-bit color escape code.

    Parameters
    ----------
    r : int
        Red component (0-255).
    g : int
        Green component (0-255).
    b : int
        Blue component (0-255).

    Returns
    -------
    str
        ANSI escape code string for the specified RGB color.
    """
    return f"\033[38;2;{r};{g};{b}m"


MATPLOTLIB_RGB = [
    (31, 119, 180),  # C0
    (255, 127, 14),  # C1
    (44, 160, 44),  # C2
    (214, 39, 40),  # C3
    (148, 103, 189),  # C4
    (140, 86, 75),  # C5
    (227, 119, 194),  # C6
    (127, 127, 127),  # C7
    (188, 189, 34),  # C8
    (23, 190, 207),  # C9
]
RESET = "\033[0m"


class RankColorFormatter(logging.Formatter):
    """A logging formatter that colors messages based on the rank.

    This formatter adds ANSI color codes to log messages, with the color
    determined by the rank (typically the GPU index). It uses the default
    matplotlib color cycle for consistent color mapping.

    Attributes
    ----------
    color : str
        The ANSI color escape code based on the rank.
    """

    def __init__(self, rank: int, fmt: str):
        """Initialize the formatter with rank-based coloring.

        Parameters
        ----------
        rank : int
            The rank (e.g., GPU index) to determine the color.
        fmt : str
            The log format string.
        """
        super().__init__(fmt)
        rgb = MATPLOTLIB_RGB[rank % 10]
        self.color = _rgb_escape(*rgb)

    def format(self, record):
        """Format the log record with color based on rank.

        Parameters
        ----------
        record : logging.LogRecord
            The log record to format.

        Returns
        -------
        str
            The formatted log message with ANSI color codes.
        """
        message = super().format(record)
        return f"{self.color}{message}{RESET}"


def _setup_logger(rank: int) -> logging.Logger:
    """Set up a logger with console and file outputs.

    Creates a logger that outputs to both a rank-specific log file and
    the console. Console output is colored based on the rank.

    Parameters
    ----------
    rank : int
        The rank (GPU index) for which to set up the logger.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(f"Rank{rank}")
    logger.setLevel(logging.INFO)

    # Base format
    fmt = "%(asctime)s - [%(name)s] %(message)s"

    # Output to file (no color)
    file_formatter = logging.Formatter(fmt)
    fh = logging.FileHandler(f"log_rank{rank}.txt")
    fh.setFormatter(file_formatter)
    logger.addHandler(fh)

    # Output to console (color according to rank)
    console_formatter = RankColorFormatter(rank, fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(console_formatter)
    logger.addHandler(ch)

    return logger


def log(msg: str) -> None:
    """Log an informational message through the simulator logger.

    If the simulator logger is not initialized, falls back to print.

    Parameters
    ----------
    msg : str
        Message to log.
    """
    if globals.logger:
        globals.logger.info(msg)
    else:
        print(msg)


def log_error(msg: str) -> None:
    """Log an error message through the simulator logger.

    If the simulator logger is not initialized, falls back to print with
    an "ERROR:" prefix.

    Parameters
    ----------
    msg : str
        Error message to log.
    """
    if globals.logger:
        globals.logger.error(msg)
    else:
        print(f"ERROR: {msg}")


def can_display_graphics():
    """Check if the current environment can display graphics.

    Determines if matplotlib can show interactive figures based on the
    current backend and display environment.

    Returns
    -------
    bool
        True if graphics display is available, False otherwise.
    """

    # Potentially interactive backends
    interactive_backends = [
        backend.lower()
        for backend in [
            "GTK3Agg",
            "GTK3Cairo",
            "MacOSX",
            "nbAgg",
            "Qt4Agg",
            "Qt5Agg",
            "QtAgg",
            "TkAgg",
            "TkCairo",
            "WebAgg",
            "WX",
            "WXAgg",
        ]
    ]
    backend = matplotlib.get_backend()

    # In Unix DISPLAY is required; in Windows/Mac it usually works
    has_display = (
        sys.platform.startswith("win")
        or sys.platform == "darwin"
        or os.environ.get("DISPLAY") is not None
    )

    return backend.lower() in interactive_backends and has_display


def show_or_save_plot(filename="output.png", log=None):
    """Display or save a matplotlib figure depending on environment capabilities.

    If the environment supports interactive graphics, shows the figure.
    Otherwise, saves it to the specified filename.

    Parameters
    ----------
    filename : str, optional
        Filename to save the figure if display is not available, by default "output.png".
    log : callable, optional
        Logging function to use for informing about the saved file, by default None.
        If None, print is used.
    """
    if can_display_graphics():
        plt.show()
    else:
        plt.savefig(filename)
        if log:
            log(f"Plot saved as '{filename}'")
        else:
            print(f"Plot saved as '{filename}'")


def smooth_spikes(spk_times, n_neurons=1, from_time=0.0, to_time=1.0, sigma=5):
    # Parámetros
    dt = 1.0  # tamaño del paso de simulación en ms
    duration = int(to_time - from_time)
    num_neurons = n_neurons
    bin_size = 1  # en pasos de simulación

    # Histograma global
    spike_counts = torch.bincount(spk_times.to(torch.int64), minlength=duration)
    rate = spike_counts.numpy() / num_neurons / (bin_size * dt / 1000)  # en Hz

    # Suavizado
    time = np.arange(duration) * dt
    smoothed_rate = gaussian_filter1d(rate, sigma=sigma)

    return time, smoothed_rate