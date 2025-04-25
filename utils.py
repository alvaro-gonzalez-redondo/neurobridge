from __future__ import annotations

from . import globals

import logging
import sys
import os

import torch.distributed as dist

import matplotlib
from matplotlib import pyplot as plt


def is_distributed() -> bool:
    """Check if the simulation is running in distributed mode.

    Returns
    -------
    bool
        True if PyTorch distributed is available and initialized.
    """
    return dist.is_available() and dist.is_initialized()


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


class _RankColorFormatter(logging.Formatter):
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
    console_formatter = _RankColorFormatter(rank, fmt)
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
