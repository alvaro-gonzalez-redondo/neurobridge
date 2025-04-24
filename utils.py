from __future__ import annotations

from . import globals

import logging
import sys
import os

import torch.distributed as dist

import matplotlib
from matplotlib import pyplot as plt


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


# ANSI 24-bit escape codes para C0–C9 (matplotlib default color cycle)
def _rgb_escape(r, g, b):
    return f"\033[38;2;{r};{g};{b}m"


MATPLOTLIB_RGB = [
    (31, 119, 180),   # C0
    (255, 127, 14),   # C1
    (44, 160, 44),    # C2
    (214, 39, 40),    # C3
    (148, 103, 189),  # C4
    (140, 86, 75),    # C5
    (227, 119, 194),  # C6
    (127, 127, 127),  # C7
    (188, 189, 34),   # C8
    (23, 190, 207),   # C9
]
RESET = "\033[0m"


class _RankColorFormatter(logging.Formatter):
    def __init__(self, rank: int, fmt: str):
        super().__init__(fmt)
        rgb = MATPLOTLIB_RGB[rank % 10]
        self.color = _rgb_escape(*rgb)

    def format(self, record):
        message = super().format(record)
        return f"{self.color}{message}{RESET}"


def _setup_logger(rank: int) -> logging.Logger:
    logger = logging.getLogger(f"Rank{rank}")
    logger.setLevel(logging.INFO)

    # Formato base
    fmt = '%(asctime)s - [%(name)s] %(message)s'

    # Salida a fichero (sin color)
    file_formatter = logging.Formatter(fmt)
    fh = logging.FileHandler(f"log_rank{rank}.txt")
    fh.setFormatter(file_formatter)
    logger.addHandler(fh)

    # Salida a consola (con color según rank)
    console_formatter = _RankColorFormatter(rank, fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(console_formatter)
    logger.addHandler(ch)

    return logger


def log(msg: str) -> None:
    if globals.logger:
        globals.logger.info(msg)
    else:
        print(msg)


def log_error(msg: str) -> None:
    if globals.logger:
        globals.logger.error(msg)
    else:
        print(f"ERROR: {msg}")


def can_display_graphics():
    # Lista de backends que podrían ser interactivos
    interactive_backends = [backend.lower() for backend in [
        'GTK3Agg', 'GTK3Cairo', 'MacOSX', 'nbAgg', 'Qt4Agg',
        'Qt5Agg', 'QtAgg', 'TkAgg', 'TkCairo', 'WebAgg', 'WX', 'WXAgg'
    ]]
    backend = matplotlib.get_backend()

    # En Unix, requiere DISPLAY; en Windows/Mac suele funcionar siempre
    has_display = (
        sys.platform.startswith('win') or
        sys.platform == 'darwin' or
        os.environ.get("DISPLAY") is not None
    )

    return backend.lower() in interactive_backends and has_display


def show_or_save_plot(filename="output.png", log=None):
    if can_display_graphics():
        plt.show()
    else:
        plt.savefig(filename)
        if log:
            log(f"Gráfico guardado como '{filename}'")
        else:
            print(f"Gráfico guardado como '{filename}'")