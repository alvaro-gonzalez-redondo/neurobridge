from __future__ import annotations

from typing import Any, Optional

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


def smooth_spikes(spk_times, n_neurons=1, from_step=0, to_step=1, dt=1e-3, sigma=0.1):
    """
    Calcula la tasa de disparo suavizada de una población neuronal.
    
    Parámetros:
    -----------
    spk_times : torch.Tensor
        Índices de tiempo de disparo (en pasos de simulación, no en segundos).
    n_neurons : int
        Número total de neuronas de la población.
    from_step, to_step : int
        Rango de pasos de simulación a considerar.
    dt : float
        Tamaño del paso temporal en segundos.
    sigma : float
        Desviación estándar del suavizado gaussiano en segundos.

    Retorna:
    --------
    time : np.ndarray
        Vector temporal en segundos.
    smoothed_rate : np.ndarray
        Tasa de disparo suavizada en Hz.
    """
    n_steps = int(to_step - from_step)
    bin_size = 1  # en pasos de simulación

    # Filtrado de spikes fuera del rango
    valid = (spk_times >= 0) & (spk_times < n_steps)
    spk_times = spk_times[valid].to(torch.long)

    # Histograma global
    spike_counts = torch.bincount(spk_times, minlength=n_steps)
    rate = spike_counts.cpu().numpy() / (n_neurons * bin_size * dt)  # en Hz

    # Suavizado gaussiano
    sigma_bins = (sigma * 1e3) / (dt * 1e3)  # conversión de ms a bins
    smoothed_rate = gaussian_filter1d(rate, sigma=sigma_bins)

    # Eje temporal en segundos
    time = np.arange(n_steps) * dt

    return time, smoothed_rate


def to_tensor(x, dtype, device=None):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.to(dtype=dtype, device=device)
    return torch.tensor(x, dtype=dtype, device=device)


def resolve_param(param:Any, src_idx:torch.Tensor, tgt_idx:torch.Tensor, src:torch.Tensor, tgt:torch.Tensor, default_val:Any, dtype:Any):
    device = src.device
    if callable(param):
        src_pos = getattr(src,"positions",None)
        tgt_pos = getattr(tgt,"positions",None)
        src_sel = src_pos[src_idx] if src_pos is not None else None
        tgt_sel = tgt_pos[tgt_idx] if tgt_pos is not None else None
        out = param(src_idx, tgt_idx, src_sel, tgt_sel)
        return out.to(device=device, dtype=dtype)
    elif torch.is_tensor(param):
        return param.to(device=device,dtype=dtype)
    elif param is None:
        return torch.full((src_idx.numel(),), default_val, device=device, dtype=dtype)
    else:
        return torch.full((src_idx.numel(),), float(param), device=device, dtype=dtype)


def block_distance_connect(
    src_pos: torch.Tensor,
    tgt_pos: torch.Tensor,
    *,
    sigma: float | None = None,
    p_max: float = 1.0,
    max_distance: float | None = None,
    fanin: int | None = None,
    fanout: int | None = None,
    prob_func = None,  # callable(pre_block, pos_all, dists_block) -> probs_block
    block_src: int = 2048,
    block_tgt: int = 2048,
) -> tuple[torch.LongTensor, torch.LongTensor]:
    """
    Devuelve (src_idx, tgt_idx) según conectividad 'distance' evitando construir la matriz Npre×Npos completa.

    Modos (mutuamente excluyentes):
      - Probabilístico: usa sigma/p_max o prob_func (pre_block, pos_all, dists_block) -> [B, Npos].
      - fanin: K vecinos más cercanos por target (itera por bloques de targets).
      - fanout: K vecinos más cercanos por source (itera por bloques de sources).
    """
    device = src_pos.device
    Nsrc, Ntgt = src_pos.shape[0], tgt_pos.shape[0]

    # Exclusividad de modos
    modes = [m is not None for m in (fanin, fanout, sigma or prob_func)]
    if sum(modes) != 1:
        raise ValueError("distance: especifica exactamente uno de {probabilístico (sigma/prob_func), fanin, fanout}")

    src_parts: list[torch.Tensor] = []
    tgt_parts: list[torch.Tensor] = []

    INF = torch.finfo(src_pos.dtype).max

    if fanin is not None:
        if fanin > Nsrc:
            raise ValueError("fanin no puede exceder pre.size")
        # Itera por bloques de targets (pos)
        for j0 in range(0, Ntgt, block_tgt):
            j1 = min(j0 + block_tgt, Ntgt)
            tgt_block = tgt_pos[j0:j1]                           # [Bpos, D]
            dists = torch.cdist(src_pos, tgt_block)               # [Npre, Bpos]
            if max_distance is not None:
                dists = dists.masked_fill_(dists > max_distance, INF)
            vals, idx = torch.topk(dists, k=fanin, dim=0, largest=False)  # [fanin, Bpos]
            # Filtra entradas inválidas (INF si no hay suficientes dentro del radio)
            valid = vals.isfinite()                                # [fanin, Bpos]
            if valid.any():
                col_idx = torch.arange(j0, j1, device=device).repeat(fanin, 1)
                src = idx[valid]
                tgt = col_idx[valid]
                src_parts.append(src.reshape(-1))
                tgt_parts.append(tgt.reshape(-1))

    elif fanout is not None:
        if fanout > Ntgt:
            raise ValueError("fanout no puede exceder pos.size")
        # Itera por bloques de sources (pre)
        for i0 in range(0, Nsrc, block_src):
            i1 = min(i0 + block_src, Nsrc)
            src_block = src_pos[i0:i1]                             # [Bpre, D]
            dists = torch.cdist(src_block, tgt_pos)               # [Bpre, Npos]
            if max_distance is not None:
                dists = dists.masked_fill_(dists > max_distance, INF)
            vals, idx = torch.topk(dists, k=fanout, dim=1, largest=False)  # [Bpre, fanout]
            valid = vals.isfinite()                                # [Bpre, fanout]
            if valid.any():
                row_idx = torch.arange(i0, i1, device=device).unsqueeze(1).repeat(1, fanout)
                src = row_idx[valid]
                tgt = idx[valid]
                src_parts.append(src.reshape(-1))
                tgt_parts.append(tgt.reshape(-1))

    else:
        # Modo probabilístico (sigma/p_max o prob_func)
        if prob_func is None and sigma is None:
            raise ValueError("distance probabilístico: define sigma o prob_func")
        for i0 in range(0, Nsrc, block_src):
            i1 = min(i0 + block_src, Nsrc)
            src_block = src_pos[i0:i1]                             # [Bpre, D]
            dists = torch.cdist(src_block, tgt_pos)               # [Bpre, Npos]
            if prob_func is not None:
                # El prob_func debe devolver matriz de probabilidades [Bpre, Npos]
                probs = prob_func(src_block, tgt_pos, dists)
            else:
                # Gaussiana p(d) = p_max * exp(-d^2 / (2 sigma^2))
                probs = p_max * torch.exp(-(dists**2) / (2.0 * (sigma**2)))
            if max_distance is not None:
                probs = probs * (dists <= max_distance)

            # Muestreo Bernoulli por bloque
            mask = torch.rand_like(probs) < probs                  # [Bpre, Npos]
            if mask.any():
                src, tgt = mask.nonzero(as_tuple=True)             # locales al bloque
                src_parts.append(src + i0)
                tgt_parts.append(tgt)

    if not src_parts:
        # Sin conexiones
        return (torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device))

    return torch.cat(src_parts), torch.cat(tgt_parts)