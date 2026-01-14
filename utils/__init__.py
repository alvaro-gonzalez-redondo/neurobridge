from __future__ import annotations

from typing import Any, Optional

from .. import globals

from .random_utils import RandomDistribution, Uniform, UniformInt, Normal, LogNormal, Constant
from .viz_client import VisualizerClient
from ..monitors import SpikeMonitor
from .sensorial_manifold import (StateMachine, catmull_rom_segment, SplineSegment, TrajectoryRunner, poisson_disk_sampling, RBFSpace, MultiScaleRBFEncoder, SensoryTrajectoryGenerator, ContinuousOUNoise)

import logging
import sys
import os

import torch
import torch.distributed as dist

import numpy as np

import matplotlib
from matplotlib import pyplot as plt

from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
import umap


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
    """Resolve connection parameters to tensors.

    Supports multiple input types:
    - Tuple (low, high): Uniform distribution for floats, UniformInt for ints
    - RandomDistribution instance: Custom distribution
    - Callable: Lambda function (pre_idx, tgt_idx, pre_pos, tgt_pos) -> values
    - Tensor: Direct tensor
    - Scalar: Broadcast to all connections
    - None: Use default_val

    Parameters
    ----------
    param : Any
        Parameter specification (see above).
    src_idx : torch.Tensor
        Source neuron indices.
    tgt_idx : torch.Tensor
        Target neuron indices.
    src : torch.Tensor
        Source neuron group.
    tgt : torch.Tensor
        Target neuron group.
    default_val : Any
        Default value if param is None.
    dtype : torch.dtype
        Target dtype for the output tensor.

    Returns
    -------
    torch.Tensor
        Resolved parameter tensor of shape (num_connections,).
    """
    from .random_utils import RandomDistribution, Uniform, UniformInt

    device = src.device
    n_connections = src_idx.numel()

    # Handle tuple: (low, high) → distribution
    if isinstance(param, tuple):
        if len(param) != 2:
            raise ValueError(f"Tuple parameter must have 2 elements (low, high), got {len(param)}")
        low, high = param

        # Use UniformInt for integer dtypes, Uniform for floats
        if dtype in (torch.long, torch.int, torch.int32, torch.int64):
            dist = UniformInt(int(low), int(high))
        else:
            dist = Uniform(float(low), float(high))

        return dist.sample(n_connections, device).to(dtype=dtype)

    # Handle RandomDistribution instances
    elif isinstance(param, RandomDistribution):
        return param.sample(n_connections, device).to(dtype=dtype)

    # Handle callable (lambda function)
    elif callable(param):
        src_pos = getattr(src,"positions",None)
        tgt_pos = getattr(tgt,"positions",None)
        src_sel = src_pos[src_idx] if src_pos is not None else None
        tgt_sel = tgt_pos[tgt_idx] if tgt_pos is not None else None
        out = param(src_idx, tgt_idx, src_sel, tgt_sel)
        return out.to(device=device, dtype=dtype)

    # Handle tensor
    elif torch.is_tensor(param):
        return param.to(device=device,dtype=dtype)

    # Handle None (use default)
    elif param is None:
        return torch.full((n_connections,), default_val, device=device, dtype=dtype)

    # Handle scalar
    else:
        return torch.full((n_connections,), float(param), device=device, dtype=dtype)


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


import numpy as np

def mean_phase_sorting(spk_times, spk_indices, T, n_neurons=None):
    """
    Ordena las neuronas por su fase media de disparo en un ciclo de duración T.

    Parameters
    ----------
    spk_times : array-like, shape (N,)
        Tiempos de los spikes.
    spk_indices : array-like, shape (N,)
        Índices de las neuronas correspondientes.
    T : float
        Duración del ciclo periódico.
    n_neurons : int, opcional
        Número total de neuronas. Si se da, se valida que los índices estén en [0, n_neurons).

    Returns
    -------
    order : ndarray (int)
        Índices de neurona ordenados por fase media.
    phases : ndarray (float)
        Fase media (en radianes) de cada neurona (mismo tamaño que `order`).
    """

    spk_times = np.asarray(spk_times, dtype=float)
    spk_indices = np.asarray(spk_indices, dtype=int)

    # ---- 0. Casos degenerados ----
    if T <= 0:
        raise ValueError("T debe ser positivo.")

    if spk_times.size == 0 or spk_indices.size == 0:
        # No hay spikes -> no hay fases
        if n_neurons is None:
            # No sabemos ni cuántas neuronas había, devolvemos vacío
            return np.array([], dtype=int), np.array([], dtype=float)
        else:
            # Sabemos cuántas neuronas hay pero ninguna dispara:
            # devolvemos todas en orden 0..n_neurons-1 con fase NaN
            order = np.arange(n_neurons, dtype=int)
            phases = np.full(n_neurons, np.nan, dtype=float)
            return order, phases

    if spk_times.shape != spk_indices.shape:
        raise ValueError("spk_times y spk_indices deben tener la misma forma 1D.")

    # ---- 1. Determinar número de neuronas y validar índices ----
    if n_neurons is None:
        n_neurons = int(spk_indices.max()) + 1
    else:
        if spk_indices.min() < 0 or spk_indices.max() >= n_neurons:
            raise ValueError(
                f"Indices de neurona fuera de rango: deben estar en [0, {n_neurons-1}]."
            )

    # ---- 2. Convertir tiempos en ángulos de fase dentro del ciclo ----
    theta = (2.0 * np.pi * (spk_times % T) / T)

    vx = np.cos(theta)
    vy = np.sin(theta)

    # ---- 3. Acumular vectores por neurona ----
    sum_x = np.zeros(n_neurons, dtype=float)
    sum_y = np.zeros(n_neurons, dtype=float)

    np.add.at(sum_x, spk_indices, vx)
    np.add.at(sum_y, spk_indices, vy)

    # ---- 4. Calcular fase media (atan2 y envolver a [0, 2π)) ----
    phases = np.mod(np.arctan2(sum_y, sum_x), 2.0 * np.pi)

    # Ojo: neuronas sin spikes tienen (sum_x,sum_y) = (0,0),
    # atan2(0,0) = 0 → fase 0. Si prefieres NaN:
    # mask_zero = (sum_x == 0) & (sum_y == 0)
    # phases[mask_zero] = np.nan

    # ---- 5. Orden según fase ----
    order = np.argsort(phases)

    return order, phases


def plot_neural_trajectory_pca(
    spk_data: torch.Tensor,
    n_neurons: int = None,
    t_max: float = None,
    dt: float = 1e-3,
    sigma_ms: float = 200.0,
    plot_3d: bool = True,
    title: str = ""
):
    """
    Genera un gráfico de PCA de la trayectoria neuronal a partir de spikes.

    Args:
        spk_data: tensor Torch (columnas spk_times, spk_index). Los tiempos deben estar en segundos.
        n_neurons: Número total de neuronas (si es None, se infiere del índice máximo).
        t_max: Tiempo total de simulación (si es None, se infiere del último spike).
        dt: Paso de tiempo para la discretización (segundos). Default 1ms.
        sigma_ms: Ancho del kernel Gaussiano para suavizar la tasa (milisegundos).
                  Valores entre 20ms y 100ms suelen funcionar bien para ver dinámicas.
        plot_3d: Si True, incluye un subplot 3D.
    
    --- Ejemplo de uso ---
    plot_neural_trajectory_pca(spk_data, sigma_ms=50.0)
    """
    
    # 1. Desempaquetar datos
    spk_data = spk_data.numpy(force=True)
    spk_steps, spk_indices = spk_data[:, 1], spk_data[:, 0]
    spk_times = spk_steps*1e-3

    if len(spk_indices)==0:
        print("No hay spikes, nada que analizar. Abortando...")
        return
    
    # Inferir parámetros si no se dan
    if n_neurons is None:
        n_neurons = int(np.max(spk_indices)) + 1
    if t_max is None:
        t_max = np.max(spk_times)

    # 2. Crear matriz de spikes (Binning)
    n_bins = int(t_max / dt) + 1
    time_axis = np.linspace(0, t_max, n_bins)
    
    # Matriz [Time, Neurons] (sklearn prefiere samples en filas)
    spike_matrix = np.zeros((n_bins, n_neurons))
    
    # Convertir tiempos a índices de bin
    bin_indices = (spk_times / dt).astype(int)
    
    # Clip por seguridad (por si algún spike cae justo en t_max)
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    # Rellenar matriz (acumulando spikes)
    # Forma rápida usando np.add.at para sumar spikes en el mismo bin
    np.add.at(spike_matrix, (bin_indices, spk_indices), 1)

    # 3. Suavizado (Smoothing) -> Obtener Rates
    # Convertir sigma de ms a bins
    sigma_bins = (sigma_ms * 1e-3) / dt
    
    # Aplicar filtro gaussiano sobre el eje temporal (axis 0)
    # Esto convierte los 0s y 1s en una curva suave de probabilidad/tasa
    rates_matrix = gaussian_filter1d(spike_matrix, sigma=sigma_bins, axis=0)

    # 4. Normalización (Z-Score)
    # Importante: Restar la media y dividir por std para que todas las neuronas contribuyan igual.
    # Neuronas mudas (std=0) se quedan en 0 para no dividir por cero.
    means = np.mean(rates_matrix, axis=0)
    stds = np.std(rates_matrix, axis=0)
    valid_neurons = stds > 0
    
    normalized_rates = np.zeros_like(rates_matrix)
    normalized_rates[:, valid_neurons] = (rates_matrix[:, valid_neurons] - means[valid_neurons]) / stds[valid_neurons]

    # 5. Aplicar PCA
    pca = PCA(n_components=3)
    projected = pca.fit_transform(normalized_rates) # [n_bins, 3]
    
    # Varianza explicada (para saber si el plot es significativo)
    var_exp = pca.explained_variance_ratio_
    print(f"Varianza explicada por PC1, PC2, PC3: {var_exp * 100}")
    print(f"Total varianza explicada (3D): {np.sum(var_exp)*100:.2f}%")

    # 6. Plotting
    fig = plt.figure(figsize=(14, 6))
    fig.canvas.manager.set_window_title(title)
    
    # Mapa de color basado en el tiempo
    colors = time_axis
    
    # --- Subplot 2D (PC1 vs PC2) ---
    ax1 = fig.add_subplot(1, 2, 1)
    sc1 = ax1.scatter(projected[:, 0], projected[:, 1], c=colors, cmap='viridis', s=2, alpha=0.6)
    ax1.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}%)")
    ax1.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}%)")
    ax1.set_title("Trayectoria Neuronal 2D")
    ax1.grid(True, alpha=0.3)
    
    # Barra de color
    cbar = plt.colorbar(sc1, ax=ax1)
    cbar.set_label("Tiempo (s)")

    # --- Subplot 3D (PC1 vs PC2 vs PC3) ---
    if plot_3d:
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        sc2 = ax2.scatter(projected[:, 0], projected[:, 1], projected[:, 2], 
                          c=colors, cmap='viridis', s=2, alpha=0.6)
        
        # Conectar puntos con una línea tenue para ver mejor el flujo
        # (Opcional, a veces ensucia si hay mucho ruido)
        ax2.plot(projected[:, 0], projected[:, 1], projected[:, 2], color='gray', alpha=0.2, lw=0.5)

        ax2.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}%)")
        ax2.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}%)")
        ax2.set_zlabel(f"PC3 ({var_exp[2]*100:.1f}%)")
        ax2.set_title("Trayectoria Neuronal 3D")
    
    plt.tight_layout()

    return pca, projected


def plot_neural_trajectory_umap(
    spk_data: torch.Tensor,
    n_neighbors: int = 50,
    min_dist: float = 0.1,
    n_components: int = 2,
    t_max: float = None,
    dt: float = 1e-3,
    sigma_ms: float = 50.0,
    random_state: int = 42,
    color_by: np.ndarray = None, 
    title: str = "",
    title_suffix: str = ""
):
    """
    Genera una proyección UMAP de la trayectoria neuronal.

    Args:
        spk_data: tensor Torch (columnas spk_times, spk_index).
        n_neighbors: Parámetro clave de UMAP. 
                     Valores bajos (5-15) enfocan en estructura local (bursts individuales).
                     Valores altos (50-200) enfocan en estructura global (el ciclo lento).
                     Para SFA/Trayectorias, usa valores altos (30-100).
        min_dist: Cuán apretados están los puntos (0.0 a 0.99).
        n_components: 2 para 2D, 3 para 3D.
        t_max: Tiempo total simulación.
        dt: Paso de tiempo binning.
        sigma_ms: Ancho del suavizado gaussiano. ¡CRÍTICO! 
                  Para ver trayectorias lentas con bursting, usa 50ms-200ms.
        random_state: Semilla para reproducibilidad (UMAP es estocástico).
        color_by: Array opcional del mismo tamaño que los bins temporales para colorear
                  (ej. el valor de tu señal senoidal de entrada). Si es None, usa el tiempo.

    --- Ejemplo de uso ---
    plot_neural_trajectory_umap(spk_data, n_neighbors=50, sigma_ms=100)
    """
    
    # --- 1. Preprocesado (Idéntico a PCA) ---
    spk_data = spk_data.numpy(force=True)
    spk_steps, spk_indices = spk_data[:, 1], spk_data[:, 0]
    spk_times = spk_steps*1e-3

    print(f"{title}:")
    if len(spk_indices) == 0:
        print("No hay spikes para analizar.")
        return

    n_neurons = int(np.max(spk_indices)) + 1
    if t_max is None:
        t_max = np.max(spk_times)

    # Binning
    n_bins = int(t_max / dt) + 1
    spike_matrix = np.zeros((n_bins, n_neurons))
    bin_indices = (spk_times / dt).astype(int)
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    np.add.at(spike_matrix, (bin_indices, spk_indices), 1)

    # Smoothing (Rates)
    sigma_bins = (sigma_ms * 1e-3) / dt
    rates_matrix = gaussian_filter1d(spike_matrix, sigma=sigma_bins, axis=0)

    # Z-Score Normalization
    means = np.mean(rates_matrix, axis=0)
    stds = np.std(rates_matrix, axis=0)
    valid = stds > 0
    normalized_rates = np.zeros((n_bins, np.sum(valid)))
    normalized_rates = (rates_matrix[:, valid] - means[valid]) / stds[valid]

    # --- 2. UMAP ---
    print(f"Ejecutando UMAP sobre matriz {normalized_rates.shape}...")
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric='euclidean', # 'cosine' a veces va mejor para spikes, prueba ambos
        random_state=random_state
    )
    
    embedding = reducer.fit_transform(normalized_rates)

    # --- 3. Plotting ---
    fig = plt.figure(figsize=(10, 8))
    fig.canvas.manager.set_window_title(title)
    
    # Definir color
    if color_by is not None:
        # Asegurarse que coincida en longitud (puede variar por 1 frame por redondeo)
        if len(color_by) != n_bins:
            # Interpolación simple para ajustar tamaño
            from scipy.interpolate import interp1d
            f = interp1d(np.linspace(0, 1, len(color_by)), color_by)
            c_vals = f(np.linspace(0, 1, n_bins))
        else:
            c_vals = color_by
        c_label = "Valor Señal Externa"
    else:
        c_vals = np.linspace(0, t_max, n_bins)
        c_label = "Tiempo (s)"

    if n_components == 2:
        ax = fig.add_subplot(111)
        sc = ax.scatter(embedding[:, 0], embedding[:, 1], c=c_vals, cmap='viridis', s=3, alpha=0.7)
        # Línea tenue para ver la trayectoria
        ax.plot(embedding[:, 0], embedding[:, 1], c='black', alpha=0.1, lw=0.5)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        
    elif n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c=c_vals, cmap='viridis', s=3, alpha=0.7)
        ax.plot(embedding[:, 0], embedding[:, 1], embedding[:, 2], c='black', alpha=0.1, lw=0.5)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_zlabel('UMAP 3')

    plt.colorbar(sc, label=c_label)
    plt.title(f"Trayectoria Neuronal (UMAP)\nneighbors={n_neighbors}, dist={min_dist} {title_suffix}")
    plt.tight_layout()
    
    return embedding


def plot_spikes(current_step, spike_monitor:SpikeMonitor, title_suffix:str="", phase_sorting_t:float=-1):
    fig, ax0 = plt.subplots()
    ax1 = ax0.twinx()
    id_sum = 0
    for idx, group in enumerate(spike_monitor.groups):
        label = group.name
        spikes = spike_monitor.get_spike_tensor(idx).cpu()
        print(f"\nAnalysis of {label} population:")
        analyze_network_state(spikes, group.size)

        # Graficas
        spk_steps, spk_neurons = spikes[:, 1], spikes[:, 0]
        spk_times = spk_steps*1e-3

        if phase_sorting_t > 0:
            order, _ = mean_phase_sorting(spk_times, spk_neurons, T=phase_sorting_t)
            inv_order = np.empty_like(order)
            inv_order[order] = np.arange(len(order))
            spk_ordered_indices = inv_order[spk_neurons]
            ax1.scatter(spk_times, spk_ordered_indices+id_sum, s=1, label=label, c=f"C{idx}")
        else:
            ax1.scatter(spk_times, spk_neurons+id_sum, s=1, label=label, c=f"C{idx}")

        try:
            n_neurons = int(spike_monitor.filters[idx].nonzero(as_tuple=True)[0][-1]) + 1
        except:
            n_neurons = 1
        id_sum += n_neurons
        times, rate = smooth_spikes(spk_steps, n_neurons=n_neurons, to_step=current_step, sigma=0.1)
        ax0.plot(times, rate, c=f"C{idx}")
    
    ax1.legend(loc="lower right")
    plt.title(f"Spikes from different subpopulations\n{title_suffix}")
    plt.xlabel("Time (seconds)")
    ax0.set_ylabel("Spiking rate (Hz)")
    ax1.set_ylabel("Neuron ID")


def analyze_network_state(
    spike_tensor: torch.Tensor,
    n_neurons: int,
    dt: float = 1e-3,
    bin_size: float = 0.01
):
    """
    spike_tensor: Tensor [N_spikes, 2] donde col 0 = neuron_id, col 1 = step
    Devuelve:
        cv_list : lista de CVs válidos
        off_diag : correlaciones fuera de la diagonal (tensor)
    """

    # ---------- Validaciones iniciales ----------
    if spike_tensor.numel() == 0 or spike_tensor.shape[0] == 0:
        print("No spikes: CV = NaN, corr = NaN")
        return [], torch.tensor([])

    if spike_tensor.ndim != 2 or spike_tensor.shape[1] != 2:
        raise ValueError("spike_tensor debe ser [N_spikes, 2]")

    neuron_ids = spike_tensor[:, 0].long()
    spike_steps = spike_tensor[:, 1].float()

    # Validar rango de neuron_ids
    if neuron_ids.min() < 0 or neuron_ids.max() >= n_neurons:
        raise ValueError(f"neuron_ids fuera de rango [0, {n_neurons-1}]")

    # Convertir steps a tiempo
    spike_times = spike_steps * dt

    # ----------
    # 1. CV de ISI
    # ----------
    cv_list = []

    for i in range(n_neurons):
        times = spike_times[neuron_ids == i]
        if len(times) > 2:
            isi = torch.diff(times)
            mean_isi = isi.mean()
            std_isi = isi.std()

            # Prevenir divisiones por cero
            if mean_isi > 0:
                cv_list.append((std_isi / mean_isi).item())

    if len(cv_list) == 0:
        mean_cv = float('nan')
    else:
        mean_cv = float(np.mean(cv_list))

    print(f"Mean CV (Target ~1.0): {mean_cv:.3f}")

    # ----------
    # 2. Correlación promedio
    # ----------

    if spike_times.numel() == 0:
        print("No spikes → sin correlación.")
        return cv_list, torch.tensor([])

    t_max = spike_times.max().item()

    if t_max < 0:
        print("t_max < 0, algo raro en los datos: corr = NaN.")
        return cv_list, torch.tensor([])

    # Si no hay suficiente duración para bins
    n_bins = max(int(t_max / bin_size) + 1, 1)

    spike_counts = torch.zeros((n_neurons, n_bins), dtype=torch.float32)

    bin_indices = (spike_times / bin_size).long()

    # Clampear para evitar index_put out-of-bounds
    bin_indices = torch.clamp(bin_indices, 0, n_bins - 1)

    # Rellenar matriz de conteos
    indices = torch.stack([neuron_ids, bin_indices], dim=0)
    values = torch.ones_like(neuron_ids, dtype=torch.float32)
    spike_counts.index_put_(tuple(indices), values, accumulate=True)

    # Centrar y normalizar
    means = spike_counts.mean(dim=1, keepdim=True)
    stds = spike_counts.std(dim=1, keepdim=True)

    valid = (stds.squeeze(1) > 0)

    if valid.sum() < 2:
        print("Menos de 2 neuronas activas → no se puede calcular correlación.")
        return cv_list, torch.tensor([])

    spike_counts = spike_counts[valid]
    means = means[valid]
    stds = stds[valid]

    spike_counts_norm = (spike_counts - means) / (stds + 1e-8)

    # Correlación
    corr_matrix = (spike_counts_norm @ spike_counts_norm.T) / max(n_bins - 1, 1)

    # Extraer off-diagonal
    mask_off = ~torch.eye(corr_matrix.shape[0], dtype=bool)
    off_diag = corr_matrix[mask_off]

    if off_diag.numel() == 0:
        mean_corr = float('nan')
    else:
        mean_corr = off_diag.mean().item()

    print(f"Mean Pairwise Correlation (Target ~0.0): {mean_corr:.4f}")

    return cv_list, off_diag


def plot_sparse_as_dense(
    conn,
    n_pre: int,
    n_post: int,
    *,
    ax: Optional[plt.Axes] = None,
    figsize=(6, 6),
    cmap="viridis",
    title: str | None = None,
    show_colorbar: bool = True,
    vmin=None,  # <--- Argumento añadido
    vmax=None,  # <--- Argumento añadido
):
    """
    Convierte la conectividad sparse a matriz densa y la dibuja.
    NO recomendado para matrices grandes.
    """
    idx_pre = conn.idx_pre.detach().cpu()
    idx_pos = conn.idx_pos.detach().cpu()
    w = conn.weight.detach().cpu()

    mat = torch.zeros((n_pre, n_post))
    mat[idx_pre, idx_pos] = w

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True

    # Pasamos vmin y vmax a imshow para fijar la escala de color
    im = ax.imshow(
        mat.T,
        origin="upper",
        cmap=cmap,
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xlabel("Neuronas pre")
    ax.set_ylabel("Neuronas post")

    if title is not None:
        ax.set_title(title)

    if show_colorbar:
        # Usamos ax.figure.colorbar para ser más robustos que plt.colorbar
        ax.figure.colorbar(im, ax=ax, label="Peso sináptico")

    if created_fig:
        plt.tight_layout()
        plt.show()

    return im


def plot_sparse_connectivity(
    conn,
    n_pre: int | None = None,
    n_post: int | None = None,
    *,
    ax: Optional[plt.Axes] = None,
    figsize=(6, 6),
    cmap="viridis",
    s=4,
    show_colorbar=True,
    title: str | None = None,
    background_color="black",
    vmin=None,  # <--- Nuevo argumento
    vmax=None,  # <--- Nuevo argumento
):
    """
    Visualiza conectividad sparse como scatter (pre -> post).
    """
    idx_pre = conn.idx_pre.detach().cpu()
    idx_pos = conn.idx_pos.detach().cpu()
    w = conn.weight.detach().cpu()

    if n_pre is None:
        n_pre = int(idx_pre.max()) + 1
    if n_post is None:
        n_post = int(idx_pos.max()) + 1

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True

    ax.set_facecolor(background_color)

    # Pasamos vmin y vmax para fijar la escala de colores externamente
    sc = ax.scatter(
        idx_pre,
        idx_pos,
        c=w,
        s=s,
        cmap=cmap,
        marker="s",
        linewidths=0,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xlim(-0.5, n_pre - 0.5)
    ax.set_ylim(n_post - 0.5, -0.5)
    ax.set_xlabel("Neuronas pre")
    ax.set_ylabel("Neuronas post")
    ax.set_aspect("equal")

    if title is not None:
        ax.set_title(title)

    # Solo mostramos colorbar interno si se pide explícitamente.
    # Cuando hacemos plots combinados, esto suele ser False.
    if show_colorbar:
        fig = ax.figure
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Peso sináptico")

        if background_color != "white":
            cbar.ax.yaxis.set_tick_params(color="white")
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
            cbar.ax.yaxis.label.set_color("white")

    if created_fig:
        plt.tight_layout()
        plt.show()

    return sc