# ============================================================
# 0) Imports
# ============================================================
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import math
import random

import torch


# ============================================================
# 1) StateMachine
# ============================================================
@dataclass
class _Transition:
    dst: str
    prob: float
    segment_id: str


class StateMachine:
    """
    Estados abstractos. Transiciones probabilísticas.
    Cada transición referencia un segmento geométrico (segment_id).
    """
    def __init__(self, initial_state: str):
        self.initial_state = initial_state
        self.current_state = initial_state
        self._states = set([initial_state])
        self._out: Dict[str, List[_Transition]] = {}

    def add_state(self, state_id: str):
        self._states.add(state_id)

    def add_transition(self, src: str, dst: str, probability: float, segment_id: str):
        assert probability >= 0.0, "probability must be non-negative"
        self._states.add(src)
        self._states.add(dst)
        self._out.setdefault(src, []).append(_Transition(dst=dst, prob=float(probability), segment_id=segment_id))

    def reset(self):
        self.current_state = self.initial_state

    def step(self) -> str:
        """
        Elige una transición saliente según probabilidad, actualiza estado actual,
        devuelve segment_id activo.
        """
        transitions = self._out.get(self.current_state, None)
        if not transitions:
            raise RuntimeError(f"No outgoing transitions from state '{self.current_state}'")

        probs = [t.prob for t in transitions]
        s = sum(probs)
        if s <= 0:
            raise RuntimeError(f"Outgoing probabilities from state '{self.current_state}' sum to 0")

        # normalización robusta
        probs = [p / s for p in probs]

        r = random.random()
        acc = 0.0
        chosen = transitions[-1]
        for t, p in zip(transitions, probs):
            acc += p
            if r <= acc:
                chosen = t
                break

        self.current_state = chosen.dst
        return chosen.segment_id


# ============================================================
# 2) Catmull-Rom helpers + SplineSegment
# ============================================================
def catmull_rom_segment(P0: torch.Tensor, P1: torch.Tensor, P2: torch.Tensor, P3: torch.Tensor, t: torch.Tensor):
    """
    P0..P3: (D,) or (..., D)
    t: (..., 1) in [0,1]
    """
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        2 * P1 +
        (-P0 + P2) * t +
        (2*P0 - 5*P1 + 4*P2 - P3) * t2 +
        (-P0 + 3*P1 - 3*P2 + P3) * t3
    )


class SplineSegment:
    """
    Segmento geométrico: Catmull–Rom por waypoints (abierta),
    reparametrizada aproximadamente por longitud de arco (LUT).
    """
    def __init__(
        self,
        waypoints: torch.Tensor,          # (N, D)
        speed: float,
        samples_per_segment: int = 100,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
    ):
        assert waypoints.ndim == 2, "waypoints must be (N, D)"
        assert waypoints.shape[0] >= 2, "need at least 2 waypoints"
        assert speed > 0, "speed must be > 0"
        assert samples_per_segment >= 10, "samples_per_segment too small"

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.speed = float(speed)
        self.samples_per_segment = int(samples_per_segment)

        P = waypoints.to(device=self.device, dtype=self.dtype)

        # duplicamos extremos (trayectoria abierta)
        self.P = torch.cat([P[0:1], P, P[-1:]], dim=0)  # (N+2, D)

        self._build_arc_length_table()

    def _build_arc_length_table(self):
        pts = []
        # generamos puntos muestreados en todos los segmentos
        for i in range(len(self.P) - 3):
            P0, P1, P2, P3 = self.P[i:i+4]  # (D,)
            t = torch.linspace(0, 1, self.samples_per_segment, device=self.device, dtype=self.dtype).unsqueeze(1)
            curve_pts = catmull_rom_segment(P0, P1, P2, P3, t)  # (S, D)
            pts.append(curve_pts)

        self.points = torch.cat(pts, dim=0)  # (T, D)

        diffs = self.points[1:] - self.points[:-1]              # (T-1, D)
        dists = torch.linalg.norm(diffs, dim=1)                 # (T-1,)

        self.arc_lengths = torch.cat([
            torch.zeros(1, device=self.device, dtype=self.dtype),
            torch.cumsum(dists, dim=0)
        ], dim=0)                                               # (T,)

        self.total_length = float(self.arc_lengths[-1].item())
        self._duration = self.total_length / self.speed if self.total_length > 0 else 0.0

    def length(self) -> float:
        return self.total_length

    def duration(self) -> float:
        return self._duration

    def _sample_by_distance(self, s: torch.Tensor) -> torch.Tensor:
        """
        s: tensor escalar o vector (..)
        """
        if self.total_length <= 0:
            # segmento degenerado
            return self.points[0].expand((*s.shape, self.points.shape[-1]))

        s = torch.clamp(s, 0.0, self.arc_lengths[-1])

        idx = torch.searchsorted(self.arc_lengths, s)
        idx = torch.clamp(idx, 1, len(self.arc_lengths) - 1)

        s0 = self.arc_lengths[idx - 1]
        s1 = self.arc_lengths[idx]

        alpha = (s - s0) / (s1 - s0 + 1e-8)
        alpha = alpha.unsqueeze(-1)

        p0 = self.points[idx - 1]
        p1 = self.points[idx]
        return p0 + alpha * (p1 - p0)

    def sample(self, t: float | torch.Tensor) -> torch.Tensor:
        """
        t en [0, duration]. Devuelve posición (D,) o (..., D)
        """
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=self.device, dtype=self.dtype)
        t = torch.clamp(t, 0.0, self._duration)
        s = t * self.speed
        return self._sample_by_distance(s)


# ============================================================
# 3) TrajectoryRunner (FSM + segmentos)
# ============================================================
class TrajectoryRunner:
    """
    Une FSM + librería de splines.
    Mantiene segmento activo y tiempo local dentro del segmento.
    Puede añadir ruido OU continuo global a la posición.
    """
    def __init__(
        self,
        fsm: StateMachine,
        spline_library: Dict[str, SplineSegment],
        ou_noise: Optional[ContinuousOUNoise] = None,
    ):
        self.fsm = fsm
        self.spline_library = spline_library
        self.ou_noise = ou_noise

        self._active_segment_id: Optional[str] = None
        self._active_segment: Optional[SplineSegment] = None
        self._t_local = 0.0

    def reset(self):
        self.fsm.reset()
        self._active_segment_id = None
        self._active_segment = None
        self._t_local = 0.0
        if self.ou_noise is not None:
            self.ou_noise.reset()

    def _ensure_segment(self):
        if self._active_segment is None:
            seg_id = self.fsm.step()
            if seg_id not in self.spline_library:
                raise KeyError(f"segment_id '{seg_id}' not found in spline_library")
            self._active_segment_id = seg_id
            self._active_segment = self.spline_library[seg_id]
            self._t_local = 0.0

    def step(self, dt: float) -> torch.Tensor:
        """
        Avanza dt segundos.
        Devuelve posición observada (con ruido si está activo).
        """
        assert dt >= 0.0
        self._ensure_segment()

        seg = self._active_segment
        assert seg is not None

        remaining = dt
        pos = None

        while True:
            dur = seg.duration()
            if dur <= 0.0:
                seg_id = self.fsm.step()
                seg = self.spline_library[seg_id]
                self._active_segment_id = seg_id
                self._active_segment = seg
                self._t_local = 0.0
                continue

            t_next = self._t_local + remaining
            if t_next < dur:
                self._t_local = t_next
                pos = seg.sample(self._t_local)
                break

            # Fin de segmento
            remaining = t_next - dur
            seg_id = self.fsm.step()
            seg = self.spline_library[seg_id]
            self._active_segment_id = seg_id
            self._active_segment = seg
            self._t_local = 0.0

            if remaining <= 1e-12:
                pos = seg.sample(self._t_local)
                break

        # ---- ruido OU continuo global
        if self.ou_noise is not None:
            offset = self.ou_noise.step(dt)
            pos = pos + offset

        return pos


# ============================================================
# 4) Poisson disk sampling en dD (Bridson generalizado)
# ============================================================
def poisson_disk_sampling(
    n_points_target: int,
    dim: int,
    domain_min: torch.Tensor,
    domain_max: torch.Tensor,
    r: float,
    k: int = 30,
    seed: Optional[int] = None,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Genera puntos en [domain_min, domain_max] (hipercubo/hiperrectángulo) en dD,
    con distancia mínima aproximada r (Poisson-disk), usando Bridson (grid hashing).

    - n_points_target: objetivo (puede devolver menos si está muy apretado).
    - r: distancia mínima.
    - k: intentos por punto activo.
    """
    assert dim >= 1
    assert n_points_target >= 1
    assert r > 0

    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    domain_min = domain_min.to(device=device, dtype=dtype)
    domain_max = domain_max.to(device=device, dtype=dtype)
    assert domain_min.shape == (dim,)
    assert domain_max.shape == (dim,)

    # cell size para grid hashing
    cell_size = r / math.sqrt(dim)
    inv_cell = 1.0 / cell_size

    def grid_coords(x: torch.Tensor) -> Tuple[int, ...]:
        g = torch.floor((x - domain_min) * inv_cell).to(torch.int64)
        return tuple(int(v.item()) for v in g)

    def in_domain(x: torch.Tensor) -> bool:
        return bool(torch.all(x >= domain_min) and torch.all(x <= domain_max))

    # grid: dict coord-> index en points
    grid: Dict[Tuple[int, ...], int] = {}

    points: List[torch.Tensor] = []
    active: List[int] = []

    # punto inicial aleatorio
    x0 = domain_min + (domain_max - domain_min) * torch.rand((dim,), device=device, dtype=dtype)
    points.append(x0)
    active.append(0)
    grid[grid_coords(x0)] = 0

    # vecinos a chequear: todos los offsets en {-2,-1,0,1,2}^d suelen bastar
    # (porque cell_size = r/sqrt(d) => radio r ocupa ~sqrt(d) celdas)
    offsets = torch.cartesian_prod(*[torch.tensor([-2, -1, 0, 1, 2], device=device) for _ in range(dim)])
    offsets = offsets.to(torch.int64)

    def is_far_enough(x: torch.Tensor) -> bool:
        gc = torch.tensor(grid_coords(x), device=device, dtype=torch.int64)
        for off in offsets:
            key = tuple(int(v.item()) for v in (gc + off))
            j = grid.get(key, None)
            if j is None:
                continue
            if torch.linalg.norm(x - points[j]) < r:
                return False
        return True

    while active and len(points) < n_points_target:
        idx = random.choice(active)
        base = points[idx]
        found = False

        for _ in range(k):
            # sample en el anillo [r, 2r] con dirección aleatoria
            # dirección: normalizar vector gaussiano
            direction = torch.randn((dim,), device=device, dtype=dtype)
            direction = direction / (torch.linalg.norm(direction) + 1e-12)
            radius = r * (1.0 + random.random())  # uniform en [r,2r]
            cand = base + direction * radius

            if not in_domain(cand):
                continue
            if not is_far_enough(cand):
                continue

            points.append(cand)
            new_i = len(points) - 1
            active.append(new_i)
            grid[grid_coords(cand)] = new_i
            found = True
            break

        if not found:
            active.remove(idx)

    return torch.stack(points, dim=0)  # (K, D)


# ============================================================
# 5) RBFSpace + MultiScaleRBFEncoder
# ============================================================
def _toroidal_delta(x: torch.Tensor, c: torch.Tensor, period: float) -> torch.Tensor:
    """
    Devuelve delta mínimo en una dimensión toroidal.
    x,c: (...,)
    """
    d = torch.abs(x - c)
    return torch.minimum(d, period - d)


def _rbf_distance_squared(
    position: torch.Tensor,  # (D,)
    centers: torch.Tensor,   # (K, D)
    toroidal_dims: Optional[Dict[int, float]] = None
) -> torch.Tensor:
    """
    Devuelve dist^2 para cada centro: (K,)
    Aplica métrica toroidal en dims indicadas (periodo).
    """
    # diff: (K, D)
    diff = centers - position.unsqueeze(0)

    if toroidal_dims:
        # asumimos que position y centers ya están dentro del dominio [0, period] o [0,1], etc.
        for dim, period in toroidal_dims.items():
            # delta mínimo en esa dimensión
            d = _toroidal_delta(centers[:, dim], position[dim].expand_as(centers[:, dim]), period)
            # reconstruimos diff en esa dimensión conservando signo (opcional) -> aquí sólo importa d^2
            diff[:, dim] = torch.sign(diff[:, dim]) * d

    return torch.sum(diff * diff, dim=1)


class RBFSpace:
    def __init__(
        self,
        centers: torch.Tensor,          # (K, D)
        basis_vectors: torch.Tensor,    # (K, M)
        sigma: float,
        toroidal_dims: Optional[Dict[int, float]] = None,
        eps: float = 1e-12,
    ):
        assert centers.ndim == 2
        assert basis_vectors.ndim == 2
        assert centers.shape[0] == basis_vectors.shape[0]
        assert sigma > 0

        self.centers = centers
        self.basis_vectors = basis_vectors
        self.sigma = float(sigma)
        self.toroidal_dims = toroidal_dims or {}
        self.eps = eps

    def encode(self, position: torch.Tensor) -> torch.Tensor:
        """
        position: (D,)
        Devuelve (M,), mezcla convexa de basis_vectors con pesos gausianos,
        normalizando pesos (y luego L1 de salida por seguridad).
        """
        d2 = _rbf_distance_squared(position, self.centers, self.toroidal_dims)  # (K,)
        w = torch.exp(-0.5 * d2 / (self.sigma * self.sigma))                    # (K,)

        w_sum = torch.sum(w) + self.eps
        w = w / w_sum  # normaliza pesos primero

        y = torch.matmul(w.unsqueeze(0), self.basis_vectors).squeeze(0)         # (M,)

        return y


class MultiScaleRBFEncoder:
    def __init__(self, spaces: List[RBFSpace], noise_std: float = 0.0, eps: float = 1e-12):
        assert len(spaces) >= 1
        assert noise_std >= 0.0
        self.spaces = spaces
        self.noise_std = float(noise_std)
        self.eps = eps

    def encode(self, position: torch.Tensor) -> torch.Tensor:
        """
        Concatena [y1|y2|...], con ruido blanco aditivo antes de renormalizar cada subvector.
        """
        outs = []
        for sp in self.spaces:
            y = sp.encode(position)
            if self.noise_std > 0:
                y = y + self.noise_std * torch.randn_like(y)
            outs.append(y)
        return torch.cat(outs, dim=0)


# ============================================================
# 6) SensoryTrajectoryGenerator (fachada)
# ============================================================
class SensoryTrajectoryGenerator:
    def __init__(self, trajectory: TrajectoryRunner, encoder: MultiScaleRBFEncoder):
        self.trajectory = trajectory
        self.encoder = encoder

    def reset(self):
        self.trajectory.reset()

    def step(self, dt: float) -> torch.Tensor:
        pos = self.trajectory.step(dt)
        return self.encoder.encode(pos)


class ContinuousOUNoise:
    """
    Ruido Ornstein–Uhlenbeck continuo en R^D.
    """
    def __init__(
        self,
        dim: int,
        sigma: float,
        tau: float,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
    ):
        assert dim >= 1
        assert sigma >= 0.0
        assert tau > 0.0

        self.dim = dim
        self.sigma = float(sigma)
        self.tau = float(tau)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        self.state = torch.zeros(dim, device=self.device, dtype=self.dtype)

    def reset(self):
        self.state.zero_()

    def step(self, dt: float) -> torch.Tensor:
        """
        Avanza el proceso OU un paso dt.
        Devuelve el desplazamiento actual (D,).
        """
        if self.sigma == 0.0:
            return self.state

        dt = float(dt)
        sqrt_dt = math.sqrt(dt)

        noise = torch.randn(
            self.dim,
            device=self.device,
            dtype=self.dtype
        )

        # Euler–Maruyama
        self.state += (
            - (dt / self.tau) * self.state
            + self.sigma * sqrt_dt * noise
        )
        return self.state