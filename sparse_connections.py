from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .neurons import NeuronGroup
from . import globals
from .group import Group
from .connection import ConnectionSpec
from .utils import resolve_param, block_distance_connect

from typing import Any

import torch
import numpy as np


class StaticSparse(Group):
    pre: NeuronGroup
    pos: NeuronGroup
    idx_pre: torch.Tensor
    idx_pos: torch.Tensor
    weight: torch.Tensor
    delay: torch.Tensor
    channel: int
    _current_buffer: torch.Tensor

    def __init__(self, spec: ConnectionSpec):
        """
        Initialize a sparse connection using a ConnectionSpec.

        Parameters
        ----------
        spec : ConnectionSpec
            Specification object with all details of the connection.
        """
        if spec.pre.device != spec.pos.device:
            raise RuntimeError("Connected populations must be on the same device.")

        super().__init__(size=spec.src_idx.numel(), device=spec.pre.device)

        self.pre = spec.pre
        self.pos = spec.pos
        self.idx_pre = spec.src_idx
        self.idx_pos = spec.tgt_idx
        self.weight = spec.weight.to(device=self.pre.device, dtype=torch.float32)
        self.delay  = spec.delay.to(device=self.pre.device, dtype=torch.long)
        self.channel = spec.params.get("channel", 0)

        if torch.any(self.delay >= self.pre.delay_max):
            raise ValueError(
                f"Connection delay too large (max {torch.max(self.delay)}) "
                f"for pre.delay_max={self.pre.delay_max}"
            )

        self._current_buffer = torch.zeros(
            self.pos.size, dtype=torch.float32, device=self.pre.device
        )


    def _process(self):
        super()._process()
        self._propagate()
        self._update()


    def _propagate(self):
        """Propagate spikes from pre-synaptic to post-synaptic neurons.

        Retrieves pre-synaptic spikes with appropriate delays, multiplies by weights,
        and injects the resulting currents into post-synaptic neurons.
        """
        spikes_mask = self.pre.get_spikes_at(self.delay, self.idx_pre)
        contrib = self.weight * spikes_mask.to(self.weight.dtype)
        self._current_buffer.zero_()
        self._current_buffer.index_add_(0, self.idx_pos, contrib)
        self.pos.inject_currents(self._current_buffer, self.channel)


    def _update(self) -> None:
        # Por defecto, nada (estático). Subclases sobreescriben.
        pass


class STDPSparse(StaticSparse):
    """Spike-Timing-Dependent Plasticity (STDP) synaptic connections.

    Implements STDP, a biologically-inspired learning rule where synaptic
    weights are modified based on the relative timing of pre- and post-synaptic spikes.

    Attributes
    ----------
    A_plus : torch.Tensor
        Learning rate for potentiation (when pre-synaptic spike precedes post-synaptic).
    A_minus : torch.Tensor
        Learning rate for depression (when post-synaptic spike precedes pre-synaptic).
    tau_plus : torch.Tensor
        Time constant for pre-synaptic trace decay.
    tau_minus : torch.Tensor
        Time constant for post-synaptic trace decay.
    w_min : torch.Tensor
        Minimum allowed weight value.
    w_max : torch.Tensor
        Maximum allowed weight value.
    x_pre : torch.Tensor
        Pre-synaptic spike traces for each connection.
    x_pos : torch.Tensor
        Post-synaptic spike traces for each connection.
    alpha_pre : torch.Tensor
        Decay factor for pre-synaptic traces.
    alpha_pos : torch.Tensor
        Decay factor for post-synaptic traces.
    _delay_1 : torch.Tensor
        Constant tensor of ones for accessing post-synaptic spikes.
    """

    A_plus: torch.Tensor
    A_minus: torch.Tensor
    tau_plus: torch.Tensor
    tau_minus: torch.Tensor
    w_min: torch.Tensor
    w_max: torch.Tensor
    oja_decay: torch.Tensor
    x_pre: torch.Tensor
    x_pos: torch.Tensor
    alpha_pre: torch.Tensor
    alpha_pos: torch.Tensor


    def __init__(self, spec: ConnectionSpec):
        super().__init__(spec)

        device = self.pre.device
        params = spec.params

        # Parámetros STDP (se pueden pasar en spec.params)
        self.A_plus = torch.tensor(params.get("A_plus",    1e-4), device=device)
        self.A_minus = torch.tensor(params.get("A_minus", -1.2e-4), device=device)
        self.tau_plus = torch.tensor(params.get("tau_plus", 20e-3), device=device)
        self.tau_minus = torch.tensor(params.get("tau_minus", 20e-3), device=device)
        self.w_min = torch.tensor(params.get("w_min", 0.0), device=device)
        self.w_max = torch.tensor(params.get("w_max", 1.0), device=device)
        self.oja_decay = torch.tensor(params.get("oja_decay", 1e-5), device=device)

        # Trazas sinápticas
        n_edges = len(self.idx_pre)
        self.x_pre = torch.zeros(n_edges, dtype=torch.float32, device=device)
        self.x_pos = torch.zeros(n_edges, dtype=torch.float32, device=device)

        self.alpha_pre = torch.exp(-1e-3 / self.tau_plus)
        self.alpha_pos = torch.exp(-1e-3 / self.tau_minus)


    def _update(self) -> None:
        """Update synaptic weights according to the STDP rule."""
        # 1. Decaimiento
        self.x_pre *= self.alpha_pre
        self.x_pos *= self.alpha_pos

        # 2. Spikes actuales
        pre_spikes = self.pre.get_spikes_at(self.delay, self.idx_pre)
        pos_spikes = self.pos.get_spikes_at(1, self.idx_pos)

        self.x_pre += pre_spikes.to(torch.float32)
        self.x_pos += pos_spikes.to(torch.float32)

        # 3. STDP updates
        potentiation = self.A_plus * self.x_pre * pos_spikes
        depression = self.A_minus * self.x_pos * pre_spikes
        homeostasis = self.x_pos*self.x_pos * self.weight * self.oja_decay
        self.weight += potentiation + depression - homeostasis

        self.weight.clamp_(self.w_min, self.w_max)


class STDPSFASparse(StaticSparse):
    """
    Sparse SFA-STDP (Mexican hat kernel per synapse).
    All traces are per-synapse, not per-neuron.
    """

    def __init__(self, spec: ConnectionSpec):
        super().__init__(spec)
        device = self.pre.device
        p = spec.params
        
        # Taus
        self.tau_fast = torch.tensor(p.get("tau_fast", 20e-3), device=device)
        self.tau_slow = torch.tensor(p.get("tau_slow", 40e-3), device=device)
        dt = globals.simulator.dt

        self.alpha_fast = torch.exp(-dt / self.tau_fast)
        self.alpha_slow = torch.exp(-dt / self.tau_slow)
        
        # Balanced kernel
        self.a1 = 1.0
        self.a2 = -(self.tau_fast / self.tau_slow)

        # Learning rate
        self.A = torch.tensor(p.get("A", -1e-5), device=device)

        # Channel sign (AMPA/NMDA = +1, GABA = -1)
        chan = p.get("channel", 0)
        self.sign = torch.tensor(1.0 if chan != 1 else -1.0, device=device)

        # Number of synapses
        num_syn = self.weight.numel()

        # --- Traces are PER SYNAPSE ---
        self.pre_fast  = torch.zeros(num_syn, device=device)
        self.pre_slow  = torch.zeros(num_syn, device=device)
        self.post_fast = torch.zeros(num_syn, device=device)
        self.post_slow = torch.zeros(num_syn, device=device)

        # Index arrays for sparse structure
        self.i = self.idx_pre     # [num_syn]
        self.j = self.idx_pos     # [num_syn]

        # Normalization parameters
        self.norm_every = int(p.get("norm_every", 200))
        self.scale = p.get("scale", 1.0)
        self.eps = 1e-12

        # Clamp
        self.w_min = torch.tensor(p.get("w_min", 1e-10), device=device)
        self.w_max = torch.tensor(p.get("w_max", 1.0), device=device)


    @torch.no_grad()
    def _update(self):

        current_step = globals.simulator.local_circuit.current_step
        dt = globals.simulator.dt

        # Spikes por neurona
        pre_neuron  = self.pre.get_spikes().float()   # [N_pre]
        post_neuron = self.pos.get_spikes().float()   # [N_post]

        # Spikes por sinapsis
        pre  = pre_neuron[self.i]    # [num_syn]
        post = post_neuron[self.j]   # [num_syn]

        # Decay traces in-place
        self.pre_fast.mul_(self.alpha_fast)
        self.pre_slow.mul_(self.alpha_slow)
        self.post_fast.mul_(self.alpha_fast)
        self.post_slow.mul_(self.alpha_slow)

        # Add current spikes
        self.pre_fast.add_(pre)
        self.pre_slow.add_(pre)
        self.post_fast.add_(post)
        self.post_slow.add_(post)

        # High-pass filtered signals per synapse
        h_pre  = self.a1 * self.pre_fast  + self.a2 * self.pre_slow
        h_post = self.a1 * self.post_fast + self.a2 * self.post_slow

        # SFA-STDP rule (per synapse)
        # Δw_k = A * (pre_i * h_post_j + h_pre_i * post_j)
        dw = self.A * (pre * h_post + h_pre * post)

        # Channel sign
        dw.mul_(self.sign)

        # Update weights
        self.weight.add_(dw)

        # Clamp
        self.weight.clamp_(self.w_min, self.w_max)

        # Column-wise L2 norm (sparse)
        if self.norm_every and (current_step % self.norm_every == 0):
            col_norm = torch.zeros(self.pos.size, device=self.weight.device)
            col_norm.index_add_(0, self.j, self.weight * self.weight)
            col_norm = col_norm.sqrt().clamp_min(self.eps)

            scale = self.scale / col_norm   # [N_post]
            self.weight.mul_(scale[self.j])


class STDPSparseNormalized(StaticSparse):
    """
    STDP con normalización periódica aferente (column-wise).
    
    En lugar de usar un término de homeostasis continua (como Oja), esta regla
    aplica STDP estándar y, cada X pasos, fuerza que la norma L2 de los pesos
    entrantes a cada neurona post-sináptica sea igual a un valor objetivo.

    Attributes
    ----------
    norm_every : int
        Intervalo de pasos de tiempo para aplicar la normalización.
    norm_target : float
        Valor objetivo de la norma (suma cuadrática) de los pesos entrantes.
    """

    A_plus: torch.Tensor
    A_minus: torch.Tensor
    tau_plus: torch.Tensor
    tau_minus: torch.Tensor
    w_min: torch.Tensor
    w_max: torch.Tensor
    
    norm_every: int
    norm_target: float
    eps: float

    x_pre: torch.Tensor
    x_pos: torch.Tensor
    alpha_pre: torch.Tensor
    alpha_pos: torch.Tensor

    def __init__(self, spec: ConnectionSpec):
        super().__init__(spec)

        device = self.pre.device
        params = spec.params

        # Parámetros STDP
        self.A_plus = torch.tensor(params.get("A_plus",    1e-4), device=device)
        self.A_minus = torch.tensor(params.get("A_minus", -1.2e-4), device=device)
        self.tau_plus = torch.tensor(params.get("tau_plus", 20e-3), device=device)
        self.tau_minus = torch.tensor(params.get("tau_minus", 20e-3), device=device)
        
        # Límites duros
        self.w_min = torch.tensor(params.get("w_min", 0.0), device=device)
        self.w_max = torch.tensor(params.get("w_max", 1.0), device=device)

        # Parámetros de Normalización
        self.norm_every = int(params.get("norm_every", 100)) # Ej: cada 100ms
        self.norm_target = float(params.get("norm_target", 1.0))
        self.eps = 1e-8 # Para evitar división por cero

        # Trazas sinápticas
        n_edges = len(self.idx_pre)
        self.x_pre = torch.zeros(n_edges, dtype=torch.float32, device=device)
        self.x_pos = torch.zeros(n_edges, dtype=torch.float32, device=device)

        self.alpha_pre = torch.exp(-1e-3 / self.tau_plus)
        self.alpha_pos = torch.exp(-1e-3 / self.tau_minus)

    def _update(self) -> None:
        """Update synaptic weights according to STDP + Periodic Normalization."""
        
        # 1. Actualización de trazas (Decay)
        self.x_pre.mul_(self.alpha_pre)
        self.x_pos.mul_(self.alpha_pos)

        # 2. Inserción de spikes actuales en las trazas
        # Nota: Asumimos que get_spikes_at devuelve bool o 0/1 float
        pre_spikes = self.pre.get_spikes_at(self.delay, self.idx_pre).float()
        pos_spikes = self.pos.get_spikes_at(1, self.idx_pos).float()

        self.x_pre.add_(pre_spikes)
        self.x_pos.add_(pos_spikes)

        # 3. Cálculo de dW por STDP
        # Potenciación: traza pre * spike post
        potentiation = self.x_pre * pos_spikes * self.A_plus
        # Depresión: traza post * spike pre
        depression = self.x_pos * pre_spikes * self.A_minus

        # 4. Aplicar cambios y Clamping
        # Eliminamos el término de Oja aquí
        self.weight.add_(potentiation).add_(depression)
        self.weight.clamp_(self.w_min, self.w_max)

        # 5. Normalización Periódica (Column-wise L2 Norm)
        current_step = globals.simulator.local_circuit.current_step
        
        if self.norm_every > 0 and (current_step % self.norm_every == 0):
            self._apply_normalization()

    def _apply_normalization(self) -> None:
        """Calcula la norma L2 de los pesos entrantes por neurona y reescala."""
        
        # Buffer para acumular la suma de cuadrados por neurona post-sináptica
        # Tamaño = número de neuronas en el grupo destino
        col_norm = torch.zeros(self.pos.size, dtype=torch.float32, device=self.weight.device)
        
        # Sumar w^2 agrupando por índice post-sináptico (idx_pos)
        # weight^2 -> acumulado en col_norm[idx_pos]
        col_norm.index_add_(0, self.idx_pos, self.weight.pow(2))
        
        # Raíz cuadrada para obtener L2 norm
        col_norm.sqrt_()
        
        # Evitar divisiones por cero o normas muy pequeñas
        col_norm.clamp_min_(self.eps)
        
        # Calcular factor de escala: Target / Actual
        # scale_factors será un vector de tamaño [N_post_neurons]
        scale_factors = self.norm_target / col_norm
        
        # Aplicar el factor a cada peso individual.
        # Usamos idx_pos para mapear cada peso a su factor de escala correspondiente.
        self.weight.mul_(scale_factors[self.idx_pos])
        
        # (Opcional) Re-clamping por si la normalización empujó algo fuera de rango,
        # aunque si norm_target es razonable y w_max es razonable, no debería pasar mucho.
        self.weight.clamp_(self.w_min, self.w_max)


class VogelsSparse(StaticSparse):
    """Inhibitory Spike-Timing-Dependent Plasticity (iSTDP) - Vogels et al. (2011) for Sparse Networks.

    Implements the symmetric inhibitory learning rule on sparse connections.
    Updates are performed element-wise on the edge list.

    Attributes
    ----------
    eta : torch.Tensor
        Learning rate.
    tau : torch.Tensor
        Time constant for synaptic traces.
    alpha_depression : torch.Tensor
        The calculated depression constant (2 * target_rate * tau).
    w_max : torch.Tensor
        Maximum allowed weight value.
    x_pre : torch.Tensor
        Pre-synaptic spike traces (one per edge).
    x_pos : torch.Tensor
        Post-synaptic spike traces (one per edge).
    trace_decay : torch.Tensor
        Decay factor per time step.
    """
    eta: torch.Tensor
    tau: torch.Tensor
    alpha_depression: torch.Tensor
    w_max: torch.Tensor
    x_pre: torch.Tensor
    x_pos: torch.Tensor
    trace_decay: torch.Tensor

    def __init__(self, spec: ConnectionSpec):
        super().__init__(spec)

        # 1. Validación de Canal (GABA = 1)
        if spec.params.get("channel") != 1:
            print(f"Warning: Vogels2011Sparse rule applied to channel {spec.params.get('channel')}. "
                  "This rule is designed for GABAergic (channel 1) connections.")

        device = self.pre.device
        params = spec.params

        # Parámetros Vogels
        self.eta = torch.tensor(params.get("eta", 1e-4), device=device)
        self.tau = torch.tensor(params.get("tau", 20e-3), device=device)
        self.w_max = torch.tensor(params.get("w_max", 1.0), device=device)

        # Cálculo de alpha (Depresión homeostática)
        target_rate = params.get("target_rate", 5.0) # Hz
        alpha_val = 2.0 * target_rate * self.tau.item()
        self.alpha_depression = torch.tensor(alpha_val, device=device)

        # Trazas sinápticas: Un valor por cada conexión existente
        n_edges = len(self.idx_pre)
        self.x_pre = torch.zeros(n_edges, dtype=torch.float32, device=device)
        self.x_pos = torch.zeros(n_edges, dtype=torch.float32, device=device)

        # Factor de decaimiento
        dt = 1e-3 
        self.trace_decay = torch.exp(torch.tensor(-dt, device=device) / self.tau)

    def _update(self) -> None:
        # 1. Decaimiento de trazas
        self.x_pre *= self.trace_decay
        self.x_pos *= self.trace_decay

        # 2. Obtener spikes alineados con las conexiones (Sparse)
        # pre_spikes y pos_spikes son vectores de tamaño [n_edges]
        # Contienen 1.0 si la neurona pre/pos correspondiente a esa conexión disparó.
        pre_spikes = self.pre.get_spikes_at(self.delay, self.idx_pre).float()
        pos_spikes = self.pos.get_spikes_at(1, self.idx_pos).float()

        # 3. Actualizar trazas
        self.x_pre += pre_spikes
        self.x_pos += pos_spikes

        # 4. Cálculo de Delta Weights (Vogels Simétrico)
        # Como todo son vectores alineados [n_edges], usamos multiplicación element-wise (*).
        
        # A. Pre-driven update: Cuando la PRE dispara, miramos el estado de la traza POST.
        #    Formula: Spike_pre * (Trace_pos - alpha)
        #    Efecto: Potenciación si post disparó recientemente, Depresión (alpha) si no.
        pre_driven = pre_spikes * (self.x_pos - self.alpha_depression)

        # B. Post-driven update: Cuando la POST dispara, miramos el estado de la traza PRE.
        #    Formula: Spike_pos * Trace_pre
        #    Efecto: Potenciación si pre disparó recientemente.
        pos_driven = pos_spikes * self.x_pre

        # Suma total y learning rate
        delta_w = self.eta * (pre_driven + pos_driven)

        # 5. Aplicar cambios
        self.weight += delta_w
        self.weight.clamp_(0.0, self.w_max)


class TripletSTDPSparse(StaticSparse):
    """
    Implementación de la regla de aprendizaje con:
    1. Tripletes de spikes (x_pre, x_post1, x_post2).
    2. Homeostasis de pesos (Normalización L1 de filas y columnas).
    3. Capacidad de invertir el aprendizaje (Anti-Hebbian) para la fase SLEEP.
    """
    
    # Tensores de estado
    x_pre: torch.Tensor
    x_post1: torch.Tensor
    x_post2: torch.Tensor
    
    # Parámetros (Tensores para permitir modificaciones in-place)
    eta_pre: torch.Tensor
    eta_post: torch.Tensor
    
    # Constantes
    w_max: torch.Tensor
    mu: float
    norm_every: int
    norm_target_in: float
    norm_target_out: float
    
    # Estado interno
    _sign_inverted: bool = False

    def __init__(self, spec: ConnectionSpec):
        super().__init__(spec)
        device = self.pre.device
        params = spec.params

        # Learning Rates (base)
        self.eta_pre_base = params.get("eta_pre", 0.005)
        self.eta_post_base = params.get("eta_post", 0.025)
        
        # Los guardamos como tensores para poder invertir su signo globalmente
        self.eta_pre = torch.tensor(self.eta_pre_base, device=device)
        self.eta_post = torch.tensor(self.eta_post_base, device=device)

        # Constantes de tiempo
        tau_pre = params.get("tau_x_pre", 20e-3)
        tau_post1 = params.get("tau_x_post1", 40e-3) # Fast (Pair)
        tau_post2 = params.get("tau_x_post2", 40e-3) # Slow (Triplet) - Nota: En Diehl a veces es igual

        self.w_max = torch.tensor(params.get("w_max", 0.5), device=device)
        self.mu = params.get("mu", 0.2)
        self.eps = 1e-8

        # Normalización
        self.norm_every = int(params.get("norm_every", 100))
        self.norm_target_in = float(params.get("norm_target_in", 1.0))
        self.norm_target_out = float(params.get("norm_target_out", 1.0))

        # Inicializar trazas (tamaño n_edges para compatibilidad Sparse)
        n_edges = len(self.idx_pre)
        self.x_pre = torch.zeros(n_edges, dtype=torch.float32, device=device)
        self.x_post1 = torch.zeros(n_edges, dtype=torch.float32, device=device)
        self.x_post2 = torch.zeros(n_edges, dtype=torch.float32, device=device)

        # Factores de decaimiento (asumiendo dt=1ms para cálculo rápido)
        dt = 1e-3
        self.decay_pre = np.exp(-dt / tau_pre)
        self.decay_post1 = np.exp(-dt / tau_post1)
        self.decay_post2 = np.exp(-dt / tau_post2)

    def set_learning_sign(self, invert: bool):
        """Invierte los learning rates para la fase de sueño (Anti-Hebbian)."""
        if invert and not self._sign_inverted:
            self.eta_pre.data.fill_(-self.eta_pre_base)
            self.eta_post.data.fill_(-self.eta_post_base)
            self._sign_inverted = True
        elif not invert and self._sign_inverted:
            self.eta_pre.data.fill_(self.eta_pre_base)
            self.eta_post.data.fill_(self.eta_post_base)
            self._sign_inverted = False
    
    def set_learning_enabled(self, enabled: bool):
        """Para la fase de Test (Learning rate = 0)."""
        if enabled:
            # Restaurar según el estado de inversión actual
            val_pre = -self.eta_pre_base if self._sign_inverted else self.eta_pre_base
            val_post = -self.eta_post_base if self._sign_inverted else self.eta_post_base
            self.eta_pre.data.fill_(val_pre)
            self.eta_post.data.fill_(val_post)
        else:
            self.eta_pre.data.fill_(0.0)
            self.eta_post.data.fill_(0.0)

    def _update(self) -> None:
        # 1. Decaimiento de trazas
        self.x_pre.mul_(self.decay_pre)
        self.x_post1.mul_(self.decay_post1)
        self.x_post2.mul_(self.decay_post2)

        # 2. Leer spikes
        pre_spikes = self.pre.get_spikes_at(self.delay, self.idx_pre).float()
        pos_spikes = self.pos.get_spikes_at(1, self.idx_pos).float()

        # 3. Actualizar trazas (Pre y Post)
        self.x_pre.add_(pre_spikes)
        self.x_post1.add_(pos_spikes)
        self.x_post2.add_(pos_spikes)

        # 4. Cálculo STDP Triplet
        # LTD (Pre-driven): -eta_pre * x_post1 * w^mu
        # Se aplica cuando dispara la presináptica
        ltd_term = -self.eta_pre * self.x_post1 * pre_spikes * torch.pow(self.weight, self.mu)

        # LTP (Post-driven): eta_post * x_pre * x_post2 * (w_max - w)^mu
        # Se aplica cuando dispara la postsináptica
        # x_post2 actúa como el tercer factor (actividad post previa)
        dist_to_max = (self.w_max - self.weight).clamp(min=0)
        ltp_term = self.eta_post * self.x_pre * self.x_post2 * pos_spikes * torch.pow(dist_to_max, self.mu)

        # 5. Actualizar pesos
        self.weight.add_(ltd_term + ltp_term)
        self.weight.clamp_(0.0, self.w_max)

        # 6. Normalización periódica
        curr_step = globals.simulator.local_circuit.current_step
        if self.norm_every > 0 and (curr_step % self.norm_every == 0):
            self._apply_normalization()

    def _apply_normalization(self):
        # Normalización de Columnas (Incoming weights per post-neuron) -> Suma L1 constante
        if self.norm_target_in > 0:
            col_sum = torch.zeros(self.pos.size, device=self.weight.device)
            col_sum.index_add_(0, self.idx_pos, self.weight)
            col_sum.clamp_min_(self.eps)
            scale_in = self.norm_target_in / col_sum
            self.weight.mul_(scale_in[self.idx_pos])

        # Normalización de Filas (Outgoing weights per pre-neuron) -> Suma L1 constante
        if self.norm_target_out > 0:
            row_sum = torch.zeros(self.pre.size, device=self.weight.device)
            row_sum.index_add_(0, self.idx_pre, self.weight)
            row_sum.clamp_min_(self.eps)
            scale_out = self.norm_target_out / row_sum
            self.weight.mul_(scale_out[self.idx_pre])