from __future__ import annotations

from typing import TYPE_CHECKING, Any
from . import globals
from .dense import Dense
from .utils import resolve_param
from .connection import ConnectionSpec

import torch


class StaticDense(Dense):
    """Dense connection between two neuron groups.
    
    Represents a fully-matrix synaptic connection. Efficient but limited:
    - All weights stored in a dense [Npre, Npos] matrix
    - Only supports a single uniform delay (int)

    Attributes
    ----------
    pre : NeuronGroup
        Pre-synaptic (source) neuron group.
    pos : NeuronGroup
        Post-synaptic (target) neuron group.
    mask : torch.Tensor
        Boolean mask indicating existing connections between neurons.
    weight : torch.Tensor
        Synaptic weights for each connection.
    delay : Int
        Synaptic delays for all connection, in time steps.
    """
    mask: torch.Tensor
    weight: torch.Tensor
    delay: torch.Tensor
    channel: int


    def __init__(self, spec: ConnectionSpec):
        # Pasamos pre/pos al padre
        super().__init__(pre=spec.pre, pos=spec.pos, device=spec.pre.device)

        # --- delay check ---
        if spec.delay is None:
            self.delay = 0
        
        elif isinstance(spec.delay, torch.Tensor):
            # Caso crítico: tensor vacío → no hay conexiones en esta capa
            if spec.delay.numel() == 0:
                self.delay = 0
            else:
                # Chequeo normal
                dmin = spec.delay.min()
                dmax = spec.delay.max()
                all_equal = bool((dmin == dmax).item())

                if all_equal:
                    self.delay = int(dmin.item())
                else:
                    raise ValueError(
                        f"Dense connections only support a uniform scalar delay, got tensor with values {spec.delay}"
                    )
        elif isinstance(spec.delay, (int, float)):
            self.delay = int(spec.delay)
        else:
            raise TypeError(
                f"Invalid delay type for Dense connection: {type(spec.delay)}"
            )

        # Inicializamos pesos
        self.weight = torch.zeros(self.shape, dtype=torch.float32, device=self.device)
        if spec.weight is not None:
             self.weight[spec.src_idx, spec.tgt_idx] = spec.weight

        # Inicializamos la MASK física basada en el Spec
        # Importante: Inicializamos mask a False y activamos solo lo definido
        self.mask.fill_(False)
        self.mask[spec.src_idx, spec.tgt_idx] = True

        self.channel = spec.params.get("channel", 0)

        # Sanity checks
        if self.delay >= self.pre.delay_max:
            raise ValueError(
                f"Dense connection delay {self.delay} must be less than pre.delay_max {self.pre.delay_max}"
            )
        if self.channel >= self.pos.n_channels:
            raise ValueError(
                f"Channel {self.channel} not valid for post-synaptic group with {self.pos.n_channels} channels"
            )

    def _process(self):
        super()._process()
        self._propagate()
        self._update()

    def _propagate(self):
        """Propagate spikes using dense weight matrix and uniform delay."""
        current_step = globals.simulator.local_circuit.current_step
        phase = (current_step - self.delay) % self.pre.delay_max

        pre_spikes = self.pre._spike_buffer[:, phase].to(self.weight.dtype).squeeze(-1)  # [Npre]
        effective_weight = self.weight.masked_fill(~self.mask, 0)
        contrib = pre_spikes @ effective_weight
        self.pos.inject_currents(contrib, self.channel)

    def _update(self):
        """To be implemented by subclasses (plasticity)."""
        pass


class STDPDense(StaticDense):
    """Spike-Timing-Dependent Plasticity (STDP) synaptic connections for dense networks.

    Implements STDP, a biologically-inspired learning rule where synaptic
    weights are modified based on the relative timing of pre- and post-synaptic spikes.
    This version is optimized for dense connectivity patterns using matrix operations.

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
        Pre-synaptic spike traces for each pre-synaptic neuron.
    x_pos : torch.Tensor
        Post-synaptic spike traces for each post-synaptic neuron.
    alpha_pre : torch.Tensor
        Decay factor for pre-synaptic traces.
    alpha_pos : torch.Tensor
        Decay factor for post-synaptic traces.
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

        # STDP parameters
        self.A_plus = torch.tensor(params.get("A_plus", 1e-4), device=device)
        self.A_minus = torch.tensor(params.get("A_minus", -1.2e-4), device=device)
        self.tau_plus = torch.tensor(params.get("tau_plus", 20e-3), device=device)
        self.tau_minus = torch.tensor(params.get("tau_minus", 20e-3), device=device)
        self.w_min = torch.tensor(params.get("w_min", 0.0), device=device)
        self.w_max = torch.tensor(params.get("w_max", 1.0), device=device)
        self.oja_decay = torch.tensor(params.get("oja_decay", 1e-5), device=self.device)

        # Traces
        self.x_pre = torch.zeros(self.pre.size, dtype=torch.float32, device=device)
        self.x_pos = torch.zeros(self.pos.size, dtype=torch.float32, device=device)

        # Decay factors
        self.alpha_pre = torch.exp(-1e-3 / self.tau_plus)
        self.alpha_pos = torch.exp(-1e-3 / self.tau_minus)

    def _update(self):
        # Decay traces
        self.x_pre *= self.alpha_pre
        self.x_pos *= self.alpha_pos

        # Get current spikes
        current_step = globals.simulator.local_circuit.current_step
        phase = (current_step - self.delay) % self.pre.delay_max
        pre_spikes = self.pre._spike_buffer[:, phase].float().squeeze(-1)  # [Npre]
        pos_spikes = self.pos.get_spikes().float()             # [Npos]

        # Update traces
        self.x_pre += pre_spikes
        self.x_pos += pos_spikes

        # Potentiation: pre spikes × pos traces
        potentiation = torch.outer(pre_spikes, self.x_pos) * self.A_plus
        # Depression: pre traces × pos spikes
        depression = torch.outer(self.x_pre, pos_spikes) * self.A_minus
        # Homeostasis: Oja's normalization rule
        homeostasis = self.x_pos*self.x_pos * self.weight * self.oja_decay

        # Apply update only where connections exist
        self.weight += (potentiation + depression - homeostasis) * self.mask
        self.weight.clamp_(self.w_min, self.w_max)


class SFADense(StaticDense):
    """
    Slow Feature Analysis (SFA) dense synapses.
    Combina:
      - Δw_ij ∝ -eta * dotx_i * doty_j        (SFA rule)
      - Δw_ij ∝ -beta * (rate - target) w_ij  (homeostasis multiplicativa)
      - signo según tipo de canal (AMPA=+1, GABA=-1, NMDA=+1)
      - normalización suave periódica de norma L2 por columna
    """

    def __init__(self, spec: ConnectionSpec):
        super().__init__(spec)

        device = self.pre.device
        params = spec.params
        self.dt = 1e-3

        # --- Parámetros SFA ---
        self.eta = torch.tensor(params.get("eta", 1e-9), device=device)
        self.w_min = torch.tensor(params.get("w_min", 0.0), device=device)
        self.w_max = torch.tensor(params.get("w_max", 0.5), device=device)
        self.scale = torch.tensor(params.get("scale", 5e-4), device=device)
        self.norm_every = int(params.get("norm_every", 100))
        self.eps = 1e-12

        # --- Canal (AMPA=0, GABA=1, NMDA=2) ---
        channel = params.get("channel", 0)
        if channel == 0:       # AMPA
            self.chan_sign = 1.0
        elif channel == 1:     # GABA
            self.chan_sign = -1.0
        else:                  # NMDA
            self.chan_sign = 1.0
        self.chan_sign = torch.tensor(self.chan_sign, device=device)

        # --- Homeostasis ---
        # tasa media objetivo (sobre firing rate o V filtrado)
        self.target_rate = torch.tensor(params.get("target_rate", 0.1), device=device)
        self.beta = torch.tensor(params.get("beta", 1e-4), device=device)

        # --- Máscara flotante ---
        self.mask_float = self.mask.float()

        # Inicializa sinapsis no negativas
        self.weight.clamp_(self.w_min, self.w_max)

    @torch.no_grad()
    def _update(self):
        """Actualización de pesos SFA + homeostasis."""

        current_step = globals.simulator.local_circuit.current_step

        # -------------------------------------------------------------------
        # 1. Obtener derivadas pre y post
        # -------------------------------------------------------------------
        dotx = self.pre.dYdt     # [N_pre]
        doty = self.pos.dYdt     # [N_post]

        # -------------------------------------------------------------------
        # 2. Regla SFA: outer product (local)
        # -------------------------------------------------------------------
        # dw_sfa_ij = -eta * dotx_i * doty_j
        dw_sfa = -self.eta * torch.outer(dotx, doty)

        # -------------------------------------------------------------------
        # 3. Homeostasis multiplicativa de Turrigiano
        # -------------------------------------------------------------------
        # actividad post promedio (usa firing-rate, o |V|, o rectificación)
        r = self.pos.r

        epsilon = r - self.target_rate

        # dw_homeo_ij = -beta * epsilon * w_ij
        dw_homeo = -self.beta * epsilon * self.weight

        # -------------------------------------------------------------------
        # 4. Combinar reglas + signo del canal
        # -------------------------------------------------------------------
        dw = self.chan_sign * (dw_sfa + dw_homeo)

        # Aplicar máscara de conectividad
        self.weight.addcmul_(dw, self.mask_float)

        # -------------------------------------------------------------------
        # 5. Clamping no negativo
        # -------------------------------------------------------------------
        self.weight.clamp_(self.w_min, self.w_max)

        # -------------------------------------------------------------------
        # 6. Normalización periódica L2 por columna (para estabilidad)
        # -------------------------------------------------------------------
        if self.scale > 0 and self.norm_every > 0:
            if current_step % self.norm_every == 0:

                squared = (self.weight * self.mask_float).pow_(2)
                norm = squared.sum(dim=0, keepdim=True).sqrt_()
                norm.clamp_min_(self.eps)

                scale_factor = self.scale / norm
                self.weight.mul_(scale_factor * self.mask_float)


class STDPSFADense(StaticDense):
    """
    SFA-STDP Rule (Cleaned Version).
    Implements the learning rule derived from Slow Feature Analysis cost function.
    
    Kernel shape: Mexican Hat (Difference of Exponentials)
    Dynamics:
        Δw ~ - A * [ Pre * (Post * K) + Post * (Pre * K) ]
        where K is the high-pass filter (integral zero).
    """

    def __init__(self, spec: ConnectionSpec):
        super().__init__(spec)
        device = self.pre.device
        p = spec.params
        
        # --- 1. SFA STDP Parameters ---
        # "Fast" tau: width of the central depression (coincidence)
        # "Slow" tau: width of the lateral potentiation (integration)
        # Use defaults if not provided in sim.connect()
        self.tau_stdp_fast = torch.tensor(p.get("tau_stdp_fast", 20e-3), device=device)
        self.tau_stdp_slow = torch.tensor(p.get("tau_stdp_slow", 40e-3), device=device)
        
        # Learning rate (keep it small)
        self.A = torch.tensor(p.get("A", -1e-5), device=device)
        
        # Homeostasis / Normalization params
        self.scale = p.get("scale", 1.0)
        self.norm_every = p.get("norm_every", 100)
        
        # Check limits
        self.w_min = torch.tensor(p.get("w_min", 1e-10), device=device)
        self.w_max = torch.tensor(p.get("w_max", 1.0), device=device)

        # --- 2. Pre-compute Decay Factors ---
        # Assuming dt = 1ms (fixed for simplicity, can be passed as param)
        dt = 1e-3
        self.alpha_fast = torch.exp(-dt / self.tau_stdp_fast)
        self.alpha_slow = torch.exp(-dt / self.tau_stdp_slow)

        # --- 3. Kernel Balance Coefficients ---
        # We need integral(Kernel) = 0 to ignore DC component (rate).
        # Kernel = a1 * exp(-t/fast) + a2 * exp(-t/slow)
        # Integral = a1*fast + a2*slow = 0  =>  a2 = -a1 * (fast/slow)
        self.a1 = 1.0
        self.a2 = - (self.tau_stdp_fast / self.tau_stdp_slow)

        # --- 4. State Variables (Traces) ---
        # We need separate traces for the two time constants
        self.trace_pre_fast = torch.zeros(self.pre.size, device=device)
        self.trace_pre_slow = torch.zeros(self.pre.size, device=device)
        self.trace_pos_fast = torch.zeros(self.pos.size, device=device)
        self.trace_pos_slow = torch.zeros(self.pos.size, device=device)


    def _update(self):
        current_step = globals.simulator.local_circuit.current_step
        
        # --- A. Get Spikes ---
        # Handle delay buffer for Pre
        phase = (current_step - self.delay) % self.pre.delay_max
        pre_spikes = self.pre._spike_buffer[:, phase].float().squeeze(-1)
        pos_spikes = self.pos.get_spikes().float()

        # --- B. Update Traces ---
        # 1. Decay
        self.trace_pre_fast *= self.alpha_fast
        self.trace_pre_slow *= self.alpha_slow
        self.trace_pos_fast *= self.alpha_fast
        self.trace_pos_slow *= self.alpha_slow
        
        # 2. Add new spikes
        self.trace_pre_fast += pre_spikes
        self.trace_pre_slow += pre_spikes
        self.trace_pos_fast += pos_spikes
        self.trace_pos_slow += pos_spikes
        
        # --- C. Form the Filtered Signal (The "Derivative" proxy) ---
        # h(t) represents the high-pass filtered spike train
        h_pre = self.a1 * self.trace_pre_fast + self.a2 * self.trace_pre_slow
        h_pos = self.a1 * self.trace_pos_fast + self.a2 * self.trace_pos_slow
        
        # --- D. SFA Update Rule ---
        # Δw_ij = A * [ Pre_i * h_pos_j + h_pre_i * Post_j ]
        # Note: A is negative for excitatory connections, positive por inhibitory ones.
        update = self.A * (torch.outer(pre_spikes, h_pos) + torch.outer(h_pre, pos_spikes))
        
        # Apply mask and update
        self.weight += update * self.mask
        
        # Clamp to non-negative weights (Excitatory logic applied to magnitude)
        # Note: If this class is used for Inhibition, strictly speaking weights should be negative 
        # in the simulator logic, but stored as positive magnitudes here. 
        self.weight.clamp_(self.w_min, self.w_max)

        # --- E. Periodic Normalization (Competition) ---
        if self.norm_every > 0 and current_step % self.norm_every == 0:
            # L2 Normalization per column (per post-synaptic neuron)
            col_norms = self.weight.norm(p=2, dim=0, keepdim=True) + 1e-12
            self.weight.mul_(self.scale / col_norms)



class VogelsDense(StaticDense):
    """Inhibitory Spike-Timing-Dependent Plasticity (iSTDP) - Vogels et al. (2011).

    Implements a symmetric learning rule for inhibitory synapses designed to 
    balance excitation and inhibition (E/I balance) by maintaining a specific 
    target firing rate in the post-synaptic neurons.

    The rule potentiates weights for any pre-post coincidences (symmetric window)
    and applies a constant depression term to achieve homeostasis.

    Reference:
    Vogels, T. P., et al. (2011). Inhibitory plasticity balances 
    excitation and inhibition in sensory pathways and memory networks. Science.

    Attributes
    ----------
    eta : torch.Tensor
        Learning rate (applies to both potentiation and depression terms).
    tau : torch.Tensor
        Time constant for synaptic traces (typically 20ms).
    alpha_depression : torch.Tensor
        The calculated depression constant derived from target_rate.
        Formula: alpha = 2 * target_rate * tau
    w_max : torch.Tensor
        Maximum allowed weight value (weights are assumed positive magnitudes).
    x_pre : torch.Tensor
        Pre-synaptic spike traces.
    x_pos : torch.Tensor
        Post-synaptic spike traces.
    trace_decay : torch.Tensor
        Decay factor for traces per time step (exp(-dt/tau)).
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
        
        # 1. Validación de Canal
        # La regla de Vogels es EXCLUSIVA para inhibición. 
        # Asumimos que channel 1 es GABA.
        if spec.params.get("channel") != 1:
            # O lanzas error o warning, depende de cuan estricta sea tu lib.
            print(f"Warning: Vogels2011 rule applied to channel {spec.params.get('channel')}. "
                  "This rule is designed for GABAergic (channel 1) connections.")

        device = self.pre.device
        params = spec.params

        # Parámetros de Vogels
        # Nota: Vogels usa una sola tasa de aprendizaje 'eta', no A_plus/A_minus separados.
        self.eta = torch.tensor(params.get("eta", 1e-4), device=device)
        
        # Constante de tiempo (usualmente simétrica para iSTDP)
        self.tau = torch.tensor(params.get("tau", 20e-3), device=device)
        
        # Homeostasis: Target Rate
        target_rate = params.get("target_rate", 5.0) # Hz
        
        # Cálculo de alpha: 2 * rho * tau
        # Esto asegura que el peso se estabilice cuando la post-neurona dispare a target_rate.
        # (Asumiendo trazas con altura 1 y decaimiento tau).
        alpha_val = 2.0 * target_rate * self.tau.item()
        self.alpha_depression = torch.tensor(alpha_val, device=device)

        self.w_max = torch.tensor(params.get("w_max", 1.0), device=device)
        # w_min es implícitamente 0.0 en esta implementación

        # Traces
        self.x_pre = torch.zeros(self.pre.size, dtype=torch.float32, device=device)
        self.x_pos = torch.zeros(self.pos.size, dtype=torch.float32, device=device)

        # Decay factor (Time-driven)
        dt = 1e-3 # Asumo 1ms, si tu simulador tiene dt variable, úsalo aquí
        self.trace_decay = torch.exp(torch.tensor(-dt, device=device) / self.tau)

    def _update(self):
        # 1. Decay traces
        self.x_pre *= self.trace_decay
        self.x_pos *= self.trace_decay

        # 2. Get current spikes
        current_step = globals.simulator.local_circuit.current_step
        phase = (current_step - self.delay) % self.pre.delay_max
        
        pre_spikes = self.pre._spike_buffer[:, phase].float().squeeze(-1) # [Npre]
        pos_spikes = self.pos.get_spikes().float()                        # [Npos]

        # 3. Update traces (Spike arrival adds 1)
        self.x_pre += pre_spikes
        self.x_pos += pos_spikes

        # 4. Calculate Delta Weights (Symmetric Vogels Rule)
        # La regla conceptual es: delta_w = eta * ( pre * (x_pos - alpha) + post * x_pre )
        
        # Termino A (Driven by Pre spikes):
        # Cuando una pre-neurona dispara, mira el trace post-sináptico.
        # Si la post disparó hace poco (x_pos alto), Potencia.
        # Siempre aplica una pequeña depresión constante (alpha).
        # Shape: [Npre, 1] * [1, Npos] -> [Npre, Npos] via outer
        # Logic: pre_spikes_i * (x_pos_j - alpha)
        pre_driven_update = torch.outer(pre_spikes, self.x_pos - self.alpha_depression)

        # Termino B (Driven by Post spikes):
        # Cuando la post-neurona dispara, mira el trace pre-sináptico.
        # Si la pre disparó hace poco (x_pre alto), Potencia.
        # Logic: x_pre_i * pos_spikes_j
        pos_driven_update = torch.outer(self.x_pre, pos_spikes)

        # Suma total
        delta_w = self.eta * (pre_driven_update + pos_driven_update)

        # 5. Apply update
        # Solo actualizamos donde existe conexión (self.mask)
        self.weight += delta_w * self.mask
        
        # Clamp: Los pesos inhibitorios (magnitud) deben mantenerse positivos y bajo w_max
        self.weight.clamp_(0.0, self.w_max)