from neurobridge.core.synaptic_group import SynapticGroup
import torch


class STDPSynapse(SynapticGroup):

    def __init__(self, pre, pos, idx_pre, idx_pos, delay, weight,
                 A_plus=0.01, A_minus=0.012,
                 tau_plus=20.0, tau_minus=20.0,
                 dt=1.0,
                 w_min=0.0, w_max=1.0):
        super().__init__(pre, pos, idx_pre, idx_pos, delay)
        self.weight = weight.to(device=self.device)

        # Parámetros de aprendizaje
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.dt = dt
        self.w_min = w_min
        self.w_max = w_max

        # Trazas por neurona (no por sinapsis)
        self.x_pre = torch.zeros(pre.size, dtype=torch.float32, device=self.device)
        self.x_pos = torch.zeros(pos.size, dtype=torch.float32, device=self.device)

        # Decaimiento precomputado
        self.alpha_pre = torch.exp(-dt / tau_plus)
        self.alpha_pos = torch.exp(-dt / tau_minus)


    def propagate(self):
        valid = self._valid_indices
        if valid is None:
            return

        tgt = self.idx_pos[valid]
        wgt = self.weight[valid]
        current = torch.zeros(self.pos.size, dtype=torch.float32, device=self.device)
        current.index_add_(0, tgt, wgt)
        self.pos.inject_currents(current)


    def update(self):
        # Tiempo actual
        t = self.pre.t

        # 1. Decaer trazas de forma global
        self.x_pre *= self.alpha_pre
        self.x_pos *= self.alpha_pos

        # 2. Recuperar spikes actuales
        pre_spikes = self.pre.spike_buffer[(t - self.delay) % self.pre.delay]
        pos_spikes = self.pos.spike_buffer[(t - 1) % self.pos.delay]

        # 3. Actualizar trazas
        self.x_pre[pre_spikes] += 1.0
        self.x_pos[pos_spikes] += 1.0

        # 4. LTP: si spike PRE, usar traza post acumulada
        valid_pre = pre_spikes[self.idx_pre]
        if valid_pre.any():
            indices = valid_pre.nonzero(as_tuple=True)[0]
            pos = self.idx_pos[indices]
            dw = self.A_plus * self.x_pos[pos]
            self.weight[indices] += dw

        # 5. LTD: si spike POST, usar traza pre acumulada
        valid_pos = pos_spikes[self.idx_pos]
        if valid_pos.any():
            indices = valid_pos.nonzero(as_tuple=True)[0]
            pres = self.idx_pre[indices]
            dw = -self.A_minus * self.x_pre[pres]
            self.weight[indices] += dw

        # 6. Limitar peso
        self.weight.clamp_(self.w_min, self.w_max)
