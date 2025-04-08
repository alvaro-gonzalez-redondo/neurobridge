from __future__ import annotations
import torch
from torch import Tensor
from typing import Union
from neurobridge.core.synaptic_group import SynapticGroup


class STDPSynapse(SynapticGroup):
    weight: Tensor
    A_plus: float
    A_minus: float
    tau_plus: float
    tau_minus: float
    dt: float
    w_min: float
    w_max: float
    x_pre: Tensor
    x_pos: Tensor
    alpha_pre: float
    alpha_pos: float

    def __init__(
        self,
        pre,
        pos,
        idx_pre: Tensor,
        idx_pos: Tensor,
        delay: Tensor,
        weight: Union[float, Tensor],
        A_plus: float = 0.01,
        A_minus: float = 0.012,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
        dt: float = 1.0,
        w_min: float = 0.0,
        w_max: float = 1.0
    ) -> None:
        super().__init__(pre, pos, idx_pre, idx_pos, delay)
        self.weight = (
            torch.full((len(idx_pre),), float(weight), dtype=torch.float32, device=self.device)
            if isinstance(weight, (int, float))
            else weight.to(device=self.device)
        )

        self.A_plus = A_plus
        self.A_minus = A_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.dt = dt
        self.w_min = w_min
        self.w_max = w_max

        self.x_pre = torch.zeros(pre.size, dtype=torch.float32, device=self.device)
        self.x_pos = torch.zeros(pos.size, dtype=torch.float32, device=self.device)

        self.alpha_pre = float(torch.exp(torch.tensor(-dt / tau_plus)))
        self.alpha_pos = float(torch.exp(torch.tensor(-dt / tau_minus)))

    def propagate(self) -> None:
        valid = self._valid_indices
        if valid is None:
            return

        tgt = self.idx_pos[valid]
        wgt = self.weight[valid]
        current = torch.zeros(self.pos.size, dtype=torch.float32, device=self.device)
        current.index_add_(0, tgt, wgt)
        self.pos.inject_currents(current)

    def update(self) -> None:
        t = self.pre.t

        self.x_pre *= self.alpha_pre
        self.x_pos *= self.alpha_pos

        pre_spikes = self.pre.spike_buffer[(t - self.delay) % self.pre.delay]
        pos_spikes = self.pos.spike_buffer[(t - 1) % self.pos.delay]

        self.x_pre[pre_spikes] += 1.0
        self.x_pos[pos_spikes] += 1.0

        valid_pre = pre_spikes[self.idx_pre]
        if valid_pre.any():
            indices = valid_pre.nonzero(as_tuple=True)[0]
            post_indices = self.idx_pos[indices]
            dw = self.A_plus * self.x_pos[post_indices]
            self.weight[indices] += dw

        valid_pos = pos_spikes[self.idx_pos]
        if valid_pos.any():
            indices = valid_pos.nonzero(as_tuple=True)[0]
            pre_indices = self.idx_pre[indices]
            dw = -self.A_minus * self.x_pre[pre_indices]
            self.weight[indices] += dw

        self.weight.clamp_(self.w_min, self.w_max)
