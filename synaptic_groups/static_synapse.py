from __future__ import annotations
import torch
from torch import Tensor
from neurobridge.core.synaptic_group import SynapticGroup
from typing import Union


class StaticSynapse(SynapticGroup):
    weight: Tensor

    def __init__(
        self,
        pre,
        pos,
        idx_pre: Tensor,
        idx_pos: Tensor,
        delay: Tensor,
        weight: Union[float, Tensor],
    ) -> None:
        super().__init__(pre, pos, idx_pre, idx_pos, delay)
        self.weight = (
            torch.full((len(idx_pre),), float(weight), dtype=torch.float32, device=pre.device)
            if isinstance(weight, (int, float))
            else weight.to(device=pre.device)
        )

    def propagate(self) -> None:
        valid = self._valid_indices
        if valid is None:
            return

        tgt = self.idx_pos[valid]
        wgt = self.weight[valid]
        net_current = torch.zeros(self.pos.size, dtype=torch.float32, device=self.device)
        net_current.index_add_(0, tgt, wgt)
        self.pos.inject_currents(net_current)

    def update(self) -> None:
        pass
