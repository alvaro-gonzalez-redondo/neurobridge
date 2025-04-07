from neurobridge.core.synaptic_group import SynapticGroup
import torch


class StaticSynapse(SynapticGroup):

    def __init__(self, pre, pos, idx_pre, idx_pos, delay, weight):
        super().__init__(pre, pos, idx_pre, idx_pos, delay)
        self.weight = weight.to(device=pre.device)


    def propagate(self):
        valid = self._valid_indices
        if valid is None:
            return

        tgt = self.idx_pos[valid]
        wgt = self.weight[valid]
        current = torch.zeros(self.pos.size, dtype=torch.float32, device=self.device)
        current.index_add_(0, tgt, wgt)
        self.pos.inject_currents(current)


    def update(self): pass

