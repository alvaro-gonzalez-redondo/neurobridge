from neurobridge.core.synaptic_group import SynapticGroup
import torch


class StaticSynapse(SynapticGroup):

    def __init__(self, pre, post, idx_pre, idx_post, delay, weight):
        super().__init__(pre, post, idx_pre, idx_post, delay)
        self.weight = weight.to(device=pre.device)


    def propagate(self):
        valid = self._valid_indices
        if valid is None:
            return

        tgt = self.idx_post[valid]
        wgt = self.weight[valid]
        current = torch.zeros(self.post.size, dtype=torch.float32, device=self.device)
        current.index_add_(0, tgt, wgt)
        self.post.inject_currents(current)


    def update(self): pass

