from typing import Any
import torch

from neurobridge.utils import block_distance_connect, resolve_param


# ---- tests ----

def test_distance_probabilistic():
    src = torch.arange(5).float().unsqueeze(1)   # posiciones 1D
    tgt= torch.arange(5).float().unsqueeze(1)
    src,tgt = block_distance_connect(src,tgt,sigma=1.0,p_max=0.5)
    print("Probabilistic: ", list(zip(src.tolist(), tgt.tolist())))

def test_distance_fanin():
    src = torch.arange(5).float().unsqueeze(1)
    tgt= torch.arange(5).float().unsqueeze(1)
    src,tgt = block_distance_connect(src,tgt,fanin=2)
    print("Fanin=2: ", list(zip(src.tolist(), tgt.tolist())))

def test_distance_fanout():
    src = torch.arange(5).float().unsqueeze(1)
    tgt= torch.arange(5).float().unsqueeze(1)
    src,tgt = block_distance_connect(src,tgt,fanout=2)
    print("Fanout=2: ", list(zip(src.tolist(), tgt.tolist())))

def test_resolve_param():
    class Dummy: pass
    src = Dummy(); tgt = Dummy()
    src.device = torch.device("cpu")
    src.positions = torch.arange(5).float().unsqueeze(1)
    tgt.positions = torch.arange(5).float().unsqueeze(1)
    src_ids = torch.tensor([0,1,2])
    tgt_ids = torch.tensor([1,2,3])

    w_const = resolve_param(0.5,src_ids,tgt_ids,src,tgt,1.0,torch.float32)
    print("Const weights:", w_const)

    w_tensor = resolve_param(torch.tensor([0.1,0.2,0.3]),src_ids,tgt_ids,src,tgt,1.0,torch.float32)
    print("Tensor weights:", w_tensor)

    w_func = resolve_param(lambda s,t,ps,pt: torch.norm(ps-pt,dim=1), src_ids,tgt_ids,src,tgt,0.0,torch.float32)
    print("Func weights:", w_func)

# Run tests
if __name__=="__main__":
    test_distance_probabilistic()
    test_distance_fanin()
    test_distance_fanout()
    test_resolve_param()
