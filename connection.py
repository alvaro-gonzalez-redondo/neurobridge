from typing import Optional, Type
from dataclasses import dataclass, field
import torch
from .core import GPUNode


@dataclass()
class ConnectionSpec:
    pre: GPUNode
    pos: GPUNode
    src_idx: torch.Tensor
    tgt_idx: torch.Tensor
    weight: Optional[torch.Tensor] = None
    delay: Optional[torch.Tensor] = None
    connection_type: Optional[Type[GPUNode]] = None
    params: dict = field(default_factory=dict)