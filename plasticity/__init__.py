"""
Modular plasticity framework for Neurobridge.

This package implements a unified framework for synaptic plasticity based on
the e-prop formulation: Δw = L' · e

All plasticity rules decompose into three components:
- Eligibility: Local synaptic traces
- LearningSignal: Error/modulation signals at postsynaptic neurons
- UpdatePolicy: How to combine them and apply constraints

Available rules (Phase 1):
- STDP (Spike-Timing-Dependent Plasticity)
- Vogels (inhibitory STDP with homeostasis)
- SFA (Spike-Frequency Adaptation)

Usage:
------
>>> from neurobridge.plasticity import PlasticSparse
>>> from neurobridge import ConnectionSpec
>>>
>>> # Simple declarative configuration
>>> spec = ConnectionSpec(
...     pre=source, pos=target,
...     src_idx=..., tgt_idx=...,
...     weight=..., delay=...,
...     params={"plasticity": {"name": "stdp"}}
... )
>>> conn = PlasticSparse(spec)
"""

from .base import EligibilityBase, LearningSignalBase, UpdatePolicyBase
from .rule import PlasticityRule

__all__ = [
    "EligibilityBase",
    "LearningSignalBase",
    "UpdatePolicyBase",
    "PlasticityRule",
]
