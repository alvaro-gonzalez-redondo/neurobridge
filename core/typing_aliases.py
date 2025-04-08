# neurobridge/core/typing_aliases.py
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neurobridge.core.neuron_group import NeuronGroup
    from neurobridge.core.synaptic_group import SynapticGroup
    from neurobridge.local_circuit import LocalCircuit
    from neurobridge.bridges import AxonalBridge
