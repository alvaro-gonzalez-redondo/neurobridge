import torch

from .bridge import BridgeNeuronGroup
from .utils import log_error, log

from .engine import Simulator

from tqdm import tqdm


class Experiment:

    sim: Simulator

    def __init__(self, sim:Simulator, **kwargs):
        self.sim = sim
        for key, value in kwargs.items():
            setattr(self, key, value)
        log("#################################################################")
        log("Neurobridge initialized. Building user network...")
        self.build_network()
        log("User network built successfully.")
        self.sim.initialize()
    
    @property
    def time(self) -> int:
        return self.sim.local_circuit.t.item()
    
    @property
    def local_rank(self) -> int:
        return self.sim.local_circuit.rank
    
    @property
    def world_size(self) -> int:
        return self.sim.world_size
    
    @property
    def local_device(self) -> torch.device:
        return self.sim.local_circuit.device

    def add_default_bridge(self, n_local_neurons: int, n_steps: int):
        """Add a default bridge for inter-GPU communication.

        Creates a BridgeNeuronGroup and adds it to the local circuit.

        Parameters
        ----------
        n_local_neurons : int
            Number of bridge neurons per GPU.
        n_steps : int
            Number of time steps to collect before synchronizing.
        """
        bridge = BridgeNeuronGroup(
            device = self.sim.local_circuit.device,
            rank = self.sim.local_circuit.rank,
            world_size = self.sim.world_size,
            n_local_neurons = n_local_neurons,
            n_bridge_steps = n_steps,
            spatial_dimensions = 2,
            delay_max = n_steps + 1,
        )
        self.sim.local_circuit.bridge = bridge
        self.sim.local_circuit.add_child(bridge)

    def build_network(self) -> None:
        """Build the user-defined neural network.

        This method must be implemented by subclasses to define the
        specific neural network structure for the simulation.

        Parameters
        ----------
        rank : int
            Rank (process index) in the distributed setup.
        world_size : int
            Number of processes in the distributed setup.
        device : str
            Device identifier used in this part of the network.

        Raises
        ------
        NotImplementedError
            If not implemented by a subclass.
        """
        raise NotImplementedError(
            "`build_network` in `Experiment` must be implemented."
        )
    
    def on_start(self) -> None:
        pass

    def pre_step(self) -> None:
        pass

    def pos_step(self) -> None:
        pass
    
    def on_finish(self) -> None:
        pass

    def run(self, steps:int) -> None:
        """Runs the current experiment.
        """
        try:
            self.on_start()
            for t in tqdm(range(steps)):
                self.pre_step()
                self.sim.step()
                self.pos_step()
            self.on_finish()

        except Exception as e:
            log_error(f"ERROR: {e}")
            import traceback

            log_error(traceback.format_exc())
        finally:
            self.sim.close()