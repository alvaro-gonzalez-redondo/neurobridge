from neurobridge.engine import SimulatorEngine
from neurobridge.neuron_groups.if_neuron_group import IandF


class MyEngine(SimulatorEngine):

    def build_user_network(self):
        with self.device(0):
            # Crear poblaciones
            popA = IandF(size=12, delay=5)
            popB = IandF(size=12, delay=5)


if __name__ == "__main__":
    engine = MyEngine(n_gpus=-1)
    for _ in range(10):
        engine.step()
