from neurobridge import SimulatorEngine, RandomSpikeNeurons, SimpleIFNeurons, SpikeMonitor
import torch
import matplotlib.pyplot as plt


# Definimos una simulación simple con una fuente de spikes aleatorios
# y un grupo de neuronas IF conectado todo-a-todo.
class DemoSimulation(SimulatorEngine):

    def build_user_network(self, rank: int, world_size: int):
        # --- Construcción del grafo de cómputo ---
        with self.autoparent("graph"):
            # Fuente de spikes aleatorios: 50 neuronas a 5 Hz
            src = RandomSpikeNeurons(
                device=self.local_circuit.device, n_neurons=50, firing_rate=5.0
            )
            # Grupo de neuronas IF: 20 unidades
            tgt = SimpleIFNeurons(
                device=self.local_circuit.device, n_neurons=20, threshold=1.0, tau_membrane=0.5
            )
            # Conexión all-to-all con peso fijo y retardo de 1 paso
            (src >> tgt)(
                pattern="all-to-all",
                weight=lambda pre, pos: torch.rand(len(pre)) * 5e-2,
                delay=1,
            )

        # --- Configuración de monitores ---
        with self.autoparent("normal"):
            # Monitorizamos las 10 primeras neuronas de cada grupo
            self.spike_monitor = SpikeMonitor(
                [src.where_id(lambda i: i < 10), tgt.where_id(lambda i: i < 10)]
            )

    def plot_spikes(self):
        # Recuperamos y dibujamos los spikes
        for idx, label in zip([0, 1], ["Fuente", "IF"]):
            spikes = self.spike_monitor.get_spike_tensor(idx).cpu()
            times, neurons = spikes[:, 1], spikes[:, 0]
            plt.figure()
            plt.scatter(times, neurons, s=10)
            plt.title(f"Spikes: {label}")
            plt.xlabel("Tiempo (pasos)")
            plt.ylabel("ID de neurona")
        plt.show()


# --- Ejecución de la simulación ---
if __name__ == "__main__":
    with DemoSimulation() as sim:
        n_steps = 1000
        for _ in range(n_steps):
            sim.step()
        sim.plot_spikes()
