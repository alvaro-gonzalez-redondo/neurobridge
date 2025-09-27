from neurobridge import *
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm


# Definimos una simulación simple con una fuente de spikes aleatorios
# y un grupo de neuronas IF conectado todo-a-todo.
class BalancedRandomNetworkExperiment(Experiment):
    n_total_neurons: int = 2000
    exc_prop: float = 0.8
    conn_prob: float = 0.1


    def build_network(self):
        n_noise_neurons = 100
        n_excitatory_neurons = int(self.n_total_neurons * self.exc_prop)
        n_inhibitory_neurons = self.n_total_neurons - n_excitatory_neurons

        # --- Construcción del grafo de cómputo ---
        with self.sim.autoparent("graph"):
            noise = RandomSpikeNeurons(n_neurons=n_noise_neurons, firing_rate=5.0)  # Fuente de spikes aleatorios: 50 neuronas a 5 Hz
            exc_neurons = IFNeurons(n_neurons=n_excitatory_neurons) # Neuronas excitadoras
            inh_neurons = IFNeurons(n_neurons=n_inhibitory_neurons) # Neuronas inhibitorias

            # Conexiones
            n2e: StaticDenseConnection = (noise >> exc_neurons)(
                pattern="specific",
                weight=lambda pre, pos: torch.rand(len(pre)) * 2e-4, #TODO: Escalar pesos según el número de aferencias
                delay=1,
                synapse_class=StaticDenseConnection,
                mask = torch.rand((noise.size, exc_neurons.size), device=self.local_device) < self.conn_prob, #TODO: Esto es feo, habría que simplificar asignar valores aleatorios a mask sin tener que poner el tamaño de la matriz
            )

            e2e: StaticDenseConnection = (exc_neurons >> exc_neurons)(
                pattern="specific",
                #weight=lambda pre, pos: torch.rand(len(pre)) * 5e-6,  #TODO: Escalar pesos según el número de aferencias
                weight=lambda pre, pos: torch.rand(len(pre)) * 1e-6,  #TODO: Escalar pesos según el número de aferencias
                delay=1,
                #synapse_class=StaticDenseConnection,
                synapse_class=STDPDenseConnection,
                w_max = 3e-6,
                mask = torch.rand((exc_neurons.size, exc_neurons.size), device=self.local_device) < self.conn_prob, #TODO: Esto es feo, habría que simplificar asignar valores aleatorios a mask sin tener que poner el tamaño de la matriz
            )

            e2i: StaticDenseConnection = (exc_neurons >> inh_neurons)(
                pattern="specific",
                weight=lambda pre, pos: torch.rand(len(pre)) * 2e-5, #TODO: Escalar pesos según el número de aferencias
                delay=1,
                synapse_class=StaticDenseConnection,
                mask = torch.rand((exc_neurons.size, inh_neurons.size), device=self.local_device) < self.conn_prob, #TODO: Esto es feo, habría que simplificar asignar valores aleatorios a mask sin tener que poner el tamaño de la matriz
            )

            i2e: StaticDenseConnection = (inh_neurons >> exc_neurons)(
                pattern="specific",
                weight=lambda pre, pos: torch.rand(len(pre)) * 4e-5, #TODO: Escalar pesos según el número de aferencias
                delay=1,
                synapse_class=StaticDenseConnection,
                mask = torch.rand((inh_neurons.size, exc_neurons.size), device=self.local_device) < self.conn_prob, #TODO: Esto es feo, habría que simplificar asignar valores aleatorios a mask sin tener que poner el tamaño de la matriz
                channel = 1,
            )

            i2i: StaticDenseConnection = (inh_neurons >> inh_neurons)(
                pattern="specific",
                weight=lambda pre, pos: torch.rand(len(pre)) * 1e-5, #TODO: Escalar pesos según el número de aferencias
                delay=1,
                synapse_class=StaticDenseConnection,
                mask = torch.rand((inh_neurons.size, inh_neurons.size), device=self.local_device) < self.conn_prob, #TODO: Esto es feo, habría que simplificar asignar valores aleatorios a mask sin tener que poner el tamaño de la matriz
                channel = 1,
            )

        # --- Configuración de monitores ---
        with self.sim.autoparent("normal"):
            # Monitorizamos un subconjunto de neuronas de cada grupo
            self.spike_monitor = SpikeMonitor(
                [
                    noise.where_id(lambda i: i < 100),
                    exc_neurons.where_id(lambda i: i < 100),
                    inh_neurons.where_id(lambda i: i < 100),
                ]
            )

            self.state_monitor = VariableMonitor(
                [
                    exc_neurons.where_id(lambda i: i<3),
                ],
                ['V']
            )

    def on_finish(self):
        # Recuperamos y dibujamos los spikes
        fig, ax0 = plt.subplots()
        ax1 = ax0.twinx()
        id_sum = 0
        for idx, label in enumerate(["Noise", "Exc", "Inh"]):
            spikes = self.spike_monitor.get_spike_tensor(idx).cpu()
            spk_times, neurons = spikes[:, 1], spikes[:, 0]
            ax1.scatter(spk_times, neurons+id_sum, s=1, label=label, c=f"C{idx}")
            n_neurons = int(self.spike_monitor.filters[idx].nonzero(as_tuple=True)[0][-1]) + 1
            id_sum += n_neurons
            times, rate = smooth_spikes(spk_times, n_neurons=n_neurons, to_time=self.time)
            ax0.plot(times, rate, c=f"C{idx}")
            
        ax1.legend()
        plt.title(f"Spikes of different subpopulations")
        plt.xlabel("Time (steps)")
        ax0.set_ylabel("Neuron ID")
        ax1.set_ylabel("Spiking rate (Hz)")

        # Mostramos el voltaje de membrana de la primera neurona
        plt.figure()
        V = self.state_monitor.get_variable_tensor(0, 'V')
        plt.plot(V)

        plt.show()



# --- Ejecución de la simulación ---
if __name__ == "__main__":
    exp = BalancedRandomNetworkExperiment(sim=Simulator())
    exp.run(steps=1000)