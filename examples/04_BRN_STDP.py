from neurobridge import *
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm


# Definimos una simulación simple con una fuente de spikes aleatorios
# y un grupo de neuronas IF conectado todo-a-todo.
class BalancedRandomNetworkExperiment(Experiment):
    n_total_neurons: int = 1_000
    exc_prop: float = 0.8
    conn_prob: float = 0.1


    def build_network(self):
        n_noise_neurons = 100
        n_excitatory_neurons = int(self.n_total_neurons * self.exc_prop)
        n_inhibitory_neurons = self.n_total_neurons - n_excitatory_neurons

        #with self.sim.autoparent("graph"):
        with self.sim.autoparent("normal"):
            noise = RandomSpikeNeurons(n_neurons=n_noise_neurons, firing_rate=5.0)
            exc_neurons = IFNeurons(n_neurons=n_excitatory_neurons)
            inh_neurons = IFNeurons(n_neurons=n_inhibitory_neurons)

            if True:
                n2e = (noise >> exc_neurons)(
                    pattern="random", p=self.conn_prob,
                    synapse_class=StaticDense,
                    weight=lambda pre_idx, tgt_idx, pre_pos, tgt_pos: torch.rand(pre_idx.numel(), device=self.local_device) * 2e-4,
                    delay=0,
                )

            if True:
                e2e_ampa = (exc_neurons >> exc_neurons)(
                    pattern="random", p=self.conn_prob,
                    synapse_class=STDPDense,
                    weight=lambda pre_idx, tgt_idx, pre_pos, tgt_pos: torch.rand(pre_idx.numel(), device=self.local_device) * 1e-6,
                    delay=5,
                    w_max=1e-6,
                    A_plus=1e-8, A_minus=1.2e-8, oja_decay=3e-3,
                )

                e2e_nmda = (exc_neurons >> exc_neurons)(
                    pattern="random", p=self.conn_prob,
                    synapse_class=STDPDense,
                    weight=lambda pre_idx, tgt_idx, pre_pos, tgt_pos: torch.rand(pre_idx.numel(), device=self.local_device) * 1e-6,
                    delay=5,
                    w_max=1e-6,
                    channel=2,
                    A_plus=1e-8, A_minus=1.2e-8, oja_decay=3e-3,
                )

            if True:
                e2i = (exc_neurons >> inh_neurons)(
                    pattern="random", p=self.conn_prob,
                    synapse_class=StaticDense,
                    weight=lambda pre_idx, tgt_idx, pre_pos, tgt_pos: torch.rand(pre_idx.numel(), device=self.local_device) * 0.075e-4,
                    delay=0,
                )

            if True:
                i2e = (inh_neurons >> exc_neurons)(
                    pattern="random", p=self.conn_prob,
                    synapse_class=StaticDense,
                    weight=lambda pre_idx, tgt_idx, pre_pos, tgt_pos: torch.rand(pre_idx.numel(), device=self.local_device) * 1e-4,
                    delay=0,
                    channel=1,
                )
            
            if False:
                i2i = (inh_neurons >> inh_neurons)(
                    pattern="random", p=self.conn_prob,
                    synapse_class=StaticDense,
                    weight=lambda pre_idx, tgt_idx, pre_pos, tgt_pos: torch.rand(pre_idx.numel(), device=self.local_device) * 1e-5,
                    delay=1,
                    channel=1,
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
                [exc_neurons.where_id(lambda i: i<1),],
                ['V', 'spikes', 'channel_currents@0', 'channel_currents@1', 'channel_currents@2']
            )


            e2e_ampa.filter[:,:] = False    ##task #neurobridge Incluir filtro tipo máscara 2d o función lambda a conexiones densas 🔺
            e2e_ampa.filter[0,:10] = True
            e2e_nmda.filter[:,:] = False
            e2e_nmda.filter[0,:10] = True
            self.weight_monitor = VariableMonitor(
                [
                    e2e_ampa,
                    e2e_nmda,
                ], 
                ["weight"]
            )


    def on_finish(self):

        # Recuperamos y dibujamos los spikes

        fig, ax0 = plt.subplots()
        ax1 = ax0.twinx()
        id_sum = 0
        for idx, label in enumerate(["Noise", "Exc", "Inh"]):
            spikes = self.spike_monitor.get_spike_tensor(idx).cpu()
            spk_steps, spk_neurons = spikes[:, 1], spikes[:, 0]
            spk_times = spk_steps*1e-3
            ax1.scatter(spk_times, spk_neurons+id_sum, s=1, label=label, c=f"C{idx}")
            n_neurons = int(self.spike_monitor.filters[idx].nonzero(as_tuple=True)[0][-1]) + 1
            id_sum += n_neurons
            times, rate = smooth_spikes(spk_steps, n_neurons=n_neurons, to_step=self.step, sigma=0.1)
            ax0.plot(times, rate, c=f"C{idx}")
            
        ax1.legend(loc='lower right')
        plt.title(f"Spikes of different subpopulations")
        plt.xlabel("Time (steps)")
        ax0.set_ylabel("Spiking rate (Hz)")
        ax1.set_ylabel("Neuron ID")


        # Mostramos el voltaje de membrana de la primera neurona

        fig, ax0 = plt.subplots()
        ax1 = ax0.twinx()
        ax2 = ax0.twinx()
        
        V = self.state_monitor.get_variable_tensor(0, 'V')
        ax0.plot(V, color='C0')
        spikes = self.state_monitor.get_variable_tensor(0, 'spikes')
        ax2.vlines(spikes.nonzero(as_tuple=True), ymin=0,ymax=1, color='black')
        ax2.get_yaxis().set_visible(False)
        
        ampa = self.state_monitor.get_variable_tensor(0, 'channel_currents@0')
        ax1.plot(ampa, color='C1', label='AMPA')
        gaba = self.state_monitor.get_variable_tensor(0, 'channel_currents@1')
        ax1.plot(gaba, color='C2', label='GABA')
        nmda = self.state_monitor.get_variable_tensor(0, 'channel_currents@2')
        ax1.plot(nmda, color='C3', label='NMDA')
        ax1.grid()
        ax1.legend()


        # Mostramos pesos

        if True:
            fig, ax0 = plt.subplots()
            w_values = self.weight_monitor.get_variable_tensor(0, "weight")
            ax0.plot(w_values[:,0,:])

        
        plt.show()



# --- Ejecución de la simulación ---
if __name__ == "__main__":
    exp = BalancedRandomNetworkExperiment(sim=Simulator(seed=0))
    simulation_time = 10.0
    exp.run(steps=int(1000 * simulation_time))