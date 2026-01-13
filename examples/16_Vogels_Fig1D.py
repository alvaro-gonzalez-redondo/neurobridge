from neurobridge import *
import torch
import numpy as np
import matplotlib.pyplot as plt

class VogelsFig1DExperiment(Experiment):
    n_input_groups = 8
    n_excitatory_inputs_per_group = 100
    n_inhibitory_inputs_per_group = 25
    n_inputs_per_group = n_excitatory_inputs_per_group + n_inhibitory_inputs_per_group
    n_total_inputs = n_input_groups * n_inputs_per_group

    change_input_each = 50
    max_firing_rate = 50.0


    def build_network(self):
        is_exc = lambda i: (i % self.n_inputs_per_group) < self.n_excitatory_inputs_per_group
        is_inh = lambda i: (i % self.n_inputs_per_group) >= self.n_excitatory_inputs_per_group

        # Entradas y neurona de salida

        #with self.sim.autoparent("graph"):
        with self.sim.autoparent("normal"):
            self.inputs = RandomSpikeNeurons(self.n_total_inputs, firing_rate=0.0)
            #self.neuron = LIFNeurons(1)
            #self.neuron = AdExNeurons(1)
            #self.neuron = ALIFNeurons(1)
            self.neuron = PowerLawALIFNeurons(1, n_basis=5, tau_min=0.010, tau_max=20.0)


            exc_conns = self.sim.connect(self.inputs.where_idx(is_exc), self.neuron, 
                StaticDense, pattern="all-to-all", delay=0, weight=Uniform(), channel=0,
            )
            inh_conns = self.sim.connect(self.inputs.where_idx(is_inh), self.neuron,
                VogelsDense, pattern="all-to-all", delay=0, weight=Uniform(), channel=1,
                eta=1e-2, target_rate=5.0, w_max=100.0
            )

            for i_group in range(self.n_input_groups):
                from_i = self.n_inputs_per_group * i_group
                to_i = self.n_inputs_per_group * (i_group+1)
                exc_conns.weight[from_i:to_i, :] = np.random.rand()
        
        # Monitores

        with self.sim.autoparent("normal"):
            #self.spike_monitor = SpikeMonitor([self.inputs, self.neuron])
            self.spike_monitor = SpikeMonitor([self.neuron])

            adaptation_vars = []
            if type(self.neuron) is ALIFNeurons:
                adaptation_vars += ['A']
            elif type(self.neuron) is PowerLawALIFNeurons:
                adaptation_vars += [f'A@{i}' for i in range(self.neuron.n_basis)]

            self.voltage_monitor = VariableMonitor(
                  [self.neuron],
                  ['V', 'spikes', 'channel_currents@0', 'channel_currents@1'] + adaptation_vars
            )

            self.weight_monitor = VariableMonitor([inh_conns], ["weight"])

    def pre_step(self):
        if (self.current_step % self.change_input_each) == 0:
            for i_group in range(self.n_input_groups):
                from_i = self.n_inputs_per_group * i_group
                to_i = self.n_inputs_per_group * (i_group+1)
                self.inputs.firing_rate[from_i:to_i] = np.random.rand()*self.max_firing_rate


    def on_finish(self):
        
        # Spikes de entradas

        _fig, ax0 = plt.subplots()
        ax1 = ax0.twinx()
        id_sum = 0
        for idx, group in enumerate(self.spike_monitor.groups):
            label = group.name
            spikes = self.spike_monitor.get_spike_tensor(idx).cpu()
            # Graficas
            spk_steps, spk_neurons = spikes[:, 1], spikes[:, 0]
            spk_times = spk_steps*1e-3

            ax1.scatter(spk_times, spk_neurons+id_sum, s=1, label=label, c=f"C{idx}")
            id_sum += group.size
            times, rate = smooth_spikes(spk_steps, n_neurons=group.size, to_step=self.current_step, sigma=1.0)
            ax0.plot(times, rate, c=f"C{idx}")
        
        ax1.legend(loc="lower right")
        plt.title(f"Spikes from inputs")
        plt.xlabel("Time (seconds)")
        ax0.set_ylabel("Spiking rate (Hz)")
        ax1.set_ylabel("Neuron ID")


        # Voltaje de membrana y canales

        fig, ax0 = plt.subplots()
        ax1 = ax0.twinx()

        #V = self.voltage_monitor.get_variable_tensor(0, 'V')
        #ax0.plot(V, color='C0')
        spikes = self.voltage_monitor.get_variable_tensor(0, 'spikes')
        ax1.vlines(spikes.nonzero(as_tuple=True), ymin=0,ymax=1, color='black')
        ax1.get_yaxis().set_visible(False)
        
        ampa = self.voltage_monitor.get_variable_tensor(0, 'channel_currents@0')
        ax0.plot(ampa, color='C1', label='AMPA')
        gaba = self.voltage_monitor.get_variable_tensor(0, 'channel_currents@1')
        ax0.plot(gaba, color='C2', label='GABA')
        net = ampa+gaba
        ax0.plot(net, color='C3', label='Net')

        ax0.grid()
        ax0.legend()


        # Trazas de la power-law o la ALIF

        adaptation_vars = []
        if type(self.neuron) is ALIFNeurons:
            adaptation_vars += ['A']
        elif type(self.neuron) is PowerLawALIFNeurons:
            adaptation_vars += [f'A@{i}' for i in range(self.neuron.n_basis)]

        if adaptation_vars:
            fig, ax0 = plt.subplots()
            ax1 = ax0.twinx()
            ax1.vlines(spikes.nonzero(as_tuple=True), ymin=0,ymax=1, color='black')
            ax1.get_yaxis().set_visible(False)

            for i, var in enumerate(adaptation_vars):
                A = self.voltage_monitor.get_variable_tensor(0, var)
                ax0.plot(A, label=f'Adaptation variable #{i}')
            ax0.legend()


        # Pesos sinápticos

        fig, ax0 = plt.subplots()
        w_values = self.weight_monitor.get_variable_tensor(0, "weight").numpy()

        for i_group in range(self.n_input_groups):
            from_i = self.n_inputs_per_group * i_group
            to_i = self.n_inputs_per_group * (i_group+1)
            mean_w = np.mean(w_values[:,from_i:to_i], axis=1)
            ax0.plot(mean_w)


        plt.show()


if __name__ == "__main__":
    sim = Simulator(seed=42)
    exp = VogelsFig1DExperiment(sim=sim)
    exp.run(time=100.0)
    