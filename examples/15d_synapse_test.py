from neurobridge import *
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

class SynapseTests(Experiment):
    def build_network(self):
        #with self.sim.autoparent("graph"):
        with self.sim.autoparent("normal"):
            self.inputs = ParrotNeurons(2)
            self.output = LIFNeurons(1)

            _ = self.sim.connect(
                self.inputs.where_idx(lambda i: i==0), self.output, 
                connection_type=StaticDense,
                weight=Constant(1e-5),
                channel=0,
            )

            _ = self.sim.connect(
                self.inputs.where_idx(lambda i: i==1), self.output, 
                connection_type=StaticDense,
                weight=Constant(1e-5),
                channel=1,
            )


        with self.sim.autoparent("normal"):
            self.spike_monitor = SpikeMonitor([self.inputs, self.output])

            self.voltage_monitor = VariableMonitor(
                  [self.output],
                  ['V', 'spikes', 'channel_currents@0', 'channel_currents@1']
            )


    def on_start(self, **kwargs):
        pass


    def pre_step(self):
        step = self.sim.local_circuit.current_step
        t = step*1e-3

        if step==100:
            self.inputs._input_spikes[0] = True
        if step==200:
            self.inputs._input_spikes[1] = True
        if step==300:
            self.inputs._input_spikes[:] = True

    def pos_step(self):
        pass


    def on_finish(self):
        if False:
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
        #ax2 = ax1.twinx()

        V = self.voltage_monitor.get_variable_tensor(0, 'V')
        ax0.plot(V, color='C0', label='Voltage')
        ax0.legend(loc='upper left')
        ax0.set_ylabel('Membrane potential (V)')
        ax0.set_xlabel('Time (ms)')

        #spikes = self.voltage_monitor.get_variable_tensor(0, 'spikes')
        #ax2.vlines(spikes.nonzero(as_tuple=True), ymin=0,ymax=1, color='black')
        #ax2.get_yaxis().set_visible(False)
        
        ampa = self.voltage_monitor.get_variable_tensor(0, 'channel_currents@0')
        ax1.plot(ampa, color='C1', label='AMPA')
        gaba = self.voltage_monitor.get_variable_tensor(0, 'channel_currents@1')
        ax1.plot(gaba, color='C2', label='GABA')
        net = ampa+gaba
        ax1.plot(net, color='C3', label='Net')
        ax1.set_ylabel('Channel current')

        ax1.grid()
        ax1.legend(loc='upper right')

        plt.show()


if __name__ == "__main__":
    exp = SynapseTests(sim=Simulator(seed=0))
    simulation_time = 0.4
    exp.run(time=simulation_time)