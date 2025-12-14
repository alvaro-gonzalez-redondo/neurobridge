from neurobridge import *
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

class SFAToyProblem(Experiment):
    def build_network(self):
        #with self.sim.autoparent("graph"):
        with self.sim.autoparent("normal"):
            self.inputs = LIFNeurons(1000)
            self.output = LIFNeurons(1)

            syn_params = {
                'connection_type': STDPSFADense,
                'weight': Constant(1e-8),
                'tau_stdp_fast': 20e-3,
                'tau_stdp_slow': 100e-3,
                'scale': 5e-4,
                'norm_every': -1,
                'A': -1e-9, #Learning rate
            }

            self.i2o = self.sim.connect(self.inputs, self.output, **syn_params)
        
        with self.sim.autoparent("normal"):
            self.spike_monitor = SpikeMonitor([self.inputs, self.output])
            self.weight_monitor = VariableMonitor([self.i2o], ["weight"])

    def on_start(self, **kwargs):
        pass

    def pre_step(self):
        step = self.sim.local_circuit.current_step
        n_neurons = self.inputs.size
        self.inputs._input_spikes[step % n_neurons] = True

        if step==500:
            self.output._input_spikes[0] = True


    def pos_step(self):
        pass


    def on_finish(self):

        # Pesos
        fig, ax0 = plt.subplots()
        w_values = self.weight_monitor.get_variable_tensor(0, "weight")
        ax0.plot(w_values[-1,:])
        ax0.legend(loc="lower right")
        plt.xlabel("Time diff (steps)")
        print(torch.sum(w_values[-1,:]-1e-8))

        # Spikes
        fig, ax0 = plt.subplots()
        ax1 = ax0.twinx()
        id_sum = 0
        for idx, label in enumerate(["Inputs", "Ouput"]):
            spikes = self.spike_monitor.get_spike_tensor(idx).cpu()
            spk_steps, spk_neurons = spikes[:, 1], spikes[:, 0]
            spk_times = spk_steps*1e-3
            ax1.scatter(spk_times, spk_neurons+id_sum, s=1, label=label, c=f"C{idx}")
            n_neurons = self.spike_monitor.filters[idx].numel()
            id_sum += n_neurons
            times, rate = smooth_spikes(spk_steps, n_neurons=n_neurons, to_step=self.current_step, sigma=0.05)
            ax0.plot(times, rate, label=label, c=f"C{idx}", linewidth=5 if label=="y" else 2)
        
        ax0.legend(loc="lower right")
        plt.title(f"Spikes from different subpopulations")
        plt.xlabel("Time (seconds)")
        ax0.set_ylabel("Spiking rate (Hz)")
        ax1.set_ylabel("Neuron ID")

        plt.show()


if __name__ == "__main__":
    exp = SFAToyProblem(sim=Simulator(seed=0))
    simulation_time = 1.0
    exp.run(steps=int(1000*simulation_time))