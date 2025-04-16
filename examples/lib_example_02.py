from neurobridge.all import *
#from all import *
from matplotlib import pyplot as plt
from tqdm import tqdm

from pyplot_tools import *


class RandomInputSimulation(SimulatorEngine):

    def __init__(self):
        super().__init__(autoparenting_nodes=True)


    def build_user_network(self, rank: int, world_size: int):
        n_src_neurons = 100
        n_tgt_neurons = 100

        src_neurons = RandomSpikeGenerator(
            device = self.local_circuit.device,
            n_neurons = n_src_neurons,
            firing_rate = 10.0,
        )

        tgt_neurons = IFNeuronGroup(
            device = self.local_circuit.device,
            n_neurons = n_tgt_neurons,
        )

        stdp_conns = (src_neurons >> tgt_neurons)(
            pattern = 'all-to-all',
            synapse_class = STDPSynapse,
            weight = lambda pre,pos: torch.rand(len(pre)) * 2e-2
        )

        # Registramos las neuronas para poder monitorearlas
        self.spike_monitor = SpikeMonitor(
            [
                src_neurons, #src_neurons.where_id(lambda i: i%10==0),
                tgt_neurons, #tgt_neurons.where_pos(lambda p: p[:,0]>0.5)
            ]
        )
        self.voltage_monitor = VariableMonitor([tgt_neurons], ['V'])
        self.weight_monitor = VariableMonitor(
            [
                stdp_conns.where_id(lambda ids: ids%101 == 0)
            ], 
            ['weight']
        )


    def plot_spikes(self):
        
        # Source spikes
        if True:
            plt.figure()
            src_spikes = self.spike_monitor.get_spike_tensor(0)
            ot, oi = src_spikes[:,1], src_spikes[:,0]
            plt.scatter(ot, oi, s=4)

            tgt_spikes = self.spike_monitor.get_spike_tensor(1)
            ot, oi = tgt_spikes[:,1], tgt_spikes[:,0]
            plt.scatter(ot, oi, s=4)

            show_or_save_plot(filename=f"rank{self.rank}_output_1.png", log=log)

        # Target voltages
        if True:
            plt.figure()
            v_values = self.voltage_monitor.get_variable_tensor(0, 'V')
            plt.plot(v_values)
            show_or_save_plot(filename=f"rank{self.rank}_output_2.png", log=log)

        # Synaptic weights
        if True:
            plt.figure()
            w_values = self.weight_monitor.get_variable_tensor(0, 'weight')
            plt.plot(w_values)
            show_or_save_plot(filename=f"rank{self.rank}_output_3.png", log=log)


# Main

try:
    with RandomInputSimulation() as engine:
        simulation_length = 1
        simulation_steps = simulation_length * 1000
        for _ in tqdm(range(simulation_steps)):
            engine.step()
        
        engine.plot_spikes()

except Exception as e:
    log_error(f"ERROR: {e}")
    import traceback
    log_error(traceback.format_exc())