from neurobridge import *
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

class NeuronTests(Experiment):
    common_params = {
        'tau_membrane': 10e-3, #10e-3, #20e-3,
        'tau_refrac': 2e-3,
        'threshold': -52e-3, #-52e-3,
        'E_rest': -65e-3,
        'n_channels': 2,
        'channel_time_constants': [(1e-3, 5e-3), (1e-3, 10e-3)], # Exc, Inh
        'channel_reversal_potentials': [0.0, -85e-3],
    }

    lif_params = {
        **common_params,
        'threshold': -52e-3,
    }

    alif_params = {
        **common_params,
        'tau_adapt': 0.2,
        'beta': 1.7e-3,
    }

    plalif_params = {
        **common_params,
        'n_basis': 6,
        'tau_min': 0.010,
        'tau_max': 50.0,
        'r_target': 10.0,      # Hz objetivo (conceptual)
        'eta_theta': 1e-6,    # learning rate intrínseco (muy lento)
    }

    ampa_params = {
        'connection_type': StaticDense,
        'weight': Constant(1e-5),
        'channel': 0,
    }

    gaba_params = {
        'connection_type': StaticDense,
        'weight': Constant(1e-5),
        'channel': 1,
    }
    
    def build_network(self):
        #with self.sim.autoparent("graph"):
        with self.sim.autoparent("normal"):
            self.inputs_ampa = RandomSpikeNeurons(1000)
            self.inputs_gaba = RandomSpikeNeurons(1000)
            self.output1 = LIFNeurons(1, **self.lif_params)
            self.output2 = ALIFNeurons(1, **self.alif_params)
            self.output3 = PowerLawALIFNeurons(1, **self.plalif_params)

            if True:
                _ = self.sim.connect(self.inputs_ampa, self.output1, **self.ampa_params)
                _ = self.sim.connect(self.inputs_ampa, self.output2, **self.ampa_params)
                _ = self.sim.connect(self.inputs_ampa, self.output3, **self.ampa_params)
            
            if False:
                _ = self.sim.connect(self.inputs_gaba, self.output1, **self.gaba_params)
                _ = self.sim.connect(self.inputs_gaba, self.output2, **self.gaba_params)
                _ = self.sim.connect(self.inputs_gaba, self.output3, **self.gaba_params)
        
        with self.sim.autoparent("normal"):
            #self.spike_monitor = SpikeMonitor([self.inputs1, self.output])
            #self.weight_monitor = VariableMonitor([self.i2o], ["weight"])

            self.viz = VisualizerClient()
            self.viz.reset()

            self.in_spikes = RealtimeSpikeMonitor(
                groups = [self.inputs_ampa.where_idx(lambda i: i<50), self.inputs_gaba.where_idx(lambda i: i<50)],
                group_names = ["AMPA", "GABA"],
                viz_client=self.viz, 
                plot_id="raster_1",
                rollover_spikes=1_000,
                rollover_lines=500,
            )

            self.out_spikes = RealtimeSpikeMonitor(
                groups = [self.output1, self.output2, self.output3],
                group_names = ["LIF", "ALIF", "PowerLawALIF"],
                viz_client=self.viz, 
                plot_id="raster_2",
                rollover_spikes=1_000,
                rollover_lines=500,
            )

            self.voltage_monitor = RealtimeVariableMonitor(
                [self.output1, self.output2, self.output3],
                #['V', 'spikes', 'channel_currents@0', 'channel_currents@1'] + (['A'] if neuron!=self.output1 else []),
                ['V', 'channel_currents@0', 'channel_currents@1'],
                viz_client=self.viz, plot_id="variable_1",
                interval=100, rollover=100
            )

            self.voltage_monitor = RealtimeVariableMonitor(
                [self.output2, self.output3], ['A'],
                viz_client=self.viz, plot_id="variable_2",
                interval=100, rollover=100
            )


    def on_start(self, **kwargs):
        pass

    def pre_step(self):
        step = self.sim.local_circuit.current_step
        t = step*1e-3
        t1 = t*11 / 50
        t2 = t*13 / 50
        self.inputs_ampa.firing_rate[:] = ((torch.cos(torch.pi*2*t1)+1)*0.5) * 10.0
        self.inputs_gaba.firing_rate[:] = ((torch.sin(torch.pi*2*t2)+1)*0.5) * 10.0

    def pos_step(self):
        pass


    def on_finish(self):
        pass


if __name__ == "__main__":
    exp = NeuronTests(sim=Simulator(seed=0))
    simulation_time = 100.0
    exp.run(time=simulation_time)