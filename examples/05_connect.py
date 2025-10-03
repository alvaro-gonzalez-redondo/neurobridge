import torch
from neurobridge import *

class SimpleNetwork(Experiment):

    def build_network(self):
        n_neurons = 10

        with self.sim.autoparent("graph"):
            self.n1 = ParrotNeurons(n_neurons)
            self.n2 = ParrotNeurons(n_neurons)

            self.sim.connect(self.n1, self.n2, StaticSparse,
                pattern="random",
                delay=1,
                weight=1,
                #p=0.2,
                fanin=1,#n_neurons//2,
                fanout=1,#n_neurons//2,
            )
            self.sim.connect(self.n2, self.n1, StaticSparse,
                pattern="random",
                delay=1,
                weight=1,
                #p=0.2,
                fanin=1,#n_neurons//2,
                fanout=1,#n_neurons//2,
            )
        
        with self.sim.autoparent("normal"):
            self.spike_monitor = SpikeMonitor([self.n1, self.n2])
    

    def pre_step(self):
        if self.time == 10:
            initial_spikes = torch.zeros(
                self.n1.size,
                dtype=torch.bool,
                device=self.local_device,
            )
            initial_spikes[0] = True
            self.n1.inject_spikes(initial_spikes)
    

    def pos_step(self):
        spk_buf = self.n1.get_spike_buffer()
        phase = (self.time-1) % self.n1.delay_max
        spks = spk_buf[:, phase].squeeze().tolist()
        spks_str = "".join(["|" if spk else "_" for spk in spks])

        spk_buf = self.n2.get_spike_buffer()
        phase = (self.time-1) % self.n2.delay_max
        spks = spk_buf[:, phase].squeeze().tolist()
        spks_str += " - " + "".join(["|" if spk else "_" for spk in spks])
        
        log(f"t={self.time:<5}: {spks_str}")
    

    def on_finish(self):
        #monitor: SpikeMonitor = self.spike_monitor
        pass


if __name__ == "__main__":
    exp = SimpleNetwork(sim=Simulator(seed=1))
    exp.run(steps=100)