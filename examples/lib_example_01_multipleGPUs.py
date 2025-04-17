from neurobridge.all import *
#from all import *
from matplotlib import pyplot as plt
from tqdm import tqdm


class PingPongRingSimulation(SimulatorEngine):

    def __init__(self):
        super().__init__(autoparenting_nodes=True)

    def build_user_network(self, rank: int, world_size: int):
        n_neurons = 1_000

        # Crear un puente neuronal (permite la comunicación entre GPUs)
        self.add_default_bridge(n_local_neurons=n_neurons, n_steps=20)
        bridge = self.local_circuit.bridge

        # Crear un grupo neuronal local
        local_neurons = ParrotGroup(self.local_circuit.device, n_neurons, delay_max=100)

        # Envía a la siguiente GPU (o a sí misma si está sola)
        (local_neurons >> bridge.where_rank(rank))(
            pattern = 'one-to-one',
            delay   = 0,
            weight  = 1.0,
        )

        # Recibe de la GPU anterior (o de sí misma si está sola)
        (bridge.where_rank((rank-1) % world_size) >> local_neurons)(
            pattern = 'one-to-one',
            delay   = 0,
            weight  = 1.0,
        )
                      
        # Registramos las neuronas para poder meterle entradas
        self.local_neurons = local_neurons

        # Añadimos un monitor
        self.neuron_monitor = SpikeMonitor([self.local_neurons])


    def feed_input(self):
        # En la primera neurona (rank 0), inyectar un spike inicial para comenzar la actividad
        if self.rank == 0:
            if self.t < self.local_neurons.size+1:
                initial_spikes = torch.zeros(self.local_neurons.size, dtype=torch.bool, device=self.local_circuit.device)
                initial_spikes[self.t-1] = True
                self.local_neurons.inject_spikes(initial_spikes)


    def plot_spikes(self):
        monitor:SpikeMonitor = self.neuron_monitor
        cpu_spikes = monitor.get_spike_tensor(0).cpu()
        ot, oi = cpu_spikes[:,1], cpu_spikes[:,0]
        plt.scatter(ot, oi, s=4)
        show_or_save_plot(filename=f"rank{self.rank}_output.png", log=log)



# Main

try:
    with PingPongRingSimulation() as engine:
        simulation_length = 1
        simulation_steps = simulation_length * 1000
        for _ in tqdm(range(simulation_steps)):
            engine.step()
            engine.feed_input()
        
        engine.plot_spikes()

except Exception as e:
    log_error(f"ERROR: {e}")
    import traceback
    log_error(traceback.format_exc())