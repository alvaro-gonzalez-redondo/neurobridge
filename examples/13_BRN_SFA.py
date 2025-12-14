from neurobridge import *
import torch
import matplotlib.pyplot as plt
import numpy as np


class BalancedRandomNetworkExperiment(Experiment):
    # Estructura general de la red
    n_input_neurons: int = 160
    n_excitatory_neurons = 800 #800
    n_inhibitory_neurons = 200 #200
    n_output_neurons: int = 10

    integration_layer:bool = True

    # Parámetros de las conexiones
    noi2exc_params = {'fanin':5,  'weight':Constant(5e-4), 'delay':UniformInt(0,19)}

    #inp2exc_params = {'fanin':20,  'weight':Constant(0e-4), 'delay':UniformInt(0,19)}
    inp2exc_params = {'fanin':140,  'weight':Uniform(0.0, 1e-4), 'delay':UniformInt(0,19), 'w_max':2e-4, 'A_plus':1e-6, 'A_minus':-1e-6, 'oja_decay':20e-6}
    inp2inh_params = {'fanin':20,  'weight':Constant(2e-4), 'delay':0}
    
    exc2exc_params = {'fanin':80,  'weight':Uniform(0.0, 1e-4), 'delay':UniformInt(0,19), 'w_max':2e-4, 'A_plus':1e-6, 'A_minus':-1e-6, 'oja_decay':20e-6}
    exc2inh_params = {'fanin':80,  'weight':Constant(2e-4), 'delay':0}
    
    sta_inh2exc_params = {'fanin':20,  'weight':Constant(1e-3), 'delay':0}
    pla_inh2exc_params = {'fanin':10, 'weight':Constant(0.0), 'delay':0, 'w_max':1e-2, 'eta':1e-6, 'target_rate':20.0}
    inh2inh_params = {'fanin':20,  'weight':Constant(1e-3), 'delay':0}
    
    exc2out_params = {'fanin':200, 'weight':Uniform(0.0, 1e-2), 'tau_stdp_fast':100e-3, 'tau_stdp_slow':200e-3, 'scale':1e-3, 'norm_every':1, 'A':-1e-6}
    #inh2out_params = {'fanin':80, 'weight':Constant(0.0), 'delay':0, 'w_max':10.0, 'tau':1000.0, 'eta':1e-5, 'target_rate':40.0}
    out2out_params = {'weight':Uniform(0.0, 1e-3), 'delay':0}
    
    def build_network(self):
        with self.sim.autoparent("normal"):
            # Capas
            self.noi_neurons:RandomSpikeNeurons = RandomSpikeNeurons(n_neurons=self.n_input_neurons, firing_rate=10.0, name='Noise')
            self.inp_neurons:RandomSpikeNeurons = RandomSpikeNeurons(n_neurons=self.n_input_neurons, firing_rate=15.0, name='Input')
            self.exc_neurons:LIFNeurons = LIFNeurons(n_neurons=self.n_excitatory_neurons, tau_refrac=3e-3, name='Excitatory')
            self.inh_neurons:LIFNeurons = LIFNeurons(n_neurons=self.n_inhibitory_neurons, tau_membrane=1e-2, tau_refrac=1e-3, name='Inhibitory')
            self.out_neurons:LIFNeurons = LIFNeurons(self.n_output_neurons, name='Ouput')

            # Conexiones
            if self.integration_layer:
                self.noi2exc:StaticSparse = self.sim.connect(self.noi_neurons, self.exc_neurons, connection_type=StaticSparse, pattern="random", **self.noi2exc_params)
                #self.inp2exc:StaticSparse = self.sim.connect(self.inp_neurons, self.exc_neurons, connection_type=StaticSparse, pattern="random", **self.inp2exc_params)
                self.inp2exc:StaticSparse = self.sim.connect(self.inp_neurons, self.exc_neurons, connection_type=STDPSparse, pattern="random", **self.inp2exc_params)
                self.inp2inh:StaticSparse = self.sim.connect(self.inp_neurons, self.inh_neurons, connection_type=StaticSparse, pattern="random", **self.inp2inh_params)
                self.exc2exc:STDPSparse = self.sim.connect(self.exc_neurons, self.exc_neurons, connection_type=STDPSparse, pattern="random", autapses=False, **self.exc2exc_params)
                self.exc2inh:StaticSparse = self.sim.connect(self.exc_neurons, self.inh_neurons, connection_type=StaticSparse, pattern="random", **self.exc2inh_params)
                self.sta_inh2exc:VogelsSparse = self.sim.connect(self.inh_neurons, self.exc_neurons, connection_type=StaticSparse, pattern="random", channel=1, **self.sta_inh2exc_params)
                self.pla_inh2exc:VogelsSparse = self.sim.connect(self.inh_neurons, self.exc_neurons, connection_type=VogelsSparse, pattern="random", channel=1, **self.pla_inh2exc_params)
                self.inh2inh:StaticSparse = self.sim.connect(self.inh_neurons, self.inh_neurons, connection_type=StaticSparse, pattern="random", channel=1, autapses=False, **self.inh2inh_params)
                self.exc2out:STDPSFASparse = self.sim.connect(self.exc_neurons, self.out_neurons,  connection_type=STDPSFASparse, pattern="random", channel=0, **self.exc2out_params)
                #self.inh2out:VogelsSparse = self.sim.connect(self.inh_neurons, self.out_neurons,  connection_type=VogelsSparse, pattern="random", channel=1, **self.inh2out_params)
            else:
                pass
                #self.inp2out_exc:STDPSFADense = self.sim.connect(self.inp_neurons, self.out_neurons,  connection_type=STDPSFADense, pattern="all-to-all", channel=0, fanin=200, weight=Uniform(0.0, 1e-2), tau_stdp_fast=100e-3, tau_stdp_slow=200e-3, scale=2.5e-3, norm_every=1, A=-1e-9)
                #self.inp2out_inh:VogelsDense = self.sim.connect(self.inp_neurons, self.out_neurons,  connection_type=VogelsDense, pattern="all-to-all", channel=0,weight=Uniform(0.0, 5e-4), delay=0, w_max=10.0, eta=1e-2, target_rate=20.0)
            
            self.out2out:VogelsDense = self.sim.connect(self.out_neurons, self.out_neurons, pattern="all-to-all", autapses=False, connection_type=StaticSparse, channel=1, **self.out2out_params)

        with self.sim.autoparent("normal"):

            # Monitores en tiempo real

            self.viz = VisualizerClient()
            self.viz.reset()

            self.rt_spikes = RealtimeSpikeMonitor(
                groups = [
                    self.inp_neurons.where_idx(lambda i: (0<=i) & (i<50) ), 
                    self.exc_neurons.where_idx(lambda i: i<20),
                    self.inh_neurons.where_idx(lambda i: i<10),
                    self.out_neurons,
                ],
                viz_client=self.viz, 
                plot_id="raster_1",
                rollover_spikes=2_000,
            )

            # Monitor de variables
            if self.integration_layer:
                self.rt_weights_inp2exc = RealtimeVariableMonitor(
                    groups=[self.exc2out.where_idx( lambda i: i%100==0 ),],
                    variable_names=["weight"], viz_client=self.viz, plot_id="w_exc2out", interval=50,rollover=200,
                )                
                self.rt_weights_inp2exc = RealtimeVariableMonitor(
                    groups=[self.inp2exc.where_idx( lambda i: i%16_000==0 ),],
                    variable_names=["weight"], viz_client=self.viz, plot_id="w_inp2exc", interval=50,rollover=200,
                )                
                self.rt_weights_exc2exc = RealtimeVariableMonitor(
                    groups=[self.exc2exc.where_idx( lambda i: i%10_000==0 ),],
                    variable_names=["weight"], viz_client=self.viz, plot_id="w_exc2exc", interval=50,rollover=200,
                )
            else:
                pass
                #self.rt_weights_exc = RealtimeVariableMonitor(groups=[self.inp2out_exc.where_idx( lambda i,j: (i<10) & (j==0) ),], variable_names=["weight"], viz_client=self.viz, plot_id="w_exc", interval=50,rollover=200)
                #self.rt_weights_inh = RealtimeVariableMonitor(groups=[self.inp2out_inh.where_idx( lambda i,j: (i<10) & (j==0) ),], variable_names=["weight"], viz_client=self.viz, plot_id="w_inh", interval=50,rollover=200)


            # ---

            # Monitorización estática
            self.spike_monitor = SpikeMonitor(
                [
                    self.inp_neurons.where_idx(lambda i: i<50),     #0
                    self.exc_neurons.where_idx(lambda i: i<100),    #1
                    self.inh_neurons.where_idx(lambda i: i<100),    #2
                    self.out_neurons,                               #3
                ]
            )
    

    def on_start(self, **kwargs):
        # 1. Definir tiempo
        T = kwargs.get('total_steps', 1000)*1e-3
        ts = np.arange(0, T, self.sim.dt)

        # 2. Configuración Multiescala
        scales = [0.05, 0.3, 1.0] # Escala 1 (Fina): Sigma 0.05, 50 centros (Alta precisión local). Escala 2 (Media): Sigma 0.3, 20 centros (Generalización). Escala 3 (Gruesa): Sigma 1.0, 10 centros (Tendencia global lenta)
        n_centers = [50, 20, 10]
        assert (np.sum(n_centers)*2)==self.n_input_neurons, f"El número de entradas ({self.n_input_neurons}) debe ser igual al número de neuronas RBF ON/OFF ({np.sum(n_centers)*2})."

        # 3. Generar y almacenar
        activities, _signal, _meta = generate_multiscale_activity(
            ts, 
            alpha=0.5, 
            scales=scales, 
            n_centers=n_centers, 
            max_rate=100.0 # Hz
        )
        self.inp_activities = torch.from_numpy(activities)
    

    def pre_step(self):
        self.inp_neurons.firing_rate[:] = self.inp_activities[:, self.current_step]


    def on_finish(self):
        
        # Raster plots de todas las poblaciones
        plot_spikes(self.current_step, self.spike_monitor) #phase_sorting_t=2*np.pi)
        
        # PCAs y UMAPs de todas las poblaciones
        for i, group in enumerate(self.spike_monitor.groups):
            spikes = self.spike_monitor.get_spike_tensor(i)
            plot_neural_trajectory_pca(spikes, title=f"PCA of {group.name} layer")
            #plot_neural_trajectory_umap(spikes, title=f"UMAP of {group.name} layer")

        plt.show(block=True)


def get_signal(t, alpha=0.5):
    """
    Genera la señal mezcla de lenta (sin) y rápida (cos^2).
    """
    return (1 - alpha) * np.sin(t) + alpha * (np.cos(11 * t)**2)

def generate_multiscale_activity(
    time_array: np.ndarray,
    alpha: float = 0.5,
    signal_range: tuple = (-1.5, 1.5),
    scales: list = [0.05, 0.2, 0.5], # Sigmas de las RBFs
    n_centers: list = [50, 20, 10],  # Número de neuronas (centros) por escala
    max_rate: float = 50.0           # Hz máximos
):
    """
    Genera la actividad neuronal para una población de entrada usando RBFs multiescala
    y codificación ON/OFF complementaria.

    Args:
        time_array: Array de tiempos t.
        alpha: Parámetro de mezcla de la señal.
        signal_range: (min, max) rango esperado de la señal para distribuir los centros.
        scales: Lista con los anchos (sigma) de las gaussianas para cada escala.
        n_centers: Lista con el número de centros (resolución) para cada escala.
        max_rate: Tasa de disparo máxima (Hz).

    Returns:
        activities: Matriz [N_total_neuronas, T_steps] con las tasas de disparo instantáneas.
        signal: La señal original generada s(t).
        meta: Diccionario con info sobre qué neurona es qué (indices).
    """
    
    # 1. Generar Señal
    signal = get_signal(time_array, alpha)
    
    # Validar entradas
    assert len(scales) == len(n_centers), "Las listas 'scales' y 'n_centers' deben tener la misma longitud."
    
    # 2. Preparar contenedores
    all_activities = []
    meta_info = [] # Para saber qué neurona pertenece a qué escala
    neuron_counter = 0
    
    # 3. Iterar por cada escala
    for scale_idx, (sigma, n_c) in enumerate(zip(scales, n_centers)):
        
        # Generar centros distribuidos uniformemente en el rango de la señal
        centers = np.linspace(signal_range[0], signal_range[1], n_c)
        
        # Broadcasting mágico de Numpy:
        # signal: [1, T]
        # centers: [N_c, 1]
        # Result: [N_c, T]
        delta = signal[None, :] - centers[:, None]
        
        # Calcular respuesta RBF (Gaussiana)
        rbf_response = np.exp(- (delta**2) / (2 * sigma**2))
        
        # --- Codificación ON (La RBF tal cual) ---
        # Tasa: va de 0 a max_rate
        rates_on = max_rate * rbf_response
        
        # --- Codificación OFF (La Complementaria) ---
        # Tasa: va de max_rate a 0 (cuando la señal está en el centro)
        # Esto mantiene la energía constante: ON + OFF = max_rate
        rates_off = max_rate * (1.0 - rbf_response)
        
        # Guardar actividades
        # Apilamos verticalmente: primero las N ON, luego las N OFF de esta escala
        scale_activity = np.vstack([rates_on, rates_off])
        all_activities.append(scale_activity)
        
        # Guardar metadatos para análisis posterior
        meta_info.append({
            'scale_idx': scale_idx,
            'sigma': sigma,
            'idx_start': neuron_counter,
            'idx_end': neuron_counter + (2 * n_c),
            'n_on': n_c,
            'n_off': n_c
        })
        neuron_counter += (2 * n_c)

    # 4. Consolidar en una única matriz gigante
    # Shape final: [Total_Neuronas, Time_Steps]
    final_activity = np.vstack(all_activities)
    
    return final_activity, signal, meta_info


if __name__ == "__main__":
    exp = BalancedRandomNetworkExperiment(sim=Simulator(seed=0))
    simulation_time = 100.0
    exp.run(steps=int(1000*simulation_time))