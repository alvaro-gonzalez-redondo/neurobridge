from neurobridge import *
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


class SFAToyProblem(Experiment):
    def build_network(self):
        #with self.sim.autoparent("graph"):
        with self.sim.autoparent("normal"):
            self.n_inputs = 10
            self.inputs = RandomSpikeNeurons(self.n_inputs, firing_rate=20.0) #20 to 180Hz
            self.output = PhaseIFNeurons(1)

            params_exc = {
                #'connection_type': STDPSparseSFA,
                'connection_type': STDPSFADense,
                #'connection_type': SFADense,
                'weight': Constant(5e-4),
                'tau_stdp_fast': 100e-3,
                'tau_stdp_slow': 200e-3,
                'scale': 5e-4,
                'norm_every': 1,
                'A': -1e-6, #Learning rate
            }
            self.i2o = self.sim.connect(self.inputs, self.output, **params_exc)
        
        with self.sim.autoparent("normal"):
            self.viz = VisualizerClient()
            self.viz.reset()

            # Monitor de Spikes en tiempo real
            self.rt_spikes = RealtimeSpikeMonitor(
                groups = [self.inputs.where_idx(lambda i: i==j) for j in range(self.n_inputs)] + [self.output],
                group_names = [f"x{i}" for i in range(self.n_inputs)] + ["y"],
                viz_client=self.viz, 
                plot_id="raster_1",
                rollover_spikes=5_000,
            )

            # Monitor de variables
            self.rt_weights = RealtimeVariableMonitor(
                groups=[self.i2o], # Asumiendo que i2o_e hereda de Group/Node compatible
                variable_names=["weight"],
                viz_client=self.viz,
                plot_id="w",
                interval=50,
                rollover=200,
            )

    def on_start(self, **kwargs):
        _ts, rates, _pca_model = generate_whitened_on_off_inputs(duration = 2*np.pi)
        self.input_rates = rates


    def pre_step(self):
        step = self.sim.local_circuit.current_step
        idx_step = step % self.input_rates.shape[0]
        for idx_input in range(self.n_inputs):
            self.inputs.firing_rate[idx_input] = self.input_rates[idx_step, idx_input]


    def pos_step(self):
        pass


    def on_finish(self):
        pass


def generate_whitened_on_off_inputs(duration=10.0, dt=0.001, gain_factor=50.0):
    """
    Genera señales blanqueadas y separadas en canales ON/OFF.
    Returns:
        t: tiempo
        rates_combined: Matriz de (Samples x 2*N_features). 
                        Las primeras N son ON, las siguientes N son OFF (o intercaladas).
    """
    t = np.arange(0, duration, dt)
    alpha = 0.1
    
    # 1. Generar señales crudas
    raw_signals = [
        ((1-alpha)*np.sin(t) + alpha*(np.cos(11 * t)**2) + 1) / 2,
        (np.cos(11 * t) + 1) / 2,
        (((1-alpha)*np.sin(t) + alpha*(np.cos(11 * t)**2))**2),
        (((1-alpha)*np.sin(t) + alpha*(np.cos(11 * t)**2)) * np.cos(11 * t) + 1) / 2,
        (np.cos(11 * t)**2)
    ]
    
    X = np.stack(raw_signals, axis=1)
    
    # 2. Whitening (Media 0, Varianza 1)
    pca = PCA(n_components=None, whiten=True)
    X_whitened = pca.fit_transform(X)
    
    # X_whitened tiene valores aprox entre -3 y +3 (desviaciones estándar)
    
    # 3. Separación ON / OFF (Rectificación de media onda)
    # Aplicamos una ganancia para convertir "desviaciones estándar" a "Hz"
    # Ej: Si la señal es 1.0 (1 sigma), disparará a 50 Hz.
    signal_scaled = X_whitened * gain_factor
    
    rates_on = np.maximum(0, signal_scaled)  # ReLU
    rates_off = np.maximum(0, -signal_scaled) # ReLU del negativo
    
    # 4. Combinar
    # Opción A: Concatenar [ON_1, ON_2... OFF_1, OFF_2...]
    # rates_combined = np.hstack([rates_on, rates_off])
    
    # Opción B (Más clara para visualizar): Intercalar [ON_1, OFF_1, ON_2, OFF_2...]
    n_samples, n_features = X_whitened.shape
    rates_interleaved = np.zeros((n_samples, n_features * 2))
    
    for i in range(n_features):
        rates_interleaved[:, 2*i] = rates_on[:, i]     # Canal par: ON
        rates_interleaved[:, 2*i+1] = rates_off[:, i]  # Canal impar: OFF
        
    return t, rates_interleaved, pca


if __name__ == "__main__":
    exp = SFAToyProblem(sim=Simulator(seed=0))
    simulation_time = 100.0
    exp.run(time=simulation_time)