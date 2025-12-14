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
            self.n_outputs = 2

            self.inputs = RandomSpikeNeurons(self.n_inputs, firing_rate=20.0) #20 to 180Hz
            self.outputs = LIFNeurons(self.n_outputs)

            # Feedforward (SFA)
            self.input2output = self.sim.connect(
                self.inputs, self.outputs, 
                connection_type=STDPSFADense,
                pattern="all-to-all",
                weight=Uniform(0.0, 1e-2),
                tau_stdp_fast=10e-3,
                tau_stdp_slow=20e-3,
                scale=5e-3,
                norm_every=1,
                A=-1e-7, #Learning rate
                channel=0,
            )

            # Competición lateral (Vogels Plástico)
            self.output2output = self.sim.connect(
                self.outputs, self.outputs,
                pattern="all-to-all",
                autapses=False,
                connection_type=VogelsDense,
                weight=Constant(0.0),
                delay=0,
                w_max=10.0,           # Permitir inhibición fuerte si hace falta
                eta=1e-5,            # Tasa de aprendizaje rápida (std Vogels)
                target_rate=100.0, # IMPORTANTE: Define el punto de operación
                channel=1,           # GABA
            )            
        
        with self.sim.autoparent("normal"):
            self.spike_monitor = SpikeMonitor([self.outputs.where_idx(lambda i: i==j) for j in range(self.n_outputs)])

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
        # Spikes
        fig, ax0 = plt.subplots()
        
        ax1 = ax0.twinx()
        id_sum = 0
        labels = [f"y{i}" for i in range(self.n_outputs)]
        for idx, label in enumerate(labels):
            spikes = self.spike_monitor.get_spike_tensor(idx).cpu()

            # Plot spikes
            spk_steps, spk_neurons = spikes[:, 1], spikes[:, 0]
            spk_times = spk_steps*1e-3
            ax1.scatter(spk_times, (spk_neurons*0)+id_sum, s=40, alpha=0.5, label=label, c=f"C{idx}")
            n_neurons = self.spike_monitor.filters[idx].numel()
            id_sum += n_neurons

            # Plot smooth curves
            times, rate = smooth_spikes(spk_steps, n_neurons=n_neurons, to_step=self.current_step, sigma=0.1)
            ax0.plot(times, rate, label=label, c=f"C{idx}", linewidth=3)

        ax0.legend(loc="lower right")
        plt.title(f"Spikes from different subpopulations")
        plt.xlabel("Time (seconds)")
        ax0.set_ylabel("Spiking rate (Hz)")
        ax1.set_ylabel("Neuron ID")

        plt.show()


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
    simulation_time = 50.0
    exp.run(time=simulation_time)