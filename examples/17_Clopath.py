import torch
import numpy as np
import matplotlib.pyplot as plt
from neurobridge import *

# Asumimos que las clases nuevas están en un archivo accesible o pegadas arriba
# from my_clopath_implementation import ClopathAdExNeurons, ClopathSTDPSparse

class ExperimentFig1H(Experiment):
    """
    Reproducción de la Figura 1h de Clopath et al. (2010).
    Dependencia del cambio de peso con el voltaje postsináptico fijo (Voltage Clamp).
    """

    # --- Configuración del Experimento ---
    duration = 50.0  # Segundos (según el paper para este experimento)
    freq_pre = 2.0   # Hz
    
    # Rango de voltaje a explorar
    v_start = -80e-3
    v_end   = -35e-3
    n_samples = 100  # Resolución de la gráfica

    def build_network(self):
        # 1. Parámetros "Visual Cortex" (Tabla 1b) convertidos a SI (Voltios, Amperios)
        # Nota: A_LTD y A_LTP se escalan porque en el paper están en mV^-1 y mV^-2
        params_visual_cortex = {
            'A_LTD': 14e-5 * 1e3,       # 0.14 V^-1
            'A_LTP': 8e-5 * 1e6,        # 80.0 V^-2
            'theta_minus': -70.6e-3,
            'theta_plus': -45.3e-3,
            'tau_x': 15e-3,
            'tau_minus': 10e-3,
            'tau_plus': 7e-3,
            'w_max': 2.0,
            
            # Homeostasis: Para reproducir la Fig 1h fielmente (que usa parámetros fijos),
            # deberíamos anular el efecto dinámico de la homeostasis o asumir que 
            # el fit del paper ya incluye el factor promedio.
            # Aquí usamos el modelo completo.
            'u_ref_sq': 60.0, # mV^2 (La clase lo convierte a V^2 internamente si es > 1e-3?)
                              # Ojo: En mi implementación de ClopathSTDPSparse puse:
                              # raw_uref * 1e-6. Así que pasamos 60.0.
        }

        with self.sim.autoparent("normal"):
            # 2. Neurona Presináptica (Generador de Spikes controlable)
            # Usamos ParrotNeurons para inyectar spikes manualmente
            self.pre_pop = ParrotNeurons(n_neurons=1, name="Pre")

            # 3. Población Postsináptica (Barrido de Voltajes)
            # Creamos N neuronas, cada una representará un punto en la gráfica X
            self.post_pop = ClopathAdExNeurons(
                n_neurons=self.n_samples,
                name="Post_Clamp",
                # Parámetros físicos (Visual cortex)
                tau_membrane=9.36e-3, 
                E_leak=-70.6e-3,
                v_rheobase=-50.4e-3,
                threshold=-20e-3,
                # Parámetros filtros Clopath
                tau_minus=params_visual_cortex['tau_minus'],
                tau_plus=params_visual_cortex['tau_plus'],
                tau_slow=1.0 # Lento para homeostasis
            )

            # 4. Conexión Plástica
            # Conectamos la única neurona PRE a TODAS las POST (broadcast)
            self.conn = self.sim.connect(
                self.pre_pop, 
                self.post_pop, 
                connection_type=ClopathSTDPSparse,
                pattern="all-to-all",
                weight=1.0, # Peso inicial
                delay=0,
                **params_visual_cortex
            )
        
        # Guardamos los pesos iniciales para calcular el cambio después
        self.w_initial = self.conn.weight.clone()


    def setup_voltage_clamp(self):
        """Configura el vector de voltajes fijos."""
        # Generar vector lineal de voltajes [-80mV ... -35mV]
        v_targets = torch.linspace(
            self.v_start, self.v_end, self.n_samples, 
            device=self.current_device
        )
        
        # Aplicar Voltage Clamp
        self.post_pop.set_voltage_clamp(active=True, values=v_targets)
        
        # IMPORTANTE: Inicializar los filtros de voltaje (u_minus, u_plus, u_slow)
        # al valor del clamp para evitar transitorios iniciales largos.
        self.post_pop.u_minus.copy_(v_targets)
        self.post_pop.u_plus.copy_(v_targets)
        self.post_pop.u_slow.copy_(v_targets)
        self.post_pop.V.copy_(v_targets)


    def pre_step(self):
        # 1. Configuración inicial (solo en el paso 0)
        if self.current_step == 0:
            self.setup_voltage_clamp()

        # 2. Generación de Spikes Presinápticos (2 Hz)
        # Periodo = 1/2Hz = 0.5s.
        # Step Period = 0.5 / dt
        steps_per_spike = int(1.0 / self.freq_pre / self.sim.dt)
        
        if self.current_step % steps_per_spike == 0 and self.current_step > 0:
            # Inyectar spike en la neurona PRE
            spikes = torch.tensor([True], device=self.current_device)
            self.pre_pop.inject_spikes(spikes)


    def on_finish(self):
        # Calcular cambio de pesos
        w_final = self.conn.weight
        delta_w = (w_final - self.w_initial)
        
        # Convertir a numpy para graficar
        v_axis = torch.linspace(self.v_start, self.v_end, self.n_samples).cpu().numpy() * 1000 # a mV
        dw_axis = delta_w.cpu().numpy()

        # Graficar
        plt.figure(figsize=(8, 6))
        plt.plot(v_axis, dw_axis, 'b-', linewidth=2, label='Modelo Clopath (Simulado)')
        plt.axhline(0, color='k', linestyle='--', alpha=0.5)
        
        # Marcar umbrales
        theta_minus_mv = -70.6
        theta_plus_mv = -45.3
        plt.axvline(theta_minus_mv, color='r', linestyle=':', label=r'$\theta_-$')
        plt.axvline(theta_plus_mv, color='g', linestyle=':', label=r'$\theta_+$')

        plt.title(f"Fig 1h: Dependencia del Voltaje (Voltage Clamp)\nFreq Pre: {self.freq_pre} Hz, Duración: {self.duration}s")
        plt.xlabel("Voltaje Postsináptico (mV)")
        plt.ylabel("Cambio de Peso $\Delta w$")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

if __name__ == "__main__":
    # dt pequeño para precisión en los filtros exponenciales
    sim = Simulator(seed=42) 
    exp = ExperimentFig1H(sim=sim)
    
    print(f"Iniciando simulación de Fase 1...")
    exp.run(exp.duration)