import time
import random
import math
from neurobridge import VisualizerClient # Ajusta el import a tu estructura

print("Conectando con el visualizador...")
# Asegúrate de que detecta/arranca el servidor
viz = VisualizerClient()
viz.reset()
time.sleep(0.5) 

# 1. Crear Raster
viz.create_raster("raster_1", title="Actividad Neuronal (Layer 1)")

# 2. Crear Lineplot (NUEVA API: requiere lista de nombres de series)
# Definimos que esta gráfica tendrá 2 líneas: "seno" y "coseno"
viz.create_lineplot(
    "monitor_1", 
    series_names=["onda_seno", "onda_coseno"], 
    title="Señales continuas"
)

print("Iniciando simulación...")
t = 0.0
dt = 0.01 

try:
    while True:
        # --- A. Spikes ---
        active_ids = []
        active_times = []
        for neuron_idx in range(50):
            if random.random() < 0.05:
                active_ids.append(neuron_idx)
                active_times.append(t)
        
        # OJO: Ahora push_spikes requiere un 3er argumento: group_ids (para el color)
        # Generamos una lista de ceros del mismo tamaño que los spikes
        group_ids = [0] * len(active_ids) 
        
        viz.push_spikes("raster_1", active_ids, active_times, group_ids)

        # --- B. Valores Continuos ---
        val1 = math.sin(t)
        val2 = math.cos(t) * 0.5
        
        # Enviar LISTA de valores [v1, v2] correspondientes a ["onda_seno", "onda_coseno"]
        viz.push_values("monitor_1", t, [val1, val2])

        t += dt
        
        # Velocidad máxima
        time.sleep(0.05) 
        
        if int(t/dt) % 100 == 0:
            print(f"t={t:.2f}s")

except KeyboardInterrupt:
    print("\nDetenido.")