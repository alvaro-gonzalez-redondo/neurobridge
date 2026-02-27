from neurobridge import *
import torch
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# 1. GESTOR DE CONECTIVIDAD (BUILDER PATTERN)
# =============================================================================

class ConnectivityManager:
    """
    Gestor genérico. Puede usarse para:
    - Normalizar entradas a Excitatorias (STDP, w_max variable).
    - Normalizar entradas a Inhibitorias (Estáticas, peso fijo).
    """
    def __init__(self, simulator, target_k, total_w_budget, name_suffix=""):
        self.sim = simulator
        self.target_k = target_k             # Presupuesto de número de conexiones (K)
        self.w_budget = total_w_budget       # Presupuesto de "fuerza" total (Suma de pesos/w_max)
        self.registry = {}                   # {target_name: [list_of_specs]}
        self.suffix = name_suffix

    def add_projection(self, source, target, params_template, connection_type, 
                       channel=0, weight_init_factor=1.0, relative_priority=1.0):
        if target.name not in self.registry:
            self.registry[target.name] = []

        self.registry[target.name].append({
            'source': source,
            'target': target,
            'type': connection_type,
            'params': params_template.copy(),
            'channel': channel,
            'w_factor': weight_init_factor,
            'priority': relative_priority
        })

    def build(self):
        created_connections = []
        print(f"\n--- Building Connectivity Group: {self.suffix} ---")

        for target_name, specs in self.registry.items():
            total_priority = sum(s['priority'] for s in specs)
            
            for spec in specs:
                # 1. Calcular K (Fan-in)
                fraction = spec['priority'] / total_priority
                k_calc = int(self.target_k * fraction)
                k_calc = max(1, k_calc)

                # 2. Calcular Peso Base
                # Peso promedio por sinapsis = Presupuesto / K_Total
                w_base = self.w_budget / self.target_k 
                
                # Ajustar peso inicial
                w_init = w_base * spec['w_factor']

                # 3. Configurar Parámetros
                params = spec['params']
                
                # Caso A: Regla Plástica (ej. Clopath) -> w_max es el límite
                if 'w_max' in params:
                    params['w_max'] = w_base # El límite superior es el peso base calculado
                
                # Caso B: Regla Estática -> No usa w_max, el peso se define en connect()
                
                # 4. Conectar
                conn = self.sim.connect(
                    spec['source'], 
                    spec['target'], 
                    connection_type=spec['type'], 
                    pattern="random", 
                    fanin=k_calc,
                    autapses=False, 
                    **params, 
                    channel=spec['channel'], 
                    weight=w_init
                )
                created_connections.append(conn)
                
                w_display = params.get('w_max', w_init) # Para el print
                print(f"Conn {spec['source'].name:6} -> {spec['target'].name:6} | "
                      f"Ch:{spec['channel']} | k={k_calc:3} | w~{w_display:.5f}")

        return created_connections


# =============================================================================
# 2. EXPERIMENTO: HUB NETWORK (ADVANCED BUDGETING)
# =============================================================================

class HubNetworkExperiment(PersistentExperiment):
    
    # === Dimensiones ===
    N_exc = 1600
    N_inh = 400
    
    # === PRESUPUESTOS DE CONECTIVIDAD ===
    
    # 1. PERIFÉRICOS (PopA, PopB, PopC)
    # Tienen input directo fuerte, necesitan menos drive recurrente/lateral.
    PERI_K_EXC_TARGET = 200
    PERI_W_EXC_BUDGET = 5e-2 

    # 2. HUB (PopH) - EL "MOTOR" CENTRAL
    # No tiene input directo. Necesita un budget mucho más alto para alcanzar el umbral 
    # y sostener actividad auto-mantenida (attractor dynamics).
    HUB_K_EXC_TARGET = 120 #300
    HUB_W_EXC_BUDGET = 0.35 #0.3
    
    # 3. INHIBICIÓN (Exc -> Inh)
    # Gestionada globalmente pero normalizada por target.
    GLOBAL_K_INH_TARGET = 150
    GLOBAL_W_INH_BUDGET = 6e-3

    # === Parámetros Generales ===
    delay_max = 20
    learning_rate = 1e-6
    target_firing_rate = 10.0 # Hz
    
    # === Constantes Auxiliares para Estáticas ===
    k_in2i = N_exc // 10
    k_e2i = 10 # Fan-in local
    k_i2e = 40 #80 # Aumentado para mayor control (Strong Inhibition)
    k_i2i = N_inh // 10
    
    one2one_exc = 2.0e-3
    one2one_inh = 3.0e-3
    
    # Distribuciones de peso
    rnd_dist = Normal(1, 0.2).clamp(0.5, 1.5) # Evitar ceros
    w_e2i_base = rnd_dist * (0.8 * one2one_inh) / k_e2i
    w_in2i_base = rnd_dist * (0.3 * one2one_inh) / k_in2i

    # === Parámetros Neuronas ===
    exc_neuron_params = {
        'tau_membrane': Uniform(15e-3, 25e-3),
        'tau_refrac': 2e-3,
        'threshold': -55e-3,
        'E_rest': -65e-3,
        'n_channels': 3,
        'channel_time_constants': [(1e-3, 5e-3), (1e-3, 10e-3), (15e-3, 100e-3)], 
        'channel_reversal_potentials': [0.0, -85e-3, 0.0],
        'delay_max': delay_max,
    }

    inh_neuron_params = {
        'tau_membrane': Uniform(5e-3, 15e-3),
        'tau_refrac': 1e-3,
        'threshold': -55e-3,
        'E_rest': -65e-3,
        'n_channels': 2,
        'channel_time_constants': [(1e-3, 5e-3), (1e-3, 10e-3)], 
        'channel_reversal_potentials': [0.0, -85e-3],
        'delay_max': delay_max,
    }

    # === Plantillas de Plasticidad ===
    delay_exc = UniformInt(0, delay_max-1)
    
    triplet_base_ampa = {
        'target_activity': target_firing_rate, 'tau_x_slow': 1.0, 'delay': delay_exc,
        'eta_pre':  0.1 * 1e4*learning_rate, 'eta_post': 0.2 * 1e4*learning_rate,
        'tau_x_pre': 20e-3, 'tau_x_post1': 20e-3, 'tau_x_post2': 40e-3,
        'w_max': None 
    }
    
    triplet_base_nmda = {
        'target_activity': target_firing_rate, 'tau_x_slow': 1.0, 'delay': delay_exc,
        'eta_pre':  0.2 * 1e4*learning_rate, 'eta_post': 0.4 * 1e4*learning_rate,
        'tau_x_pre': 30e-3, 'tau_x_post1': 50e-3, 'tau_x_post2': 50e-3,
        'w_max': None
    }

    static_base = {'delay': delay_exc}
    
    # === Control ===
    steps_per_example = 500
    n_examples_wake = 100      
    sleep_duration_steps = 4000
    input_gaussian_sigma = 0.08
    input_gaussian_max_rate = 90.0
    base_noise = 2.0
    sleep_noise = 25.0
    
    phase = "WAKE"
    phase_timer = 0
    example_counter = 0
    test_start_step = 0
    total_test_steps = 0
    
    stimulus_history = [] 
    curr_stim_start = 0; curr_vals = (0,0,0); curr_stim_type = None

    def build_network(self):
        self.pops_exc = {}
        self.pops_inh = {}
        self.inputs = {}
        self.triplet_conns = [] 
        
        self.module_names = ["PopA", "PopB", "PopC", "PopH"]
        self.peripherals = ["PopA", "PopB", "PopC"]
        
        # --- 1. GESTORES DIFERENCIADOS ---
        
        # A. Gestor para Periféricos (A, B, C) - Presupuesto Normal
        self.peri_manager = ConnectivityManager(
            self.sim, target_k=self.PERI_K_EXC_TARGET, 
            total_w_budget=self.PERI_W_EXC_BUDGET, name_suffix="Peripherals_Exc"
        )
        
        # B. Gestor para HUB (H) - Presupuesto AUMENTADO
        self.hub_manager = ConnectivityManager(
            self.sim, target_k=self.HUB_K_EXC_TARGET, 
            total_w_budget=self.HUB_W_EXC_BUDGET, name_suffix="HUB_Exc_Boosted"
        )
        
        # C. Gestor para Inhibidoras (E->I) - Presupuesto Normal
        self.inh_manager = ConnectivityManager(
            self.sim, target_k=self.GLOBAL_K_INH_TARGET, 
            total_w_budget=self.GLOBAL_W_INH_BUDGET, name_suffix="To_Inhibitory"
        )

        with self.sim.autoparent("normal"):
            # 2. CREACIÓN DE POBLACIONES Y CONEXIONES ESTÁTICAS LOCALES
            for name in self.module_names:
                pop_exc = LIFNeurons(n_neurons=self.N_exc, name="Exc_"+name, **self.exc_neuron_params)
                pop_inh = LIFNeurons(n_neurons=self.N_inh, name="Inh_"+name, **self.inh_neuron_params)
                rate_noise = 0.1 if name=="PopH" else self.base_noise
                inp = RandomSpikeNeurons(n_neurons=self.N_exc, firing_rate=rate_noise, name="Inp_"+name)
                
                self.pops_exc[name] = pop_exc
                self.pops_inh[name] = pop_inh
                self.inputs[name] = inp

                # Input -> Exc/Inh (Static)
                self.sim.connect(inp, pop_exc, connection_type=StaticSparse, pattern="one-to-one", channel=0, weight=self.one2one_exc, delay=self.delay_exc)
                self.sim.connect(inp, pop_inh, connection_type=StaticSparse, pattern="random", fanin=50, channel=0, weight=self.w_in2i_base, delay=self.delay_exc)
                
                # Exc -> Inh (Local estática base + Boost para Hub)
                # NOTA: Además de lo que pondrá el inh_manager (que viene de fuera), ponemos una base local fuerte para el Hub
                weight_e2i_local = self.w_e2i_base
                if name == 'PopH':
                    # Inhibición local muy fuerte en Hub para forzar competencia (WTA)
                    weight_e2i_local = self.w_e2i_base * 2.5 #8.0 
                
                self.sim.connect(pop_exc, pop_inh, connection_type=StaticSparse, pattern="random", fanin=self.k_e2i, channel=0, weight=weight_e2i_local, delay=self.delay_exc)
                
                # --- INHIBICIÓN ESTÁTICA (ESTABILIZACIÓN) ---
                
                # 1. Inh -| Exc (GABA ch1)
                # Estática para encauzar actividad. Fuerte para evitar epilepsia dado el budget alto del Hub.
                w_i2e = self.one2one_exc * 6 #1.5 # Bastante fuerte
                self.sim.connect(pop_inh, pop_exc, connection_type=StaticSparse, pattern="random", fanin=self.k_i2e, channel=1, weight=w_i2e, delay=self.delay_exc)
                
                # 2. Inh -| Inh (GABA ch1)
                # Desincronización. Rompe oscilaciones globales excesivas.
                w_i2i = self.one2one_exc * 0.5 
                self.sim.connect(pop_inh, pop_inh, connection_type=StaticSparse, pattern="random", fanin=self.k_i2i, channel=1, weight=w_i2i, delay=self.delay_exc)


            # 3. DEFINIR TOPOLOGÍA EXCITATORIA (Usando los 2 Gestores)
            popH_exc = self.pops_exc["PopH"]
            popH_inh = self.pops_inh["PopH"]
            
            # --- A. RECURRENCIA (Self) ---
            for name in self.module_names:
                p_exc = self.pops_exc[name]
                p_inh = self.pops_inh[name]
                
                # Seleccionar gestor adecuado
                manager = self.hub_manager if name == "PopH" else self.peri_manager
                
                # Exc->Exc (Recurrente)
                # Prioridad 1.0 base.
                manager.add_projection(p_exc, p_exc, self.triplet_base_ampa, ClopathTripletSparse, channel=0, weight_init_factor=0.05, relative_priority=1.0)
                manager.add_projection(p_exc, p_exc, self.triplet_base_nmda, ClopathTripletSparse, channel=2, weight_init_factor=0.05, relative_priority=1.0)
                
                # Exc->Inh (Recurrente cruzada al manager de inhibición global)
                self.inh_manager.add_projection(p_exc, p_inh, self.static_base, StaticSparse, channel=0, weight_init_factor=1.0)

            # --- B. CONEXIONES ESTRELLA ---
            for p_name in self.peripherals:
                popP_exc = self.pops_exc[p_name]
                popP_inh = self.pops_inh[p_name]
                
                # 1. PERIFÉRICO -> HUB
                # Usamos hub_manager (Budget Alto).
                # Priority alta (5.0) para que el Hub escuche más a los periféricos que a sí mismo inicialmente.
                self.hub_manager.add_projection(popP_exc, popH_exc, self.triplet_base_ampa, ClopathTripletSparse, channel=0, weight_init_factor=0.1, relative_priority=5.0)
                self.hub_manager.add_projection(popP_exc, popH_exc, self.triplet_base_nmda, ClopathTripletSparse, channel=2, weight_init_factor=0.1, relative_priority=5.0)
                
                # Feedforward Inhibition (Peri -> InhHub)
                self.inh_manager.add_projection(popP_exc, popH_inh, self.static_base, StaticSparse, channel=0, weight_init_factor=1.0)

                # 2. HUB -> PERIFÉRICO
                # Usamos peri_manager (Budget Normal).
                # Priority normal/baja (2.0) para que el periférico no sea dominado totalmente por el Hub (input externo manda).
                self.peri_manager.add_projection(popH_exc, popP_exc, self.triplet_base_ampa, ClopathTripletSparse, channel=0, weight_init_factor=0.1, relative_priority=2.0)
                self.peri_manager.add_projection(popH_exc, popP_exc, self.triplet_base_nmda, ClopathTripletSparse, channel=2, weight_init_factor=0.1, relative_priority=2.0)
                
                # Feedback Inhibition (Hub -> InhPeri)
                self.inh_manager.add_projection(popH_exc, popP_inh, self.static_base, StaticSparse, channel=0, weight_init_factor=1.0)

            # 4. CONSTRUIR
            self.triplet_conns.extend(self.peri_manager.build())
            self.triplet_conns.extend(self.hub_manager.build())
            self.inh_manager.build() # Static only


        # Monitores
        with self.sim.autoparent("normal"):
            is_sample = lambda x: lambda i: i % x == 0

            self.viz = VisualizerClient()
            self.viz.reset()

            # Monitores en Tiempo Real
            self.pop_monitor_exc = RealtimeSpikeMonitor(
                groups = [self.pops_exc[pop].where_idx(is_sample(200)) for pop in self.module_names],
                group_names = [f"Exc {pop}" for pop in self.module_names],
                viz_client=self.viz, plot_id="Excitatory neurons", rollover_spikes=1_000
            )
            self.pop_monitor_inh = RealtimeSpikeMonitor(
                groups = [self.pops_inh[pop].where_idx(is_sample(100)) for pop in self.module_names],
                group_names = [f"Inh {pop}" for pop in self.module_names],
                viz_client=self.viz, plot_id="Inhibitory neurons", rollover_spikes=1_000
            )

            # Monitor estático para plot final (muestreo 1 de cada 2 neuronas para reducir memoria)
            self.pop_monitor_static = SpikeMonitor(
                [
                    self.pops_exc["PopA"].where_idx(is_sample(2)), # idx 0
                    self.pops_exc["PopB"].where_idx(is_sample(2)), # idx 1
                    self.pops_exc["PopC"].where_idx(is_sample(2)), # idx 2
                    self.pops_exc["PopH"].where_idx(is_sample(2)), # idx 3
                ]
            )

            try:
                self.weight_monitor = VariableMonitor(
                    [
                        self.triplet_conns[0].where_idx(lambda i: (i<10)), # Interna
                        self.vogels_conns[0].where_idx(lambda i: (i<10)),  # Homeostasis
                    ], 
                    ["weight"]
                )
            except:
                pass

    # --- Lógica de Control ---
    def set_phase(self, phase_name, test_steps=0):
        self.phase = phase_name
        if phase_name == "WAKE":
            for c in self.triplet_conns: c.set_learning_enabled(True); c.set_sleep_mode(False)
            self.example_counter = 0; self.present_wake_example()
        elif phase_name == "SLEEP":
            for c in self.triplet_conns: c.set_learning_enabled(True); c.set_sleep_mode(True)
            self.phase_timer = self.sleep_duration_steps; self.present_sleep_noise()
        elif phase_name == "TEST":
            for c in self.triplet_conns: c.set_learning_enabled(False)
            self.test_start_step = self.current_step; self.total_test_steps = test_steps
            self.present_test_example(condition="AB->C")
        print(f"--- Phase changed to {phase_name} at step {self.current_step} ---")

    def pre_step(self):
        if self.phase == "WAKE":
            if self.current_step % self.steps_per_example == 0:
                self.example_counter += 1
                if self.example_counter >= self.n_examples_wake: self.set_phase("SLEEP")
                else: self.present_wake_example()
        elif self.phase == "SLEEP":
            if self.current_step % 250 == 0: self.present_sleep_noise()
            self.phase_timer -= 1
            if self.phase_timer <= 0: self.set_phase("WAKE")
        elif self.phase == "TEST":
            steps_elapsed = self.current_step - self.test_start_step
            if steps_elapsed % self.steps_per_example == 0:
                self._record_history()
                condition = "AB->C" if steps_elapsed < (self.total_test_steps / 2) else "AC->B"
                self.present_test_example(condition)

    def _record_history(self):
        if self.curr_stim_type is not None:
             self.stimulus_history.append((self.curr_stim_start, self.current_step, self.curr_vals[0], self.curr_vals[1], self.curr_vals[2], self.curr_stim_type))

    def _get_gaussian_rates(self, value, sigma=None, max_rate=None):
        sigma = self.input_gaussian_sigma if sigma is None else sigma
        max_rate = self.input_gaussian_max_rate if max_rate is None else max_rate
        x = torch.linspace(0, 1, self.N_exc, device=self.current_device)
        delta = torch.abs(x - value); delta = torch.minimum(delta, 1.0 - delta) 
        rates = max_rate * torch.exp(-(delta**2)/(2*sigma**2))
        return rates

    def _set_input(self, pop_name, val=None):
        if val is not None: self.inputs[pop_name].firing_rate = self._get_gaussian_rates(val) + self.base_noise
        else: self.inputs[pop_name].firing_rate[:] = self.base_noise

    def present_wake_example(self):
        val_a, val_b = np.random.rand(), np.random.rand()
        val_c = (val_a + val_b) % 1.0
        self._set_input("PopA", val_a); self._set_input("PopB", val_b); self._set_input("PopC", val_c)
        self._set_input("PopH", None)

    def present_sleep_noise(self):
        target = np.random.choice(self.module_names)
        for name in self.module_names:
            if name == target: self.inputs[name].firing_rate = (torch.rand(self.N_exc, device=self.current_device) * self.sleep_noise) + self.base_noise
            else: self.inputs[name].firing_rate[:] = self.base_noise

    def present_test_example(self, condition):
        val_a, val_b = np.random.rand(), np.random.rand()
        val_c = (val_a + val_b) % 1.0
        self.curr_stim_start = self.current_step; self.curr_vals = (val_a, val_b, val_c); self.curr_stim_type = condition
        if condition == "AB->C":
            self._set_input("PopA", val_a); self._set_input("PopB", val_b); self._set_input("PopC", None); self._set_input("PopH", None)
        elif condition == "AC->B":
            self._set_input("PopA", val_a); self._set_input("PopB", None); self._set_input("PopC", val_c); self._set_input("PopH", None)

    def on_finish(self):
        self._record_history()
        self.plot_activity_hub()
        self.plot_final_connections()
        plt.show()

    def plot_activity_hub(self):
        fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 12))
        names = ["A", "B", "C", "Hub"]
        
        for i, name in enumerate(names):
            spikes = self.pop_monitor_static.get_spike_tensor(i).cpu()
            if len(spikes) > 0:
                axes[i].scatter(spikes[:, 1]*1e-3, spikes[:, 0], s=0.3, c='black', alpha=0.5)
            
            axes[i].set_ylabel(name)
            axes[i].set_ylim(0, self.N_exc)

            # Líneas de referencia
            for (start, end, va, vb, vc, cond) in self.stimulus_history:
                t0, t1 = start*1e-3, end*1e-3
                vals = [va, vb, vc, None]
                is_input = True
                
                # Definir si es input o target para pintar sólido o punteado
                if cond == "AB->C": is_input = (i != 2) # C es target
                elif cond == "AC->B": is_input = (i != 1) # B es target
                
                if vals[i] is not None:
                    style = 'solid' if is_input else 'dashed'
                    color = 'C0' if is_input else 'C1'
                    axes[i].hlines(vals[i]*self.N_exc, t0, t1, colors=color, linestyles=style, lw=3)

        axes[0].set_title("Hub Network Inference (Solid=Input, Dashed=Inferred)")
        axes[-1].set_xlabel("Time (s)")

    def plot_final_connections(self):
        it = iter(self.triplet_conns)
        pairs = zip(it, it)
        
        for conn_ampa, conn_nmda in pairs:
            title = f"Connections from {conn_ampa.pre.name} to {conn_ampa.pos.name}"

            # Calcular límites globales para que los colores sean comparables
            w_ampa = conn_ampa.weight.detach().cpu()
            w_nmda = conn_nmda.weight.detach().cpu()
            
            # Obtenemos min y max combinados
            global_min = min(w_ampa.min(), w_nmda.min())
            global_max = max(w_ampa.max(), w_nmda.max())

            is_hub_conn = 'H' in conn_ampa.pre.name or 'H' in conn_ampa.pos.name

            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), constrained_layout=True)
            fig.suptitle(title, fontsize=14)
            
            # Plot AMPA (sin colorbar individual, pasando vmin/vmax)
            sc1 = plot_sparse_connectivity(
                conn_ampa, self.N_exc, self.N_exc, 
                ax=axs[0], 
                title="AMPA", 
                show_colorbar=False,
                vmin=global_min, 
                vmax=global_max,
                order=is_hub_conn
            )
            
            # Plot NMDA (sin colorbar individual, pasando vmin/vmax)
            _ = plot_sparse_connectivity(
                conn_nmda, self.N_exc, self.N_exc, 
                ax=axs[1], 
                title="NMDA", 
                show_colorbar=False,
                vmin=global_min, 
                vmax=global_max
            )

            # Crear el colorbar común
            # Usamos sc1 porque comparte la misma norma (vmin/vmax) y cmap que sc2
            cbar = fig.colorbar(sc1, ax=axs, shrink=0.8, location='right')
            cbar.set_label("Peso sináptico")


if __name__ == "__main__":
    sim = Simulator(seed=999)
    exp = HubNetworkExperiment(sim=sim)
    
    print("--- STARTING WAKE PHASE ---")
    exp.set_phase("WAKE")
    exp.run(steps=100_000, close_on_finish=False)
    
    print("--- STARTING TEST PHASE ---")
    exp.set_phase("TEST", test_steps=10_000)
    exp.run(steps=10_000, close_on_finish=True)