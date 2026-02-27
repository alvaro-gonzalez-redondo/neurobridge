from neurobridge import *
import torch
import matplotlib.pyplot as plt
import numpy as np


class RelationalNetworkAdvanced(PersistentExperiment):
    
    # === Parámetros Generales ===
    N_total = 2000
    N_exc = 1600
    N_inh = 400
    delay_max = 20
    
    # === Parámetros Neuronas (Igual que antes) ===
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

    # === Parámetros Conectividad ===
    learning_rate = 3e-6 #1e-2
    target_firing_rate = 5.0

    delay_exc = UniformInt(0, delay_max-1)
    delay_inh = UniformInt(0, 4)

    k_in2e = 1
    k_in2i = N_exc // 10
    k_e2e = N_exc // 10
    k_e2i = 8
    k_i2e = N_inh // 10
    
    one2one_exc = 1.85e-3
    one2one_inh = 3.27e-3

    rnd = Normal(1, 0.2).clamp(0,2)
    w_in2e = rnd * (1.0*one2one_exc) / k_in2e
    w_in2i = rnd * (0.3*one2one_inh) / k_in2i
    w_e2i = rnd * (0.5*one2one_inh) / k_e2i
    w_i2e = rnd * (3*one2one_exc) / k_i2e
    
    internode_factor = 1.0 

    w_e2e_nmda = rnd * (0.1*one2one_exc)/k_e2e #Constant(0.0) #rnd * (0.5*one2one_exc) / k_e2e
    w_e2e_ampa = rnd * (0.1*one2one_exc)/k_e2e #Constant(0.0) #rnd * (0.5*one2one_exc) / k_e2e
    
    # --- Plasticidad ---
    triplet_params_common = {
        'target_activity': target_firing_rate,
        'tau_x_slow': 1.0, 
        'w_max': 3e-2/k_e2e,
        'delay': delay_exc,
    }
    triplet_params_nmda = {**triplet_params_common,
        'eta_pre':  0.5 * 1e3*learning_rate,
        'eta_post': 2 * 1e3*learning_rate,

        'tau_x_pre': 30e-3, #50e-3,
        'tau_x_post1': 50e-3,
        'tau_x_post2': 50e-3, #100e-3,
    }
    triplet_params_ampa = {**triplet_params_common,
        'eta_pre':  0.1 * 1e3*learning_rate,
        'eta_post': 0.2 * 1e3*learning_rate,

        'tau_x_pre': 20e-3,
        'tau_x_post1': 20e-3,
        'tau_x_post2': 40e-3,
    }
        
    vogels_params = {
        'eta': learning_rate, 
        'target_rate': target_firing_rate,
        'weight': 0.0,
        'w_max': 1e-1,
        'delay': delay_inh
    }

    # === Control Wake-Sleep y Test ===
    steps_per_example = 500
    n_examples_wake = 50      
    sleep_duration_steps = 5000

    # Entrada y ruido
    input_gaussian_sigma = 0.075 #0.15
    input_gaussian_max_rate = 80.0 #40.0
    base_noise = 1.0
    sleep_noise = 15.0
    
    # Estado interno
    phase = "WAKE"
    phase_timer = 0
    example_counter = 0
    
    # Variables para el Test Dinámico
    test_start_step = 0
    total_test_steps = 0
    
    # Historial para plotear: lista de tuplas (start, end, val_a, val_b, direction)
    stimulus_history = [] 
    current_stimulus_start = 0
    current_val_a = 0
    current_val_b = 0
    current_direction = None


    def build_network(self):
        self.pops_exc = {}
        self.pops_inh = {}
        self.inputs = {}
        self.triplet_conns = [] 
        self.vogels_conns = []

        module_names = ["PopA", "PopB"]
                
        with self.sim.autoparent("normal"):
            for name in module_names:
                pop_exc = LIFNeurons(n_neurons=self.N_exc, **self.exc_neuron_params, name="Exc_"+name)
                pop_inh = LIFNeurons(n_neurons=self.N_inh, **self.inh_neuron_params, name="Inh_"+name)
                self.pops_exc[name] = pop_exc
                self.pops_inh[name] = pop_inh
                
                inp = RandomSpikeNeurons(n_neurons=self.N_exc, firing_rate=self.base_noise, name="RndSpk"+name)
                self.inputs[name] = inp

                self._build_internal_connectivity(pop_exc, pop_inh, inp)

            popA_exc = self.pops_exc["PopA"]
            popA_inh = self.pops_inh["PopA"]
            popB_exc = self.pops_exc["PopB"]
            popB_inh = self.pops_inh["PopB"]

            self._connect_modules(source_exc=popA_exc, target_exc=popB_exc, target_inh=popB_inh)
            self._connect_modules(source_exc=popB_exc, target_exc=popA_exc, target_inh=popA_inh)

        # Monitores
        with self.sim.autoparent("normal"):
            is_multiple_of = lambda x: lambda i: i % x == 0
            self.viz = VisualizerClient()
            self.viz.reset()

            # Monitores en Tiempo Real
            self.pop_monitor_exc = RealtimeSpikeMonitor(
                groups = [
                    self.pops_exc["PopA"].where_idx(is_multiple_of(100)),
                    self.pops_exc["PopB"].where_idx(is_multiple_of(100)),
                ],
                group_names = [
                    "Exc A",
                    "Exc B",
                ],
                viz_client=self.viz, plot_id="Excitatory neurons", rollover_spikes=1_000
            )

            self.pop_monitor_inh = RealtimeSpikeMonitor(
                groups = [
                    self.pops_inh["PopA"].where_idx(is_multiple_of(50)),
                    self.pops_inh["PopB"].where_idx(is_multiple_of(50)),
                ],
                group_names = [
                    "Inh A",
                    "Inh B",
                ],
                viz_client=self.viz, plot_id="Inhibitory neurons", rollover_spikes=1_000
            )

            self.pop_monitor_static = SpikeMonitor(
                [
                    self.pops_exc["PopA"].where_idx(is_multiple_of(1)), # idx 0
                    self.pops_inh["PopA"].where_idx(is_multiple_of(1)), # idx 1
                    self.pops_exc["PopB"].where_idx(is_multiple_of(1)), # idx 2
                    self.pops_inh["PopB"].where_idx(is_multiple_of(1)), # idx 3
                ]
            )

            self.weight_monitor = VariableMonitor(
                [
                    self.triplet_conns[0].where_idx(lambda i: (i<20)),
                    self.triplet_conns[-1].where_idx(lambda i: (i<20)),
                    self.vogels_conns[0].where_idx(lambda i: (i<20)),
                ], 
                ["weight"]
            )


    def _build_internal_connectivity(self, pop_exc, pop_inh, inp):
        self.sim.connect(inp, pop_exc, connection_type=StaticSparse, pattern="one-to-one", channel=0, weight=self.w_in2e, delay=self.delay_exc)
        self.sim.connect(inp, pop_inh, connection_type=StaticSparse, pattern="random", fanin=self.k_in2i, channel=0, weight=self.w_in2i, delay=self.delay_inh)
        
        conn_ampa = self.sim.connect(pop_exc, pop_exc, connection_type=ClopathTripletSparse, pattern="random", fanin=self.k_e2e, autapses=False, **self.triplet_params_ampa, channel=0, weight=self.w_e2e_ampa)
        self.triplet_conns.append(conn_ampa)
        conn_nmda = self.sim.connect(pop_exc, pop_exc, connection_type=ClopathTripletSparse, pattern=conn_ampa, **self.triplet_params_nmda, channel=2, weight=self.w_e2e_nmda)
        self.triplet_conns.append(conn_nmda)

        self.sim.connect(pop_exc, pop_inh, connection_type=StaticSparse, pattern="random", fanin=self.k_e2i, channel=0, weight=self.w_e2i, delay=self.delay_inh)
        self.sim.connect(pop_inh, pop_exc, connection_type=StaticSparse, pattern="random", fanin=self.k_i2e, channel=1, weight=self.w_i2e, delay=self.delay_inh)
        conn = self.sim.connect(pop_inh, pop_exc, connection_type=VogelsSparse, pattern="random", fanin=self.k_i2e, channel=1, **self.vogels_params)
        self.vogels_conns.append(conn)

    def _connect_modules(self, source_exc, target_exc, target_inh):
        factor = self.internode_factor
        
        params_ampa = self.triplet_params_ampa
        conn_ampa = self.sim.connect(source_exc, target_exc, connection_type=ClopathTripletSparse, pattern="random", fanin=self.k_e2e, autapses=False, **params_ampa, channel=0, weight=self.w_e2e_ampa)
        self.triplet_conns.append(conn_ampa)

        params_nmda = self.triplet_params_nmda
        conn_nmda = self.sim.connect(source_exc, target_exc, connection_type=ClopathTripletSparse, pattern=conn_ampa, **params_nmda, channel=2, weight=self.w_e2e_nmda)
        self.triplet_conns.append(conn_nmda)

        self.sim.connect(source_exc, target_inh, connection_type=StaticSparse, pattern="random", fanin=self.k_e2i, channel=0, weight=self.w_e2i, delay=self.delay_inh)


    # --- Lógica de Control ---

    def set_phase(self, phase_name, test_steps=0):
        self.phase = phase_name
        
        if phase_name == "WAKE":
            for c in self.triplet_conns:
                c.set_learning_enabled(True)
                c.set_sleep_mode(False)
            self.example_counter = 0
            self.present_wake_example()
            
        elif phase_name == "SLEEP":
            for c in self.triplet_conns:
                c.set_learning_enabled(True)
                c.set_sleep_mode(True)
            self.phase_timer = self.sleep_duration_steps
            self.present_sleep_noise()

        elif phase_name == "TEST":
            # Congelar pesos
            for c in self.triplet_conns:
                c.set_learning_enabled(False)
            
            # Configurar contadores para el test
            self.test_start_step = self.current_step
            self.total_test_steps = test_steps
            self.present_test_example(direction="A2B")

        print(f"--- Phase changed to {phase_name} at step {self.current_step} ---")


    def pre_step(self):
        # Lógica WAKE
        if self.phase == "WAKE":
            if self.current_step % self.steps_per_example == 0:
                self.example_counter += 1
                if self.example_counter >= self.n_examples_wake:
                    self.set_phase("SLEEP")
                else:
                    self.present_wake_example()
        
        # Lógica SLEEP
        elif self.phase == "SLEEP":
            if self.current_step % 250 == 0:
                self.present_sleep_noise()
            self.phase_timer -= 1
            if self.phase_timer <= 0:
                self.set_phase("WAKE")

        # Lógica TEST (Nueva implementación dinámica)
        elif self.phase == "TEST":
            # Cambiamos el ejemplo cada cierto tiempo para ver varios casos
            if (self.current_step - self.test_start_step) % self.steps_per_example == 0:
                
                # Guardamos el ejemplo anterior en el historial para plotear
                if self.current_direction is not None:
                    self.stimulus_history.append((
                        self.current_stimulus_start, self.current_step, 
                        self.current_val_a, self.current_val_b, self.current_direction
                    ))

                # Determinamos dirección: Primera mitad A->B, Segunda mitad B->A
                steps_elapsed = self.current_step - self.test_start_step
                direction = "A2B" if steps_elapsed < (self.total_test_steps / 2) else "B2A"
                
                self.present_test_example(direction)


    # --- Generación de Inputs ---

    def _get_gaussian_rates(self, value, sigma=None, max_rate=None):
        sigma = self.input_gaussian_sigma if sigma is None else sigma
        max_rate = self.input_gaussian_max_rate if max_rate is None else max_rate
        x = torch.linspace(0, 1, self.N_exc, device=self.current_device)
        delta = torch.abs(x - value)
        delta = torch.minimum(delta, 1.0 - delta) 
        rates = max_rate * torch.exp(-(delta**2)/(2*sigma**2))
        return rates

    def present_wake_example(self):
        val_a = np.random.rand()
        val_b = (1.0 - val_a) % 1.0
        self.inputs["PopA"].firing_rate = self._get_gaussian_rates(val_a) + self.base_noise
        self.inputs["PopB"].firing_rate = self._get_gaussian_rates(val_b) + self.base_noise

    def present_sleep_noise(self):
        target = np.random.choice(["PopA", "PopB"])
        if target == "PopA":
            noise = torch.rand(self.N_exc, device=self.current_device) * self.sleep_noise
            self.inputs["PopA"].firing_rate = noise + self.base_noise
            self.inputs["PopB"].firing_rate[:] = self.base_noise
        else:
            noise = torch.rand(self.N_exc, device=self.current_device) * self.sleep_noise
            self.inputs["PopB"].firing_rate = noise + self.base_noise
            self.inputs["PopA"].firing_rate[:] = self.base_noise

    def present_test_example(self, direction):
        """Genera un ejemplo de test aleatorio en la dirección especificada"""
        val_a = np.random.rand()
        val_b = (1.0 - val_a) % 1.0
        
        # Guardar estado actual para plotear luego
        self.current_stimulus_start = self.current_step
        self.current_val_a = val_a
        self.current_val_b = val_b
        self.current_direction = direction

        if direction == "A2B":
            # Input fuerte en A, Silencio en B (para que B infiera)
            self.inputs["PopA"].firing_rate = self._get_gaussian_rates(val_a) + self.base_noise
            self.inputs["PopB"].firing_rate[:] = self.base_noise
        elif direction == "B2A":
            # Input fuerte en B, Silencio en A (para que A infiera)
            self.inputs["PopB"].firing_rate = self._get_gaussian_rates(val_b) + self.base_noise
            self.inputs["PopA"].firing_rate[:] = self.base_noise


    def on_finish(self):
        # Guardar el último estímulo pendiente
        if self.phase == "TEST" and self.current_direction is not None:
             self.stimulus_history.append((
                self.current_stimulus_start, self.current_step, 
                self.current_val_a, self.current_val_b, self.current_direction
            ))

        self.plot_activity()
        self.plot_history_connections()
        self.plot_final_connections()
        plt.show()
    

    def plot_activity(self):
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 10))
        
        # --- Obtener Spikes ---
        spikes_a = self.pop_monitor_static.get_spike_tensor(0).cpu() # PopA Exc
        spikes_b = self.pop_monitor_static.get_spike_tensor(2).cpu() # PopB Exc
        
        # Convertir tiempos a segundos (aproximado, usando 1ms = 1 step para visualización simple o usar self.sim.dt si disponible)
        # Asumimos timestep 1e-3 para plot
        t_a = spikes_a[:, 1] * 1e-3
        id_a = spikes_a[:, 0]
        
        t_b = spikes_b[:, 1] * 1e-3
        id_b = spikes_b[:, 0]

        # --- Scatter Plots (Puntos) ---
        # Usamos s=0.5 y alpha para que se vea mejor la densidad
        axes[0].scatter(t_a, id_a, s=0.5, c='black', alpha=0.6, label='Spikes')
        axes[1].scatter(t_b, id_b, s=0.5, c='black', alpha=0.6, label='Spikes')

        # --- Dibujar Líneas de Input y Target ---
        # Iteramos sobre el historial de estímulos del TEST
        # (Si hubo entrenamiento antes, no se plotearán líneas en esa zona, lo cual está bien)
        
        for (start, end, val_a, val_b, direction) in self.stimulus_history:
            t_start_s = start * 1e-3
            t_end_s = end * 1e-3
            
            # Mapear valor [0, 1] a ID [0, 1600]
            y_a = val_a * self.N_exc
            y_b = val_b * self.N_exc
            
            if direction == "A2B":
                # A es Input
                axes[0].hlines(y_a, t_start_s, t_end_s, colors='C0', linestyles='solid', linewidth=4)
                # B es Target - Lo que esperamos que la red infiera
                axes[1].hlines(y_b, t_start_s, t_end_s, colors='C1', linestyles='solid', linewidth=4)
                
            elif direction == "B2A":
                # B es Input
                axes[1].hlines(y_b, t_start_s, t_end_s, colors='C0', linestyles='solid', linewidth=4)
                # A es Target
                axes[0].hlines(y_a, t_start_s, t_end_s, colors='C1', linestyles='solid', linewidth=4)

        # Decoración
        axes[0].set_title("Population A Activity")
        axes[0].set_ylabel("Neuron ID")
        axes[0].set_ylim(0, self.N_exc)
        # Solo mostrar leyenda una vez
        axes[0].hlines([], [], [], colors='C0', linestyles='solid', linewidth=4, label='Input')
        axes[0].hlines([], [], [], colors='C1', linestyles='solid', linewidth=4, label='Target')
        axes[0].legend(loc='upper right')

        axes[1].set_title("Population B Activity")
        axes[1].set_ylabel("Neuron ID")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylim(0, self.N_exc)
    
    def plot_history_connections(self):
        _fig, ax0 = plt.subplots()
        w_values = self.weight_monitor.get_variable_tensor(0, "weight")
        ax0.plot(w_values, c='C0')
        w_values = self.weight_monitor.get_variable_tensor(1, "weight")
        ax0.plot(w_values, c='C1')
        w_values = self.weight_monitor.get_variable_tensor(2, "weight")
        ax0.plot(w_values, c='C2')
        
        ax0.plot([],[], c='C0', label='A->A')
        ax0.plot([],[], c='C1', label='B->A')
        ax0.plot([],[], c='C2', label='A-|A')
        plt.legend()        
    
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

            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), constrained_layout=True)
            fig.suptitle(title, fontsize=14)
            
            # Plot AMPA (sin colorbar individual, pasando vmin/vmax)
            sc1 = plot_sparse_connectivity(
                conn_ampa, self.N_exc, self.N_exc, 
                ax=axs[0], 
                title="AMPA", 
                show_colorbar=False,
                vmin=global_min, 
                vmax=global_max
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


# -------------------------------------------------------------------------
# Ejecución
# -------------------------------------------------------------------------
if __name__ == "__main__":
    sim = Simulator(seed=555)
    exp = RelationalNetworkAdvanced(sim=sim)
    
    # 1. Entrenamiento (Corto para demo)
    print("--- STARTING TRAINING ---")
    exp.set_phase("WAKE")
    # Entrenamos 100 segundos
    exp.run(steps=100_000, close_on_finish=False)
    
    # 2. Test Bidireccional
    print("\n--- STARTING BIDIRECTIONAL TEST ---")
    # Probamos durante 10 segundos (5s A->B, 5s B->A)
    steps_test = 10_000
    exp.set_phase("TEST", test_steps=steps_test)
    
    exp.run(steps=steps_test, close_on_finish=True)