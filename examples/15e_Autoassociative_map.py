from neurobridge import *
import torch
import matplotlib.pyplot as plt
import numpy as np


class RelationalNetworkExperiment(PersistentExperiment):
    
    # === Parámetros del Artículo ===
    N_total = 2000
    N_exc = 1600
    N_inh = 400

    delay_max = 20
    
    # Parámetros LIF (Conductancia)
    exc_neuron_params = {
        'tau_membrane': Uniform(15e-3, 25e-3),
        'tau_refrac': 2e-3,
        'threshold': -55e-3, #-52e-3
        'E_rest': -65e-3,
        'n_channels': 3,
        #'channel_time_constants': [(1e-3, 5e-3), (1e-3, 10e-3), (4e-3, 70e-3)], # AMPA, GABA, NMDA
        'channel_time_constants': [(1e-3, 5e-3), (1e-3, 10e-3), (15e-3, 100e-3)], # AMPA, GABA, NMDA
        'channel_reversal_potentials': [0.0, -85e-3, 0.0],
        'delay_max': delay_max,
    }

    inh_neuron_params = {
        'tau_membrane': Uniform(5e-3, 15e-3),
        'tau_refrac': 1e-3,
        'threshold': -55e-3, #-52e-3
        'E_rest': -65e-3,
        'n_channels': 2,
        'channel_time_constants': [(1e-3, 5e-3), (1e-3, 10e-3)], # Exc, Inh
        'channel_reversal_potentials': [0.0, -85e-3],
        'delay_max': delay_max,
    }


    # Parámetros Conectividad
    learning_rate = 1e-8

    base_noise = 1.0

    delay_exc = UniformInt(0,delay_max-1) #UniformInt(0,4)
    delay_inh = UniformInt(0,4)

    k_in2e = 1 #implícito en las conexiones 1:1
    k_in2i = N_exc // 10
    k_e2e = N_exc // 10 #40
    k_e2i = 8
    k_i2e = N_inh // 10
    #k_i2i = 0 #20 #40
    
    one2one_exc = 1.85e-3 #Constante empírica que hace que las neuronas excitadoras disparen un spike cuando reciben un spike con este peso
    one2one_inh = 3.27e-3 #Igual para inhibidoras

    rnd = Normal(1, 0.2).clamp(0,2)
    w_in2e = rnd * (1.0*one2one_exc) / k_in2e
    w_in2i = rnd * (0.5*one2one_inh) / k_in2i
    w_e2i = rnd * (0.3*one2one_inh) / k_e2i
    w_i2e = rnd * (3*one2one_exc) / k_i2e
    #w_i2i = w_e2i
    internode_factor = 1.0
    
    mu_log = np.log((1.0*one2one_exc) / k_e2e) - 0.5 # Ajuste a ojo para centrar la media real
    sigma_log = 1.0 
    w_e2e_nmda = LogNormal(mean=mu_log, std=sigma_log)    
    w_e2e_ampa = rnd * (0.5*one2one_exc) / k_e2e
    norm_target_nmda = k_e2e * w_e2e_nmda.mean(force_estimate=True)
    norm_target_ampa = k_e2e * w_e2e_ampa.mean(force_estimate=True)

    # Plasticidad Exc->Exc
    triplet_params_nmda = {
        #'eta_pre': 0.0, 'eta_post': 0.0,
        'eta_pre': 15*learning_rate, 'eta_post': 25*learning_rate,
        'tau_x_pre': 50e-3, 'tau_x_post1': 50e-3, 'tau_x_post2': 100e-3,
        'w_max': w_e2e_nmda.mean(force_estimate=True)*10, 'mu': 0.2,
        'norm_every': 100, 'norm_target_in': norm_target_nmda, 'norm_target_out': norm_target_nmda,
        'delay': delay_exc, #'weight': w_e2e_nmda,
    }
    triplet_params_ampa = {
        **triplet_params_nmda,
        'tau_x_pre': 20e-3, 'tau_x_post1': 20e-3, 'tau_x_post2': 40e-3,
        'w_max': w_e2e_ampa.mean(force_estimate=True)*10, 
        'norm_target_in': norm_target_ampa, 'norm_target_out': norm_target_ampa,
    }
        
    # Plasticidad Inhibitoria I->E
    vogels_params = {
        'eta': 50*learning_rate, 'target_rate': 10.0, 'w_max': 1e-1,
        'weight': 0.0,
        'delay': delay_inh
    }

    # Control Wake-Sleep
    steps_per_example = 250   # 250ms por input
    n_examples_wake = 50      # Cuántos ejemplos antes de dormir
    sleep_duration_steps = 5000 # 5 segundos de sueño
    
    # Estado
    phase = "WAKE"
    phase_timer = 0
    example_counter = 0


    def build_network(self):
        # 1. Definición de Neuronas
        self.pops_exc = {}
        self.pops_inh = {}
        self.inputs = {}
        self.triplet_conns = [] # Guardamos referencias a conexiones plásticas para control rápido
        self.vogels_conns = []

        # Módulos: A, B, C (Periféricos) y H (Oculto)
        module_names = ["PopA", "PopB", "PopC", "PopH"]
                
        with self.sim.autoparent("normal"):
            for name in module_names:
                # Neuronas LIF (Exc 0-1599, Inh 1600-1999 implícitamente por lógica de conexión)
                pop_exc = LIFNeurons(n_neurons=self.N_exc, **self.exc_neuron_params, name="Exc_"+name)
                pop_inh = LIFNeurons(n_neurons=self.N_inh, **self.inh_neuron_params, name="Inh_"+name)
                self.pops_exc[name] = pop_exc
                self.pops_inh[name] = pop_inh
                
                # Input Layer (Solo Excitatoria, Poisson)
                # Creamos inputs para todos, aunque PopH no reciba input externo directo
                # nos permite inyectar ruido si quisiéramos.
                inp = RandomSpikeNeurons(n_neurons=self.N_exc, firing_rate=self.base_noise, name="RndSpk"+name)
                self.inputs[name] = inp

                # --- Conectividad Interna del Módulo ---
                self._build_internal_connectivity(pop_exc, pop_inh, inp, is_hidden=(name=="PopH"))

            # 2. Conectividad entre Módulos (Topología Relacional)
            # A <-> H, B <-> H, C <-> H
            h_pop_exc = self.pops_exc["PopH"]
            h_pop_inh = self.pops_inh["PopH"]
            peripherals = ["PopA", "PopB", "PopC"]
            
            for p_name in peripherals:
                p_pop_exc = self.pops_exc[p_name]
                p_pop_inh = self.pops_inh[p_name]
                self._connect_bidirectional(p_pop_exc, p_pop_inh, h_pop_exc, h_pop_inh, n_peripherals=len(peripherals))

        # 3. Monitores
        with self.sim.autoparent("normal"):
            # Monitorizamos PopC para evaluar la inferencia
            is_multiple_of = lambda x: lambda i: i % x == 0

            self.viz = VisualizerClient()
            self.viz.reset()

            self.input_monitor = RealtimeSpikeMonitor(
                groups = [
                    self.inputs["PopA"].where_idx(is_multiple_of(200)),
                    self.inputs["PopB"].where_idx(is_multiple_of(200)),
                    self.inputs["PopC"].where_idx(is_multiple_of(200)),
                    self.inputs["PopH"].where_idx(is_multiple_of(200)),
                ],
                group_names = ["PopA", "PopB", "PopC", "PopH"],
                viz_client=self.viz, 
                plot_id="Inputs",
                rollover_spikes=1_000,
                rollover_lines=500,
            )

            self.hidden_monitor_exc = RealtimeSpikeMonitor(
                groups = [
                    self.pops_exc["PopA"].where_idx(is_multiple_of(200)),
                    self.pops_exc["PopB"].where_idx(is_multiple_of(200)),
                    self.pops_exc["PopC"].where_idx(is_multiple_of(200)),
                    self.pops_exc["PopH"].where_idx(is_multiple_of(200)),
                ],
                group_names = ["PopA Exc", "PopB Exc", "PopC Exc", "PopH Exc"],
                viz_client=self.viz, 
                plot_id="Hidden excitatory units",
                rollover_spikes=1_000,
                rollover_lines=500,
            )

            self.hidden_monitor_inh = RealtimeSpikeMonitor(
                groups = [
                    self.pops_inh["PopA"].where_idx(is_multiple_of(50)),
                    self.pops_inh["PopB"].where_idx(is_multiple_of(50)),
                    self.pops_inh["PopC"].where_idx(is_multiple_of(50)),
                    self.pops_inh["PopH"].where_idx(is_multiple_of(50)),
                ],
                group_names = ["PopA Inh", "PopB Inh", "PopC Inh", "PopH Inh"],
                viz_client=self.viz, 
                plot_id="Hidden inhibitory units",
                rollover_spikes=1_000,
                rollover_lines=500,
            )


            if True:
                if True:
                    self.inp_monitor = SpikeMonitor(
                        [
                            self.inputs["PopA"].where_idx(is_multiple_of(20)),
                            self.inputs["PopB"].where_idx(is_multiple_of(20)),
                            self.inputs["PopC"].where_idx(is_multiple_of(20)),
                            self.inputs["PopH"].where_idx(is_multiple_of(20)),
                        ]
                    )

                    self.pop_monitor = SpikeMonitor(
                        [
                            self.pops_exc["PopA"].where_idx(is_multiple_of(20)),
                            self.pops_inh["PopA"].where_idx(is_multiple_of(20)),

                            self.pops_exc["PopB"].where_idx(is_multiple_of(20)),
                            self.pops_inh["PopB"].where_idx(is_multiple_of(20)),

                            self.pops_exc["PopC"].where_idx(is_multiple_of(20)),
                            self.pops_inh["PopC"].where_idx(is_multiple_of(20)),

                            self.pops_exc["PopH"].where_idx(is_multiple_of(20)),
                            self.pops_inh["PopH"].where_idx(is_multiple_of(20)),
                        ]
                    )

                if True:
                    self.state_monitor = VariableMonitor(
                        [self.pops_exc["PopC"].where_idx(0)],
                        [
                            'V', 'spikes', 
                            'channel_currents@0', 'channel_currents@1', 'channel_currents@2'
                        ]
                    )


                if True and len(self.triplet_conns)>0:
                    self.weight_monitor = VariableMonitor(
                        [
                            self.triplet_conns[0].where_idx(lambda i: (i<20)),
                            self.triplet_conns[-1].where_idx(lambda i: (i<20)),
                            self.vogels_conns[0].where_idx(lambda i: (i<20)),
                        ], 
                        ["weight"]
                    )


    def _build_internal_connectivity(self, pop_exc, pop_inh, inp, is_hidden):
        """Crea conexiones E-E, E-I, I-E, I-I e Input->E"""
        
        # Input -> Exc (Estático, fuerte para conducir la red)
        # Solo si no es oculto, o si queremos permitir ruido en H
        self.sim.connect(inp, pop_exc, connection_type=StaticSparse, pattern="one-to-one", channel=0, weight=self.w_in2e, delay=self.delay_exc)
        self.sim.connect(inp, pop_inh, connection_type=StaticSparse, pattern="random", fanin=self.k_in2i, channel=0, weight=self.w_in2i, delay=self.delay_inh)
        
        # Exc -> Exc (Triplet STDP)
        conn_ampa = self.sim.connect(
            pop_exc, pop_exc, connection_type=TripletSTDPSparse, pattern="random", fanin=self.k_e2e, autapses=False, 
            **self.triplet_params_ampa, channel=0, weight=self.w_e2e_ampa,
        )
        self.triplet_conns.append(conn_ampa)
        conn_nmda = self.sim.connect(
            pop_exc, pop_exc, connection_type=TripletSTDPSparse, pattern=conn_ampa,
            **self.triplet_params_nmda, channel=2, weight=self.w_e2e_nmda,
        )
        self.triplet_conns.append(conn_nmda)

        # Exc -> Inh (Estático)
        if is_hidden:
            self.sim.connect(pop_exc, pop_inh, connection_type=StaticSparse, pattern="random", fanin=self.k_e2i, channel=0, weight=self.w_e2i, delay=self.delay_inh)
        else:
            self.sim.connect(pop_exc, pop_inh, connection_type=StaticSparse, pattern="random", fanin=self.k_e2i, channel=0, weight=self.w_e2i, delay=self.delay_inh)

        # Inh -> Exc (Estático)
        self.sim.connect(pop_inh, pop_exc, connection_type=StaticSparse, pattern="random", fanin=self.k_i2e, channel=1, weight=self.w_i2e, delay=self.delay_inh)
        # Inh -> Exc (Plasticidad Inhibitoria Vogels)
        conn = self.sim.connect(pop_inh, pop_exc, connection_type=VogelsSparse, pattern="random", fanin=self.k_i2e, channel=1, **self.vogels_params)
        self.vogels_conns.append(conn)
        
        # Inh -> Inh (Estático)
        ##self.sim.connect(pop_inh, pop_inh, connection_type=StaticSparse, pattern="random", fanin=self.k_i2i, autapses=False, channel=1, weight=self.w_i2i, delay=self.delay_default)


    def _connect_bidirectional(self, per_exc, per_inh, hid_exc, hid_inh, n_peripherals=1):
        """Conecta Periférico <-> Oculto"""
        # Periph -> Hidden (Exc -> Exc+Inh)
        k_e2e_fanin_hid = self.k_e2e // n_peripherals
        k_e2i_fanin_hid = self.k_e2i // n_peripherals

        # Exc -> Exc (Triplet STDP)
        norm_target_ampa = (self.internode_factor * self.norm_target_ampa) / n_peripherals
        per2hid_dict_ampa = {**self.triplet_params_ampa, 'norm_target_in': norm_target_ampa, 'norm_target_out': norm_target_ampa}
        conn_ampa = self.sim.connect(
            per_exc, hid_exc, connection_type=TripletSTDPSparse, pattern="random", fanin=k_e2e_fanin_hid, autapses=False, 
            **per2hid_dict_ampa, channel=0, weight=self.w_e2e_ampa,
        )
        self.triplet_conns.append(conn_ampa)

        norm_target_nmda = (self.internode_factor * self.norm_target_nmda) / n_peripherals
        per2hid_dict_nmda = {**self.triplet_params_nmda, 'norm_target_in': norm_target_nmda, 'norm_target_out': norm_target_nmda}
        conn_nmda = self.sim.connect(
            per_exc, hid_exc, connection_type=TripletSTDPSparse, pattern=conn_ampa,
            **per2hid_dict_nmda, channel=2, weight=self.w_e2e_nmda,
        )
        self.triplet_conns.append(conn_nmda)


        # Exc->Inh (Estático)
        self.sim.connect(per_exc, hid_inh, connection_type=StaticSparse, pattern="random", fanin=k_e2i_fanin_hid, channel=0, weight=self.w_e2i, delay=self.delay_inh)

        # Hidden -> Periph (Exc -> Exc+Inh)
        # Exc->Exc (Plástico)
        hid2per_norm_target_ampa = (self.internode_factor * self.norm_target_ampa)
        hid2per_dict_ampa = {**self.triplet_params_ampa, 'norm_target_in': hid2per_norm_target_ampa, 'norm_target_out': hid2per_norm_target_ampa}
        conn_ampa = self.sim.connect(
            hid_exc, per_exc, connection_type=TripletSTDPSparse, pattern="random", fanin=self.k_e2e, autapses=False, 
            **hid2per_dict_ampa, channel=0, weight=self.w_e2e_ampa,
        )
        self.triplet_conns.append(conn_ampa)

        hid2per_norm_target_nmda = (self.internode_factor * self.norm_target_nmda)
        hid2per_dict_nmda = {**self.triplet_params_nmda, 'norm_target_in': hid2per_norm_target_nmda, 'norm_target_out': hid2per_norm_target_nmda}
        conn_nmda = self.sim.connect(
            hid_exc, per_exc, connection_type=TripletSTDPSparse, pattern=conn_ampa,
            **hid2per_dict_nmda, channel=2, weight=self.w_e2e_nmda,
        )
        self.triplet_conns.append(conn_nmda)
        
        # Exc->Inh (Estático)
        self.sim.connect(hid_exc, per_inh, connection_type=StaticSparse, pattern="random", fanin=self.k_e2i, channel=0, weight=self.w_e2i, delay=self.delay_inh)

    # --- Lógica de Control ---

    def set_phase(self, phase_name):
        self.phase = phase_name
        
        if phase_name == "WAKE":
            for c in self.triplet_conns:
                conn:TripletSTDPSparse = c
                conn.set_learning_enabled(True)
                conn.set_learning_sign(invert=False)
            self.example_counter = 0
            self.present_wake_example()
            
        elif phase_name == "SLEEP":
            for c in self.triplet_conns:
                conn:TripletSTDPSparse = c
                conn.set_learning_enabled(True)
                conn.set_learning_sign(invert=True)
            self.phase_timer = self.sleep_duration_steps
            self.present_sleep_noise()

        elif phase_name == "TEST":
            for c in self.triplet_conns:
                conn:TripletSTDPSparse = c
                conn.set_learning_enabled(False) # Congelar pesos
            self.present_inference_input()

        print(f"--- Phase changed to {phase_name} at step {self.current_step} ---")


    def pre_step(self):
        # Lógica Wake/Sleep automática durante entrenamiento
        if self.phase == "WAKE":
            if self.current_step % self.steps_per_example == 0:
                self.example_counter += 1
                if self.example_counter >= self.n_examples_wake:
                    self.set_phase("SLEEP")
                else:
                    self.present_wake_example()
        
        elif self.phase == "SLEEP":
            # Cambiar el patrón de ruido cada cierto tiempo (ej. cada 250ms)
            if self.current_step % 250 == 0:
                self.present_sleep_noise()
            
            self.phase_timer -= 1
            if self.phase_timer <= 0:
                self.set_phase("WAKE")

    # --- Generación de Inputs ---

    def _get_gaussian_rates(self, value, sigma=0.15, max_rate=40.0):
        """Population Coding circular"""
        x = torch.linspace(0, 1, self.N_exc, device=self.current_device)
        delta = torch.abs(x - value)
        delta = torch.minimum(delta, 1.0 - delta) # Distancia circular
        rates = max_rate * torch.exp(-(delta**2)/(2*sigma**2))
        return rates


    def present_wake_example(self):
        # Relación A + B = C (mod 1)
        val_a = np.random.rand()
        val_b = np.random.rand()
        val_c = (val_a + val_b) % 1.0
        
        self.inputs["PopA"].firing_rate = self._get_gaussian_rates(val_a) + self.base_noise
        self.inputs["PopB"].firing_rate = self._get_gaussian_rates(val_b) + self.base_noise
        self.inputs["PopC"].firing_rate = self._get_gaussian_rates(val_c) + self.base_noise
        self.inputs["PopH"].firing_rate[:] = 15.0 #self.base_noise #15.0


    def present_sleep_noise(self):
        # Ruido en UN módulo, silencio en los demás
        target = np.random.choice(["PopA", "PopB", "PopC"])
        for name in ["PopA", "PopB", "PopC", "PopH"]:
            if name == target:
                # Ruido uniforme ~15Hz
                noise = torch.rand(self.N_exc, device=self.current_device) * 30.0 #15.0
                self.inputs[name].firing_rate = noise + self.base_noise
            else:
                #self.inputs[name].firing_rate = torch.zeros(self.N_exc, device=self.current_device)
                self.inputs[name].firing_rate[:] = 15.0 #self.base_noise #15.0


    def present_inference_input(self):
        # Test: A=0.2, B=0.3 -> Esperamos C=0.5
        print("Presenting Inference Input: A=0.2, B=0.3 -> C=?")
        self.inputs["PopA"].firing_rate = self._get_gaussian_rates(0.2) + self.base_noise
        self.inputs["PopB"].firing_rate = self._get_gaussian_rates(0.3) + self.base_noise
        self.inputs["PopC"].firing_rate[:] = 15.0 #self.base_noise #15.0
        self.inputs["PopH"].firing_rate[:] = 15.0 #self.base_noise #15.0


    def on_finish(self):
        for i in range(10):
            self.plot_connections(i)

        # Decodificación final (solo útil si acabamos en TEST)
        if self.phase == "TEST":
            self.analyze_inference()
        self.plot_activity()
        plt.show()
    

    def plot_activity(self):

        # Spikes de entradas

        _fig, ax0 = plt.subplots()
        ax1 = ax0.twinx()
        id_sum = 0
        for idx, group in enumerate(self.inp_monitor.groups):
            label = group.name
            spikes = self.inp_monitor.get_spike_tensor(idx).cpu()
            # Graficas
            spk_steps, spk_neurons = spikes[:, 1], spikes[:, 0]
            spk_times = spk_steps*1e-3

            unique_ids = np.unique(spk_neurons.numpy())
            n_active_neurons = len(unique_ids)
            id_map = {old: new for new, old in enumerate(unique_ids)}
            spk_neurons_remapped = np.array([id_map[i] for i in spk_neurons.numpy()])

            ax1.scatter(spk_times, spk_neurons_remapped+id_sum, s=1, label=label, c=f"C{idx}")
            id_sum += n_active_neurons
            times, rate = smooth_spikes(spk_steps, n_neurons=n_active_neurons, to_step=self.current_step, sigma=0.1)
            ax0.plot(times, rate, c=f"C{idx}")
        
        ax1.legend(loc="lower right")
        plt.title(f"Spikes from inputs")
        plt.xlabel("Time (seconds)")
        ax0.set_ylabel("Spiking rate (Hz)")
        ax1.set_ylabel("Neuron ID")

        # Spikes de poblaciones neuronales

        _fig, ax0 = plt.subplots()
        ax1 = ax0.twinx()
        id_sum = 0
        for idx, group in enumerate(self.pop_monitor.groups):
            label = group.name
            spikes = self.pop_monitor.get_spike_tensor(idx).cpu()
            # Graficas
            spk_steps, spk_neurons = spikes[:, 1], spikes[:, 0]
            spk_times = spk_steps*1e-3

            unique_ids = np.unique(spk_neurons.numpy())
            n_active_neurons = len(unique_ids)
            id_map = {old: new for new, old in enumerate(unique_ids)}
            spk_neurons_remapped = np.array([id_map[i] for i in spk_neurons.numpy()])

            ax1.scatter(spk_times, spk_neurons_remapped+id_sum, s=1, label=label, c=f"C{idx}")
            id_sum += n_active_neurons
            times, rate = smooth_spikes(spk_steps, n_neurons=n_active_neurons, to_step=self.current_step, sigma=0.1)
            ax0.plot(times, rate, c=f"C{idx}")
        
        ax1.legend(loc="lower right")
        plt.title(f"Spikes from different subpopulations")
        plt.xlabel("Time (seconds)")
        ax0.set_ylabel("Spiking rate (Hz)")
        ax1.set_ylabel("Neuron ID")

        # Mostramos el voltaje de membrana de la primera neurona

        if hasattr(self, "state_monitor"):
            fig, ax0 = plt.subplots()
            ax1 = ax0.twinx()
            ax2 = ax0.twinx()
        
            V = self.state_monitor.get_variable_tensor(0, 'V')
            ax0.plot(V, color='C0')
            spikes = self.state_monitor.get_variable_tensor(0, 'spikes')
            ax2.vlines(spikes.nonzero(as_tuple=True), ymin=0,ymax=1, color='black')
            ax2.get_yaxis().set_visible(False)
            
            ampa = self.state_monitor.get_variable_tensor(0, 'channel_currents@0')
            ax1.plot(ampa, color='C1', label='AMPA')
            gaba = self.state_monitor.get_variable_tensor(0, 'channel_currents@1')
            ax1.plot(gaba, color='C2', label='GABA')
            nmda = self.state_monitor.get_variable_tensor(0, 'channel_currents@2')
            ax1.plot(nmda, color='C3', label='NMDA')
            ax1.grid()
            ax1.legend()

        if hasattr(self, "weight_monitor"):
            fig, ax0 = plt.subplots()
            w_values = self.weight_monitor.get_variable_tensor(0, "weight")
            ax0.plot(w_values, c='C0')
            w_values = self.weight_monitor.get_variable_tensor(1, "weight")
            ax0.plot(w_values, c='C1')
            w_values = self.weight_monitor.get_variable_tensor(2, "weight")
            ax0.plot(w_values, c='C2')
            
            ax0.plot([],[], c='C0', label='A->A')
            ax0.plot([],[], c='C1', label='H->C')
            ax0.plot([],[], c='C2', label='H-|H')
            plt.legend()


    def analyze_inference(self):
        # Obtener spikes de los últimos 500ms
        spikes = self.pop_monitor.get_spike_tensor(group_index=4).cpu()
        if len(spikes) == 0:
            print("No spikes in PopC during inference.")
            return

        last_step = spikes[:, 1].max()
        window = 500
        mask = spikes[:, 1] > (last_step - window)
        relevant_spikes = spikes[mask]
        
        if len(relevant_spikes) == 0:
            print("No spikes in PopC during inference.")
            return

        # Population Vector Decoding
        ids = relevant_spikes[:, 0].long()
        # Mapear ID a ángulo (0 a 2pi)
        angles = (ids.float() / self.N_exc) * 2 * np.pi
        
        # Suma vectorial
        x = torch.cos(angles).sum()
        y = torch.sin(angles).sum()
        
        decoded_angle = torch.atan2(y, x)
        if decoded_angle < 0: decoded_angle += 2*np.pi
        decoded_val = decoded_angle / (2*np.pi)
        
        print(f"\n=== INFERENCE RESULT ===")
        print(f"Target: 0.5")
        print(f"Decoded PopC: {decoded_val.item():.4f}")
        
        # Plot
        plt.figure(figsize=(10, 4))
        plt.scatter(relevant_spikes[:, 1], relevant_spikes[:, 0], s=2, c='k', alpha=0.5)
        plt.title(f"PopC Activity (Decoded: {decoded_val.item():.2f})")
        plt.xlabel("Time step")
        plt.ylabel("Neuron ID")
        plt.ylim(0, self.N_exc)


    def plot_connections(self, index:int):
        conn_ampa = self.triplet_conns[index*2]
        conn_nmda = self.triplet_conns[index*2+1]
        title = f"Connections from {conn_ampa.pre.name} to {conn_ampa.pos.name}"

        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12,6), constrained_layout=True)
        fig.suptitle(title, fontsize=14)
        
        plot_sparse_connectivity(conn_ampa, self.N_exc, self.N_exc, ax=axs[0])
        plot_sparse_connectivity(conn_nmda, self.N_exc, self.N_exc, ax=axs[1])
        
        #plt.tight_layout()
        plt.show()

# -------------------------------------------------------------------------
# 4. Ejecución
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Configurar
    sim = Simulator(seed=1234) # Semilla fija para reproducibilidad
    exp = RelationalNetworkExperiment(sim=sim)
    
    # 2. Entrenamiento (Ejemplo corto: 2 ciclos Wake/Sleep)
    # 1 Ciclo = (50 ejemplos * 250ms) + 5000ms Sleep = 12500 + 5000 = 17500 steps
    n_cycles = 10
    #steps_train = 17500 * n_cycles
    steps_train = 2000
    
    print("--- STARTING TRAINING ---")
    exp.set_phase("WAKE")
    #exp.set_phase("SLEEP")
    exp.run(steps=steps_train, close_on_finish=False)
    
    # 3. Test / Inferencia
    print("\n--- STARTING INFERENCE ---")
    exp.set_phase("TEST")
    # Corremos 10 segundo de simulación para dejar que la red converja
    #exp.run(steps=10*1000, close_on_finish=True)
    exp.run(steps=1*1000, close_on_finish=True)