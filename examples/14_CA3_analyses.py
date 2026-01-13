from neurobridge import *
import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List


class CA3Experiment(Experiment):
    # Estructura general de la red
    n_noise: int = 100
    n_inputs: int = 200
    n_excitatory: int = 1000
    n_inhibitory: int = 100

    # Parámtros de las neuronas
    noise_firing_rate: float = 15.0

    pc_params: Dict = {
        'tau_membrane': 0.041793503480278424,
        'E_leak': -0.07519,      # EL: Potencial de reposo físico
        'v_reset': -0.02974,     # Vr: Potencial tras el disparo
        'v_rheobase': -0.02442,  # VT: Umbral de inicio de la exponencial
        'v_peak': 0.0,      # Nivel de corte para detectar el spike
        'delta_T': 0.00423,      # Factor de pendiente
        'a': -0.06264501160092809,              # Adaptación sub-umbral (nS si gL fuera real)
        'b': 0.047990719257540615,            # Adaptación activada por disparos (pA)
        'tau_w': 0.08493,        # Constante de tiempo de w
        'n_channels': 3,
        'channel_time_constants': [(0.00130, 0.0095), (0.0003, 0.0033), (0.002, 0.02)],
    }

    pvbc_params: Dict = {
        'tau_membrane': 0.015781624500665777,
        'E_leak': -0.07474,      # EL: Potencial de reposo físico
        'v_reset': -0.06499,     # Vr: Potencial tras el disparo
        'v_rheobase': -0.057710000000000004,  # VT: Umbral de inicio de la exponencial
        'v_peak': 0.0,      # Nivel de corte para detectar el spike
        'delta_T': 0.00458,      # Factor de pendiente
        'a': 0.40612516644474034,              # Adaptación sub-umbral (nS si gL fuera real)
        'b': 0.00012117177097203729,            # Adaptación activada por disparos (pA)
        'tau_w': 0.178580,        # Constante de tiempo de w
        'n_channels': 2,
        'channel_time_constants': [(0.0005, 0.002), (0.0005, 0.005)],
        'channel_reversal_potentials': [0.0, -0.07],        
    }

    # Parámetros de las conexiones
    norm_target = 5e-3
    lr = norm_target * 5e-3
    static_params = {'connection_type':StaticSparse}
    stdp_params = {'connection_type':STDPSparseNormalized, 'w_max':1.0, 'A_plus':lr, 'A_minus':-lr*1.2, 'norm_every':100, 'norm_target':norm_target}

    noi2exc_conn =     {**static_params, 'pattern':"random", 'fanin':5,   'weight':Constant(1e-3),     'delay':UniformInt(0,19)}
    inp2exc_conn =     {**stdp_params,   'pattern':"random", 'fanin':100, 'weight':Uniform(0.0, 2e-4), 'delay':UniformInt(0,19)}
    inp2inh_conn =     {**static_params, 'pattern':"random", 'fanin':50,  'weight':Uniform(0.0, 2e-4), 'delay':Constant(0)}

    exc2inh_conn =     {**static_params, 'pattern':"random", 'fanin':50,  'weight':Uniform(0.0, 5e-4), 'delay':Constant(0)}
    inh2exc_sta_conn = {**static_params, 'pattern':"random", 'fanin':10,  'weight':Uniform(0.0, 1e-3), 'delay':Constant(0),      'channel':1}

    
    def build_network(self):
        
        with self.sim.autoparent("normal"):
            # Capas
            self.noi_neurons:RandomSpikeNeurons = RandomSpikeNeurons(n_neurons=self.n_noise, firing_rate=self.noise_firing_rate, name='Noise')
            self.inp_neurons:RandomSpikeNeurons = RandomSpikeNeurons(n_neurons=self.n_inputs, firing_rate=0.0, name='Input')
            self.pc_neurons:AdExNeurons = AdExNeurons(self.n_excitatory, name='PC', **self.pc_params)
            self.pvbc_neurons:AdExNeurons = AdExNeurons(self.n_inhibitory, name='PVBC', **self.pvbc_params)

            # Conexiones
            self.noi2exc:StaticSparse = self.sim.connect(self.noi_neurons, self.pc_neurons, **self.noi2exc_conn)
            self.inp2exc:StaticSparse = self.sim.connect(self.inp_neurons, self.pc_neurons, **self.inp2exc_conn)
            self.inp2inh:StaticSparse = self.sim.connect(self.inp_neurons, self.pvbc_neurons, **self.inp2inh_conn)
            self.exc2inh:StaticSparse = self.sim.connect(self.pc_neurons, self.pvbc_neurons, **self.exc2inh_conn)
            self.inh2exc_sta:StaticSparse = self.sim.connect(self.pvbc_neurons, self.pc_neurons, **self.inh2exc_sta_conn)

        with self.sim.autoparent("normal"):

            # Monitores en tiempo real

            self.viz = VisualizerClient()
            self.viz.reset()

            self.rt_spikes = RealtimeSpikeMonitor(
                groups = [
                    self.inp_neurons.where_idx( lambda i: (0<=i) & (i<50) ),
                    self.pc_neurons.where_idx(  lambda i: (0<=i) & (i<50) ),
                    self.pvbc_neurons.where_idx(lambda i: (0<=i) & (i<50) ),
                ],
                viz_client=self.viz, 
                plot_id="raster_1",
                rollover_spikes=2_000,
            )

            self.rt_weights_inp2exc = RealtimeVariableMonitor(
                groups=[self.inp2exc.where_idx( lambda i: i<20 ),],
                variable_names=["weight"], viz_client=self.viz, plot_id="w_inp2exc", interval=50,rollover=200,
            )                


            # ---

            # Monitorización estática
            self.spike_monitor = SpikeMonitor(
                [
                    self.inp_neurons.where_idx( lambda i: i<50 ),   #0
                    self.pc_neurons.where_idx(  lambda i: i<50 ),   #1
                    self.pvbc_neurons.where_idx(lambda i: i<50 ),   #2
                ]
            )
    

    def on_start(self, **kwargs):
        total_steps = kwargs.get('total_steps', 1000)
        inputs = self._create_input_stream(total_steps)

        # Mapeamos los inputs a su distribución final
        inputs = inputs - torch.mean(inputs)
        inputs = inputs / torch.std(inputs)
        inputs = (torch.tanh(inputs)+1)/2
        inputs = inputs**1.5
        inputs = inputs*55 + 5.0

        self.inp_activities = inputs
    

    def _create_input_stream(self, total_steps):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float32

        # ---- Espacio 2D acotado [0,1]^2 (dim 1 toroidal con periodo 1, por ejemplo)
        D = 2
        domain_min = torch.zeros(D)
        domain_max = 10*torch.ones(D)

        # ---- 3 escalas (fine/mid/coarse) con Poisson radii distintos
        O = self.n_inputs // 2          # ON-OFF / OFF-ON
        M = O // 3                      # por escala
        remainder = O - 3 * M           # sobrantes reales

        scales = [
            dict(name="fine",   r=1.0, M=M + remainder, sigma_alpha=1.0),
            dict(name="mid",    r=3.0, M=M,             sigma_alpha=1.0),
            dict(name="coarse", r=9.0, M=M,             sigma_alpha=1.0),
        ]
        toroidal_dims = {1: 1.0}  # por ejemplo: dimensión y es circular (ángulo normalizado), opcional

        spaces: List[RBFSpace] = []
        for sc in scales:
            centers = poisson_disk_sampling(
                n_points_target=200,
                dim=D,
                domain_min=domain_min,
                domain_max=domain_max,
                r=sc["r"],
                k=30,
                seed=123,
                device=device,
                dtype=dtype,
            )
            K = centers.shape[0]
            basis = torch.rand((K, sc["M"]), device=device, dtype=dtype)

            sigma = sc["sigma_alpha"] * sc["r"]
            spaces.append(RBFSpace(centers=centers, basis_vectors=basis, sigma=sigma, toroidal_dims=toroidal_dims))

        encoder = MultiScaleRBFEncoder(spaces=spaces, noise_std=0.01)

        # ---- Librería de segmentos spline
        # Dos segmentos (S0 y S1) para ilustrar bifurcación
        seg_library: Dict[str, SplineSegment] = {}
        seg_library["seg_A"] = SplineSegment(
            waypoints=torch.tensor([[2.0, 2.0], [8.0, 2.0], [8.0, 7.0], [8.0, 2.0], [2.0, 2.0]], device=device),
            speed=10.0,
            samples_per_segment=200,
            device=device,
        )
        seg_library["seg_B"] = SplineSegment(
            waypoints=torch.tensor([[2.0, 2.0], [2.0, 8.0], [7.0, 8.0], [2.0, 8.0], [2.0, 2.0]], device=device),
            speed=10.0,
            samples_per_segment=200,
            device=device,
        )

        # ---- FSM con bifurcación probabilística
        fsm = StateMachine(initial_state="S")
        fsm.add_transition("S", "S", probability=0.5, segment_id="seg_A")
        fsm.add_transition("S", "S", probability=0.5, segment_id="seg_B")

        ou_noise = ContinuousOUNoise(
            dim=D,
            sigma=0.1,   # amplitud del ruido (en unidades del espacio)
            tau=5.0,      # segundos de correlación
            device=device
        )
        #ou_noise.sigma = 0.0 # Para replay determinista

        trajectory = TrajectoryRunner(fsm=fsm, spline_library=seg_library, ou_noise=ou_noise)
        gen = SensoryTrajectoryGenerator(trajectory=trajectory, encoder=encoder)
        gen.reset()

        # ---- Generar unas muestras
        dt = 1e-3
        xs = []
        ys = []
        inputs = []
        segment_ids = []
        for _ in range(total_steps):
            pos = trajectory.step(dt)
            segment_ids.append(trajectory._active_segment_id)

            xs.append(pos[0].item())
            ys.append(pos[1].item())

            on = encoder.encode(pos)            # shape: (O,)
            on = torch.clamp(on, 0.0, 1.0)      # por seguridad numérica
            off = 1.0 - on
            inp = torch.cat([on, off], dim=0)   # shape: (2*O,)

            if inp.numel() < self.n_inputs:
                pad = torch.zeros(
                    self.n_inputs - inp.numel(),
                    device=inp.device,
                    dtype=inp.dtype
                )
                inp = torch.cat([inp, pad], dim=0)

            inputs.append(inp)
        inputs = torch.stack(inputs)
        return inputs
    

    def pre_step(self):
        self.inp_neurons.firing_rate[:] = self.inp_activities[self.current_step, :]


    def on_finish(self):
        # Raster plots de todas las poblaciones
        plot_spikes(self.current_step, self.spike_monitor) #phase_sorting_t=2*np.pi)
        
        # PCAs y UMAPs de todas las poblaciones
        for i, group in enumerate(self.spike_monitor.groups):
            spikes = self.spike_monitor.get_spike_tensor(i)
            plot_neural_trajectory_pca(spikes, title=f"PCA of {group.name} layer")
            #plot_neural_trajectory_umap(spikes, title=f"UMAP of {group.name} layer")

        plt.show(block=True)



if __name__ == "__main__":
    exp = CA3Experiment(sim=Simulator(seed=0))
    simulation_time = 100.0
    exp.run(steps=int(1000*simulation_time))