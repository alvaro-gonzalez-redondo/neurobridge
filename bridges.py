import torch
import torch.distributed as dist


class AxonalBridge:
    def __init__(self, size: int, n_steps: int, spike_buffer_steps: int,
                 rank: int, world_size: int, device: torch.device) -> None:
        """
        size: Nº de axones (A)
        n_steps: frecuencia de sincronización entre GPUs (T pasos)
        spike_buffer_steps: tamaño del buffer circular de spikes (S pasos)
        rank: ID de este proceso
        world_size: total de procesos (GPUs)
        device: debe ser torch.device('cuda:X')
        """
        self.size = size
        self.n_bridge_steps = n_steps  # T
        self.spike_buffer_steps = spike_buffer_steps  # S
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.t = 0

        self.write_buffer = torch.zeros((size, n_steps), dtype=torch.uint8, device=device)
        self.read_buffer = torch.zeros((size, n_steps), dtype=torch.uint8, device=device)
        self.spike_buffer = torch.zeros((size, spike_buffer_steps), dtype=torch.uint8, device=device)

    def write_spikes(self, spikes: torch.Tensor) -> None:
        """Escribe spikes locales en el write_buffer."""
        assert spikes.shape[0] == self.size
        self.write_buffer[:, self.t % self.n_bridge_steps] |= spikes.to(torch.uint8)

    def read_spikes(self, delay_steps: torch.Tensor, axon_indices: torch.Tensor) -> torch.Tensor:
        """
        Lee del spike_buffer circular.
        delay_steps: tensor de retrasos
        axon_indices: tensor de índices de axones
        """
        time_indices = (self.t - delay_steps) % self.spike_buffer_steps
        return self.spike_buffer[axon_indices, time_indices]

    def step(self) -> None:
        t = self.t

        # Solo sincroniza cada T pasos
        if t % self.n_bridge_steps == 0:
            all_write_buffers = [torch.zeros_like(self.write_buffer) for _ in range(self.world_size)]
            dist.all_gather(all_write_buffers, self.write_buffer)

            # OR binario entre todos los buffers
            combined = torch.stack(all_write_buffers, dim=0).sum(dim=0).clamp(max=1).to(torch.uint8)
            self.read_buffer.copy_(combined)

            # Coloca en el futuro del spike_buffer
            slots = (torch.arange(self.n_bridge_steps, device=self.device) + t + 1) % self.spike_buffer_steps
            self.spike_buffer[:, slots] = self.read_buffer

            self.write_buffer.zero_()

        self.t += 1

    def reset(self) -> None:
        self.t = 0
        self.write_buffer.zero_()
        self.read_buffer.zero_()
        self.spike_buffer.zero_()
