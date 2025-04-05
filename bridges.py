import torch

class AxonalBridge:

    def __init__(self, size, n_bridge_steps, rank, world_size, device):
        """
        size: número de axones (N)
        n_bridge_steps: número de pasos por ciclo (también es el desfase de transmisión)
        rank: ID de la GPU local (entre 0 y world_size-1)
        world_size: número total de GPUs
        device: 'cuda:0', 'cuda:1', etc.
        """
        self.size = size
        self.n_bridge_steps = n_bridge_steps
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(device)
        self.t = 0

        self.n_packed = (self.size + 7) // 8

        # Buffer local de spikes escritos en esta GPU (n_bridge_steps x N)
        self.local_buffer = torch.zeros((n_bridge_steps, size), dtype=torch.uint8, device=self.device)

        # Buffer de lectura (spikes que han llegado desde otras GPUs)
        self.read_buffer = torch.zeros((n_bridge_steps, size), dtype=torch.uint8, device=self.device)


    def write_spikes(self, spikes: torch.Tensor):
        """
        Añade spikes generados en esta GPU en el paso actual.
        spikes: (N,), uint8, valores 0 o 1.
        """
        assert spikes.shape[0] == self.size
        self.local_buffer[self.t % self.n_bridge_steps] |= spikes.to(dtype=torch.uint8)


    def read_spikes(self) -> torch.Tensor:
        """
        Devuelve los spikes recibidos hace `n_bridge_steps` pasos.
        """
        return self.read_buffer[self.t % self.n_bridge_steps]


    def step(self):
        """
        Avanza un paso de simulación:
        - En t % n_bridge_steps == rank → esta GPU transmite.
        - En t % n_bridge_steps == otro rank → esta GPU recibe y acumula.
        """
        current_slot = self.t % self.n_bridge_steps
        sender_rank = self.t % self.n_bridge_steps
        is_distributed = self.world_size > 1 and dist.is_initialized()

        if is_distributed:
            # Enviar si es nuestro turno
            if self.rank == sender_rank:
                packed = self._pack_spikes(self.local_buffer)
                for dst in range(self.world_size):
                    if dst != self.rank:
                        dist.send(tensor=packed, dst=dst)

            # Recibir si otro transmite
            elif self.rank != sender_rank:
                recv_buf = torch.empty((self.n_bridge_steps, self.n_packed), dtype=torch.uint8, device=self.device)
                dist.recv(tensor=recv_buf, src=sender_rank)
                unpacked = self._unpack_spikes(recv_buf)
                self.read_buffer |= unpacked  # acumulación binaria
        else:
            # Sin comunicación: replicar local_buffer como si fuera recibido
            self.read_buffer[self.t % self.n_bridge_steps] = self.local_buffer[self.t % self.n_bridge_steps]

        self.t += 1


    def reset(self):
        self.t = 0
        self.local_buffer.zero_()
        self.read_buffer.zero_()


    def _pack_spikes(self, tensor: torch.Tensor) -> torch.Tensor:
        padded_len = self.n_packed * 8
        pad = padded_len - self.size
        if pad > 0:
            tensor = torch.cat([tensor, torch.zeros((self.n_bridge_steps, pad), dtype=torch.uint8, device=self.device)], dim=1)
        reshaped = tensor.view(self.n_bridge_steps, -1, 8)
        powers = torch.tensor([1,2,4,8,16,32,64,128], dtype=torch.uint8, device=self.device)
        return (reshaped * powers).sum(dim=2)


    def _unpack_spikes(self, packed: torch.Tensor) -> torch.Tensor:
        unpacked = packed.unsqueeze(-1).bitwise_and(
            torch.tensor([1,2,4,8,16,32,64,128], dtype=torch.uint8, device=packed.device)
        ).ne(0).to(torch.uint8)
        return unpacked.view(self.n_bridge_steps, -1)[:, :self.size]
