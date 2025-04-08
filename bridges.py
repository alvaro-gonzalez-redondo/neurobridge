import torch
import torch.distributed as dist


class AxonalBridge:

    def __init__(self, size: int, n_steps: int, rank: int, world_size: int, device: str) -> None:
        """
        size: número de axones (N)
        n_bridge_steps: número de pasos por ciclo (también es el desfase de transmisión)
        rank: ID de la GPU local (entre 0 y world_size-1)
        world_size: número total de GPUs
        device: 'cuda:0', 'cuda:1', etc.
        """
        self.size: int = size
        self.n_bridge_steps: int = n_steps
        self.rank: int = rank
        self.world_size: int = world_size
        self.device: torch.device = torch.device(device)
        self.t: int = 0

        self.n_packed: int = (self.size + 7) // 8

        # Buffer local de spikes escritos en esta GPU (n_bridge_steps x N)
        self.local_buffer: torch.Tensor = torch.zeros((n_steps, size), dtype=torch.uint8, device=self.device)

        # Buffer de lectura (spikes que han llegado desde otras GPUs)
        self.read_buffer: torch.Tensor = torch.zeros((n_steps, size), dtype=torch.uint8, device=self.device)


    def write_spikes(self, spikes: torch.Tensor) -> None:
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


    def step(self) -> None:
        sender_rank = self.t % self.n_bridge_steps
        is_distributed = self.world_size > 1 and dist.is_initialized()

        if is_distributed:
            if self.device.type != "cpu":
                raise RuntimeError("AxonalBridge debe estar en CPU si usas Gloo.")

            received = broadcast_spike_buffer(
                local_spikes=self.local_buffer,
                size=self.size,
                n_steps=self.n_bridge_steps,
                sender_rank=sender_rank,
                rank=self.rank,
                world_size=self.world_size,
                device=self.device
            )
            self.read_buffer |= received

        else:
            slot = self.t % self.n_bridge_steps
            self.read_buffer[slot] = self.local_buffer[slot]

        self.t += 1


    def reset(self) -> None:
        self.t = 0
        self.local_buffer.zero_()
        self.read_buffer.zero_()


    def _pack_spikes(self, tensor: torch.Tensor) -> torch.Tensor:
        padded_len = self.n_packed * 8
        pad = padded_len - self.size
        if pad > 0:
            tensor = torch.cat([
                tensor,
                torch.zeros((self.n_bridge_steps, pad), dtype=torch.uint8, device=self.device)
            ], dim=1)
        reshaped = tensor.view(self.n_bridge_steps, -1, 8)
        powers = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8, device=self.device)
        return (reshaped * powers).sum(dim=2)


    def _unpack_spikes(self, packed: torch.Tensor) -> torch.Tensor:
        unpacked = packed.unsqueeze(-1).bitwise_and(
            torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8, device=packed.device)
        ).ne(0).to(torch.uint8)
        return unpacked.view(self.n_bridge_steps, -1)[:, :self.size]


import torch
import torch.distributed as dist


def broadcast_spike_buffer(
    local_spikes: torch.Tensor,
    size: int,
    n_steps: int,
    sender_rank: int,
    rank: int,
    world_size: int,
    device: torch.device
) -> torch.Tensor:
    """
    Transmite un buffer de spikes desde un rank a todos los demás mediante `broadcast`.

    Args:
        local_spikes: tensor local de spikes (n_steps, size), dtype=uint8
        size: número de axones (N)
        n_steps: número de pasos de retardo
        sender_rank: rank que transmite en este paso
        rank: rank actual
        world_size: número total de procesos
        device: CPU obligatoriamente (requerido por Gloo)

    Returns:
        Un tensor de forma (n_steps, size) con los spikes recibidos.
    """
    assert local_spikes.shape == (n_steps, size)
    n_packed = (size + 7) // 8
    flat = torch.zeros(n_steps * n_packed, dtype=torch.uint8, device=device)

    if rank == sender_rank:
        padded_len = n_packed * 8
        pad = padded_len - size
        if pad > 0:
            local_spikes = torch.cat([
                local_spikes,
                torch.zeros((n_steps, pad), dtype=torch.uint8, device=device)
            ], dim=1)
        reshaped = local_spikes.view(n_steps, -1, 8)
        powers = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8, device=device)
        packed = (reshaped * powers).sum(dim=2)
        flat.copy_(packed.flatten().contiguous())

    # Comunicación global
    dist.broadcast(flat, src=sender_rank)

    # Todos desempaquetan
    packed = flat.view(n_steps, n_packed)
    unpacked = packed.unsqueeze(-1).bitwise_and(
        torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8, device=device)
    ).ne(0).to(torch.uint8)

    return unpacked.view(n_steps, -1)[:, :size]
