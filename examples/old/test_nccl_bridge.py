import os
import torch
import torch.distributed as dist
from neurobridge.bridges import AxonalBridge  # Usa versión con broadcast()


def test_axonal_bridge_nccl():
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    size = 64
    n_steps = world_size
    device = torch.device(f"cuda:{rank}")

    bridge = AxonalBridge(
        size=size, n_steps=n_steps, rank=rank, world_size=world_size, device=device
    )

    # Spike único por rank, en una posición única
    spike = torch.zeros(size, dtype=torch.uint8, device=device)
    spike_index = 3 + rank * 2
    spike[spike_index] = 1
    print(f"[Rank {rank}] envía spike en índice {spike_index}")
    bridge.write_spikes(spike)

    for _ in range(n_steps):
        bridge.step()

    received = bridge.read_spikes()

    # Mover a CPU para imprimir (opcional)
    r_cpu = received.cpu()
    print(f"[Rank {rank}] recibió:\n{r_cpu}")

    if rank != 0:
        total = r_cpu.sum().item()
        assert (
            total == world_size
        ), f"[Rank {rank}] ERROR: se esperaban {world_size} spikes, se recibieron {total}"
        for i in range(world_size):
            expected = 3 + i * 2
            assert (
                r_cpu[expected] == 1
            ), f"[Rank {rank}] ERROR: falta spike de {i} en índice {expected}"
        print(f"[Rank {rank}] ✅ Comunicación múltiple correcta en CUDA.")


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Inicializar NCCL
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:29500",
        rank=rank,
        world_size=world_size,
    )

    # Seleccionar GPU
    torch.cuda.set_device(rank)

    test_axonal_bridge_nccl()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
