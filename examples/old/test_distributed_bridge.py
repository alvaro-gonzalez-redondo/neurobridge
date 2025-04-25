import os
import torch
import torch.distributed as dist
from neurobridge.bridges import AxonalBridge  # Asegúrate de que usa broadcast()


def test_axonal_bridge_all_ranks():
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    size = 64
    n_steps = world_size
    device = torch.device("cpu")

    bridge = AxonalBridge(
        size=size, n_steps=n_steps, rank=rank, world_size=world_size, device="cpu"
    )

    # Cada rank dispara su spike único en una neurona diferente
    spike = torch.zeros(size, dtype=torch.uint8)
    spike_index = 3 + rank * 2  # Separados por claridad
    spike[spike_index] = 1
    print(f"[Rank {rank}] envió spike en índice {spike_index}")
    bridge.write_spikes(spike)

    # Ejecutar un ciclo completo de comunicación
    for _ in range(n_steps):
        bridge.step()

    # Leer los spikes acumulados
    received = bridge.read_spikes()
    total_spikes = received.sum().item()

    # Validar: deben haberse recibido todos los spikes
    if rank != 0:
        print(f"[Rank {rank}] recibió:\n{received}")
        assert (
            total_spikes == world_size
        ), f"[Rank {rank}] ❌ ERROR: se esperaban {world_size} spikes, se recibieron {total_spikes}"
        for i in range(world_size):
            expected_idx = 3 + i * 2
            assert (
                received[expected_idx] == 1
            ), f"[Rank {rank}] ❌ ERROR: falta spike de rank {i} en índice {expected_idx}"
        print(f"[Rank {rank}] ✅ Comunicación múltiple correcta.")


def main():
    dist.init_process_group(
        backend="gloo",
        init_method="tcp://127.0.0.1:29500",
        rank=int(os.environ["RANK"]),
        world_size=int(os.environ["WORLD_SIZE"]),
    )
    test_axonal_bridge_all_ranks()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
