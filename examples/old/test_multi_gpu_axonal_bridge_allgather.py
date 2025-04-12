import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import time


def run(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Parámetros
    A = 32  # número total de axones
    T = world_size  # desfase de sincronización
    S = 10  # tamaño del spike buffer circular
    steps = 2 * T

    write_buffer = torch.zeros((A, T), dtype=torch.uint8, device=device)
    spike_buffer = torch.zeros((A, S), dtype=torch.uint8, device=device)

    for t in range(steps):
        spike_index = rank * 2 + t % 2  # Alternar entre dos spikes
        write_buffer[spike_index, t % T] = 1

        if (t + 1) % T == 0:
            gathered = [torch.zeros_like(write_buffer) for _ in range(world_size)]
            dist.all_gather(gathered, write_buffer)

            for other in gathered:
                for dt in range(T):
                    future_slot = (t + 1 + dt) % S
                    spike_buffer[:, future_slot] |= other[:, dt]

            write_buffer.zero_()

        dist.barrier()

    dist.barrier()
    time.sleep(0.1 * rank)

    expected = torch.zeros(A, dtype=torch.uint8, device=device)
    for r in range(world_size):
        expected[r * 2] = 1
        expected[r * 2 + 1] = 1

    latest = spike_buffer[:, (steps - 1) % S]
    print(f"[Rank {rank}] Último slot del spike_buffer:\n{latest}")
    assert torch.all(latest == expected), f"[Rank {rank}] ❌ Spike buffer incorrecto"


def main():
    world_size = torch.cuda.device_count()
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
