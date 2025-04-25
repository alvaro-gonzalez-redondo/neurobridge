import os
import torch
import torch.distributed as dist


def init_distributed():
    dist.init_process_group(
        backend="gloo",
        init_method="tcp://127.0.0.1:29500",
        rank=int(os.environ["RANK"]),
        world_size=int(os.environ["WORLD_SIZE"]),
    )


def pack_spikes(spikes: torch.Tensor) -> torch.Tensor:
    n_steps, size = spikes.shape
    n_packed = (size + 7) // 8
    padded_len = n_packed * 8
    pad = padded_len - size

    if pad > 0:
        spikes = torch.cat(
            [spikes, torch.zeros((n_steps, pad), dtype=torch.uint8)], dim=1
        )

    reshaped = spikes.view(n_steps, -1, 8)
    powers = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8)
    return (reshaped * powers).sum(dim=2)


def unpack_spikes(packed: torch.Tensor, size: int) -> torch.Tensor:
    n_steps, n_packed = packed.shape
    unpacked = (
        packed.unsqueeze(-1)
        .bitwise_and(torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8))
        .ne(0)
        .to(torch.uint8)
    )
    return unpacked.view(n_steps, -1)[:, :size]


def main():
    init_distributed()

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.manual_seed(rank)

    size = 64  # número de neuronas
    n_steps = 2  # pasos de tiempo
    n_packed = (size + 7) // 8
    buffer_shape = (n_steps, size)
    flat_shape = (n_steps * n_packed,)

    # Cada proceso crea su buffer de spikes
    local_spikes = torch.zeros(buffer_shape, dtype=torch.uint8)
    if rank == 0:
        local_spikes[0, 3] = 1  # El spike a enviar
    else:
        local_spikes[:] = 0  # Vacío

    # Empaquetar en Rank 0
    flat_packed = torch.zeros(flat_shape, dtype=torch.uint8)
    if rank == 0:
        packed = pack_spikes(local_spikes).to(torch.uint8)
        flat_packed.copy_(packed.flatten().contiguous())
    # Difundir el tensor plano desde Rank 0 a todos
    print(
        f"[Rank {rank}] flat_packed.shape = {flat_packed.shape}, dtype = {flat_packed.dtype}"
    )
    dist.broadcast(flat_packed, src=0)

    # Desempaquetar
    received_packed = flat_packed.view(n_steps, n_packed)
    unpacked = unpack_spikes(received_packed, size)

    # Verificación
    if rank != 0:
        print(f"[Rank {rank}] recibió:\n{unpacked}")
        assert unpacked.sum().item() == 1, f"[Rank {rank}] recibió datos incorrectos"
        assert unpacked[0, 3] == 1, f"[Rank {rank}] falta el spike esperado"
        print(f"[Rank {rank}] ✅ comunicación exitosa.")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
