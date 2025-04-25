import torch
import torch.distributed as dist
import os
import time
import argparse


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def bool_to_uint8(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.uint8)
    pad_len = (8 - x.numel() % 8) % 8
    if pad_len:
        x = torch.cat([x, torch.zeros(pad_len, dtype=torch.uint8, device=x.device)])
    x = x.reshape(-1, 8)
    weights = 2 ** torch.arange(8, dtype=torch.uint8, device=x.device)
    return (x * weights).sum(dim=1)


def uint8_to_bool(x: torch.Tensor, num_bits: int) -> torch.Tensor:
    bits = ((x.unsqueeze(1) >> torch.arange(8, device=x.device)) & 1).to(torch.bool)
    return bits.flatten()[:num_bits]


def run_benchmark(
    A: int, B: int, iters: int, rank: int, world_size: int, device: torch.device
):
    total_time = 0.0
    num_bits = A * B

    for _ in range(iters):
        # 1. Crear matriz booleana local aleatoria
        bool_matrix = torch.randint(0, 2, (A, B), dtype=torch.bool, device=device)

        # 2. Codificar a uint8
        packed = bool_to_uint8(bool_matrix.flatten())

        # 3. Medir tiempo de all_gather + reconstrucción
        start = time.perf_counter()

        if is_distributed():
            gathered = [torch.empty_like(packed) for _ in range(world_size)]
            dist.all_gather(gathered, packed)
        else:
            gathered = [packed]

        # 4. Decodificar
        bool_list = [uint8_to_bool(p, num_bits).reshape(A, B) for p in gathered]
        _ = torch.cat(bool_list, dim=0)

        if is_distributed():
            dist.barrier()

        end = time.perf_counter()
        total_time += end - start

    return total_time / iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--A", type=int, default=100, help="Filas de la matriz booleana local"
    )
    parser.add_argument(
        "--B", type=int, default=100, help="Columnas de la matriz booleana local"
    )
    parser.add_argument("--iters", type=int, default=50, help="Número de repeticiones")
    args = parser.parse_args()

    if torch.cuda.is_available():
        local_gpu_count = torch.cuda.device_count()
    else:
        raise RuntimeError("CUDA no está disponible")

    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    device = torch.device(f"cuda:{rank % local_gpu_count}")
    torch.cuda.set_device(device)

    # Benchmark
    avg_time = run_benchmark(args.A, args.B, args.iters, rank, world_size, device)

    if rank == 0:
        total_MB = args.A * args.B * world_size / 8 / 1024 / 1024  # bits → MB approx
        print(f"\n[Rank 0] Matriz local: {args.A}x{args.B} | GPUs: {world_size}")
        print(f"[Rank 0] Tamaño total de matriz compartida: ~{total_MB:.2f} MB")
        print(f"[Rank 0] Tiempo medio por iteración: {avg_time*1000:.3f} ms")

    if is_distributed():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
