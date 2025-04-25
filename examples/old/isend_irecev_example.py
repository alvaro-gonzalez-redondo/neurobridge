import torch
import torch.distributed as dist
import os
import logging
import random
import numpy as np


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def setup_logger(rank: int) -> logging.Logger:
    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger(f"rank{rank}")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - [Rank %(name)s] %(message)s")

    fh = logging.FileHandler(f"logs/log_rank{rank}.txt")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def bool_to_uint8(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.uint8)
    pad_len = (8 - x.numel() % 8) % 8
    if pad_len:
        x = torch.cat([x, torch.zeros(pad_len, dtype=torch.uint8, device=x.device)])
    x = x.reshape(-1, 8)
    weights = torch.tensor(
        [1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8, device=x.device
    )
    return (x * weights).sum(dim=1)


def uint8_to_bool(x: torch.Tensor, num_bits: int) -> torch.Tensor:
    bits = ((x.unsqueeze(1) >> torch.arange(8, device=x.device)) & 1).to(torch.bool)
    return bits.flatten()[:num_bits]


def define_ring_topology(rank: int, world_size: int) -> list[dict]:
    """Topología en anillo bidireccional con diferentes tamaños por canal"""
    base = 20  # bits base por conexión (puedes cambiar esto)
    return [
        {
            "target": (rank + 1) % world_size,
            "direction": "send",
            "num_bits": base + rank,
        },
        {
            "target": (rank - 1 + world_size) % world_size,
            "direction": "send",
            "num_bits": base + rank,
        },
        {
            "target": (rank - 1 + world_size) % world_size,
            "direction": "recv",
            "num_bits": base + ((rank - 1 + world_size) % world_size),
        },
        {
            "target": (rank + 1) % world_size,
            "direction": "recv",
            "num_bits": base + ((rank + 1) % world_size),
        },
    ]


def main():
    try:
        # Seguridad: verificar número de GPUs disponibles
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA no está disponible.")
        local_gpu_count = torch.cuda.device_count()
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size > local_gpu_count:
            raise RuntimeError(
                f"Se requieren {world_size} GPUs pero solo hay {local_gpu_count}."
            )
        if local_gpu_count < 2:
            raise RuntimeError(
                f"Se requieren al menos 2 GPUs, pero sólo hay {local_gpu_count}."
            )

        # Inicialización distribuida
        if "RANK" in os.environ:
            dist.init_process_group(
                backend="nccl"
            )  # , init_method="env://") #Se asume que se ha ejecutado el script con `torchrun`
            rank = dist.get_rank()
        else:
            raise RuntimeError("Use `torchrun`.")

        device = torch.device(f"cuda:{rank % local_gpu_count}")
        torch.cuda.set_device(device)

        # Configurar logger
        logger = setup_logger(rank)

        # Reproducibilidad
        torch.manual_seed(42 + rank)
        random.seed(42 + rank)
        np.random.seed(42 + rank)

        # Comunicación asíncrona punto a punto
        topology = define_ring_topology(rank, world_size)

        send_reqs = []
        recv_results = {}

        for conn in topology:
            tgt = conn["target"]
            direction = conn["direction"]
            num_bits = conn["num_bits"]

            if direction == "send":
                bool_data = torch.randint(
                    0, 2, (num_bits,), dtype=torch.bool, device=device
                )
                packed = bool_to_uint8(bool_data)
                req = dist.isend(packed.clone(), dst=tgt)
                send_reqs.append(req)
                logger.info(
                    f"Enviando {num_bits} bits a rank {tgt}: {bool_data.to(torch.uint8)}"
                )

            elif direction == "recv":
                padded_len = (num_bits + 7) // 8  # número de bytes
                recv_buf = torch.empty(padded_len, dtype=torch.uint8, device=device)
                req = dist.irecv(recv_buf, src=tgt)
                recv_results[tgt] = (recv_buf, req, num_bits)

        for src, (buf, req, nbits) in recv_results.items():
            req.wait()
            result = uint8_to_bool(buf, nbits)
            logger.info(
                f"Recibido {nbits} bits de rank {src}: {result.to(torch.uint8)}"
            )

        for req in send_reqs:
            req.wait()

    except Exception as e:
        print(f"[Rank {os.environ.get('RANK', 0)}] ERROR: {e}")
    finally:
        if is_distributed():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
