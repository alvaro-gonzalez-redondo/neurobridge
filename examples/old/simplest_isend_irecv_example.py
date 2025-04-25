import torch
import torch.distributed as dist
import os

# Safety checkings

if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available.")
if "RANK" not in os.environ:
    raise RuntimeError("Run with `torchrun`.")


# Inicialización distribuida

dist.init_process_group(backend="nccl")

local_gpu_count = torch.cuda.device_count()
world_size = int(os.environ.get("WORLD_SIZE", "1"))
rank = dist.get_rank()
device = torch.device(f"cuda:{rank % local_gpu_count}")
torch.cuda.set_device(device)


# Sending data

try:
    peer = 1 - rank  # Solo válido para 2 procesos
    if rank == 0:
        data = torch.randint(10, size=(1000,), dtype=torch.long, device=device)
        req = dist.isend(data, dst=peer)
        req.wait()
    else:
        data = torch.randint(10, size=(1000,), dtype=torch.long, device=device)
        recv_buf = torch.empty_like(data)
        req = dist.irecv(recv_buf, src=peer)
        req.wait()
        print("Recibido:", recv_buf[:5])

except Exception as e:
    print(f"[Rank {os.environ.get('RANK', 0)}] ERROR: {e}")

finally:
    dist.destroy_process_group()
