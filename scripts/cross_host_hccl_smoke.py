#!/usr/bin/env python3
"""Cross-host HCCL smoke test for 2-rank distributed all_reduce + send/recv."""
import os
import time
import torch
import torch.distributed as dist
import torch_npu  # noqa

rank = int(os.environ['RANK'])
world = int(os.environ['WORLD_SIZE'])
local = int(os.environ['LOCAL_RANK'])

print(f"[rank{rank}] Initializing: LOCAL_RANK={local}, RANK={rank}, WORLD_SIZE={world}")
print(f"[rank{rank}] MASTER_ADDR={os.environ.get('MASTER_ADDR')}, MASTER_PORT={os.environ.get('MASTER_PORT')}")

torch.npu.set_device(local)
print(f"[rank{rank}] Set device to npu:{local}")

dist.init_process_group(backend='hccl', init_method='env://', rank=rank, world_size=world)
print(f"[rank{rank}] Process group initialized")

# All-reduce test
x = torch.ones(1024, device=f'npu:{local}') * (rank + 1)
torch.npu.synchronize()
t0 = time.time()
dist.all_reduce(x)
torch.npu.synchronize()
t1 = time.time()

expected_sum = 1024 * (world * (world + 1) / 2)
actual_sum = x.sum().item()
print(f'[rank{rank}] all_reduce sum={actual_sum} expect={expected_sum} dt={(t1-t0)*1000:.3f}ms')

# Send/recv pingpong
peer = 1 - rank
buf = torch.zeros(1024, device=f'npu:{local}')
torch.npu.synchronize()
t0 = time.time()
if rank == 0:
    dist.send(x, peer)
    dist.recv(buf, peer)
else:
    dist.recv(buf, peer)
    dist.send(x, peer)
torch.npu.synchronize()
t1 = time.time()

# Verify received data
ok = torch.allclose(buf, torch.ones_like(buf) * (peer + 1))
print(f'[rank{rank}] pingpong dt={(t1-t0)*1000:.3f}ms ok={ok} recv_sum={buf.sum().item()}')

dist.destroy_process_group()
print(f'[rank{rank}] Test completed successfully')
