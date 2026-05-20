#!/usr/bin/env python3
"""D3 smoke: cross-host HCCL + DeepEP Buffer construction.

Run on each host (rank 0 on Host1, rank 1 on Host2) with env:
  MASTER_ADDR=192.168.0.125 MASTER_PORT=29555
  HCCL_IF_BASE_PORT=24500 HCCL_IF_IP=<this host's RoCE IP>
  RANK=<0|1>  WORLD_SIZE=2  LOCAL_RANK=0
"""
import os
import sys
import time
import traceback

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

rank = int(os.environ['RANK'])
world = int(os.environ['WORLD_SIZE'])
local = int(os.environ['LOCAL_RANK'])

print(f"[rank{rank}] starting: LOCAL={local} WORLD={world}", flush=True)
print(f"[rank{rank}] MASTER={os.environ.get('MASTER_ADDR')}:{os.environ.get('MASTER_PORT')} "
      f"HCCL_IF_IP={os.environ.get('HCCL_IF_IP')} BASE_PORT={os.environ.get('HCCL_IF_BASE_PORT')}",
      flush=True)

torch.npu.set_device(local)
dist.init_process_group(backend='hccl', init_method='env://', rank=rank, world_size=world)
print(f"[rank{rank}] HCCL pg initialized", flush=True)

# Sanity: all_reduce
x = torch.ones(8, device=f'npu:{local}') * (rank + 1)
torch.npu.synchronize()
t0 = time.time()
dist.all_reduce(x)
torch.npu.synchronize()
print(f"[rank{rank}] all_reduce ok sum={x.sum().item():.0f} (expected {8 * (world * (world + 1) // 2)}) dt={(time.time()-t0)*1000:.1f}ms", flush=True)

# Now try DeepEP Buffer
try:
    import deep_ep
    print(f"[rank{rank}] deep_ep version: {getattr(deep_ep, '__version__', 'n/a')}", flush=True)

    ep_group = dist.group.WORLD
    nvl_bytes = 64 * 1024 * 1024  # 64MB

    t0 = time.time()
    buf = deep_ep.Buffer(group=ep_group, num_nvl_bytes=nvl_bytes, low_latency_mode=True)
    torch.npu.synchronize()
    print(f"[rank{rank}] DeepEP Buffer (low_latency) constructed in {(time.time()-t0)*1000:.1f}ms", flush=True)
    print(f"[rank{rank}] buffer dir: {[a for a in dir(buf) if not a.startswith('_')][:15]}", flush=True)

    # Free + try normal mode
    del buf
    t0 = time.time()
    buf = deep_ep.Buffer(group=ep_group, num_nvl_bytes=nvl_bytes, low_latency_mode=False)
    torch.npu.synchronize()
    print(f"[rank{rank}] DeepEP Buffer (normal) constructed in {(time.time()-t0)*1000:.1f}ms", flush=True)

    print(f"[rank{rank}] D3 PASS", flush=True)
except Exception as e:
    print(f"[rank{rank}] D3 FAIL: {type(e).__name__}: {e}", flush=True)
    traceback.print_exc()
    sys.exit(2)
finally:
    if dist.is_initialized():
        dist.destroy_process_group()
