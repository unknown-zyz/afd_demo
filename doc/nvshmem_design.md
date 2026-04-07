# NVSHMEM P2P 通信方案设计

## 背景

当前 NCCL `isend`/`irecv` 在每层首次发送（MB0）有 20-25ms 冷启动延迟，
根因是 NCCL proxy thread 休眠 + P2P 通道懒加载。

## NVSHMEM vs NCCL 架构对比

| 特性 | NCCL P2P | NVSHMEM |
|------|----------|---------|
| 通信模型 | 双边 (isend/irecv 配对) | 单边 (put/get, 无需接收端参与) |
| 数据搬运 | CPU proxy thread 协调 | GPU 直接发起 DMA (NVLink/PCIe) |
| 通道建立 | 懒加载, 每轮首次有冷启动 | 对称内存预分配, 无冷启动 |
| 同步机制 | handle.wait() + proxy 唤醒 | signal_wait_until (GPU 端自旋) |
| 内存管理 | 用户分配 + NCCL 内部 buffer | 对称堆 (所有 PE 相同虚拟地址) |

## 核心原理

```
Attention PE (rank 0)                     FFN PE (rank 1)
    │                                         │
    ├─ nvshmem_putmem_nbi ──────────────────► │ (数据直接写入远端对称内存)
    ├─ nvshmem_fence                          │
    ├─ nvshmem_signal(remote_flag) ─────────► │ (设置远端信号标志)
    │                                         ├─ nvshmem_signal_wait_until (GPU 自旋等待)
    │                                         ├─ 数据已在本地对称内存, 零拷贝读取
```

关键优势:
1. **无 proxy thread**: GPU kernel 直接调用 NVLink/PCIe DMA, 无 CPU 中介
2. **无冷启动**: 对称内存在 init 时分配, put 操作无需建立通道
3. **单边操作**: 接收端无需 post irecv, 数据直接写入远端内存
4. **GPU 端同步**: signal_wait_until 在 GPU 上自旋, 无 CPU 调度延迟

## 预期性能

- 1MB put (NVLink V2): ~0.01-0.02ms (硬件极限)
- 信号延迟: ~1-5μs
- 总延迟: ~0.02-0.05ms
- vs NCCL MB0: 20-25ms → **提升 400-1000×**

## 实现方案

### 初始化
```python
nvshmem_init()           # 通过 MPI/PMI bootstrap
sym_buf = nvshmem_malloc(max_size)  # 对称内存分配
signal_flags = nvshmem_malloc(num_slots * 8)  # 信号标志
```

### 发送 (isend 替代)
```python
def isend(tensor, dst_pe, tag):
    slot = tag_to_slot(tag)
    # 1. 拷贝到本地对称 buffer (如果 tensor 不在对称堆)
    sym_buf[slot].copy_(tensor)
    # 2. 远端写入 (GPU 直接 DMA, 非阻塞)
    nvshmem_putmem_nbi(remote_sym_buf + slot_offset, local_ptr, size, dst_pe)
    # 3. 保序
    nvshmem_fence()
    # 4. 通知远端 (写信号标志)
    nvshmem_uint64_p(remote_signal + slot, seq_num, dst_pe)
    return handle  # wait() 调用 nvshmem_quiet()
```

### 接收 (irecv 替代)
```python
def irecv(tensor, src_pe, tag):
    slot = tag_to_slot(tag)
    # wait() 时执行:
    # 1. GPU 自旋等待信号
    nvshmem_uint64_wait_until(local_signal + slot, NVSHMEM_CMP_GE, expected_seq)
    # 2. 数据已在本地对称内存, 直接读取 (零拷贝)
    tensor.copy_(sym_buf[slot])
    # 3. 重置信号
    return handle
```

### Slot 管理
- 预分配 N 个 slot (默认 8), 每个 max_tensor_bytes
- tag → slot 映射: `slot_idx = tag % num_slots`
- 序列号防止 ABA 问题

## 安装要求

NVSHMEM 需要:
- CUDA Toolkit 11.0+
- NCCL 2.x (作为 transport 后端之一)
- 安装方式:
  ```bash
  # 方式1: NVIDIA HPC SDK
  apt install nvhpc-*
  
  # 方式2: 独立安装
  # 从 https://developer.nvidia.com/nvshmem 下载
  tar xf nvshmem_*.txz
  export NVSHMEM_HOME=/path/to/nvshmem
  
  # 方式3: pip (如果可用)
  pip install nvidia-nvshmem
  ```

## 与 PyTorch 集成

NVSHMEM 通过 ctypes 调用 C API:
```python
libnvshmem = ctypes.CDLL("libnvshmem.so")
```

对称内存指针可包装为 PyTorch tensor:
```python
ptr = nvshmem_malloc(size)
# 通过 torch.from_blob 或 ctypes+numpy 创建视图
```

## 局限性

1. 需要安装 NVSHMEM (非标准 PyTorch 依赖)
2. 对称内存必须通过 nvshmem_malloc 分配, 不能用普通 CUDA 内存
3. 需要 MPI 或 PMI 启动器 (mpirun / srun)
4. V100 NVLink V2 不支持 nvshmem_signal 原子操作的最优路径 (需 Volta+)
