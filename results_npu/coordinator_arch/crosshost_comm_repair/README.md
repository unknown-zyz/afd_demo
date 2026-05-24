# Cross-Host Communication Repair Log

日期: 2026-05-21
分支: `feat/fallback-default-comm`
范围: Host1 `afd-npu-test` (192.168.0.125) ↔ Host2 `afd-npu-test-h2` (192.168.0.192)

## 结论 (TL;DR)

| 层 | 状态 | 证据 |
|---|---|---|
| 基础 HCCL all_reduce / pingpong | ✅ **PASS** | `h{1,2}_rank{0,1}_attempt1.log` 全部 `recv_sum=3072.0` |
| Fallback `all_to_all_single` RT | ✅ **PASS** | `h1_fb_rank0.log` mean=406µs p50=389µs p99=603µs (512 KiB payload) |
| DeepEP normal `dispatch` | ❌ 仍阻塞 | `h{1,2}_deepep_normal_rank*.log` warmup iter0 静默退出 |
| DeepEP low_latency `dispatch` | ❌ 仍阻塞 | `h{1,2}_deepep_ll_rank*.log` warmup SIGKILL + `rdma_size_hint=64B` 异常 |

**核心结论**: 之前记录的 "Host1↔Host2 HCCL 全失败" 已修复。当前 DeepEP 失败不再是基础 HCCL/RoCE 问题，已下沉为 DeepEP runtime 自身问题（normal: `aclnnNotifyDispatch` warmup 静默卡死；low_latency: TBE 子进程报 main disappeared）。

## 修复方法 (可复用配方)

之前失败时用的端口组合 `MASTER_PORT=29555 / HCCL_IF_BASE_PORT=24500` 和 `HCCL_CONNECT_TIMEOUT=60`（非法值，HCCL 要求 120-7200）。**改用全新端口 + 合法超时即修复**:

```bash
export MASTER_ADDR=192.168.0.125
export MASTER_PORT=29701          # 全新, 避开历史 TIME_WAIT
export HCCL_IF_BASE_PORT=29711    # 全新
export HCCL_CONNECT_TIMEOUT=600   # 合法值
export HCCL_EXEC_TIMEOUT=600
export HCCL_IF_IP=<本机 192.168.0.x>
export ASCEND_VISIBLE_DEVICES=0
export LOCAL_RANK=0
```

后续每次跑新实验都换 fresh 端口（+20）即可:
- HCCL smoke: 29701/29711
- Fallback RT: 29721/29731
- DeepEP normal: 29741/29751
- DeepEP low_latency: 29761/29771

## 复现命令

### Host2 rank1 (先起, detached)
```bash
sudo ssh -i /root/ssh_key/KeyPair-f1dd.pem root@192.168.0.192 \
  docker exec -d afd-npu-test-h2 bash -lc 'cd /workspace/afd_demo_repo &&
    source /usr/local/Ascend/ascend-toolkit/set_env.sh &&
    export RANK=1 WORLD_SIZE=2 LOCAL_RANK=0 \
      MASTER_ADDR=192.168.0.125 MASTER_PORT=29721 \
      HCCL_IF_BASE_PORT=29731 HCCL_IF_IP=192.168.0.192 \
      HCCL_CONNECT_TIMEOUT=600 HCCL_EXEC_TIMEOUT=600 ASCEND_VISIBLE_DEVICES=0 &&
    python3 scripts/cross_host_fallback_rt_bench.py --iters 50 --warmup 10 \
      --num-tokens 64 --hidden 4096 > results/h2_fb.log 2>&1'
```

### Host1 rank0 (随后起)
```bash
docker exec afd-npu-test bash -lc 'cd /workspace/afd_demo && ... 同上 RANK=0 HCCL_IF_IP=192.168.0.125 ...'
```

## 排除项 (已确认不是问题)

- ❌ RDMA 硬件: 4 NPU NIC up、`net_health=Success`、`hccn_tool -ping` 67-136µs
- ❌ 残留 src.main 进程: Phase 2 检查无残留
- ❌ DeepEP OPP vendor: `libcust_opapi.so` 含 `aclnnDispatchLayout` 符号
- ❌ `FallbackMoECommunicator` 代码: 之前怀疑通信层 bug，跨机 RT bench 已 PASS

## 剩余 DeepEP runtime 问题 (需平台/上游联调)

1. **normal 模式**: dispatch warmup 第 0 轮静默卡死。即使开 `ASCEND_GLOBAL_LOG_LEVEL=2` 也无 traceback。怀疑 `aclnnNotifyDispatch` 的 RDMA notify socket 与 HCCL bootstrap socket 走的网卡不同，需 `HCCL_NET_DEVICE`/NIC pinning 调查。
2. **low_latency 模式**: `get_low_latency_rdma_size_hint(...)` 持续返回 64-128B (与 hidden=4096/topk=8 不相称)，warmup 触发 TBE 子进程「main process disappeared」, 父进程被 SIGKILL。强烈怀疑 wheel 与 CANN 8.5.0.b232 之间 low_latency 内存布局协议错配。

## 文件

```
crosshost_comm_repair/
├── README.md                          (this file)
├── h1_rank0_attempt1.log              HCCL smoke PASS
├── h2_rank1_attempt1.log              HCCL smoke PASS
├── h1_fb_rank0.log                    Fallback RT PASS
├── h2_fb_rank1.log                    Fallback RT PASS
├── h1_deepep_normal_rank0.log         DeepEP normal FAIL (silent)
├── h2_deepep_normal_rank1.log         DeepEP normal FAIL (silent)
├── h1_deepep_ll_rank0.log             DeepEP low_latency FAIL (SIGKILL)
└── h2_deepep_ll_rank1.log             DeepEP low_latency FAIL (SIGKILL)
```

注: `*.log` 被 `.gitignore` 排除，本地保留作存证；CI/远程审阅请参考本 README 摘要。
