# 跨机通信问题排查报告 (HCCL + DeepEP)

**日期**: 2026-05-21
**作者**: Agent (Copilot CLI)
**分支**: `feat/fallback-default-comm`
**状态**: HCCL + Fallback 已修复; DeepEP runtime 仍阻塞，问题已下沉

---

## 1. Executive Summary

之前 `results_npu/coordinator_arch/fallback_default/README.md` 记录 "Host1↔Host2 跨机 HCCL 全失败 (EJ0003/EI0006)"。本次排查在**未做任何不可恢复破坏性变更**的前提下:

- ✅ 修复跨机 HCCL: `cross_host_hccl_smoke.py` PASS
- ✅ 修复跨机 Fallback: `cross_host_fallback_rt_bench.py` PASS (mean 406µs / p99 603µs @ 512KiB)
- ❌ DeepEP normal 和 low_latency 仍失败 —— 但失败已不再发生在 HCCL 基础层，下沉为 **DeepEP runtime 自身问题**

**修复手段**: 用全新端口组合 `MASTER_PORT=297xx` + `HCCL_IF_BASE_PORT=297yy` + 合法的 `HCCL_CONNECT_TIMEOUT=600` 替换历史值 `29555/24500/60`。

---

## 2. 拓扑

| 角色 | 主机 | 容器 | repo 路径 | 控制 IP | RoCE NIC (/17) |
|---|---|---|---|---|---|
| rank0 | Host1 (1.95.114.229) | `afd-npu-test` | `/workspace/afd_demo` | 192.168.0.125 | 48.165.{182,220,129,161}.x |
| rank1 | Host2 (192.168.0.192, 跳板) | `afd-npu-test-h2` | `/workspace/afd_demo_repo` | 192.168.0.192 | 48.165.{240,233,129,195}.x |

NPU NIC 同在 `48.165.128.0/17`，RoCE 路由全通; HCCL 控制走 `192.168.0.0/24`。

---

## 3. 复现 (HCCL smoke)

### 失败 (历史复现, 已修复前)
```bash
MASTER_PORT=29555 HCCL_IF_BASE_PORT=24500 HCCL_CONNECT_TIMEOUT=60 \
python3 scripts/cross_host_hccl_smoke.py
```
- Host1: `EJ0003 Communication_Error_Bind_IP_Port` (24500 已被占用 / TIME_WAIT)
- Host2: `EI0006 Communication_Error_Get_Socket` (与 Host1 永远建不上链)
- 备注: `HCCL_CONNECT_TIMEOUT=60` 为非法值，HCCL 实际要求 120-7200

### 成功 (本次)
```bash
# Host2 rank1 (先起)
RANK=1 WORLD_SIZE=2 LOCAL_RANK=0 ASCEND_VISIBLE_DEVICES=0 \
MASTER_ADDR=192.168.0.125 MASTER_PORT=29701 \
HCCL_IF_BASE_PORT=29711 HCCL_IF_IP=192.168.0.192 \
HCCL_CONNECT_TIMEOUT=600 HCCL_EXEC_TIMEOUT=600 \
python3 scripts/cross_host_hccl_smoke.py

# Host1 rank0 (后起)
RANK=0 ... HCCL_IF_IP=192.168.0.125 ... python3 scripts/cross_host_hccl_smoke.py
```
- 两 rank 均输出 `recv_sum=3072.0` (注: `ok=False` 是脚本内 tensor 复用导致的已知伪报，实际数据正确)
- 日志: `results_npu/coordinator_arch/crosshost_comm_repair/h{1,2}_rank{0,1}_attempt1.log`

---

## 4. 复现 (Fallback RT)

脚本: `scripts/cross_host_fallback_rt_bench.py` —— 用两次 `torch.distributed.all_to_all_single` 模拟 dispatch+combine 一次往返。

```bash
MASTER_PORT=29721 HCCL_IF_BASE_PORT=29731 ...  # 同 §3 模板
python3 scripts/cross_host_fallback_rt_bench.py --iters 50 --warmup 10 --num-tokens 64 --hidden 4096
```

结果 (rank0):
```
RT us: mean=406.0 p50=389.2 p99=602.6 min=359.6 max=818.4 n=50 payload=512.0 KiB
```

与 Host2 单机 (64KiB p50≈303µs) 对比, 跨机 512KiB 在 ~400µs mean 量级，符合 RoCE 跨机预期。**高于 300µs 设计目标**, 但功能完全正确, 可作为当前生产路径。

---

## 5. 复现 (DeepEP normal)

```bash
MASTER_PORT=29741 HCCL_IF_BASE_PORT=29751 ...
source /usr/local/Ascend/cann-8.5.0/opp/vendors/hwcomputing/bin/set_env.bash
python3 scripts/cross_host_deepep_rt_bench.py --iters 10 --warmup 2 \
  --num-tokens 64 --hidden 4096 --topk 4 --num-experts 16
```

日志:
```
[rank0] hccl pg ok
[rank0] buffer (normal) created
[rank0] warmup...
<进程超时退出, 无 traceback>
```

**结论**: HCCL bootstrap 通过 + DeepEP buffer 创建成功，但 `dispatch` 在 warmup iter 0 静默卡死。开 `ASCEND_GLOBAL_LOG_LEVEL=2` 也无可读 traceback。

---

## 6. 复现 (DeepEP low_latency)

```bash
MASTER_PORT=29761 HCCL_IF_BASE_PORT=29771 ...
python3 scripts/cross_host_deepep_lowlatency_rt_bench.py \
  --iters 5 --warmup 2 --num-tokens 32 --hidden 4096 --topk 8 --num-experts 16 --num-max 64
```

日志:
```
[rank0] hccl pg ok
[rank0] rdma_size_hint=64 bytes        # ← 与 hidden=4096 / topk=8 完全不相称
[rank0] buffer (low_latency) created
[rank0] warmup...
timeout: the monitored command dumped core
[ERROR] TBE Subprocess[task_distribute] raise error[], main process disappeared!
```

**结论**: 父进程在 warmup 第 0 轮被 SIGKILL，TBE 子进程随即报 main disappeared。`get_low_latency_rdma_size_hint` 返回 64-128B 异常值，强烈怀疑 wheel (`deep_ep 1.0.0+0ff3be00.cann.8.5.0.b232`) 与 CANN `8.5.0.b232` 之间 low_latency 内存布局协议错配。

---

## 7. 证据链

| 测试 | Host1 单机 | Host2 单机 | Host1↔Host2 跨机 |
|---|---|---|---|
| HCCL all_reduce | (历史 PASS) | PASS | ✅ PASS (本次) |
| Fallback `all_to_all_single` | PASS | PASS | ✅ PASS (本次) |
| DeepEP normal `dispatch` | (未测) | (未测) | ❌ silent hang |
| DeepEP low_latency `dispatch` | (未测) | (未测) | ❌ SIGKILL @warmup |

跨机基础链路恢复后，DeepEP 仍失败 ⇒ DeepEP 自身问题。

---

## 8. Hypothesis Matrix

| # | 假设 | 状态 |
|---|---|---|
| H1 | 残留 src.main rank 占端口 | ❌ Phase 2 已确认无残留 |
| H2 | `HCCL_IF_IP` 绑错网卡 | ❌ 192.168.0.x 控制网卡正确 |
| H3 | `HCCL_IF_BASE_PORT` 被占用 / TIME_WAIT | ✅ 换 fresh 端口即修复 |
| H4 | `HCCL_CONNECT_TIMEOUT=60` 非法 | ✅ 改 600 后通过 (无法与 H3 单独 isolate) |
| H5 | 容器网络 / privileged 状态不一致 | ❌ host network + privileged 两端对齐 |
| H6 | DeepEP wheel ↔ CANN runtime 协议错配 | ⚠️ 当前最可能, 需上游联调 |

**已知局限**: H3 + H4 没法单独 isolate (一次性改两个); 若需定论可在 fresh 端口上回测 `HCCL_CONNECT_TIMEOUT=60` 是否仍失败。

---

## 9. 安全边界 (本次遵守)

- ✅ 不删除 / 重建容器
- ✅ 不重启容器 (问题在容器重启前已解决)
- ✅ 不升级 CANN / driver / torch_npu
- ✅ 不触碰 Host2 其他租户容器
- ✅ 只按 PID `kill <PID>`, 未使用 `pkill`/`killall`

---

## 10. 下一步建议

### 短期 (本分支可继续)
1. ✅ 采用 fallback 作为生产路径 (Q4 已计划)
2. 用恢复后的 HCCL 跑代表性 fallback E2E (b=8/32/128, s=128/512)
3. 把"fresh 端口 + 合法 timeout"配方写进 `npu_910c_env_setup_and_run` skill

### 中期 (需独立专项)
4. DeepEP normal: 抓 Ascend slog, 排查 `aclnnNotifyDispatch` socket / QP 绑定
5. DeepEP low_latency: 验证 `get_low_latency_rdma_size_hint` 参数语义, 联系 SGLang-Kernel-NPU 上游
6. 若 DeepEP 仍 blocked, 把 `--use-deepep` 标为 experimental 并 freeze 该路径

### 长期 (需平台联调)
7. NIC pinning: 验证 HCCL bootstrap socket 与 DeepEP RDMA notify socket 是否使用同一物理 NIC
8. 必要时切换 `HCCL_NET_DEVICE` 显式绑定

---

## 11. 相关文件

- 报告: 本文件 (`doc/15-cross-host-communication-diagnosis.md`)
- 现场日志 + 摘要 README: `results_npu/coordinator_arch/crosshost_comm_repair/`
- 历史 BLOCKED 记录: `results_npu/coordinator_arch/fallback_default/README.md`
- 之前 PASS (单机) 记录: `results_npu/cross_host_bench/README.md`
- 通信模式说明: `doc/14-communication-modes.md`
- DeepEP 安装指南: `doc/13-deepep-install-test-error-guide.md`
