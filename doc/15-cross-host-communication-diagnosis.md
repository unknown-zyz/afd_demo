# 跨机通信问题排查报告 (HCCL + DeepEP)

**日期**: 2026-05-21
**作者**: Agent (Copilot CLI)
**分支**: `feat/fallback-default-comm`
**状态**: HCCL + Fallback 已修复; DeepEP 运行时仍阻塞; P3 真实 decode 当前阻塞在 TBE JIT 冷编译

---

## 1. 摘要

之前 `results_npu/coordinator_arch/fallback_default/README.md` 记录 "Host1↔Host2 跨机 HCCL 全失败 (EJ0003/EI0006)"。本次排查在**未做任何不可恢复破坏性变更**的前提下:

- ✅ 修复跨机 HCCL: `cross_host_hccl_smoke.py` PASS
- ✅ 修复跨机 Fallback: `cross_host_fallback_rt_bench.py` PASS (mean 406µs / p99 603µs @ 512KiB)
- ❌ DeepEP normal 和 low_latency 仍失败 —— 但失败已不再发生在 HCCL 基础层，下沉为 **DeepEP 运行时自身问题**
- ⚠️ 真实 Qwen3-30B-A3B 1A7F 跨机 decode 已通过 HCCL bootstrap，但首次 prefill warmup 卡在 **TBE JIT 冷编译 > 60min**，暂未产出端到端 timing 数据

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

## 8. 假设矩阵

| # | 假设 | 状态 |
|---|---|---|
| H1 | 残留 src.main rank 占端口 | ❌ Phase 2 已确认无残留 |
| H2 | `HCCL_IF_IP` 绑错网卡 | ❌ 192.168.0.x 控制网卡正确 |
| H3 | `HCCL_IF_BASE_PORT` 被占用 / TIME_WAIT | ✅ 换 fresh 端口即修复 |
| H4 | `HCCL_CONNECT_TIMEOUT=60` 非法 | ✅ 改 600 后通过 (无法与 H3 单独 isolate) |
| H5 | 容器网络 / privileged 状态不一致 | ❌ host network + privileged 两端对齐 |
| H6 | DeepEP wheel ↔ CANN 运行时协议错配 | ⚠️ 当前最可能, 需上游联调 |

**已知局限**: H3 + H4 没法单独 isolate (一次性改两个); 若需定论可在 fresh 端口上回测 `HCCL_CONNECT_TIMEOUT=60` 是否仍失败。

---

## 9. 当前多机真实 decode 阻塞: TBE JIT / `kernel_meta`

本节说明 P3 真实 Qwen3-30B-A3B 跨机 1A7F 实验为什么仍然跑不完。这里的阻塞点**不是已经修复的 HCCL 端口/建链问题**，而是 Ascend 后端首次运行时的 TBE JIT 冷编译。

### 9.1 TBE JIT 编译是什么

TBE (Tensor Boost Engine) 是 Ascend/CANN 用来为 NPU 生成算子 kernel 的编译体系。PyTorch/torch_npu 在执行某些算子时，并不总是提前拥有所有 shape、dtype、layout、融合策略和后端配置对应的二进制 kernel；第一次遇到具体组合时，CANN 会在运行期触发编译，这就是这里说的 **TBE JIT**。

在本项目里，首次 prefill 会触发大量与以下路径相关的 kernel 编译:

- Qwen3-30B-A3B 的 48 层 attention / MLP / MoE 路径
- bf16 dtype 下的 NPU 算子
- MoE router、expert FFN、EP broadcast/reduce 等形状相关路径
- `batch_size`、`seq_len`、micro-batch 切分共同决定的具体 shape

因此，第一次跑某个新配置时，日志里看到 `Running 1 prefill warmup round(s) to absorb JIT compile cost` 之后长时间没有继续输出，并不一定表示 HCCL 已死锁；也可能是 CANN/TBE 正在编译或等待编译任务完成。

### 9.2 为什么当前实验这么慢

本次 P3 v6 的最新状态是:

- H1 rank0 + H2 ranks 1-7 都已经完成 `init_process_group`、FFN EP group 创建、权重加载和 `AFDCommunicator` 初始化
- 8 个业务 rank 全部进入第一轮 prefill warmup，并打印 `Split batch into 2 micro-batches: sizes=[1, 1]`
- commit `6c42639` 已把 torch distributed 默认 30min collective timeout 提升到 2h，二次验证中所有 rank 已经越过原来的 30min 死亡点
- 之后进程仍在第一轮 warmup 内等待，直到脚本级 `timeout 3600` 在 60min wall clock 后杀掉进程
- 现场 `kernel_meta/` 只有约数百 KB，说明冷编译没有完整跑完，缓存产物很少

慢的主要原因有三点:

1. **模型和 shape 复杂**: Qwen3-30B-A3B 是 48 层 MoE 模型，prefill 会一次性触达大量 attention、router、expert FFN 和通信相关 kernel。
2. **首次冷编译不能复用缓存**: 当前 `kernel_meta/` 基本为空，每个新 shape/dtype/backend 组合都要从头编。
3. **跨机 rank 互相等待**: 1A7F 拓扑下 H1 attention rank 与 H2 7 个 FFN rank 的进度不完全一致。某些 rank 可能已经进入 collective 等待，而另一些 rank 仍在 TBE 编译，最终整体被最慢 rank 卡住。

所以当前结论是: **HCCL bootstrap 已经通过，Fallback 通信路径也可用；P3 真实 decode 的新阻塞点是 Qwen3-30B-A3B 首次 TBE JIT 冷编译耗时超过当前脚本 60min 限制。**

### 9.3 “每台机器单机跑一次 prefill warmup 灌满 `kernel_meta/`”是什么意思

`kernel_meta/` 是 CANN/TBE 保存编译产物和元信息的缓存目录。单机 prefill warmup 的目的，是先在每台机器本地把常用 kernel 编出来，避免第一次跨机实验时所有 rank 一边编译一边等待跨机 collective。

建议流程和可直接执行的命令如下（都在各自容器内执行）。不要直接手敲
`run_npu.sh` 长命令观察前台输出：`run_npu.sh` 会把每个 rank 的日志重定向到
`results/logs/npu_*.log`，Host2 EP7 冷编译时前台长时间没有新输出是预期现象。
统一使用 `scripts/run_tbe_cache_warmup_npu.sh`，它会自动轮询 rank 数、`kernel_meta/`
大小和 rank 日志尾部；如果达到 `--timeout-sec`，脚本会返回 `124` 并按本次实验
suffix 清理残留 `python -m src.main` rank，避免后续 HCCL 端口或进程状态被污染。

1. 在 Host1 本地跑代表性 prefill 配置，让 attention 相关 kernel 先编译并落到 `/workspace/afd_demo/kernel_meta/`
   ```bash
   cd /workspace/afd_demo
   source venv/bin/activate
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   bash scripts/run_tbe_cache_warmup_npu.sh \
     --profile host1-attn \
     --batch 2 --seq 128 --tokens 5 \
     --model-name /models/Qwen3-30B-A3B \
     --timeout-sec 7200 --poll-sec 60
   ```
   若最终看到 `Exit=0` / `Warmup completed`，这是成功，不是报错。
2. 在 Host2 本地跑 EP7 / FFN 侧代表性 prefill 配置，让 FFN/MoE/EP 相关 kernel 先编译并落到 `/workspace/afd_demo_repo/kernel_meta/`
   ```bash
   cd /workspace/afd_demo_repo
   source venv/bin/activate
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   bash scripts/run_tbe_cache_warmup_npu.sh \
     --profile host2-ep7 \
     --batch 2 --seq 128 --tokens 5 \
     --model-name /models/Qwen3-30B-A3B \
     --timeout-sec 7200 --poll-sec 60
   ```
   Host2 会启动 8 个本地 ranks，前台由 wrapper 每 60 秒打印一次
   `active_src_main`、`kernel_meta` 和 rank log tail；如果日志停在
   `Running 1 prefill warmup round(s)`，通常表示仍在 TBE 冷编译。
   若 7200 秒仍未完成并返回 `exit=124`，说明本轮仍未灌满 cache；确认
   `active_src_main` 已清零后，可增大 `--timeout-sec` 或缩小 shape 继续预热。
3. 确认两边 `kernel_meta/` 明显增大后，再启动跨机 1A7F 真实 decode

这样做的好处是把“编译耗时”和“跨机通信验证”拆开: 单机 warmup 时没有 Host1/Host2 互相等待，失败也更容易定位；跨机正式运行时则可以复用已编译 kernel，减少 layer 0 / 首轮 prefill 的冷启动开销。

需要注意: warmup 的 shape 要尽量覆盖后续跨机实验。至少应覆盖计划中的 `batch_size`、`prefill_seq_len`、dtype 和 micro-batch 组织；如果后续换了很不一样的 shape，仍可能触发新的 TBE 编译。

### 9.4 cache 能不能存储下来，避免每次重编

可以。`kernel_meta/` 应该作为可复用缓存保留下来，避免每次清理工作目录或重建容器后重新冷编译。建议把它放在持久化目录或宿主机挂载卷中，并在实验记录里保存对应的软件栈信息。

但 `kernel_meta/` 不是跨所有场景通用的“永久二进制包”。它的可复用性通常绑定以下条件:

- CANN / Ascend driver / firmware 版本
- torch_npu / PyTorch 版本
- 模型代码和算子实现
- dtype、shape、layout、micro-batch 切分
- NPU 架构和编译环境变量
- 工作目录路径和容器环境中 CANN 对 cache 的解析方式

因此推荐策略是:

1. **每台机器各自保留 cache**: Host1 和 Host2 分别保留自己的 `kernel_meta/`，不要默认互相覆盖。
2. **记录 cache 元数据**: 至少记录 CANN、torch_npu、模型 commit、主要 shape、生成时间。
3. **谨慎跨机复制**: 只有在两台机器硬件、CANN、torch_npu、模型代码、路径和环境变量完全一致时，才考虑复制；否则优先每台机器本地 warmup。
4. **明确失效条件**: 升级 CANN/torch_npu、换模型、改算子实现、换 dtype/shape、切换 NPU 架构后，应删除或隔离旧 cache，重新 warmup。

对当前 P3 来说，下一步不是继续盲目拉长跨机脚本 timeout，而是先做每台机器的单机 prefill warmup，把 `kernel_meta/` 缓存真正生成出来，再用 warm cache 复跑跨机 1A7F。

### 9.5 当前 P3 真实 decode 阻塞的复现命令

下面命令复现的是当前最新状态：**HCCL bootstrap 已通过，但首次 prefill warmup 卡在 TBE JIT 冷编译**。两边都在容器内执行，先起 Host2 的 7 个 FFN ranks，再起 Host1 的 attention rank。

Host2：

```bash
cd /workspace/afd_demo_repo
source venv/bin/activate
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir -p results_npu/coordinator_arch/p3_real_xhost

export MASTER_ADDR=192.168.0.125
export MASTER_PORT=29795
export HCCL_IF_BASE_PORT=30400
export HCCL_IF_IP=192.168.0.192
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_EXEC_TIMEOUT=3600
export HCCL_BUFFSIZE=200
export AFD_DIST_TIMEOUT_SEC=7200

for R in 1 2 3 4 5 6 7; do
  LRANK=$((R - 1))
  ASCEND_VISIBLE_DEVICES=$LRANK ASCEND_RT_VISIBLE_DEVICES=$LRANK \
  nohup timeout 3600 python3 -u -m src.main \
    --backend npu --role ffn --world-size 8 --rank $R --local-rank 0 \
    --attn-node-rank 0 --ffn-node-rank 1 \
    --attn-size 1 --ffn-size 7 --ffn-tp-size 1 --ffn-ep-size 7 \
    --ffn-ep-backend broadcast_reduce_overlap --ffn-coordinator-rank 1 \
    --ep-expert-policy round_robin \
    --batch-size 2 --prefill-seq-len 128 --max-new-tokens 5 \
    --num-micro-batches 2 --timing \
    --timing-suffix xhost_real_b2_s128_t5_v6 \
    --master-addr $MASTER_ADDR --master-port $MASTER_PORT \
    --model-name /models/Qwen3-30B-A3B \
    --no-dbo --no-generate \
    > results_npu/coordinator_arch/p3_real_xhost/h2_rank${R}_xhost_real_b2_s128_t5_v6.log 2>&1 &
done
```

Host1：

```bash
cd /workspace/afd_demo
source venv/bin/activate
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir -p results_npu/coordinator_arch/p3_real_xhost

export MASTER_ADDR=192.168.0.125
export MASTER_PORT=29795
export HCCL_IF_BASE_PORT=30300
export HCCL_IF_IP=192.168.0.125
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_EXEC_TIMEOUT=3600
export HCCL_BUFFSIZE=200
export AFD_DIST_TIMEOUT_SEC=7200
export ASCEND_VISIBLE_DEVICES=0
export ASCEND_RT_VISIBLE_DEVICES=0

nohup timeout 3600 python3 -u -m src.main \
  --backend npu --role attention --world-size 8 --rank 0 --local-rank 0 \
  --attn-node-rank 0 --ffn-node-rank 1 \
  --attn-size 1 --ffn-size 7 --ffn-tp-size 1 --ffn-ep-size 7 \
  --ffn-ep-backend broadcast_reduce_overlap --ffn-coordinator-rank 1 \
  --ep-expert-policy round_robin \
  --batch-size 2 --prefill-seq-len 128 --max-new-tokens 5 \
  --num-micro-batches 2 --timing \
  --timing-suffix xhost_real_b2_s128_t5_v6 \
  --master-addr $MASTER_ADDR --master-port $MASTER_PORT \
  --model-name /models/Qwen3-30B-A3B \
  --no-dbo --no-generate \
  > results_npu/coordinator_arch/p3_real_xhost/h1_rank0_xhost_real_b2_s128_t5_v6.log 2>&1 &
```

复现时的预期现象：

1. 两边都能越过 `Distributed initialized`
2. 日志进入 `Running 1 prefill warmup round(s) to absorb JIT compile cost`
3. 冷编译场景下可能 60 分钟内仍停在第一轮 prefill warmup，随后被 `timeout 3600` 杀掉

如果已经先执行了 §9.3 的单机 warmup，再复跑本节命令，理论上应更有机会跳过冷编译阻塞，尽快进入真实 prefill / decode。

---

## 10. 安全边界

本节记录的是 §3-§8 中 HCCL/DeepEP 基础链路排查阶段遵守的边界。后续 P3 真实 decode 调试中，为清理 HCCL device 侧残留端口状态，曾在用户授权范围内重启过 `afd-npu-test` / `afd-npu-test-h2` 容器；该操作不涉及删除容器、重建环境或升级系统软件。

- ✅ 不删除 / 重建容器
- ✅ 基础链路排查阶段未重启容器 (HCCL/Fallback 问题在容器重启前已解决)
- ✅ 不升级 CANN / driver / torch_npu
- ✅ 不触碰 Host2 其他租户容器
- ✅ 只按 PID `kill <PID>`, 未使用 `pkill`/`killall`

---

## 11. 下一步建议

### 短期 (本分支可继续)
1. ✅ 采用 fallback 作为生产路径 (Q4 已计划)
2. P3 真实 decode 先按 §9 做 Host1/Host2 单机 prefill warmup，生成 `kernel_meta/` 后再复跑跨机 1A7F
3. 用恢复后的 HCCL 跑代表性 fallback E2E (b=8/32/128, s=128/512)
4. 把"fresh 端口 + 合法 timeout + TBE warm cache"配方写进 `npu_910c_env_setup_and_run` skill

### 中期 (需独立专项)
5. DeepEP normal: 抓 Ascend slog, 排查 `aclnnNotifyDispatch` socket / QP 绑定
6. DeepEP low_latency: 验证 `get_low_latency_rdma_size_hint` 参数语义, 联系 SGLang-Kernel-NPU 上游
7. 若 DeepEP 仍 blocked, 把 `--use-deepep` 标为 experimental 并 freeze 该路径

### 长期 (需平台联调)
8. NIC pinning: 验证 HCCL bootstrap socket 与 DeepEP RDMA notify socket 是否使用同一物理 NIC
9. 必要时切换 `HCCL_NET_DEVICE` 显式绑定

---

## 12. 相关文件

- 报告: 本文件 (`doc/15-cross-host-communication-diagnosis.md`)
- 现场日志 + 摘要 README: `results_npu/coordinator_arch/crosshost_comm_repair/`
- 历史 BLOCKED 记录: `results_npu/coordinator_arch/fallback_default/README.md`
- 之前 PASS (单机) 记录: `results_npu/cross_host_bench/README.md`
- 通信模式说明: `doc/14-communication-modes.md`
- DeepEP 安装指南: `doc/13-deepep-install-test-error-guide.md`
