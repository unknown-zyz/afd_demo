# DeepEP 安装、测试与报错排查指南

本文记录本仓库在双机 910C 上接入 DeepEP-Ascend 的**真实过程**：怎么安装、怎么测、遇到了什么问题、哪些问题已经排除、为什么当前生产路径仍建议先走 Fallback。

---

## 1. 结论先看

当前结论分成两层：

1. **安装层已经解锁**
   - `deep_ep` wheel 本身可以安装、导入。
   - 之前 `aclnnDispatchLayout` 缺失，不是 CANN 版本不匹配，而是 **DeepEP 自定义 OPP vendor 没安装**。
   - 现在 Host1 / Host2 都已经能解析这些自定义 op。

2. **运行层仍然阻塞**
   - **normal 模式**：跨机 warmup 首轮卡在 `aclnnNotifyDispatch`，报 `EI0006 socket timeout`，随后 AICore `507014`。
   - **low_latency 模式**：buffer 创建成功，但 warmup 首轮父进程被 SIGKILL；同时 `rdma_size_hint=128 bytes` 明显异常。
   - 因此当前这套环境下，**DeepEP 还不能作为稳定的双机生产数据面**。

3. **Fallback 是当前可工作的替代路径**
   - Fallback 用的是 **`torch.distributed.all_to_all_single`**。
   - 后端取决于传入的 process group；在我们的双机 NPU 实验里，这个组用 `dist.init_process_group(backend="hccl")` 建立，所以**实际走的是 HCCL**。
   - 只有 CPU / 本地无 NPU 测试时，才可能退到 `gloo`。

---

## 2. 相关文件

| 类型 | 文件 |
|---|---|
| 安装报告 | `doc/13-deepep-install-test-error-guide.md`（本文） |
| normal 模式 RT bench | `scripts/cross_host_deepep_rt_bench.py` |
| low_latency 模式 RT bench | `scripts/cross_host_deepep_lowlatency_rt_bench.py` |
| fallback RT bench | `scripts/cross_host_fallback_rt_bench.py` |
| Fallback communicator 实现 | `src/coordinator_arch/comm/fallback_a2a.py` |
| 通信方式切换说明 | `doc/14-communication-modes.md` |
| 本轮结果汇总 | `results_npu/coordinator_arch/d3_lowlatency/README.md` |
| NPU 环境 skill | `.github/skills/npu_910c_env_setup_and_run/SKILL.md` |

---

## 3. 环境与前置条件

本轮验证环境：

| 项 | 值 |
|---|---|
| 硬件 | 双机 Ascend 910C |
| 容器 | `afd-npu-test` / `afd-npu-test-h2` |
| CANN | `8.5.0.b232` |
| torch_npu | `2.6.0` |
| PyTorch 分布式 backend | `hccl` |
| DeepEP wheel | `deep_ep-1.0.0+0ff3be00.cann.8.5.0.b232` |

注意：本文讨论的是**跨机 Attention ↔ FFN 数据面**，不是单机 import smoke。

---

## 4. 安装步骤

### 4.1 安装 DeepEP wheel

如果容器里还没装 wheel，先参考旧安装报告：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /workspace/sglang-kernel-npu
bash build.sh -a deepep
pip install output/deep_ep*.whl
cd "$(pip show deep-ep | grep -E '^Location:' | awk '{print $2}')"
ln -sf deep_ep/deep_ep_cpp*.so .
```

安装后至少要能通过：

```bash
python3 -c "import deep_ep; print(deep_ep.__path__)"
```

### 4.2 安装 DeepEP 自定义 OPP vendor（**关键**）

这是之前最容易漏掉的一步。

DeepEP-Ascend 除了 wheel 之外，还额外提供了一组自定义 MoE op，包括：

- `aclnnDispatchLayout`
- `aclnnDispatchNormalA2`
- `aclnnDispatchFFNCombine`
- `aclnnMoeDistributeCombineA2`

这些 op 不会因为 `pip install deep_ep` 自动进入 CANN 的 vendor 目录；必须手工安装。

#### Host1：有源码时

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash /workspace/sglang-kernel-npu/csrc/deepep/ops/build_out/custom_opp_ubuntu_aarch64.run \
     --quiet --install-path=/usr/local/Ascend/cann-8.5.0/opp
```

#### Host2：没有源码时

```bash
DEEPEP_DIR=/usr/local/python3.11.14/lib/python3.11/site-packages/deep_ep/vendors/hwcomputing
VENDORS=/usr/local/Ascend/cann-8.5.0/opp/vendors
mkdir -p "$VENDORS"
cp -a "$DEEPEP_DIR" "$VENDORS/"
cat > "$VENDORS/hwcomputing/bin/set_env.bash" <<'EOF'
#!/bin/bash
export ASCEND_CUSTOM_OPP_PATH=/usr/local/Ascend/cann-8.5.0/opp/vendors/hwcomputing:${ASCEND_CUSTOM_OPP_PATH}
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-8.5.0/opp/vendors/hwcomputing/op_api/lib/:${LD_LIBRARY_PATH}
EOF
```

### 4.3 每个 shell 都要激活 vendor 环境

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/cann-8.5.0/opp/vendors/hwcomputing/bin/set_env.bash
```

### 4.4 验证自定义 op 已可见

```bash
nm -D /usr/local/Ascend/cann-8.5.0/opp/vendors/hwcomputing/op_api/lib/libcust_opapi.so \
  | grep aclnnDispatchLayout
```

期望输出包含：

```text
T aclnnDispatchLayout
T aclnnDispatchLayoutGetWorkspaceSize
```

---

## 5. 脚本功能与使用方法

### 5.1 三个 bench 脚本分别做什么

| 脚本 | 作用 | 通信方法 | 适用场景 |
|---|---|---|---|
| `scripts/cross_host_deepep_rt_bench.py` | 测 DeepEP **normal mode** 的 dispatch+combine round-trip | `deep_ep.Buffer(..., low_latency_mode=False)` | prefill / 大 batch 风格的 DeepEP 路径验证 |
| `scripts/cross_host_deepep_lowlatency_rt_bench.py` | 测 DeepEP **low_latency mode** 的 dispatch+combine round-trip | `low_latency_dispatch + low_latency_combine` | decode / 小 batch / 低延迟场景验证 |
| `scripts/cross_host_fallback_rt_bench.py` | 用 PyTorch distributed 模拟 dispatch+combine | `torch.distributed.all_to_all_single` | DeepEP 跑不通时的 fallback 路径验证 |

这 3 个脚本都不是端到端 Qwen3 推理，而是**最小通信 microbenchmark**：

- 输入是一批随机 hidden states
- 中间不做真实 FFN，只做 echo
- 目标是只看 Attention ↔ FFN 数据面的 round-trip 能否跑通，以及时延大概是多少

### 5.2 每个脚本的关键参数

#### `cross_host_deepep_rt_bench.py`

| 参数 | 含义 |
|---|---|
| `--num-tokens` | 每个 rank 本轮参与 dispatch/combine 的 token 数 |
| `--hidden` | hidden size，Qwen3-30B 当前测的是 `4096` |
| `--topk` | 每个 token 路由到多少个 expert |
| `--num-experts` | 全局 expert 总数 |
| `--iters` | 正式计时轮数 |
| `--warmup` | 预热轮数 |
| `--nvl-bytes` | normal-mode buffer 预分配的 NVL/HCCS 缓冲区大小 |

#### `cross_host_deepep_lowlatency_rt_bench.py`

| 参数 | 含义 |
|---|---|
| `--num-tokens` | 当前输入 token 数 |
| `--hidden` | hidden size |
| `--topk` | 每个 token 的 expert 选择数 |
| `--num-experts` | 全局 expert 总数 |
| `--num-max` | `num_max_dispatch_tokens_per_rank`，用于 low-latency RDMA buffer 规划 |
| `--iters` | 正式计时轮数 |
| `--warmup` | 预热轮数 |
| `--use-fp8` | 是否让 low_latency_dispatch 走 fp8 路径；当前测试未开启 |

#### `cross_host_fallback_rt_bench.py`

| 参数 | 含义 |
|---|---|
| `--num-tokens` | 每 rank token 数 |
| `--hidden` | hidden size |
| `--iters` | 正式计时轮数 |
| `--warmup` | 预热轮数 |

额外约束：

- fallback 脚本要求 `num_tokens % world_size == 0`
- 因为它直接把 token **均分**到每个 rank，再用两次 `all_to_all_single` 做“发出去 / 取回来”的 round-trip

### 5.3 运行前必须设置的环境变量

三个脚本都依赖 `env://` 初始化分布式，所以不能只写 `python3 xxx.py`，还必须配套环境变量。

| 环境变量 | 含义 |
|---|---|
| `RANK` | 当前进程的全局 rank |
| `WORLD_SIZE` | 总进程数；当前跨机 bench 是 `2` |
| `LOCAL_RANK` | 当前机器上的本地 device 编号 |
| `MASTER_ADDR` | rank0 所在机器 IP；通常设 Host1 的互通 IP |
| `MASTER_PORT` | torch distributed 主端口；每次 bench 选一个未占用值 |
| `HCCL_IF_BASE_PORT` | HCCL / DeepEP 使用的基础端口；不同轮 bench 要避免复用冲突 |
| `HCCL_IF_IP` | 当前 rank 绑定的 RoCE/HCCL 互通 IP |

此外每个 shell 都要先执行：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/cann-8.5.0/opp/vendors/hwcomputing/bin/set_env.bash
```

### 5.4 双机启动模板

下面给的是**每台机器各起 1 个 rank** 的最小模板。

#### 约定

- Host1 IP：`192.168.0.125`
- Host2 IP：`192.168.0.192`
- rank0 在 Host1，rank1 在 Host2
- `MASTER_ADDR=192.168.0.125`
- 如果 Host2 仓库路径不是 `/workspace/afd_demo`，改成实际路径（例如 `/workspace/afd_demo_repo`）

#### normal-mode DeepEP bench

**Host1 / rank0**

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/cann-8.5.0/opp/vendors/hwcomputing/bin/set_env.bash
cd /workspace/afd_demo
export RANK=0 WORLD_SIZE=2 LOCAL_RANK=0
export MASTER_ADDR=192.168.0.125 MASTER_PORT=29611
export HCCL_IF_BASE_PORT=29621 HCCL_IF_IP=192.168.0.125
python3 scripts/cross_host_deepep_rt_bench.py \
  --iters 50 --warmup 10 \
  --num-tokens 64 --hidden 4096 \
  --topk 4 --num-experts 16
```

**Host2 / rank1**

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/cann-8.5.0/opp/vendors/hwcomputing/bin/set_env.bash
cd /workspace/afd_demo_repo
export RANK=1 WORLD_SIZE=2 LOCAL_RANK=0
export MASTER_ADDR=192.168.0.125 MASTER_PORT=29611
export HCCL_IF_BASE_PORT=29621 HCCL_IF_IP=192.168.0.192
python3 scripts/cross_host_deepep_rt_bench.py \
  --iters 50 --warmup 10 \
  --num-tokens 64 --hidden 4096 \
  --topk 4 --num-experts 16
```

#### low-latency DeepEP bench

**Host1 / rank0**

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/cann-8.5.0/opp/vendors/hwcomputing/bin/set_env.bash
cd /workspace/afd_demo
export RANK=0 WORLD_SIZE=2 LOCAL_RANK=0
export MASTER_ADDR=192.168.0.125 MASTER_PORT=29631
export HCCL_IF_BASE_PORT=29641 HCCL_IF_IP=192.168.0.125
python3 scripts/cross_host_deepep_lowlatency_rt_bench.py \
  --iters 50 --warmup 10 \
  --num-tokens 64 --hidden 4096 \
  --topk 8 --num-experts 128 \
  --num-max 128
```

**Host2 / rank1**

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/cann-8.5.0/opp/vendors/hwcomputing/bin/set_env.bash
cd /workspace/afd_demo_repo
export RANK=1 WORLD_SIZE=2 LOCAL_RANK=0
export MASTER_ADDR=192.168.0.125 MASTER_PORT=29631
export HCCL_IF_BASE_PORT=29641 HCCL_IF_IP=192.168.0.192
python3 scripts/cross_host_deepep_lowlatency_rt_bench.py \
  --iters 50 --warmup 10 \
  --num-tokens 64 --hidden 4096 \
  --topk 8 --num-experts 128 \
  --num-max 128
```

#### fallback bench

**Host1 / rank0**

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /workspace/afd_demo
export RANK=0 WORLD_SIZE=2 LOCAL_RANK=0
export MASTER_ADDR=192.168.0.125 MASTER_PORT=29651
export HCCL_IF_BASE_PORT=29661 HCCL_IF_IP=192.168.0.125
python3 scripts/cross_host_fallback_rt_bench.py \
  --iters 50 --warmup 10 \
  --num-tokens 64 --hidden 4096
```

**Host2 / rank1**

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /workspace/afd_demo_repo
export RANK=1 WORLD_SIZE=2 LOCAL_RANK=0
export MASTER_ADDR=192.168.0.125 MASTER_PORT=29651
export HCCL_IF_BASE_PORT=29661 HCCL_IF_IP=192.168.0.192
python3 scripts/cross_host_fallback_rt_bench.py \
  --iters 50 --warmup 10 \
  --num-tokens 64 --hidden 4096
```

### 5.5 输出怎么读

#### DeepEP normal bench

正常情况下会打印：

- `hccl pg ok`
- `buffer (normal) created`
- `warmup ok, starting timed iters`
- `RT us: mean=... p50=... p99=...`

如果 warmup 就失败，会直接打印：

- `WARMUP FAIL @ iter <i>: ...`

#### DeepEP low_latency bench

额外会打印：

- `rdma_size_hint=<N> bytes`

这个值很重要，因为 low-latency buffer 的大小直接由它决定。

#### fallback bench

输出里的：

- `payload`：单次 round-trip 处理的数据量，单位 KiB
- `mean / p50 / p99`：round-trip 延迟统计，单位微秒

---

## 6. 当前仍存在的问题

### 6.1 normal 模式：`aclnnNotifyDispatch` 超时

测试脚本：`scripts/cross_host_deepep_rt_bench.py`

实测日志片段：

```text
[rank0] hccl pg ok
[rank0] buffer (normal) created
[rank0] warmup...
EZ9999: Kernel task happen error, retCode=0x25, [aicore timeout]
[rank0] WARMUP FAIL @ iter 0: aclnnNotifyDispatch failed,
        EI0006 Communication_Error_Get_Socket: Getting socket times out.
        srcRank[192.168.0.125/0] connect destRank[192.168.0.192/0] fail.
```

当前判断：

- `get_dispatch_layout` 和 `dispatch` 已经开始执行，说明安装层没问题
- 失败点在 `aclnnNotifyDispatch`
- 这更像是 **DeepEP 自己的跨机 notify / RDMA 建链** 没打通
- HCCL 进程组初始化成功，并不能证明这条 notify 通道也成功

当前建议：

- 优先检查 `HCCL_NET_DEVICE`、NIC pinning、`hccn_tool` 配置
- 每次 bench 更换 `MASTER_PORT` / `HCCL_IF_BASE_PORT`
- 抓更详细的 Ascend slog 再判断是否是 RoCE 路径问题

### 6.2 low_latency 模式：warmup 首轮父进程 SIGKILL

测试脚本：`scripts/cross_host_deepep_lowlatency_rt_bench.py`

实测日志片段：

```text
[rank0] rdma_size_hint=128 bytes
[rank0] buffer (low_latency) created
[rank0] warmup...
[ERROR] TBE Subprocess[task_distribute] raise error[], main process disappeared!
resource_tracker: There appear to be 30 leaked semaphore objects to clean up at shutdown
```

当前判断：

- low_latency buffer 构造本身是成功的
- 失败发生在 warmup iter 0 的 dispatch/combine 执行期
- `rdma_size_hint=128 bytes` 对 `hidden=4096, num_max=128, world=2, experts=128` 来说明显异常

当前最可疑的方向：

1. `get_low_latency_rdma_size_hint(...)` 的参数语义和当前 wheel 预期不一致
2. 当前 DeepEP build 与本机 CANN runtime 的 low-latency 内存布局不匹配
3. RDMA QP 建链阶段触发硬错误，导致父进程直接退出

### 6.3 fallback 路径可跑通

测试脚本：`scripts/cross_host_fallback_rt_bench.py`

这条路径不是 DeepEP，而是：

- 通信原语：`torch.distributed.all_to_all_single`
- NPU 双机 backend：`hccl`

当前作用：

- 证明基础 cross-host distributed collective 是通的
- 证明 coordinator / communicator 的生产可交付路径可以先落在 fallback 上

当前结果：

- 64 KiB payload：约 **330 µs mean / 414 µs p99**
- 512 KiB payload：约 **379 µs mean / 453 µs p99**

---

## 7. Fallback 方案说明

### 7.1 它用的是什么方法

Fallback 用的是 **`torch.distributed.all_to_all_single`**。

在我们的双机 NPU bench 里，进程组通过：

```python
dist.init_process_group(backend="hccl", init_method="env://", ...)
```

建立，所以 fallback 实际走的是 **HCCL backend**，不是 gloo。

### 7.2 为什么脚本里用了两次 `all_to_all_single`

`scripts/cross_host_fallback_rt_bench.py` 里 round-trip 的逻辑是：

1. 第一次 `all_to_all_single`：模拟 dispatch，把 token 发到对端
2. 第二次 `all_to_all_single`：模拟 combine，把 echo 后的结果再发回来

因此它测到的是**完整 round-trip**，不是单程时延。

### 7.3 为什么它现在更适合作为交付路径

- 它已经跨机跑通
- 它不依赖 DeepEP 自定义 kernel
- 虽然延迟高于目标里的理想值，但比 DeepEP 当前的“无法稳定执行”更适合作为阶段性交付方案

---

## 8. 当前建议

当前建议是：

1. 把 **DeepEP** 标记为 `experimental`
2. 把 **Fallback (`all_to_all_single` over HCCL)** 作为当前生产/验证路径
3. 把 DeepEP 的 normal / low_latency 运行时问题，作为单独的 RoCE/HCCL/CANN 联调课题推进
