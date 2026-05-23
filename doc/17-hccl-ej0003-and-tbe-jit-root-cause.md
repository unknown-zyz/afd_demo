# HCCL `EJ0003` 与 TBE JIT 编译过久根因分析

**状态**：跨机 1A7F coordinator 继续推进前的阻塞拆解文档。当前 immediate blocker 是 **Host2 本地 EP7 / HCCL runtime 在 fresh ports 下仍触发 `EJ0003`**；历史上真实 Qwen3 EP7 路径还存在 TBE JIT 冷编译过久风险。

---

## 1. 结论摘要

当前不能再把跨机 1A7F 卡住简单归因为“首次编译太慢”：

1. **历史 TBE 现象确实存在**：真实 Qwen3-30B-A3B EP7 路径能完成 distributed init、权重加载和通信器初始化，并进入 `Running 1 prefill warmup round(s)`；随后长时间没有继续输出，`kernel_meta/` 只增长到数百 KB。
2. **最新 immediate blocker 更低层**：Host2 干净 worktree + fresh `MASTER_PORT` + fresh `HCCL_IF_BASE_PORT`，甚至关闭 `--prefill-warmup-rounds 0` 后，rank0/rank1 仍在模型加载后的 `ctx.barrier()` 触发 HCCL `EJ0003`。
3. **下一步应先隔离 HCCL**：新增 `scripts/repro_hccl_ep7_ej0003.py` 绕开 Qwen 模型和 TBE，只复现 default process group + FFN EP groups + barriers。若该脚本也复现 `EJ0003`，根因收敛到 HCCL runtime / 多 group / 端口状态，而不是模型或 TBE。

---

## 2. 拓扑与路径

| 项 | Host1 | Host2 |
|---|---|---|
| 机器 | `schedTeam@1.95.114.229` | `root@192.168.0.192` via Host1 |
| 容器 | `afd-npu-test` | `afd-npu-test-h2` |
| 原 worktree | `/workspace/afd_demo` | `/workspace/afd_demo_repo` |
| clean worktree | `/workspace/afd_demo_exp_1a7f` | `/workspace/afd_demo_repo_exp_1a7f` |
| 控制网 IP | `192.168.0.125` | `192.168.0.192` |
| 模型路径 | `/models/Qwen3-30B-A3B` | `/models/Qwen3-30B-A3B` |

1A7F / EP7 rank layout：

| rank | 角色 | 说明 |
|---|---|---|
| 0 | Attention | attention rank / cross-host P3 时在 Host1 |
| 1 | FFN coordinator | FFN EP rank0 / cross-host P3 时在 Host2 |
| 2-7 | FFN expert | FFN EP rank1-6 / cross-host P3 时在 Host2 |

真实代码路径：

```text
scripts/run_npu.sh
  -> python -m src.main
  -> src/model/disaggregated.py
  -> src/distributed/__init__.py
     init_process_group(hccl)
     dist.new_group(ranks=[1..7])  # ffn_ep_group
     dist.new_group(ranks=[1..7])  # ffn_ep_dispatch_group
     dist.new_group(ranks=[1..7])  # ffn_ep_reduce_group
```

`scripts/repro_hccl_ep7_ej0003.py` 只保留上面 distributed/HCCL 部分，不加载模型、不进入 prefill、不触发 TBE。

---

## 3. 错误信息原文

### 3.1 当前 Host2 local EP7 `EJ0003`

关键报错：

```text
Communication_Error_Bind_IP_Port(EJ0003): Failed to bind the IP port.
Reason: The IP address and port have been bound already.
```

伴随 HCCL 初始化错误：

```text
hcclCommInitRootInfoConfig(...), error code is 7
```

最新 isolate 的重要点：

- 使用 clean worktree `/workspace/afd_demo_repo_exp_1a7f`
- fresh `MASTER_PORT=29950 HCCL_IF_BASE_PORT=30950`
- fresh `MASTER_PORT=31950 HCCL_IF_BASE_PORT=33950`
- `--prefill-warmup-rounds 0` 仍会在模型加载后的 `ctx.barrier()` 报错

这说明问题不应继续被描述成“只是 TBE 编译太慢”。

### 3.2 历史 cross-host socket 超时

历史跨机失败中 Host2 侧曾出现：

```text
EI0006 Communication_Error_Get_Socket
```

这通常对应 rank 间 socket 建链超时。此前通过 fresh `MASTER_PORT` / fresh `HCCL_IF_BASE_PORT` / 合法 `HCCL_CONNECT_TIMEOUT=600` 已修复 2-rank cross-host HCCL smoke 和 fallback RT，但不能推导 Host2 local EP7 多 group 当前也健康。

### 3.3 TBE JIT / `kernel_meta` 冷编译过久

真实模型路径中的日志卡点通常是：

```text
Running 1 prefill warmup round(s) to absorb JIT compile cost
```

观察现象：

- 前台 `run_npu.sh` 没有新输出，因为 rank 日志写入 `results/logs/npu_*.log`
- `kernel_meta/` 长时间只在数百 KB
- 进程可能停在 TBE multiprocessing 编译相关栈或等待 collective
- 60min wrapper timeout 后仍没有 timing 产物

---

## 4. 可复现命令

### 4.1 Host2 local 最小 HCCL base：只测 default process group

目的：不创建 FFN EP groups，只验证 Host2 8-rank default HCCL group + world barrier 是否健康。

```bash
cd /workspace/afd_demo_repo_exp_1a7f
source venv/bin/activate
source /usr/local/Ascend/ascend-toolkit/set_env.sh

ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MASTER_ADDR=127.0.0.1 \
MASTER_PORT=35001 \
HCCL_IF_BASE_PORT=36001 \
HCCL_CONNECT_TIMEOUT=600 \
HCCL_EXEC_TIMEOUT=600 \
python3 scripts/repro_hccl_ep7_ej0003.py \
  --spawn-local \
  --world-size 8 \
  --master-addr 127.0.0.1 \
  --master-port 35001 \
  --timeout-sec 600 \
  --skip-ep-groups
```

### 4.2 Host2 local EP group 分层

目的：分别判断 `dist.new_group(ranks=[1..7])` 的单 group、多 group 或 group barrier 是否触发 `EJ0003`。

```bash
# 只创建/测试 ffn_ep_group
ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MASTER_ADDR=127.0.0.1 MASTER_PORT=35011 HCCL_IF_BASE_PORT=36011 \
HCCL_CONNECT_TIMEOUT=600 HCCL_EXEC_TIMEOUT=600 \
python3 scripts/repro_hccl_ep7_ej0003.py \
  --spawn-local --world-size 8 --master-port 35011 --groups ep --timeout-sec 600

# 只创建/测试 dispatch group
ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MASTER_ADDR=127.0.0.1 MASTER_PORT=35021 HCCL_IF_BASE_PORT=36021 \
HCCL_CONNECT_TIMEOUT=600 HCCL_EXEC_TIMEOUT=600 \
python3 scripts/repro_hccl_ep7_ej0003.py \
  --spawn-local --world-size 8 --master-port 35021 --groups dispatch --timeout-sec 600

# 只创建/测试 reduce group
ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MASTER_ADDR=127.0.0.1 MASTER_PORT=35031 HCCL_IF_BASE_PORT=36031 \
HCCL_CONNECT_TIMEOUT=600 HCCL_EXEC_TIMEOUT=600 \
python3 scripts/repro_hccl_ep7_ej0003.py \
  --spawn-local --world-size 8 --master-port 35031 --groups reduce --timeout-sec 600

# 按真实路径顺序创建 ep / dispatch / reduce 三个 group
ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MASTER_ADDR=127.0.0.1 MASTER_PORT=35041 HCCL_IF_BASE_PORT=36041 \
HCCL_CONNECT_TIMEOUT=600 HCCL_EXEC_TIMEOUT=600 \
python3 scripts/repro_hccl_ep7_ej0003.py \
  --spawn-local --world-size 8 --master-port 35041 --groups all --timeout-sec 600
```

### 4.3 Host1 local 对照

在 Host1 clean worktree `/workspace/afd_demo_exp_1a7f` 使用同样命令，只更换端口区间，例如：

```bash
cd /workspace/afd_demo_exp_1a7f
source venv/bin/activate
source /usr/local/Ascend/ascend-toolkit/set_env.sh

ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MASTER_ADDR=127.0.0.1 MASTER_PORT=35141 HCCL_IF_BASE_PORT=36141 \
HCCL_CONNECT_TIMEOUT=600 HCCL_EXEC_TIMEOUT=600 \
python3 scripts/repro_hccl_ep7_ej0003.py \
  --spawn-local --world-size 8 --master-port 35141 --groups all --timeout-sec 600
```

如果 Host1 通过而 Host2 失败，优先怀疑 Host2 HCCL runtime / 端口 / device 状态；如果两边都失败，再回头查脚本、torch_npu/CANN 版本或多 `new_group` 通用问题。

### 4.4 真实 `src.main` no-warmup 对照

仅在最小 HCCL 脚本通过后再跑真实模型 no-warmup，否则会重新把模型加载、TBE 和 HCCL 混在一起。

```bash
cd /workspace/afd_demo_repo_exp_1a7f
source venv/bin/activate
source /usr/local/Ascend/ascend-toolkit/set_env.sh

ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MASTER_ADDR=127.0.0.1 \
MASTER_PORT=31950 \
HCCL_IF_BASE_PORT=33950 \
HCCL_CONNECT_TIMEOUT=600 \
HCCL_EXEC_TIMEOUT=1800 \
AFD_DIST_TIMEOUT_SEC=7200 \
bash scripts/run_npu.sh \
  --preset npu-ep7 \
  --ffn-ep-backend broadcast_reduce_overlap \
  --batch 2 --seq 64 --tokens 5 \
  --model-name /models/Qwen3-30B-A3B \
  --no-dbo --no-generate \
  --prefill-warmup-rounds 0
```

### 4.5 TBE cache warmup 复查

仅在 Host2 HCCL barrier 恢复后，再复查 `kernel_meta` 是否能被灌满：

```bash
cd /workspace/afd_demo_repo_exp_1a7f
source venv/bin/activate
source /usr/local/Ascend/ascend-toolkit/set_env.sh

bash scripts/run_tbe_cache_warmup_npu.sh \
  --profile host2-ep7 \
  --batch 2 --seq 128 --tokens 5 \
  --model-name /models/Qwen3-30B-A3B \
  --timeout-sec 7200 --poll-sec 60
```

---

## 5. 实验记录表

| 步骤 | 主机 | 命令/端口 | 当前结果 | 结论 |
|---|---|---|---|---|
| default group only | Host2 | `MASTER_PORT=35001`, `HCCL_IF_BASE_PORT=36001` | 待跑 | 判断 default HCCL 是否健康 |
| EP group only | Host2 | `35011/36011` | 待跑 | 判断单个 FFN subgroup 是否健康 |
| dispatch group only | Host2 | `35021/36021` | 待跑 | 判断第二类 subgroup 是否健康 |
| reduce group only | Host2 | `35031/36031` | 待跑 | 判断第三类 subgroup 是否健康 |
| all groups | Host2 | `35041/36041` | 待跑 | 模拟真实 group 创建顺序 |
| all groups | Host1 | `35141/36141` | 待跑 | Host1 local 对照 |
| `src.main` no-warmup | Host2 | `31950/33950` | 已知失败 | 模型加载后 `ctx.barrier()` 触发 `EJ0003` |

---

## 6. 判读规则

1. **default group only 失败**：Host2 8-rank local HCCL 基础层已经不健康。优先清理残留 rank、检查端口、HCCL/NPU runtime 状态，必要时运维介入或重启本项目容器。
2. **default 通过但 EP group 失败**：问题收敛到 `dist.new_group(ranks=[1..7])` / subgroup bootstrap。继续比较 Host1 对照和 group 数量。
3. **最小脚本全通过但 `src.main` 失败**：HCCL 基础多 group 可用；触发点可能是模型加载后的 torch_npu runtime 状态、AFD communicator 使用顺序、真实模型占用 HBM 后的 HCCL barrier，或需要下一层“模型加载后 barrier 最小脚本”。
4. **`src.main` no-warmup 通过但 warmup 卡住**：再回到 TBE JIT / `kernel_meta` 冷编译排查。

---

## 7. 当前结论边界

- 已经修复过的 2-rank cross-host HCCL / fallback RT 不能证明 Host2 local EP7 多 group 当前健康。
- TBE JIT 冷编译仍是历史真实 decode 的重要风险，但当前 latest blocker 更早发生在 Host2 local EP7 HCCL barrier。
- 在 Host2 local EP7 HCCL 未恢复前，不应继续扩大跨机 1A7F coordinator matrix。
- DeepEP runtime 仍保持 deferred；当前主线是先恢复 HCCL/fallback EP7 真实路径。
