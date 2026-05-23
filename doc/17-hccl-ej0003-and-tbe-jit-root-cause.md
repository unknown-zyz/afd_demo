# HCCL `EJ0003` 与 TBE JIT 编译过久根因分析

**状态**：跨机 1A7F coordinator 继续推进前的阻塞拆解文档。2026-05-23 的最小复现实验显示，Host2 本地 EP7 `EJ0003` 与一个残留 `python -m src.main` rank 相关；清理该 PID 后，Host2 / Host1 的 8-rank default group 与 EP/dispatch/reduce subgroups 都能通过。随后 Host2 真实 `src.main` EP7 prefill warmup 的 b2/s64 与 b2/s128 也完成，当前阻塞已从“Host2 HCCL barrier 过不去”解除，下一步应回到跨机 1A7F 或更大 shape 的真实路径验证。

---

## 1. 结论摘要

当前不能再把跨机 1A7F 卡住简单归因为“首次编译太慢”：

1. **历史 TBE 现象确实存在**：真实 Qwen3-30B-A3B EP7 路径能完成 distributed init、权重加载和通信器初始化，并进入 `Running 1 prefill warmup round(s)`；随后长时间没有继续输出，`kernel_meta/` 只增长到数百 KB。
2. **最新 isolate 先复现、再恢复**：Host2 最小脚本在残留 rank 存在时，8-rank default world barrier 即复现 `EJ0003`；清理明确属于本项目的残留 PID `54952` 后，Host2 default group 与 all-groups 复测均通过。
3. **EP subgroup 本身不是当前根因**：Host2 和 Host1 都能按真实顺序创建 `ffn_ep_group`、`ffn_ep_dispatch_group`、`ffn_ep_reduce_group` 并完成 group barrier；当前应把 `EJ0003` 根因收敛到残留实验进程 / HCCL runtime 状态污染，而不是多 `new_group` 逻辑错误。
4. **真实 Host2 local EP7 小 shape 已恢复**：清理后，`run_tbe_cache_warmup_npu.sh --profile host2-ep7` 在 b2/s64 与 b2/s128 上均 `Exit=0`，rank 日志显示 `Running 1 prefill warmup round(s)` 后约 2.7s 完成 warmup 并写出 timing。

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

### 3.1 Host2 local EP7 `EJ0003`

关键报错：

```text
Communication_Error_Bind_IP_Port(EJ0003): Failed to bind the IP port.
Reason: The IP address and port have been bound already.
```

伴随 HCCL 初始化错误：

```text
hcclCommInitRootInfoConfig(...), error code is 7
```

此前真实 `src.main` isolate 的重要点：

- 使用 clean worktree `/workspace/afd_demo_repo_exp_1a7f`
- fresh `MASTER_PORT=29950 HCCL_IF_BASE_PORT=30950`
- fresh `MASTER_PORT=31950 HCCL_IF_BASE_PORT=33950`
- `--prefill-warmup-rounds 0` 仍会在模型加载后的 `ctx.barrier()` 报错

这说明问题不应继续被描述成“只是 TBE 编译太慢”。但 2026-05-23 的最小脚本进一步证明：在清理残留 `src.main` rank 后，同机 8-rank HCCL default group 和 EP subgroups 可以恢复通过，因此这次 `EJ0003` 更像是残留实验进程 / HCCL runtime 状态污染，而不是代码层多 group 创建必然失败。

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
[ -f venv/bin/activate ] && source venv/bin/activate
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
[ -f venv/bin/activate ] && source venv/bin/activate
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
[ -f venv/bin/activate ] && source venv/bin/activate
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
[ -f venv/bin/activate ] && source venv/bin/activate
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
| stale process check | Host2 | `ps -ef \| grep src.main` | 发现 PID `54952` | 5/22 遗留 `python -m src.main` rank0，属于本项目 EP7 warmup |
| default group only | Host2 | `MASTER_PORT=35001`, `HCCL_IF_BASE_PORT=36001` | 失败 | 残留 PID 存在时，rank0 在 `world_barrier_after_init` 触发 `EJ0003`；其他 rank 等到 600s watchdog |
| stale process cleanup | Host2 | `kill 54952` | 已执行 | 只杀明确属于本项目的 PID；之后仅剩 defunct 记录，不再是运行中 rank |
| default group only | Host2 | `MASTER_PORT=35071`, `HCCL_IF_BASE_PORT=36071` | PASS | 清理后 8-rank default HCCL barrier 通过 |
| all groups | Host2 | `MASTER_PORT=35081`, `HCCL_IF_BASE_PORT=36081` | PASS | default world + EP/dispatch/reduce groups + group barriers 均通过 |
| default group only | Host1 | `MASTER_PORT=35101`, `HCCL_IF_BASE_PORT=36101` | PASS | Host1 8-rank default group 对照通过 |
| all groups | Host1 | `MASTER_PORT=35181`, `HCCL_IF_BASE_PORT=36181` | PASS | Host1 EP/dispatch/reduce groups 对照通过 |
| real EP7 warmup | Host2 | wrapper b2/s64, default `MASTER_PORT=29650` | PASS | `Exit=0`，rank0 warmup `2759.4 ms`，prefill timing 写入 `results/prefill_dbo/` |
| real EP7 warmup | Host2 | wrapper b2/s128, default `MASTER_PORT=29650` | PASS | `Exit=0`，rank0 warmup `2676.5 ms`，prefill timing 写入 `results/prefill_dbo/` |
| `src.main` no-warmup | Host2 | `31950/33950` | 可选复测 | 旧失败发生在残留 rank 存在背景下；小 shape warmup 已证明真实模型路径当前可越过 HCCL barrier |

关键日志片段：

```text
# Host2 stale PID 存在时
2026-05-23 03:16:23 [rank0] FAILED
RuntimeError: ... hcclCommInitRootInfoConfig(...), error code is 7
Communication_Error_Bind_IP_Port(EJ0003): Failed to bind the IP port.

# Host2 清理 PID 54952 后
2026-05-23 03:29:14 [rank0] END world_barrier_after_init: 1008.712 ms
2026-05-23 03:29:15 [rank0] completed successfully

# Host2 all-groups
2026-05-23 03:29:45 [rank0] END world_barrier_after_group_creation: 2.809 ms
2026-05-23 03:29:46 [rank0] completed successfully

# Host2 real EP7 b2/s128 after cleanup
Running 1 prefill warmup round(s) to absorb JIT compile cost
  warmup 1/1: 2676.5 ms
Timing saved: results/prefill_dbo/timing_attention_serial-prefill_npu_ep7_broadcast_reduce_overlap_b2_s128_t5.json
```

---

## 6. 判读规则

1. **default group only 失败**：Host2 8-rank local HCCL 基础层已经不健康。优先清理残留 rank、检查端口、HCCL/NPU runtime 状态，必要时运维介入或重启本项目容器。
2. **default 通过但 EP group 失败**：问题收敛到 `dist.new_group(ranks=[1..7])` / subgroup bootstrap。继续比较 Host1 对照和 group 数量。
3. **最小脚本全通过但 `src.main` 失败**：HCCL 基础多 group 可用；触发点可能是模型加载后的 torch_npu runtime 状态、AFD communicator 使用顺序、真实模型占用 HBM 后的 HCCL barrier，或需要下一层“模型加载后 barrier 最小脚本”。
4. **`src.main` no-warmup 通过但 warmup 卡住**：再回到 TBE JIT / `kernel_meta` 冷编译排查。

---

## 7. 当前结论边界

- 已经修复过的 2-rank cross-host HCCL / fallback RT 不能证明 Host2 local EP7 多 group 当前健康；但本次最小脚本已补齐 8-rank local EP7 group 证据。
- Host2 local EP7 最小 HCCL 已恢复；此前 `EJ0003` 与残留 `src.main` rank / HCCL runtime 状态污染强相关。
- Host2 local EP7 真实 warmup 小 shape 已恢复；b2/s64 与 b2/s128 不再复现 60min 级别卡死，且没有再触发 `EJ0003`。
- TBE JIT 冷编译仍是历史真实 decode 的重要风险，但当前证据显示本轮主要 blocker 是残留 rank 造成的 HCCL 状态污染。后续扩大 shape 或跨机 1A7F 前仍应先检查/清理 `src.main` 残留。
- 跨机 1A7F coordinator 可以进入下一轮小配置复测，但不应直接扩大 full matrix；先用 b2/s128/t5 或更小 shape 验证跨机真实路径和日志/metrics。
- DeepEP runtime 仍保持 deferred；当前主线是先恢复 HCCL/fallback EP7 真实路径。
