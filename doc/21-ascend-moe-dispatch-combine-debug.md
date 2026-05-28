# 昇腾 MoE dispatch/combine 算子调研与 EPFFN 接入排查

## 结论摘要

当前不建议把昇腾 `npu_moe_distribute_dispatch_v2/combine_v2` 作为 `EPFFNLayer` 的默认或推荐 backend。

原因是：torch_npu 2.9 / CANN 8.5.1 隔离容器里，`dispatch_v2 + combine_v2 + comm_alg=fullmesh_v2` 已能跑通 Qwen3-like synthetic/replay case；但接入真实 `src.main` 的 Qwen3 decode 路径后，真实 expert GEMM 输出进入 `combine_v2` 仍会触发 NPU `aicore timeout` / `MTE out-of-range` / `ACL stream synchronize failed`。因此 `--ffn-ep-backend npu_moe_v2` 目前只能保留为显式 opt-in 的实验诊断路径。

当前稳定生产/验证路径仍是：

- `broadcast_reduce_sync`
- `broadcast_reduce_overlap`

这两条路径由 `EPFFNLayer` 执行 full hidden broadcast + dense reduce，已经接入真实 decode EP overlap；`npu_moe_v2` 还没有通过真实 decode-DBO TPOT gate。

## 环境与版本结论

### 现有生产/验证容器

| 容器 | CANN | torch | torch_npu | 结论 |
|---|---|---|---|---|
| Host1 `afd-npu-test` | 8.5.0 B232 | 2.6.0+cpu | 2.6.0 | base API 可见但 Qwen3 H=2048 不适配；v2 schema 无法传 `comm_alg`。 |
| Host2 `afd-npu-test-h2` | 8.5.0 B232 | 2.6.0+cpu | 2.6.0 | 与 Host1 同类，不适合直接接入 official MoE v2。 |

torch_npu 2.6 不是“完全没有 API”，而是当前组合对本项目形状/参数支持不完整：

1. `npu_moe_distribute_dispatch` base API 在 `H=7168` sanity case 可跑通，但对 Qwen3 `hidden_size=2048` 报 shape/tiling 错误。
2. `npu_moe_distribute_dispatch_v2/combine_v2` 存在，但 Python schema 没有 `comm_alg` 参数；CANN 默认 `commAlg=0` 被拒绝。
3. 因此失败不能简单归因为 PyTorch `torch` 包版本问题，更准确地说是 `torch_npu/CANN/op-plugin` 版本配套和 Python binding 暴露能力不满足当前 Qwen3 MoE 路径。

### 隔离 torch_npu 2.9 容器

为排查版本问题，创建了隔离容器：

| 项 | 值 |
|---|---|
| Host | Host2 `liteserver-910c-2-00001.novalocal` |
| Container | `afd-npu-epffn-v2-test` |
| Image | `quay.io/ascend/vllm-ascend:v0.18.0rc1-a3` |
| CANN | 8.5.1 |
| torch | 2.9.0+cpu |
| torch_npu | 2.9.0.post1 |
| Qwen3 权重 | `/models/Qwen3-30B-A3B` 只读挂载 |

该环境的 `npu_moe_distribute_dispatch_v2/combine_v2` schema 暴露 `comm_alg`，可以传 `comm_alg="fullmesh_v2"`。已有版本探测结果见 `results_npu/moe_version_probe/summary.md`。

关键结果：

| Case | 结果 |
|---|---|
| base H=2048/top_k=8 | 失败，base 仍不适合 Qwen3 hidden size。 |
| base H=7168/top_k=2 | 通过，说明容器/HCCL/base API sanity 没问题。 |
| v2 H=2048/default comm_alg | 可通过但极慢，不适合作为候选路径。 |
| v2 H=2048/`comm_alg=fullmesh_v2` | 通过，raw probe total median 约 `0.482 ms`；更新 benchmark warm 后约 `0.860 ms`。 |

## 当前实现改动

本分支新增了实验 backend：

```bash
--ffn-ep-backend npu_moe_v2
```

默认值仍是 `broadcast_reduce_sync`，不会静默切换到 official v2。显式选择 `npu_moe_v2` 时，如果环境不满足要求会直接报错，不 silent fallback。

主要代码点：

| 文件 | 作用 |
|---|---|
| `src/main.py` | `--ffn-ep-backend` choices 增加 `npu_moe_v2`。 |
| `scripts/run_npu.sh` | `npu_moe_v2` backend 下，如用户未显式设置 expert policy，自动使用 `contiguous`。 |
| `src/pipeline/decode_scheduler.py` | `npu_moe_v2` 进入 EP overlap decode path，允许 `num_mb >= 1`。 |
| `src/model/ep_moe.py` | 新增 dispatch/combine v2 path、metadata 生命周期、hcomm 获取、packed expert compute 和诊断 env。 |
| `src/model/ffn_worker.py` | 兼容新版 Qwen3 `ModuleList` experts 与 logits-only gate output。 |
| `scripts/check_epffn_npu_moe_v2.py` | synthetic/replay correctness harness。 |

`EPFFNLayer` 中新增的核心语义是：

1. coordinator 仍做 post-attention RMSNorm 和 gate，得到 `hidden_2d`、`selected_experts`、`routing_weights`。
2. 所有 FFN EP ranks 调 `npu_moe_distribute_dispatch_v2`，得到 packed `expand_x` 与 combine metadata。
3. `ShardedExperts.forward_packed()` 在 packed expert token 段上执行本地 experts。
4. `npu_moe_distribute_combine_v2` 使用 dispatch metadata 聚合回 coordinator 输出。

这不是把 `broadcast` / `reduce` 两行替换成 official API，而是完全不同的 packed token dispatch/combine 协议。

## 已做尝试

### 1. torch_npu 2.6 API 探测与通信 benchmark

新增：

- `scripts/probe_moe_distribute_npu.py`
- `scripts/bench_moe_dispatch_npu.py`

在 Host1 `afd-npu-test` 上测得：

| backend | tokens | hidden | top_k | 结果 |
|---|---:|---:|---:|---|
| current broadcast/reduce | 4 | 2048 | 8 | total median `0.364 ms` |
| current broadcast/reduce | 16 | 2048 | 8 | total median `0.302 ms` |
| current broadcast/reduce | 64 | 2048 | 8 | total median `0.294 ms` |
| official base | 4 | 2048 | 8 | dispatch 失败，`H` 只支持 7168 类 shape |
| official base | 4 | 7168 | 2 | total median `0.279 ms` |
| official v2 | 4 | 2048 | 8 | dispatch 失败，`commAlg=0` 无效且 schema 无 `comm_alg` |

结果文件：

- `results_npu/ffn_moe_dispatch_bench/summary.md`
- `results_npu/ffn_moe_dispatch_bench/*.json`

结论：torch_npu 2.6 / CANN 8.5.0 下，official MoE distribute 不能直接用于 Qwen3 `H=2048, top_k=8` 的真实 EPFFN 路径。

### 2. torch_npu 2.9 隔离容器版本 probe

在 `afd-npu-version-probe-torch29-npu` / 后续 `afd-npu-epffn-v2-test` 环境中确认：

1. v2 schema 暴露 `comm_alg`。
2. `comm_alg=fullmesh_v2` 可以跑 Qwen3-like `H=2048, top_k=8, experts=128, EP=8`。
3. base API 仍不适合 H=2048。

结果文件：

- `results_npu/moe_version_probe/summary.md`
- `results_npu/moe_version_probe/schema_torch29_npu.txt`
- `results_npu/moe_version_probe/torch29_v2_h2048_fullmesh_v2.json`

### 3. 接入 `EPFFNLayer` 的 experimental backend

已完成的关键适配：

1. `EPWorkItem` 保存 v2 metadata：
   - `assist_info_for_combine`
   - `expert_token_nums`
   - `ep_recv_counts`
   - `tp_recv_counts`
   - `expand_scales`
   - `combined`
2. `ExpertShardPlan` 要求 `contiguous`，因为当前 `expert_shard_type=0` 路径只按连续 expert ownership 验证。
3. HCCL comm name 使用 FFN EP subgroup 的 EP-local rank 获取，而不是 global rank。
4. `global_bs` 默认设为 `local_tokens * ep_world_size`，也可用 `AFD_NPU_MOE_V2_GLOBAL_BS` 覆盖。
5. `ShardedExperts.forward_packed()` 支持按 `expert_token_nums` 的 packed 段运行 local experts，并跳过 padding zero rows。
6. `parse_gate_output()` 兼容新版 Qwen3 gate 只返回 logits tensor 的情况。

### 4. synthetic / replay correctness harness

新增 `scripts/check_epffn_npu_moe_v2.py`，覆盖：

- tokens `1/4/32`
- hidden `2048`
- intermediate `768`
- experts `128`
- top_k `8`
- EP8
- 9-rank A/F-like topology：attention rank + 8 FFN EP ranks
- random / hot routing
- extra EP group 创建
- 模拟 A/F P2P
- replay dumped real layer input
- 加载真实 Qwen expert weights

代表命令：

```bash
MASTER_PORT=29916 ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 \
torchrun --nproc_per_node=9 scripts/check_epffn_npu_moe_v2.py \
  --tokens 1 --hidden 2048 --intermediate 768 --experts 128 --top-k 8 \
  --ffn-subgroup-offset 1 --extra-ep-groups 2 --routing-pattern random \
  --global-bs-multiplier 8 --use-transfer-to-npu --simulate-af-p2p \
  --output results_npu/epffn_moe_v2/sim_p2p_tokens1_random.json
```

这些 synthetic/replay case 多数通过，说明 v2 算子在隔离形状和单次调用语义上并非完全不可用。

### 5. 真实 Qwen `src.main` smoke 与失败缩小

已通过：

1. real Qwen `b1/s1/t1`。
2. real Qwen `b1/s32/t2` + dispatch-only。
3. real Qwen `b1/s32/t2` + `AFD_NPU_MOE_V2_ZERO_EXPERTS=1`。
4. dumped real layer0 input 在 harness replay 中通过。
5. 加载真实 Qwen expert weights 的 replay 通过。

仍失败：

1. real Qwen `b1/s32/t2`，非零真实 expert 输出进入 `combine_v2`。
2. 失败点通常在 `combine_returned` 后、`combine_synced` 前后，即 combine kernel 返回后同步暴露 NPU 错误。
3. `combine_input.contiguous()`、`torch.empty(...).copy_(item.partial)` materialize、固定 `global_bs=256` 都不能彻底解决。

## 最终失败复现命令

以下命令应在 Host2 隔离容器 `afd-npu-epffn-v2-test` 内运行，使用 fresh `MASTER_PORT`，不要复用旧端口。

### 失败：真实 expert 输出 + combine_v2

```bash
MASTER_PORT=29918 ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 \
AFD_DIST_TIMEOUT_SEC=7200 AFD_DEBUG_MAX_LAYERS=2 \
AFD_NPU_MOE_V2_TRACE=1 AFD_NPU_MOE_V2_FORCE_SYNC=1 \
bash scripts/run_npu.sh --attn-size 1 --ffn-size 8 --ffn-tp-size 1 \
  --ffn-ep-size 8 --ffn-ep-backend npu_moe_v2 \
  --batch 1 --seq 32 --tokens 2 --num-micro-batches 1 \
  --model-name /models/Qwen3-30B-A3B
```

典型现象：

- trace 可看到 dispatch 和 local compute 已完成；
- combine kernel 返回后同步失败；
- 错误包括 `ACL stream synchronize failed`、`npuSynchronizeDevice ... 507014`、`aicore timeout`；
- 部分日志还会出现 `The DDR address of the MTE instruction is out of range`。

### 对照通过：zero experts

```bash
MASTER_PORT=29915 ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 \
AFD_DIST_TIMEOUT_SEC=7200 AFD_DEBUG_MAX_LAYERS=2 \
AFD_NPU_MOE_V2_TRACE=1 AFD_NPU_MOE_V2_FORCE_SYNC=1 \
AFD_NPU_MOE_V2_ZERO_EXPERTS=1 \
bash scripts/run_npu.sh --attn-size 1 --ffn-size 8 --ffn-tp-size 1 \
  --ffn-ep-size 8 --ffn-ep-backend npu_moe_v2 \
  --batch 1 --seq 32 --tokens 2 --num-micro-batches 1 \
  --model-name /models/Qwen3-30B-A3B
```

这个对照说明：

- dispatch/combine 控制路径不是必然失败；
- zero packed output 进入 combine 可以完成；
- 问题更像是 combine_v2 对真实 expert output tensor 的格式、layout、padding、写入方式或连续多次调用状态有额外约束。

### 对照通过：synthetic A/F-like tokens=1

```bash
MASTER_PORT=29916 ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 \
torchrun --nproc_per_node=9 scripts/check_epffn_npu_moe_v2.py \
  --tokens 1 --hidden 2048 --intermediate 768 --experts 128 --top-k 8 \
  --ffn-subgroup-offset 1 --extra-ep-groups 2 --routing-pattern random \
  --global-bs-multiplier 8 --use-transfer-to-npu --simulate-af-p2p \
  --output results_npu/epffn_moe_v2/sim_p2p_tokens1_random.json
```

这个对照说明：单独的 tokens=1、EP8、A/F-like subgroup、H=2048、top_k=8 并不是失败充分条件。

## 根因分析

### 已基本排除的方向

1. **不是 dispatch_v2 必然不可用**
   dispatch-only real path 可以返回并同步。

2. **不是 H=2048 / top_k=8 / experts=128 本身不可用**
   torch_npu 2.9 v2 synthetic 与 replay case 可以通过。

3. **不是 FFN EP subgroup hcomm rank 获取错误**
   已改为用 EP-local rank 获取 HCCL comm name；synthetic 9-rank + extra groups 可通过。

4. **不是 A/F P2P 与 FFN subgroup collective 并存必然失败**
   harness 中 `--simulate-af-p2p` 仍可通过。

5. **不是真实 layer0 input 或真实 expert weights 单独导致**
   dumped real layer0 input + true Qwen expert weights replay 通过。

6. **不是 combine_v2 对任何输入都失败**
   zero-experts path 可以完整通过。

7. **不是单纯缺少 `contiguous()` 或简单 materialize**
   `contiguous()` 与 `empty+copy_` 都尝试过，只能移动或缓解失败点，不能使真实 decode path 稳定。

### 当前最可信判断

问题集中在：

```text
真实 Qwen expert GEMM 输出 -> packed partial buffer -> combine_v2
```

可能原因包括：

1. `combine_v2` 对 `expand_x` 的 NPU 内存 format/layout 有未文档化要求。
2. 自定义 Python expert GEMM 写出的 packed output 与 CANN combine kernel 期望的 buffer 格式不完全一致。
3. `expert_token_nums` 中存在大量 padding rows，虽然 compute 已跳过 padding zero rows，但 combine 仍可能要求某些 padding/segment 状态满足更严格约束。
4. 完整真实进程中 prefill 后连续 decode、多个 layer 连续调用、HCCL subgroup 与 P2P 交织后的 runtime 状态，和单独 harness replay 不完全等价。
5. CANN/torch_npu 2.9 当前 `combine_v2` 在该使用方式下存在 kernel bug 或缺少公开约束说明。

因此现在不能把它描述成“已经跑通、只差性能评估”。更准确的状态是：**通信算子在 isolated probe 中可用，但接入真实 EPFFN 后 combine_v2 对非零 expert output 不稳定**。

## 当前 blocked 项

| 项 | 状态 | 阻塞原因 | 解锁条件 |
|---|---|---|---|
| `bench-moe-dispatch-real-routing` | blocked | torch_npu 2.6 环境不支持可用 v2；2.9 真实路径 combine_v2 不稳定。 | 真实 Qwen nonzero expert output + combine_v2 smoke 通过。 |
| `run-epffn-v2-decode-dbo` | blocked | `npu_moe_v2` 未通过 b1/s32/t2 real smoke。 | 至少通过 b1/s32/t2、b2/s64/t4，再进入 TPOT 矩阵。 |
| `plot-epffn-v2-pipelines` | blocked | 没有稳定 decode-DBO timing JSON。 | decode-DBO 跑通并生成 attention/ffn timing。 |
| `evaluate-moe-distribute-backend` | blocked | 不能在真实路径比较 TPOT 或 pipeline overlap。 | correctness + stability gate 通过。 |
| `same-env-moe-comm-compare` | pending | 可做纯通信 microbench，但不能代表真实 EPFFN 收益。 | 放到低优先级，作为后续性能参考。 |

## 待做工作

### P0：最小复现与 bug 边界

1. 写一个更小的 sequential reproducer：
   - 同一进程内先跑 prefill-like tokens；
   - 再跑 decode-like tokens=1；
   - 使用真实 Qwen expert weights 或等价 expert GEMM；
   - 不启动完整 `src.main`，只保留 dispatch -> expert GEMM -> combine。
2. 尝试在 combine 前执行：
   - `torch_npu.npu_format_cast(output, torch_npu.Format.ND)`；
   - 或其他可查询到的 base-format materialize。
3. 固定更小变量：
   - layer0 only；
   - top_k=1 vs top_k=8；
   - single active expert vs random routing；
   - no padding vs padding-heavy `expert_token_nums`。
4. 如果仍失败，整理给 Ascend/torch_npu 的最小 bug report：
   - 环境版本；
   - schema；
   - 最小脚本；
   - NPU 错误日志；
   - `combine_v2` 输入 tensor shape/dtype/format。

### P1：代码清理边界

1. 保留 `--ffn-ep-backend npu_moe_v2` 为实验 backend，但文档和错误信息都要强调不稳定。
2. 诊断 env 继续加 `AFD_NPU_MOE_V2_*` 前缀，避免被误认为稳定用户接口。
3. `AFD_DEBUG_MAX_LAYERS` 是临时 debug 开关；合入前需要决定保留、重命名为正式 debug，或移除。
4. 不把 `npu_moe_v2` 加入默认矩阵实验，避免误导结果。

### P2：继续 FFN 优化的实际方向

在 `npu_moe_v2` 解锁前，FFN 优化应继续集中在现有真实路径：

1. `src/model/ep_moe.py`
   - 优化 `ShardedExperts.forward_local()` 的 per-expert Python loop、小 GEMM、host sync 和 `index_add_`。
   - 评估 grouped GEMM / batched GEMM，但必须用 Qwen3 shape 和 NPU 结果验证。
2. `EPFFNLayer` broadcast/reduce path
   - 继续优化 overlap、payload、通信等待隐藏。
   - 保持 layer-major、micro-batch-major collective 顺序。
3. `src/pipeline/decode_scheduler.py`
   - 分析 FFN compute、A2F/F2A wait、reduce wait 的 pipeline bubble。
4. `src/coordinator_arch/comm/fallback_a2a.py`
   - 可作为 future token-aware all-to-all 方向，但当前不是 `src.main` 静态 decode 的真实 FFN compute 路径。

## 当前可引用命令清单

### torch_npu 2.6 base/v2 probe

```bash
ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --nproc_per_node=8 scripts/bench_moe_dispatch_npu.py \
  --backend broadcast_reduce \
  --tokens-list 4,16,64 --hidden 2048 --experts 128 --top-k 8 \
  --output results_npu/ffn_moe_dispatch_bench/broadcast_reduce_ep8_h2048.json
```

```bash
ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --nproc_per_node=8 scripts/bench_moe_dispatch_npu.py \
  --backend npu_moe_distribute --tokens-list 4 \
  --hidden 2048 --experts 128 --top-k 8 \
  --output results_npu/ffn_moe_dispatch_bench/official_base_ep8_h2048_fail.json
```

```bash
ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --nproc_per_node=8 scripts/bench_moe_dispatch_npu.py \
  --backend npu_moe_distribute_v2 --tokens-list 4 \
  --hidden 2048 --experts 128 --top-k 8 \
  --output results_npu/ffn_moe_dispatch_bench/official_v2_ep8_h2048_fail.json
```

### torch_npu 2.9 v2 fullmesh_v2 probe

```bash
ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --nproc_per_node=8 scripts/bench_moe_dispatch_npu.py \
  --backend npu_moe_distribute_v2 --tokens-list 4 \
  --hidden 2048 --experts 128 --top-k 8 \
  --comm-alg fullmesh_v2 \
  --output results_npu/moe_version_probe/bench_script_v2_h2048_fullmesh_v2_warm.json
```

### 真实 Qwen `npu_moe_v2` 失败/对照

失败：

```bash
MASTER_PORT=29918 ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 \
AFD_DIST_TIMEOUT_SEC=7200 AFD_DEBUG_MAX_LAYERS=2 \
AFD_NPU_MOE_V2_TRACE=1 AFD_NPU_MOE_V2_FORCE_SYNC=1 \
bash scripts/run_npu.sh --attn-size 1 --ffn-size 8 --ffn-tp-size 1 \
  --ffn-ep-size 8 --ffn-ep-backend npu_moe_v2 \
  --batch 1 --seq 32 --tokens 2 --num-micro-batches 1 \
  --model-name /models/Qwen3-30B-A3B
```

zero-experts 对照：

```bash
MASTER_PORT=29915 ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 \
AFD_DIST_TIMEOUT_SEC=7200 AFD_DEBUG_MAX_LAYERS=2 \
AFD_NPU_MOE_V2_TRACE=1 AFD_NPU_MOE_V2_FORCE_SYNC=1 \
AFD_NPU_MOE_V2_ZERO_EXPERTS=1 \
bash scripts/run_npu.sh --attn-size 1 --ffn-size 8 --ffn-tp-size 1 \
  --ffn-ep-size 8 --ffn-ep-backend npu_moe_v2 \
  --batch 1 --seq 32 --tokens 2 --num-micro-batches 1 \
  --model-name /models/Qwen3-30B-A3B
```

## 后续决策门槛

只有同时满足以下条件，才应继续把 `npu_moe_v2` 推向 decode-DBO 性能实验：

1. real Qwen `b1/s32/t2` 非零 expert output 通过。
2. real Qwen `b2/s64/t4` 或同级 smoke 通过。
3. `npu_moe_v2` 输出与 broadcast/reduce reference 数值误差可接受。
4. 至少一个小矩阵的 `decode_tpot_ms` 不明显回退。
5. pipeline 图中 dispatch/combine 或 local expert compute 有可解释收益。

在这些条件满足前，`npu_moe_v2` 的状态应保持为：

```text
experimental / blocked for real decode path
```
