# NPU EP4 broadcast_reduce_sync 同步版实验结果

本目录是 `1 Attention + 4 FFN EP ranks`、`broadcast_reduce_sync` 后端的同步版
EP4 负结果记录。它不是待删除临时目录，而是后续 `broadcast_reduce_overlap` 和
EP7 正收益探索的对照基线。

## 1. 分支与实现状态

- 分支：`exp/npu-ep-prototype`
- 当前拓扑：`1 Attention rank + 4 FFN EP ranks`
- EP 后端：`broadcast_reduce_sync`
- 专家分配：`round_robin`，Qwen3-30B-A3B 的 128 个 routed experts 按 `expert_id % ep_size` 分到 4 个 FFN rank，每个 rank 32 个 experts。

本轮实现的是 correctness-first 同步版 EP：

1. Attention rank 只和 FFN coordinator 通信。
2. FFN coordinator 做 post-attention norm/router。
3. coordinator 向 EP group broadcast hidden/router metadata。
4. 每个 FFN EP rank 只计算本地 experts 的 partial output。
5. EP group reduce SUM 到 coordinator。
6. coordinator 完成 residual 后发回 Attention rank。

## 2. 已修复的问题

- generation prefill/decode 的同步路径中，`ffn_expert` ranks 之前会错误等待 Attention P2P，导致 decode smoke 卡住；现在 expert-only ranks 只参与 FFN EP collectives。
- 多个 `ffn_expert` rank 的 timing 文件名现在包含 rank，避免互相覆盖。
- decode timing JSON 新增 EP 细分：
  - `total_ep_dispatch_ms`
  - `total_ep_local_experts_ms`
  - `total_ep_reduce_ms`
  - 对应 events：`ep_dispatch`、`ep_local_experts`、`ep_reduce`

## 3. 运行命令

远端环境：

```bash
ssh schedTeam@1.95.114.229 -p 22 -i ~/.ssh/id_rsa_second
docker exec -it afd-npu-test bash
cd /workspace/afd_demo_ep_prototype
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
```

smoke：

```bash
./scripts/run_npu.sh --preset npu-ep4 --batch 1 --seq 4 --tokens 2 --model-name /models/Qwen3-30B-A3B
./scripts/run_npu.sh --preset npu-ep4 --batch 2 --seq 8 --tokens 3 --model-name /models/Qwen3-30B-A3B
```

可比 t20 配置：

```bash
./scripts/run_npu.sh --preset npu-ep4 --batch 4 --seq 128 --tokens 20 --model-name /models/Qwen3-30B-A3B
./scripts/run_npu.sh --preset npu-ep4 --batch 8 --seq 512 --tokens 20 --model-name /models/Qwen3-30B-A3B
```

## 4. 结果汇总

`t3` 只用于 smoke，不用于性能结论。下表使用 `tokens=20`，并与既有 NPU 2-rank baseline 报告对比。

| 配置 | Serial TPOT (ms) | 2-rank Decode DBO TPOT (ms) | EP4 sync TPOT (ms) | EP4 vs Serial | EP4 vs 2-rank DBO |
|---|---:|---:|---:|---:|---:|
| b4/s128/t20 | 252.722 | 273.469 | 783.681 | 0.322x | 0.349x |
| b8/s512/t20 | 351.484 | 332.727 | 931.040 | 0.378x | 0.357x |

结论：同步 broadcast/reduce EP4 已跑通，但性能明显负优化，不能作为最终方案。
该负结果保留用于说明为什么后续进入 EP overlap 与 token-aware dispatch/combine。

## 5. 每层平均细分

单位为 `ms / layer / micro-batch`。

| 配置 | Attention compute | Attention F2A recv-wait | FFN wall compute | EP dispatch | EP local experts | EP reduce | FFN/Attention compute ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| b4/s128/t20 | 1.029 | 6.808 | 7.799 | 0.487 | 4.745 | 1.608 | 7.58x |
| b8/s512/t20 | 1.005 | 7.865 | 8.883 | 0.394 | 5.053 | 2.530 | 8.84x |

观察：

- EP4 没有让 FFN 接近 Attention；FFN wall compute 仍是 Attention compute 的 `7.6x~8.8x`。
- Attention 侧 F2A recv-wait 被放大到 `6.8~7.9 ms/layer/MB`，说明同步 EP 内部 fan-out/reduce 把 Attention rank 长时间阻塞在等待 FFN 返回。
- EP reduce 在 b8/s512 下达到 `2.53 ms/layer/MB`，比 b4/s128 的 `1.61 ms/layer/MB` 更高，说明当前 reduce 路径随 batch 放大明显。
- 当前实现中的 dispatch/reduce 是同步集体通信，还没有跨 micro-batch overlap，因此无法用下一 MB 的 local expert 计算覆盖上一 MB 的 reduce wait。

## 6. 图

- `fig_ep4_tpot_comparison.png`：Serial、旧 2-rank Decode DBO、EP4 sync 的 TPOT 对比。
- `fig_ep4_ffn_breakdown.png`：EP4 sync FFN coordinator 的 dispatch/local experts/reduce 每层细分。
- `fig_ep4_balance_wait.png`：Attention compute、FFN wall compute、Attention recv-wait 的平衡关系。

## 7. 下一步建议

先不要继续扩大到 EP8/EP15。原因是 EP4 同步版已经把 TPOT 拉低到旧 2-rank DBO 的约 `0.35x`，继续增加 EP degree 大概率只会进一步放大 collective fan-out 和 reduce 成本。

下一步应先实现 `broadcast_reduce_overlap`：

- 拆分 dispatch/reduce process group。
- decode 两个 micro-batch 采用 `dispatch MB0 -> dispatch MB1 -> compute MB0 -> reduce MB0 || compute MB1 -> reduce MB1`。
- timing 中验证 `ep_reduce_wait_ms` 是否被下一 MB 的 `ep_local_experts_ms` 覆盖。

只有 overlap 后仍无法把 FFN/Attention ratio 降到约 `1.5x~2x`，才进入 token-level dispatch/combine，避免 broadcast 全 hidden 到所有 expert ranks。
