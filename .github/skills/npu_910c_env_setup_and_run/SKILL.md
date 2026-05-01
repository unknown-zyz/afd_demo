---
name: npu_910c_env_setup_and_run
description: |
  连接远程 NPU-910C 服务器，并在长期 Docker 容器 afd-npu-test 内运行
  AFD-DBO NPU/HCCL 实验。触发场景：用户要求在 NPU/910C 上跑实验、连接
  NPU 环境、运行 NPU matrix、收集 results_npu 等。
---

# NPU-910C 远程实验 Skill

## 固定环境

| 项 | 值 |
|---|---|
| SSH | `ssh schedTeam@1.95.114.229 -p 22 -i ~/.ssh/id_rsa_second` |
| 远端宿主目录 | `/home/schedTeam/zhangyz` |
| 长期容器 | `afd-npu-test` |
| 容器工作目录 | `/workspace/afd_demo` |
| 模型路径 | `/models/Qwen3-30B-A3B` |
| Python / NPU 栈 | torch 2.6.0 + torch_npu 2.6.0 + CANN 8.5 |

## 红线

- 必须在 `afd-npu-test` 容器内运行 NPU 计算，禁止宿主机直接跑。
- 不要删除、重建或停止 `afd-npu-test`，除非用户明确要求。
- 不要移动、覆盖或重新下载模型权重。
- 当前验证拓扑只使用 `--attn-size 1 --ffn-size 1 --ffn-tp-size 1`；不要用旧 4-rank preset。
- 不要用 `pkill` / `killall`，需要清理时只 `kill <PID>`。

## 标准连接

```bash
ssh -p 22 -i ~/.ssh/id_rsa_second schedTeam@1.95.114.229 \
  "docker exec afd-npu-test bash -lc '
     source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
     cd /workspace/afd_demo
     git status --short --branch
  '"
```

如需同步最新 `npu` 分支：

```bash
ssh -p 22 -i ~/.ssh/id_rsa_second schedTeam@1.95.114.229 \
  "docker exec afd-npu-test bash -lc '
     source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
     cd /workspace/afd_demo
     git fetch origin
     git checkout npu
     git merge --ff-only origin/npu
  '"
```

## 冒烟测试

```bash
ssh -p 22 -i ~/.ssh/id_rsa_second schedTeam@1.95.114.229 \
  "docker exec afd-npu-test bash -lc '
     source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
     cd /workspace/afd_demo
     ASCEND_VISIBLE_DEVICES=0,1 bash scripts/run_npu.sh \
       --attn-size 1 --ffn-size 1 --ffn-tp-size 1 \
       --batch 1 --seq 32 --tokens 4 \
       --model-name /models/Qwen3-30B-A3B
  '"
```

## 矩阵实验

```bash
ssh -p 22 -i ~/.ssh/id_rsa_second schedTeam@1.95.114.229 \
  "docker exec afd-npu-test bash -lc '
     source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
     cd /workspace/afd_demo
     ./scripts/run_experiment_matrix_npu.sh \
       --modes serial,prefill-dbo,decode-dbo,decode-dbo-crosslayer \
       --batches 2,4,8,16,32,64,128,256 \
       --seqs 128,256,512 \
       --tokens 20 \
       --comm-timing-mode completion \
       --no-cache
  '"
```

## 通信 microbenchmark

```bash
ssh -p 22 -i ~/.ssh/id_rsa_second schedTeam@1.95.114.229 \
  "docker exec afd-npu-test bash -lc '
     source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
     cd /workspace/afd_demo
     ASCEND_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
       scripts/bench_comm_transfer.py \
       --backend npu --sizes-mib 0.004,0.031,1,16,32 \
       --warmup 5 --iters 50 --blocking \
       --output results_npu/comm_bench/npu_comm.json
  '"
```

## 结果收集

结果默认写入容器内仓库：

- `results_npu/`
- `results_npu/comm_bench/`
- `results/prefill_dbo/logs/`（`run_npu.sh` 中间日志）

需要拉回本地时优先只拉 JSON、CSV、Markdown、PNG，不复制模型或大 cache。
