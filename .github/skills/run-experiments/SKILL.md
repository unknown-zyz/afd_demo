---
name: run-experiments
description: |
  AFD-DBO 实验运行 skill。定义了在本项目里跑 prefill / decode DBO benchmarks
  的统一流程:配置矩阵、串行结果缓存、自动生成 markdown 报告。
  触发场景:用户要求"重跑实验""对比 serial vs DBO""跑 batch 扫频""生成实验报告"等。
---

# Run-Experiments Skill

## 目的

为 AFD + DBO 项目提供统一的实验流程,避免每次手动拼 `run_single.sh` 命令、
手动收集结果、手动整理报告。Serial baseline 缓存后可供后续 DBO 实验直接读取。

## 目录约定

| 路径                                   | 内容                                         |
|----------------------------------------|----------------------------------------------|
| `results/serial/`                      | Serial baseline timing + report              |
| `results/serial/cache/b<B>_s<S>_t<T>.json` | 缓存的 serial attention timing (供对比)  |
| `results/prefill-dbo/`                 | Prefill DBO timing + report                  |
| `results/decode-dbo/`                  | Decode DBO (crosslayer OFF) timing + report  |
| `results/decode-dbo-crosslayer/`       | Decode DBO (crosslayer ON) timing + report   |
| `results/experiment_matrix_summary.csv`| 最新矩阵运行的汇总                           |
| `results/archive/`                     | 历史实验存档 (只读,勿覆盖)                  |

文件名规范:`<mode>_b<B>_s<S>_t<T>.{json,md}`,mode ∈ {serial, prefill-dbo,
decode-dbo, decode-dbo-crosslayer}。

## 入口

```bash
# 全矩阵 (默认 modes=all, batches=2..64, seqs={128,256,512}, tokens=20)
./scripts/run_experiment_matrix.sh

# 子集示例
./scripts/run_experiment_matrix.sh --modes decode-dbo,decode-dbo-crosslayer \
    --batches 8,16,32,64 --seqs 128

# 强制重跑 serial (忽略 cache)
./scripts/run_experiment_matrix.sh --modes serial --no-cache

# 试运行 (只打印命令)
./scripts/run_experiment_matrix.sh --dry-run --batches 2,4 --seqs 128
```

单次运行仍可直接用 `scripts/run_single.sh`,脚本末尾会自动生成 markdown
报告(若 `results/serial/cache/` 中有同 config 的 baseline 会自动对比)。

## Modes 含义

| Mode                      | 命令语义                                           |
|---------------------------|----------------------------------------------------|
| `serial`                  | `--no-dbo --generate`  完整自回归生成,禁用 DBO   |
| `prefill-dbo`             | 默认 `--no-generate`, 仅 prefill,开启 prefill DBO |
| `decode-dbo`              | `--generate`,crosslayer OFF(默认)               |
| `decode-dbo-crosslayer`   | `--generate --crosslayer`,开启跨层流水线         |

## 报告内容

`scripts/gen_experiment_report.py` 每次运行都会生成 markdown,包含:

1. **Configuration** — mode / batch / seq / tokens / layers / micro-batches
2. **End-to-end decode timing (representative step)** — attention & FFN
   各自的 total/compute/recv-wait/MoE 分项
3. **Compared to serial baseline** — 若有缓存,自动 per-step 归一化后对比
   (serial 的 `total_time_ms` 是整次 gen 时间,需除以 `max_new_tokens`)
4. **Per-layer breakdown** — L0..L<N-1> × {Attention, A2F, FFN, F2A,
   F2A recv-wait} 的 mean/min/max。L0 在求和行被排除(流水线 warmup)

## OOM / 失败处理

- `run_single.sh` 返回码 `2` 表示 OOM(通过日志 grep 检测)
- 矩阵脚本遇到 OOM 时,**停止该 (mode, seq) 的更大 batch**,继续下一对
- 失败行在 `summary.csv` 里标记 `FAIL`,不中断整个矩阵

## 关键环境

- 本机 4×V100-32GB: GPU 0,1 → attention, GPU 2,3 → FFN
- GPU 多机: 另一机 `ssh zyz@192.168.5.32 -p 31310 -i ~/.ssh/id_rsa_second`,
  用 `--deployment multinode` 触发；这不是 NPU 910C 入口
- NPU 910C: 使用 `ssh schedTeam@1.95.114.229 -p 22 -i ~/.ssh/id_rsa_second`
  进入长期容器 `afd-npu-test`，详见 `npu_910c_env_setup_and_run` skill
- Qwen3-30B-A3B 在 V100 上须 asymmetric split(21/27 层),这是模型代码
  里固定的
- `NCCL_BUFFSIZE=33554432` 由 `run_single.sh` 设置,避免 A2F 发送时的
  flow-control 阻塞;修改 batch/seq 远大于 256 时需确认是否仍足够

## 常见检查清单

运行完矩阵后先看三件事:
1. `results/experiment_matrix_summary.csv` — 总览 ok/OOM/FAIL
2. 对每个关键 config 打开 `report_<mode>_<cfg>.md` 看 speedup 是否合理
3. 若 DBO 慢于 serial:检查 report 里 `Recv wait` 列,过大多半是 NCCL
   冷启动(加 `--warmup-p2p`)或跨层/非跨层 irecv 时机问题

## 相关 skill

- `testing-workflow` — 更底层的分支管理/资源监控/对比流程,长期跨实验会话
  使用;`run-experiments` 只关注"跑出一批结果 + 生成报告"本身
