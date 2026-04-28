# NPU-910C 远程测试报告 — `feat/npu-910c` 分支

## 摘要

在远程 NPU-910C（16 × Ascend910）真实硬件上验证了 `feat/npu-910c` 分支的端到端可运行性。

| 项 | 结果 |
|---|---|
| 容器/驱动/HCCL 环境 | ✅ 跑通 |
| 单进程 NPU import / matmul | ✅ |
| 2-rank serial prefill (48 层) | ✅ |
| 2-rank prefill **DBO**（micro-batch 重叠） | ✅ |
| 2-rank decode generate（KV cache + 自回归） | ✅ |
| 4-rank `npu-4card` preset（attn=2, ffn=2, ffn_tp=2） | ❌ OOM（设备映射 bug） |

**关键发现**：当 `torch_npu==2.6.0`（而非 2.5.1）时，分支提供的 attention/FFN 路径在 NPU 上**首次端到端跑通**，包括 MoE FFN、HCCL P2P、自回归 generate。4 卡 preset 仍存在历史已知的 role/device 映射缺陷。

---

## 1. 环境快照

| 项 | 值 |
|---|---|
| 主机 | `schedTeam@1.95.114.229` (aarch64, kernel 5.10) |
| NPUs | 16 × Ascend910 (8 板 × 2 chip), 每 chip 65 GB HBM |
| Driver / 固件 | 25.2.1 (`/usr/local/Ascend/driver`) |
| 镜像 | `deepep-ascend-bench-dev` (CANN 8.5.0, Python 3.11.14) |
| 容器 | **`afd-npu-test`**（**长驻，不删**，按用户要求复用） |
| 模型 | `/home/schedTeam/Qwen3-30B-A3B`（host ro 挂载到 `/models/`） |

### 容器创建参数（关键，缺一不可）

```bash
docker run -d --name afd-npu-test --privileged --network host --ipc host --shm-size 16g \
  --device /dev/davinci0 --device /dev/davinci1 \
  --device /dev/davinci2 --device /dev/davinci3 \
  --device /dev/davinci_manager --device /dev/hisi_hdc \
  --device /dev/devmm_svm \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /home/schedTeam/Qwen3-30B-A3B:/models/Qwen3-30B-A3B:ro \
  deepep-ascend-bench-dev sleep infinity
```

> 没加 `--privileged --network host` 时，`torch_npu.npu.is_available()` 报 `drvGetDevNum failed errCode=87`。

### 安装链（容器内一次性）

```bash
pip3 config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
pip3 install numpy pyyaml decorator attrs psutil scipy        # CANN TBE 必需
pip3 install torch==2.6.0 torch_npu==2.6.0                     # ⚠ 2.5.1 会缺 kernel 二进制
pip3 install transformers accelerate sentencepiece protobuf
source /usr/local/Ascend/ascend-toolkit/set_env.sh             # 每次新 shell 都要执行
```

### HBM 隔离悖论（备注）
`npu-smi info` 在宿主显示每 chip 已用 56 GB，但 privileged 容器内 `torch.npu.mem_get_info(0)` 报 free=62 GB。结论：宿主 npu-smi 是全局视角，容器内通过 cgroup 隔离实际可用 ~62 GB/chip，足够装 30B MoE。

---

## 2. 远程操作流水（按时间）

```bash
# 1. 推送代码
git bundle create /tmp/afd_npu.bundle feat/npu-910c
scp -P 22 -i ~/.ssh/id_rsa_second /tmp/afd_npu.bundle schedTeam@1.95.114.229:/home/schedTeam/zhangyz/

# 2. 起容器（见上）
ssh ... "docker cp /home/schedTeam/zhangyz/afd_npu.bundle afd-npu-test:/tmp/"
ssh ... "docker exec afd-npu-test bash -lc 'git clone /tmp/afd_npu.bundle /workspace/afd_demo \
                                            && cd /workspace/afd_demo && git checkout feat/npu-910c'"

# 3. 装 torch_npu 2.6.0 + transformers
ssh ... "docker exec afd-npu-test pip3 install torch==2.6.0 torch_npu==2.6.0 transformers accelerate ..."

# 4. 校验 NPU import
ssh ... "docker exec afd-npu-test python3 -c \
  'import torch, torch_npu; print(torch_npu.npu.is_available(), torch_npu.npu.device_count())'"
# → True 16

# 5. 测试矩阵（C2-C7，见下）
ssh ... "docker exec afd-npu-test bash -lc 'cd /workspace/afd_demo && \
   ASCEND_VISIBLE_DEVICES=0,1 ./scripts/run_npu.sh ...'"
```

完整脚本见容器内 `/workspace/afd_demo/results/logs/npu_*.log`。

---

## 3. 测试矩阵

| # | 用例 | 配置 | 结果 |
|---|---|---|---|
| C1 | CPU smoke | `tests/test_cpu_smoke.py` | ✅（之前在 zhangyz-npu-1 已 PASS）|
| C2 | NPU import + matmul | bf16/fp16, npu:0 | ✅ avail=True, count=16 |
| C3 | hello-forward, `init_backend(npu)` | b=1, s=16, single layer | ✅ logits shape OK |
| C4 | **2-rank serial prefill + 1-token gen** | attn=1 ffn=1, b=1 s=32 | ✅ 1441.6 ms, 输出 `'Human'` |
| C5 | **2-rank prefill DBO** | b=4 s=64 (2 micro-batch) | ✅ 2137.1 ms, compute=909 ms (42.5%) |
| C6 | **2-rank decode (autoregressive)** | b=1 s=32 tokens=4, KV cache | ✅ "Hello…" → "I want to talk", 2.1 tok/s |
| C7 | **4-rank `npu-4card` preset** | attn=2 ffn=2 ffn_tp=2, b=2 s=64 | ❌ OOM @ npu:1 (4 ranks 共抢同一 device) |

### C4-C6 关键日志（节选）

```
# C4 serial
[ATTENTION] model_type=qwen3_moe, moe=True, router=True, moe_timing=True
Prefill seq_len set to 32
Output[0]: 'Hello, how are you today?' → 'Human'

# C5 DBO
=== ATTENTION 节点 ===
端到端: 2137.11ms
计算: 909.04ms (42.5%)
等待(recv): 1482.63ms

# C6 generate
KV Cache initialized (DynamicCache)
Generated 4 tokens in 1886.89ms (2.1 tok/s)
=== Generated Text ===
Hello, how are you today? I want to talk
======================
```

### C7 失败堆栈（关键行）

```
RuntimeError: NPU out of memory. Tried to allocate 386.00 MiB
(NPU 1; 61.28 GiB total capacity; 40.14 GiB already allocated; ...)
```

4 个 rank 全部把权重 `.to('npu:1')` —— **`AttentionWorker` 把 layer 0..47 平铺到 `npu:0..15`，每个 rank 都做同样的事**，于是 4 个进程在每个 device 上各放一份完整切片，第 4 个 rank OOM。

---

## 4. Bug 清单（按优先级）

### Bug 1 — 4-rank role 映射缺陷（**阻塞 4 卡 preset**）
- **位置**：`src/distributed/__init__.py:160-167`（`role` property）
- **现象**：`role` 只判断 `rank == attn_node_rank` 或 `== ffn_node_rank`；中间 rank 1/3 → `"unknown"`；并且 `AttentionWorker` 用 `layer_devices = ['npu:0'..'npu:15']`（不分 rank），4 rank 同时往同一 device 装权重 → OOM。
- **建议修复**：
  1. 让 `role` 接受连续 rank 段（`attn_node_rank` 到 `attn_node_rank+attn_size-1` 全是 attention）；
  2. `layer_devices` 按 `(global_rank, local_size)` 切片，每 rank 只持自己负责的 device 子集。

### Bug 2 — `torch_npu==2.5.1` kernel 二进制缺失
- **现象**：`BinaryGetFunction failed for Index_*_kernel0` / `aclnnArange` / `aclnnNeg`，`retCode=0x7080001`。
- **根因**：CANN 8.5 镜像里的 TBE 二进制缓存与 torch_npu 2.5.1 不匹配。
- **修复**：升级到 `torch_npu==2.6.0`（已在 `afd-npu-test` 容器中固化）。**应在 `requirements_npu.txt` 与 skill 中明确版本下限 ≥ 2.6.0**。

### Bug 3 — `aclnnNeg` 在 DBO 路径的 `.to(non_blocking=True)`
- 之前用 2.5.1 时复现于 `attention_worker.py:149`；切到 2.6.0 后**自动消失**（同根因，归为 Bug 2 的衍生症状）。

---

## 5. 资源清理记录

| 项 | 处理 |
|---|---|
| `/tmp/afd_npu.bundle`（本地） | 已删 |
| `/home/schedTeam/zhangyz/afd_npu.bundle`（远端宿主） | 已删 |
| 容器 `afd-npu-test` | **保留**（按用户要求长驻复用，已写入 skill）|
| 容器 `zhangyz-npu-1` / `deepep-ascend-benchmark-npu-1` | 未触碰 |
| 镜像、模型权重 `/home/schedTeam/Qwen3-30B-A3B` | 未触碰 |
| 其他租户的 NPU 进程 | 未受影响（仅占用 davinci0..3） |

---

## 6. 后续建议（不在本次 scope）

1. 修 Bug 1（4-rank role 与 device 切分），即可复现 `npu-4card` preset 的设计目标。
2. 把 `requirements_npu.txt` 钉到 `torch==2.6.0, torch_npu==2.6.0`。
3. `scripts/run_npu.sh` 默认导出 `ASCEND_VISIBLE_DEVICES`、`source set_env.sh`。
4. 加一行 `npu-smi info -t board -i ...` 到启动脚本前置检查（避免 OOM 时排查浪费时间）。
