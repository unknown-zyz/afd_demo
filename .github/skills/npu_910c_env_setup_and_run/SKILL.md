# NPU-910C 远程实验 Skill

> 本 skill 描述唯一允许的 NPU-910C 远程实验流程。不要使用 GPU 多机地址
> `zyz@192.168.5.32:31310` 作为 NPU 入口。
>
> 现有两台 910C 主机：**Host1**（默认，环境已就绪）与 **Host2**（备用，跳板路径已通，
> 但容器/模型尚未就绪）。当 Host1 共享 chip 被其他租户挤占无法跑实验时，
> 按下方"切换决策树"评估是否切到 Host2。

## 固定环境

### Host1（主机，afd-npu-test 已就绪）

| 项 | 值 |
|---|---|
| SSH | `ssh schedTeam@1.95.114.229 -p 22 -i ~/.ssh/id_rsa_second` |
| 远端宿主目录 | `/home/schedTeam/zhangyz` |
| 模型权重宿主路径 | `/home/schedTeam/Qwen3-30B-A3B`，只读，禁止移动/覆盖/重新下载 |
| 首选长期容器 | `afd-npu-test`，已装 torch_npu 2.6.0 + transformers，已挂模型，禁止删除 |
| 备用容器 | `zhangyz-npu-1`，仅备用/CPU 冒烟，通常无模型挂载 |
| 容器工作目录 | `/workspace/afd_demo` |
| 容器模型路径 | `/models/Qwen3-30B-A3B` |
| CANN | `/usr/local/Ascend/` |

### Host2（备用，需经 Host1 跳板）

| 项 | 值 |
|---|---|
| 跳板 SSH | 先登 Host1，然后 `sudo ssh -i /root/ssh_key/KeyPair-f1dd.pem root@192.168.0.192` |
| 主机名 | `liteserver-910c-2-00001.novalocal` (aarch64) |
| 用户 | `root`（通过 sudo，无密码 key） |
| **宿主工作目录** | **`/root/schedTeam/zhangyz/`**（与 Host1 `/home/schedTeam/zhangyz` 对称） |
| **Qwen3 权重** | **`/root/schedTeam/zhangyz/Qwen3-30B-A3B`（57G，2026-05-20 确认就位）** |
| 硬件 | 8×Ascend910（16 chips，目前几乎空闲，HBM ~3 GB/65 GB） |
| 既有容器（非本项目） | `GL-offload` / `odd` / `aiserver-agent`（vllm-ascend，torch_npu 2.9.0，**未挂 Qwen3-30B**） |
| afd-npu-test 等价容器 | **`afd-npu-test-h2` 已就绪**（2026-05-20 创建，image `deepep-ascend-bench-dev:latest`，privileged + host net，挂 16 颗 davinci + 模型 ro + `/root/schedTeam/zhangyz/workspace`，torch_npu 2.6.0、grpc 1.80.0、deep_ep 1.0.0 已装） |
| 容器工作目录 | `/workspace`（仓库 clone 到 `/workspace/afd_demo_repo`） |
| 模型路径（容器内） | `/models/Qwen3-30B-A3B` |
| CANN 路径 | 容器内 `/usr/local/Ascend/`（继承自 vllm-ascend 镜像） |

## 红线

- 必须在 `afd-npu-test` 容器内运行 NPU 计算，禁止宿主机直接跑。
- 不要删除、重建或停止 `afd-npu-test`，除非用户明确要求。
- 不要移动、覆盖或重新下载模型权重；**禁止在 Host1 与 Host2 之间擅自传输权重**（动权重需用户授权）。
- 不要触碰 Host2 上已有的 `GL-offload` / `odd` / `aiserver-agent` 容器（属于其他租户）。
- 当前验证拓扑只使用 `--attn-size 1 --ffn-size 1 --ffn-tp-size 1`；不要用旧 4-rank preset。
- 不要用 `pkill` / `killall`，需要清理时只 `kill <PID>`。

## 两台机器切换决策树

每次开始 NPU 实验时按以下顺序决定使用哪台主机：

1. **检查 Host1 HBM 占用**（共享 16 颗 chip）：

   ```bash
   ssh -p 22 -i ~/.ssh/id_rsa_second schedTeam@1.95.114.229 \
     "docker exec afd-npu-test bash -lc 'npu-smi info | grep -E \"/ 65536\"'"
   ```

   - 找空闲 HBM（`65536 - used`）。每个 attn/ffn rank 约需 12 GB 余量。
   - 至少 2 颗 chip 剩余 HBM > 25 GB → **走 Host1**（用既有命令）。
   - 仅小批量（b ≤ 4）短序列冒烟可以接受 >15 GB 余量。

2. **Host1 不满足且需要更多空闲算力** → 检查 Host2 状态：

   ```bash
   ssh -p 22 -i ~/.ssh/id_rsa_second schedTeam@1.95.114.229 \
     "sudo ssh -o StrictHostKeyChecking=no -i /root/ssh_key/KeyPair-f1dd.pem \
        root@192.168.0.192 'docker ps --format \"{{.Names}}\" | grep -E \"afd-npu|qwen\" || echo NO_AFD_CONTAINER'"
   ```

   - 输出 `NO_AFD_CONTAINER` → Host2 尚未就绪，**报告阻塞**，请用户决定是否准备 Host2 容器，**不要自己 docker run**。
   - 已存在 afd-npu 等价容器且挂了模型 → 用下面 "Host2 接入示例" 模板替换 Host1 命令。

3. **两台都不可用** → 总结当前 HBM 占用和阻塞原因，等待租户释放或用户介入；**绝不重启他人容器**。

## 标准连接（Host1）

> **新 contributor 必读**：DeepEP-Ascend 实验前**必须** source 自定义 OPP vendor 环境，
> 否则 `aclnnDispatchLayout` 等自定义 op 找不到（详见 `results_npu/coordinator_arch/d3_lowlatency/README.md`）。
> 标准的连接命令应包含两条 source：
> ```bash
> source /usr/local/Ascend/ascend-toolkit/set_env.sh
> source /usr/local/Ascend/cann-8.5.0/opp/vendors/hwcomputing/bin/set_env.bash
> ```
> 若 `vendors/hwcomputing/` 缺失，需先跑安装命令（见下方"DeepEP 自定义 OPP 安装"小节）。

```bash
ssh -p 22 -i ~/.ssh/id_rsa_second schedTeam@1.95.114.229 \
  "docker exec afd-npu-test bash -lc '
     source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
     source /usr/local/Ascend/cann-8.5.0/opp/vendors/hwcomputing/bin/set_env.bash 2>/dev/null
     cd /workspace/afd_demo
     git status --short --branch
  '"
```

### DeepEP 自定义 OPP 安装（首次配新容器或重装 CANN 后必跑）

**Host1**（有 SGLang-Kernel-NPU 源码）：

```bash
bash /workspace/sglang-kernel-npu/csrc/deepep/ops/build_out/custom_opp_ubuntu_aarch64.run \
     --quiet --install-path=/usr/local/Ascend/cann-8.5.0/opp
```

**Host2**（无源码 — 从 site-packages cp vendor 目录）：

```bash
DEEPEP_DIR=/usr/local/python3.11.14/lib/python3.11/site-packages/deep_ep/vendors/hwcomputing
VENDORS=/usr/local/Ascend/cann-8.5.0/opp/vendors
mkdir -p "$VENDORS" && cp -a "$DEEPEP_DIR" "$VENDORS/"
cat > "$VENDORS/hwcomputing/bin/set_env.bash" <<'EOF'
#!/bin/bash
export ASCEND_CUSTOM_OPP_PATH=/usr/local/Ascend/cann-8.5.0/opp/vendors/hwcomputing:${ASCEND_CUSTOM_OPP_PATH}
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-8.5.0/opp/vendors/hwcomputing/op_api/lib/:${LD_LIBRARY_PATH}
EOF
```

验证：`nm -D /usr/local/Ascend/cann-8.5.0/opp/vendors/hwcomputing/op_api/lib/libcust_opapi.so | grep aclnnDispatchLayout` 应输出 `T aclnnDispatchLayout`。

如需同步最新分支：

```bash
ssh -p 22 -i ~/.ssh/id_rsa_second schedTeam@1.95.114.229 \
  "docker exec afd-npu-test bash -lc '
     source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
     cd /workspace/afd_demo
     git fetch origin
     git checkout main
     git merge --ff-only origin/main
  '"
```

## Host2 接入示例（容器/模型就绪后才可用）

> **占位**：以下命令在 Host2 上 afd-npu 等价容器（暂称 `<HOST2_CONTAINER>`）就绪、
> Qwen3-30B 权重挂到容器 `/models/Qwen3-30B-A3B` 之后才可执行。当前 Host2
> 容器名 / 模型路径未定，需要用户在准备好后回填。

```bash
ssh -p 22 -i ~/.ssh/id_rsa_second schedTeam@1.95.114.229 \
  "sudo ssh -o StrictHostKeyChecking=no -i /root/ssh_key/KeyPair-f1dd.pem root@192.168.0.192 '
     docker exec <HOST2_CONTAINER> bash -lc \"
       source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
       cd /workspace/afd_demo
       ASCEND_VISIBLE_DEVICES=0,1 bash scripts/run_npu.sh \
         --attn-size 1 --ffn-size 1 --ffn-tp-size 1 \
         --batch 1 --seq 32 --tokens 4 \
         --model-name /models/Qwen3-30B-A3B
     \"
  '"
```

Host2 准备清单（待用户/后续会话执行）：

1. scp/rsync `Qwen3-30B-A3B` 权重到 Host2 宿主，约 60 GB。
2. 在 Host2 上 `docker run` 一个 privileged 容器，挂 `davinci` 设备 + 权重 ro，
   参考 `doc/10-npu-910c-container-deployment.md`。容器名建议 `afd-npu-test`
   保持与 Host1 一致；若不行用 `afd-npu-test-h2`。
3. 容器内 `pip install torch_npu`（Host2 现有容器是 torch_npu 2.9.0，
   本项目验证拓扑用 2.6.0，二者均可，但若用 2.9.0 须先重跑冒烟回归）。
4. clone 本仓库到容器 `/workspace/afd_demo`，`git checkout main`。
5. 冒烟跑通后回填本 skill 的 `<HOST2_CONTAINER>` 占位。

## 冒烟测试（Host1）

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

## 矩阵实验（Host1）

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

## 通信 microbenchmark（Host1）

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

## 失败排查

- 失败时收集命令退出码、相关日志末尾 100 行、`npu-smi info`。
- 若单侧 OOM 导致 peer 挂住，只能按 PID `kill <PID>`。
- CANN 8.5 必须使用 torch_npu >= 2.6.0；不要切回 2.5.1。
- 全部 chip 都被其他租户占满 → 切 Host2 决策树第 2 步；若 Host2 未就绪则报告阻塞。
