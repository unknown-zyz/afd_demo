# 10. 910C 远程容器与环境部署指南

本文档说明如何在远程 910C 机器上创建 / 进入 NPU Docker 容器，部署本仓库所需运行环境，并执行第一次冒烟测试。其设计目的是让任何新接手的实验人员在不破坏现有长期容器的前提下，快速复现 NPU 实验。

> 配套读物：
> - `doc/04-deployment.md` —— 通用部署（GPU/NPU 共用）
> - `doc/06-npu-910c-adaptation.md` —— 910C / HCCL 适配技术细节
> - `.github/skills/npu_910c_env_setup_and_run/SKILL.md` —— Agent 流程契约（与本文一致）

## 1. 远程主机与凭据

| 项 | 值 |
|---|---|
| 主机 | `1.95.114.229` |
| SSH 用户 | `schedTeam` |
| SSH 端口 | `22` |
| SSH 私钥 | `~/.ssh/id_rsa_second` |
| 宿主工作目录 | `/home/schedTeam/zhangyz` |
| 模型权重宿主路径 | `/home/schedTeam/Qwen3-30B-A3B`（只读，禁止移动 / 覆盖 / 重下载） |
| compose 文件 | `/home/schedTeam/zhangyz/docker-compose.yml` |

标准登录：

```bash
ssh -p 22 -i ~/.ssh/id_rsa_second schedTeam@1.95.114.229
```

## 2. 推荐容器：`afd-npu-test`

**默认请使用已在运行的 `afd-npu-test` 容器，不要删除、重建或停止它。**
该容器已预装 `torch_npu 2.6.0` + `transformers`，并挂载 Qwen3-30B-A3B 权重；多数实验只需 `docker exec` 进入即可。

容器关键属性（在 compose 中固化）：

| 属性 | 值 |
|---|---|
| 名称 | `afd-npu-test` |
| 镜像基线 | CANN 8.5 + Python 3.11 |
| 网络模式 | `host` |
| 特权 | `privileged: true`（必须，否则 NPU 初始化报 `Resource_Busy` / drv ret=87） |
| NPU 设备 | `/dev/davinci0..15` + `/dev/davinci_manager` + `/dev/devmm_svm` + `/dev/hisi_hdc` |
| 卷挂载 | `/home/schedTeam/Qwen3-30B-A3B → /models/Qwen3-30B-A3B (ro)`，`/usr/local/Ascend → /usr/local/Ascend (ro)`，工作区双向挂载到 `/workspace` |
| 工作目录 | `/workspace/afd_demo`（已 clone，git origin 指向本地 bundle） |
| 模型路径（容器内） | `/models/Qwen3-30B-A3B` |
| CANN 根 | `/usr/local/Ascend/` |

确认容器状态：

```bash
ssh -p 22 -i ~/.ssh/id_rsa_second schedTeam@1.95.114.229 \
  'docker ps --format "{{.Names}}\t{{.Status}}" | grep afd-npu-test'
```

进入：

```bash
ssh -p 22 -i ~/.ssh/id_rsa_second schedTeam@1.95.114.229 \
  'docker exec -it afd-npu-test bash'
```

备用容器：`zhangyz-npu-1`，仅用于无模型挂载的 CPU 冒烟，不建议作为日常实验入口。

## 3. 仅当 `afd-npu-test` 缺失时：重新创建

只有在该容器**确实不存在**（`docker ps -a` 也查不到）时，才执行下列操作。**禁止 `docker rm` 已有同名容器**。

```bash
ssh -p 22 -i ~/.ssh/id_rsa_second schedTeam@1.95.114.229 \
  'cd /home/schedTeam/zhangyz && docker compose -f docker-compose.yml up -d afd-npu-test'
```

`docker-compose.yml` 中 `afd-npu-test` 服务的必备字段（节选示意，以宿主上实际文件为准）：

```yaml
services:
  afd-npu-test:
    image: ascendhub.huawei.com/public-ascendhub/ascend-mindspore:8.0.0-cann8.5-py311-ubuntu22.04  # 或同等 CANN 8.5 镜像
    container_name: afd-npu-test
    network_mode: host
    privileged: true
    restart: unless-stopped
    devices:
      - /dev/davinci0
      - /dev/davinci1
      - /dev/davinci2
      - /dev/davinci3
      - /dev/davinci4
      - /dev/davinci5
      - /dev/davinci6
      - /dev/davinci7
      - /dev/davinci8
      - /dev/davinci9
      - /dev/davinci10
      - /dev/davinci11
      - /dev/davinci12
      - /dev/davinci13
      - /dev/davinci14
      - /dev/davinci15
      - /dev/davinci_manager
      - /dev/devmm_svm
      - /dev/hisi_hdc
    volumes:
      - /usr/local/Ascend:/usr/local/Ascend:ro
      - /home/schedTeam/Qwen3-30B-A3B:/models/Qwen3-30B-A3B:ro
      - /home/schedTeam/zhangyz/workspace:/workspace
    working_dir: /workspace
    command: sleep infinity
```

> 注意：必须 `privileged: true`，否则 NPU 初始化将报 `drvGetPlatformInfo ret=87 / Resource_Busy`。

## 4. 容器内环境校验

```bash
docker exec afd-npu-test bash -lc '
  source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
  python3 -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__, torch_npu.npu.is_available())"
  npu-smi info | head -20
'
```

预期：

```
2.6.0+cpu 2.6.0 True
```

> CANN 8.5 必须配 `torch_npu >= 2.6.0`；不要切回 `2.5.1`（缺 TBE kernel binary，运行 `Index_*_kernel0` / `aclnnNeg` 时会崩）。

## 5. 同步本仓库代码

容器内 `/workspace/afd_demo` 的 `git origin` 指向本地 bundle，**不能直接 `git fetch github` 拿到新分支**。推荐用源码包工作流：

宿主端（开发机本地）：

```bash
# 在本地 main 或目标分支上
git archive --format=tar.gz -o /tmp/afd_src.tgz HEAD
scp -P 22 -i ~/.ssh/id_rsa_second /tmp/afd_src.tgz \
  schedTeam@1.95.114.229:/tmp/afd_src.tgz
```

远端：

```bash
ssh -p 22 -i ~/.ssh/id_rsa_second schedTeam@1.95.114.229 \
  'docker cp /tmp/afd_src.tgz afd-npu-test:/tmp/afd_src.tgz && \
   docker exec afd-npu-test bash -lc "
     mkdir -p /workspace/afd_demo_$(date +%s)
     cd /workspace/afd_demo_$(date +%s) || exit 1
     tar -xzf /tmp/afd_src.tgz
     ls src scripts | head
   "'
```

> 用独立时间戳目录，**不要覆盖 `/workspace/afd_demo`**（里面可能有别人未提交的实验产物）。

如果一定要在 `/workspace/afd_demo` 内同步分支，先 `git status` 确认干净，再让用户确认。

## 6. 冒烟测试

最小拓扑（attn=1 / ffn=1 / ffn_tp=1，**不要用旧 4-rank preset**）：

```bash
ssh -p 22 -i ~/.ssh/id_rsa_second schedTeam@1.95.114.229 \
  'docker exec afd-npu-test bash -lc "
     source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
     cd /workspace/afd_demo
     ASCEND_VISIBLE_DEVICES=0,1 bash scripts/run_npu.sh \
       --attn-size 1 --ffn-size 1 --ffn-tp-size 1 \
       --batch 1 --seq 32 --tokens 4 \
       --model-name /models/Qwen3-30B-A3B
   "'
```

预期：30~60s 完成；尾部出现 `Generated 4 tokens in <N>ms` 和 `Decode timing saved`。

## 7. 矩阵实验

完整 NPU EP7 矩阵：

```bash
ssh -p 22 -i ~/.ssh/id_rsa_second schedTeam@1.95.114.229 \
  'docker exec afd-npu-test bash -lc "
     source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
     cd /workspace/afd_demo
     ./scripts/run_experiment_matrix_npu.sh \
       --modes serial,prefill-dbo,decode-dbo,decode-dbo-crosslayer \
       --batches 2,4,8,16,32,64,128,256 \
       --seqs 128,256,512 \
       --tokens 20 \
       --comm-timing-mode completion \
       --preset npu-ep7 \
       --no-cache
   "'
```

长任务建议 `docker exec -d` 启动 + `nohup ... > /tmp/run.log 2>&1 &`，并隔一段时间用 `tail -f /tmp/run.log` 查看进度，避免被 SSH 断线（远端 22 端口偶尔会主动关闭长连接）拖死。

## 8. 红线与运维约束

- **禁止删除 / 重建 / 停止 `afd-npu-test`。**
- **禁止 `docker rm` 同名容器后重建**，除非用户明确批准。
- 不要 `pkill` / `killall`；只 `kill <PID>`。
- 不要移动 / 覆盖 / 重新下载 `/home/schedTeam/Qwen3-30B-A3B`。
- 必须 `privileged: true`，否则 NPU 初始化失败。
- CANN 8.5 + torch_npu 2.6.0 是已验证组合；不要回退 `torch_npu 2.5.1`。
- 当前唯一验证拓扑：`attn-size=1 / ffn-size=1 / ffn-tp-size=1`；4-rank preset 会在 layer-placement 时 OOM。
- HCCL 端口被旧 `python -m src.main` 进程占用时会报 `EJ0003`；用 `ps -eo pid,cmd | grep src.main` 找残留 PID 后逐个 `kill`。

## 9. 故障排查速查

| 症状 | 可能原因 | 处理 |
|---|---|---|
| `Resource_Busy` / `drvGetPlatformInfo ret=87` | 容器没有 `privileged: true` | 检查 compose；重建（征得同意） |
| `EJ0003 hccl bind` | 上一次 rank 残留 | `ps -eo pid,cmd | grep src.main`，逐个 `kill <PID>` |
| `Index_*_kernel0` / `aclnnNeg` 缺失 | torch_npu 2.5.1 与 CANN 8.5 不匹配 | 切回 `torch_npu==2.6.0` |
| `Unknown backend: 'npu'` | `src/utils/device.py` 缺 NPU 分支 | 确认你的代码包含 `_has_npu()` 与 `npu` 后端路径 |
| SSH 频繁 `Connection closed` | 远端 sshd 限速 / 长任务 | 用 `docker exec -d` + `nohup` 把任务后台化 |

## 10. 拉回实验结果

只拉小型产物：

```bash
scp -P 22 -i ~/.ssh/id_rsa_second -r \
  schedTeam@1.95.114.229:/home/schedTeam/zhangyz/workspace/afd_demo/results_npu/<exp_dir> \
  ./results_npu/<exp_dir>
```

或先在容器内：

```bash
docker exec afd-npu-test bash -lc \
  'cd /workspace/afd_demo && \
   tar --exclude="*.pt" --exclude="*.bin" -czf /tmp/results_npu_subset.tgz \
       results_npu/<exp_dir>'
docker cp afd-npu-test:/tmp/results_npu_subset.tgz /home/schedTeam/zhangyz/
```

再从开发机 `scp` 回来。**不要复制模型权重 / 大 cache。**
