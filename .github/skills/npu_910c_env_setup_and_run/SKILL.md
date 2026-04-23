---
name: npu_910c_env_setup_and_run
description: |
  连接远程 NPU-910C 服务器，并在已有 Docker 容器内完成实验运行的标准流程。
  触发场景：用户要求"在 NPU 上跑实验"、"连远程 910C"、"在 zhangyz-npu-1 容器里跑"、
  "NPU 冒烟测试"、"adapter NPU 验证"等。
allowed-tools: shell
---

# NPU-910C 远程实验 Skill

> 本 skill 描述**唯一允许**的远程实验流程。任何在 NPU-910C 上跑实验的请求，都应严格按此 skill 操作，禁止绕过约束。

## 0. 固定环境信息（不要改）

| 项 | 值 |
|---|---|
| SSH 命令 | `ssh schedTeam@1.95.114.229 -p 22 -i ~/.ssh/id_rsa_second` |
| 远端工作目录 | `/home/schedTeam/zhangyz` |
| 模型权重路径 | `/home/schedTeam/Qwen3-30B-A3B`（**只读，禁止移动/重新下载**） |
| 默认容器 | `zhangyz-npu-1` |
| docker-compose 文件 | `/home/schedTeam/zhangyz/docker-compose.yml` |

## 1. 红线（⛔ 必读）

- ⛔ **必须**在 Docker 容器中运行训练/推理；禁止在宿主机直接执行任何 NPU 计算。
- ⛔ 禁止 `docker rm` / `docker rmi` 任何已有容器或镜像。
- ⛔ 禁止修改、移动、复制覆盖 `/home/schedTeam/Qwen3-30B-A3B`。
- ⛔ 禁止重新下载模型（无网络也不要尝试）；HF cache 必须指向已有模型路径。
- ⛔ 禁止创建重复的大规模环境（venv / conda 新环境 / 新容器）；尽量复用 `zhangyz-npu-1`。
- ⛔ 禁止用 `pkill` / `killall` 等按名字杀进程；只能 `kill <PID>`。

## 2. 标准执行步骤

### Step 1 — SSH 连接

```bash
ssh schedTeam@1.95.114.229 -p 22 -i ~/.ssh/id_rsa_second
```

> 若需要非交互执行命令，可在 ssh 命令后追加单条命令；批量命令建议 `ssh ... bash -lc '...'`。

### Step 2 — 进入工作目录

```bash
cd /home/schedTeam/zhangyz
ls -la                                   # 确认 docker-compose.yml 与代码目录存在
```

### Step 3 — 检查容器状态

```bash
docker ps -a --filter "name=zhangyz-npu-1"
# 期望看到 STATUS = Up / Exited，IMAGE 为 NPU 镜像
```

如果 `Exited`，启动它（**不要**重建）：

```bash
docker start zhangyz-npu-1
```

### Step 4 — 进入容器

```bash
docker exec -it zhangyz-npu-1 bash
# 进入后立即校验关键路径
ls /home/schedTeam/Qwen3-30B-A3B | head     # 模型挂载存在
python3 -c "import torch_npu; print(torch_npu.__version__)"   # NPU 运行时存在
```

> 如果模型路径在容器内不可见，说明该容器没挂载宿主目录，**先停下来**汇报，再决定是否走 Step 5。

### Step 5 — 容器不满足时（参考 docker-compose 重建）

仅在 Step 4 校验失败时使用，并且 **不要 `docker rm` 已有 `zhangyz-npu-1`**。允许的做法：

```bash
cat docker-compose.yml                   # 先看现有配置
docker compose up -d --no-recreate       # 启动 compose 中其他服务（不会动已有同名容器）
# 或者临时起一个名字不同的容器（例如 zhangyz-npu-1-tmp），跑完即停
```

任何容器变更都要在汇报里写清楚原因和回滚方法。

### Step 6 — 同步代码到容器

容器内通常已经 bind-mount 了宿主 `/home/schedTeam/zhangyz`，所以宿主上拉好分支即可：

```bash
# 在宿主上：
cd /home/schedTeam/zhangyz/<repo-dir>
git fetch origin
git checkout feat/npu-910c
git pull --ff-only origin feat/npu-910c
```

> 如果该 repo 没在远端 git 上，使用 `scp` 增量同步关键文件，**不要** `rsync --delete` 整库。

### Step 7 — 在容器内跑实验

固定使用本仓库脚本，所有路径用环境变量指向已有模型：

```bash
docker exec -it zhangyz-npu-1 bash -lc '
  cd /home/schedTeam/zhangyz/<repo-dir>
  export HF_HOME=/home/schedTeam/.cache/huggingface
  export QWEN3_MODEL_PATH=/home/schedTeam/Qwen3-30B-A3B
  export TRANSFORMERS_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
  ./scripts/run_single.sh local <BATCH> <SEQ> --tokens <N> [flags...]
'
```

冒烟测试推荐的最小配置（验证流程能跑通即可）：

```text
BATCH=2  SEQ=128  --tokens 4  --warmup-p2p --warmup-rounds 2
```

### Step 8 — 收集结果与日志

- 实验产出统一落到 `/home/schedTeam/zhangyz/<repo-dir>/results/`。
- 用 `scp` 拉关键 JSON / 报告回本地，例如：
  ```bash
  scp -P 22 -i ~/.ssh/id_rsa_second \
      schedTeam@1.95.114.229:/home/schedTeam/zhangyz/<repo-dir>/results/<file> ./
  ```
- 失败时收集：
  - 容器内 stderr / stdout（脚本退出码 + 末 200 行）
  - `dmesg | tail`（NPU OOM/驱动报错）
  - `npu-smi info`（NPU 设备状态）

## 3. Skill 输出格式

每次按本 skill 跑完实验后，必须给用户一段**结构化报告**：

```text
✅/❌ NPU 冒烟测试

环境
- 主机: 1.95.114.229
- 容器: zhangyz-npu-1 (image: <image>, status: Up)
- 分支: feat/npu-910c @ <commit-sha>
- 模型: /home/schedTeam/Qwen3-30B-A3B

实验
- 命令: ./scripts/run_single.sh local 2 128 --tokens 4 ...
- 退出码: 0
- 关键耗时: prefill XX ms / decode XX ms/step

产出 / 日志
- results/<...>.json
- 失败时贴最相关的 20-30 行 stderr
```

## 4. 常用命令速查

| 目的 | 命令 |
|---|---|
| 列容器 | `docker ps -a` |
| 进入容器 | `docker exec -it zhangyz-npu-1 bash` |
| NPU 状态 | `npu-smi info` |
| NPU 进程 | `npu-smi info -t proc-mem` |
| 看模型挂载 | `ls /home/schedTeam/Qwen3-30B-A3B` |
| 容器内 Python NPU 自检 | `python3 -c "import torch_npu; print(torch_npu.npu.is_available())"` |
| 同步代码 | `git -C /home/schedTeam/zhangyz/<repo-dir> pull --ff-only` |
| 拉日志回本地 | `scp -P 22 -i ~/.ssh/id_rsa_second schedTeam@1.95.114.229:<path> ./` |

## 5. 失败时的诊断顺序

1. 容器是否 Up？`docker ps -a`
2. 模型路径是否可见？`docker exec ... ls /home/schedTeam/Qwen3-30B-A3B`
3. NPU 是否健康？容器内 `npu-smi info`
4. Python 能 import `torch_npu` 吗？
5. 脚本本身在 GPU 分支跑得通吗？（先在本地复盘）
6. 看完以上仍不通，**保留现场**，把日志贴出来再决定是否重建容器。
