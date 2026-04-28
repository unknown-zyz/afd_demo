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
| 远端工作目录（宿主） | `/home/schedTeam/zhangyz` |
| 模型权重路径（宿主） | `/home/schedTeam/Qwen3-30B-A3B`（**只读，禁止移动/重新下载**） |
| **首选容器** | `afd-npu-test`（**长驻**，已装 torch_npu 2.6.0 + transformers，已挂模型；**禁止删除**） |
| 备用容器 | `zhangyz-npu-1`（image `deepep-ascend-bench-dev`，无模型挂载，仅适合 CPU 冒烟） |
| 容器内工作目录 | `/workspace`（来自 docker volume `zhangyz_workspace`） |
| docker-compose 文件 | `/home/schedTeam/zhangyz/docker-compose.yml` |
| 容器 NPU 设备 | `/dev/davinci0..15` 已透传，CANN 8.5 装在 `/usr/local/Ascend/` |
| Python | `/usr/local/python3.11.14/bin/python3` |
| 网络代理 | 容器内 `HTTP_PROXY=127.0.0.1:20173`（经 v2raya 服务转发） |

## 1. 红线（⛔ 必读）

- ⛔ **必须**在 Docker 容器中运行训练/推理；禁止在宿主机直接执行任何 NPU 计算。
- ⛔ 禁止 `docker rm` / `docker rmi` 任何已有容器或镜像。
- ⛔ 禁止修改、移动、复制覆盖 `/home/schedTeam/Qwen3-30B-A3B`。
- ⛔ 禁止重新下载模型（无网络也不要尝试）；HF cache 必须指向已有模型路径。
- ⛔ 禁止创建重复的大规模环境（venv / conda 新环境 / 新容器）；尽量复用 `zhangyz-npu-1`。
- ⛔ 禁止用 `pkill` / `killall` 等按名字杀进程；只能 `kill <PID>`。

## 2. 标准执行步骤

> **优先使用长驻容器 `afd-npu-test`**：已经装好 torch_npu 2.6.0、transformers、所有 CANN 依赖，并以 ro 方式挂载了模型 `/models/Qwen3-30B-A3B`。多数情况下只需 `docker exec afd-npu-test bash -lc '...'` 即可跑实验，**跳过 Step 5/6/7a**。
>
> ### 已验证可跑的最小命令（直接复用）
>
> ```bash
> ssh -p 22 -i ~/.ssh/id_rsa_second schedTeam@1.95.114.229 \
>   "docker exec afd-npu-test bash -lc '
>      source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
>      cd /workspace/afd_demo
>      ASCEND_VISIBLE_DEVICES=0,1 ./scripts/run_npu.sh \
>        --attn-size 1 --ffn-size 1 --ffn-tp-size 1 \
>        --batch 1 --seq 32 --tokens 4 \
>        --model-name /models/Qwen3-30B-A3B
>   '"
> ```
>
> ### 若 `afd-npu-test` 不存在或损坏，重建命令（**严禁删除其他容器**）
>
> ```bash
> docker run -d --name afd-npu-test --privileged --network host --ipc host --shm-size 16g \
>   --device /dev/davinci0 --device /dev/davinci1 \
>   --device /dev/davinci2 --device /dev/davinci3 \
>   --device /dev/davinci_manager --device /dev/hisi_hdc --device /dev/devmm_svm \
>   -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
>   -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
>   -v /etc/ascend_install.info:/etc/ascend_install.info \
>   -v /usr/local/dcmi:/usr/local/dcmi \
>   -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
>   -v /home/schedTeam/Qwen3-30B-A3B:/models/Qwen3-30B-A3B:ro \
>   deepep-ascend-bench-dev sleep infinity
> # privileged + network host 缺一不可，否则 drvGetDevNum=87
>
> # 装包顺序（务必 2.6.0，不要用 2.5.1）
> docker exec afd-npu-test pip3 install numpy pyyaml decorator attrs psutil scipy
> docker exec afd-npu-test pip3 install torch==2.6.0 torch_npu==2.6.0
> docker exec afd-npu-test pip3 install transformers accelerate sentencepiece protobuf
> ```
>
> ### 已知 BUG（避免重复踩）
> 1. **不要用 `torch_npu==2.5.1`**：CANN 8.5 镜像的 TBE 二进制缓存与之不匹配，会报 `BinaryGetFunction failed for Index_*` / `aclnnNeg` 等。必须 ≥ 2.6.0。
> 2. **不要用 4 卡 preset (`--attn-size 2 --ffn-size 2`)**：当前 `src/distributed/__init__.py:160-167` 的 role 映射缺陷会让 4 个 rank 抢同一个 device 导致 OOM。仅支持 `--attn-size 1 --ffn-size 1`（2 rank）跑通。

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
ls /workspace                                               # 工作目录
ls /usr/local/Ascend/cann-8.5.0 2>&1 | head                 # CANN 已装
python3 -c "import torch_npu; print(torch_npu.__version__)" # 通常未装
```

⚠ **实测**：当前 `zhangyz-npu-1`（compose 中的 `local` 服务，非 `npu`
服务）**没有**挂载 `/home/schedTeam/Qwen3-30B-A3B`，**也没有**装
`torch_npu`。要真正在 NPU 上跑模型，必须：

- 用 compose 中的 `npu` 服务（带 davinci 设备 + CANN 路径），或
- 自己 `docker run` 时加 `-v /home/schedTeam/Qwen3-30B-A3B:/models/Qwen3-30B-A3B:ro`
  并 `pip install torch_npu==<匹配 CANN 8.5 的版本>`。

不要 `docker rm` 现有容器；新起一个名字不同的容器（比如
`zhangyz-npu-1-run`）即可。

### Step 5 — 容器不满足时（参考 docker-compose 重建）

仅在 Step 4 校验失败时使用，并且 **不要 `docker rm` 已有 `zhangyz-npu-1`**。允许的做法：

```bash
cat docker-compose.yml                   # 先看现有配置
docker compose up -d --no-recreate       # 启动 compose 中其他服务（不会动已有同名容器）
# 或者临时起一个名字不同的容器（例如 zhangyz-npu-1-tmp），跑完即停
```

任何容器变更都要在汇报里写清楚原因和回滚方法。

### Step 6 — 同步代码到容器

容器使用 docker named volume（不是 bind-mount 宿主目录），所以宿主修改 **不会** 自动出现在容器里。推荐用 `git bundle` 注入：

```bash
# 在本地：
git bundle create /tmp/afd_npu.bundle feat/npu-910c
scp -P 22 -i ~/.ssh/id_rsa_second /tmp/afd_npu.bundle \
    schedTeam@1.95.114.229:/home/schedTeam/zhangyz/

# 在远端宿主：
docker cp /home/schedTeam/zhangyz/afd_npu.bundle zhangyz-npu-1:/workspace/

# 在容器内：
cd /workspace
git clone afd_npu.bundle afd_demo
cd afd_demo && git checkout feat/npu-910c
```

后续增量同步：在本地 `git bundle create /tmp/inc.bundle <last-sha>..HEAD`，
然后容器里 `git bundle unbundle inc.bundle && git merge --ff-only FETCH_HEAD`。

### Step 7 — 在容器内跑实验

#### 7a — Python 依赖（首次准备）

容器自带 Python 3.11.14，但 `torch / transformers / numpy` 等全部需要装。
通过容器内已配的代理装：

```bash
docker exec zhangyz-npu-1 bash -lc '
  pip install --quiet numpy pyyaml
  pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
  pip install --quiet transformers accelerate
  # NPU 路径：再装 torch_npu（必须与上面 torch 版本严格匹配）
  # pip install torch_npu==<X.Y.Z>
'
```

#### 7b — CPU 冒烟测试（验证 backend 抽象 / argparse / 模块导入）

不需要 NPU 设备，也不需要模型权重，分钟级跑完：

```bash
docker exec zhangyz-npu-1 bash -lc '
  cd /workspace/afd_demo
  python3 scripts/npu_smoke_test.py
'
```

期望最后输出 `PASS: all smoke checks OK`。

#### 7c — NPU 实跑（需要 torch_npu + 模型挂载）

固定用环境变量指向已有模型，**不要**让 transformers 联网：

```bash
docker exec zhangyz-npu-1 bash -lc '
  cd /workspace/afd_demo
  export HF_HOME=/workspace/.cache/huggingface
  export QWEN3_MODEL_PATH=/models/Qwen3-30B-A3B   # 需要先 -v 挂进来
  export TRANSFORMERS_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
  ./scripts/run_npu.sh --preset npu-4card --tokens 4 --batch 2 --seq 128
'
```

冒烟最小配置：`--preset npu-4card --batch 2 --seq 128 --tokens 4`。

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
| NPU 状态 | `npu-smi info`（宿主）；容器内 `/usr/local/Ascend/driver/tools/` 下没有 npu-smi |
| NPU 进程 | `npu-smi info -t proc-mem`（宿主） |
| 看模型挂载 | `ls /home/schedTeam/Qwen3-30B-A3B` |
| 容器内 Python NPU 自检 | `python3 -c "import torch_npu; print(torch_npu.npu.is_available())"` |
| NPU 状态（**宿主**执行） | `npu-smi info`（容器内通常没有，要在宿主跑） |
| 同步代码 | `git -C /home/schedTeam/zhangyz/<repo-dir> pull --ff-only` |
| 拉日志回本地 | `scp -P 22 -i ~/.ssh/id_rsa_second schedTeam@1.95.114.229:<path> ./` |

## 5. 失败时的诊断顺序

1. 容器是否 Up？`docker ps -a`
2. 模型路径是否可见？`docker exec ... ls /home/schedTeam/Qwen3-30B-A3B`
3. NPU 是否健康？容器内 `npu-smi info`
4. Python 能 import `torch_npu` 吗？
5. 脚本本身在 GPU 分支跑得通吗？（先在本地复盘）
6. 看完以上仍不通，**保留现场**，把日志贴出来再决定是否重建容器。
