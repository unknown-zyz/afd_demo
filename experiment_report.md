# AFD Demo 实验报告

## 1. 环境配置

### 硬件

| 节点 | GPU | 内网IP |
|-----|-----|--------|
| 本地 | V100 × 2 | 10.244.64.173 |
| 远程 | V100 × 1 | 10.244.24.18 |

### 软件

```
Python 3.10, CUDA 12.6 runtime
torch==2.7.0+cu126
transformers==5.4.0
accelerate==1.13.0
numpy==1.26.4
```

### 安装命令

```bash
python3 -m venv venv && source venv/bin/activate
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

---

## 2. 单机测试

### 2.1 DBO 对比测试

```bash
./scripts/run_local.sh --scheduler compare
```

**升级后结果（示例）：**
```
[ATTENTION] rank=0, device=cuda:0, dtype=torch.bfloat16
[FFN] rank=1, device=cuda:1, dtype=torch.bfloat16
========================================
DBO Comparison:
  Sync:    788.31ms
  Async:   144.38ms
  Speedup: 5.46x
========================================
Output[0]: 'Hello, how are you today?' → ' I'
```

### 2.2 详细计时测试

```bash
./scripts/run_local.sh --scheduler async --timing
python scripts/visualize_dbo.py results/
```

生成文件：
- `results/timing_attention.json`
- `results/timing_ffn.json`
- `results/dbo_timeline.png`
- `results/dbo_timeline_analysis.png`

---

## 3. 多机测试

### 启动命令

**Attention Node (本地):**
```bash
./scripts/run_attn_node.sh 10.244.64.173 29500 --scheduler async
```

**FFN Node (远程):**
```bash
ssh -p 30549 -i ~/.ssh/id_rsa_v100 zyz@192.168.5.32
./scripts/run_ffn_node.sh 10.244.64.173 29500 --scheduler async
```

---

## 4. 结论

| 指标 | 结果 |
|-----|------|
| DBO 功能 | ✅ 升级后可运行 |
| 单机测试 | ✅ 通过 |
| 多机测试 | ✅ 通过 |
| 输出正确性 | ✅ 验证 |

DBO 通过异步通信 (isend/irecv) 实现计算与通信重叠。升级到 PyTorch 2.7 后仍可完成单机/多机推理与生成。

> 已知问题：当前环境中 `PyTorch 2.7.0 + NCCL 2.26` 在进程退出阶段可能触发 NCCL 清理崩溃。代码已采用规避策略，运行阶段结果不受影响。
