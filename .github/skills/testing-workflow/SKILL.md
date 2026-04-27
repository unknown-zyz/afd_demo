---
name: testing-workflow
description: >
  测试与实验的标准化工作流。当需要运行测试、性能实验、benchmark，或涉及代码变更后的验证时使用此 skill。
  包括：分支管理、资源监控、性能对比、合并决策。
allowed-tools: shell
---

# Testing Workflow Skill

在进行任何测试、实验或 benchmark 时，严格遵循以下流程。

## 1. 分支管理

### 测试前
- **创建新分支**：任何功能开发或实验，必须在新分支上进行，禁止直接在 main 分支修改代码。
  ```bash
  git checkout -b feat/<功能描述>    # 功能开发
  git checkout -b exp/<实验描述>     # 性能实验
  git checkout -b fix/<修复描述>     # Bug 修复
  ```
- **代码完成后立即 push**：测试前确保代码已推送到远程，防止实验过程中机器异常导致代码丢失。
  ```bash
  git add -A && git commit -m "<描述>" && git push -u origin HEAD
  ```

### 测试后
- 测试通过且性能达标后，才可以合并到 main 分支。
- 合并前执行 `git diff main..HEAD --stat` 确认变更范围。

## 2. 资源监控

### 测试前检查
在启动任何 GPU 测试之前，**必须**先检查资源状态：
```bash
# 运行 skill 目录下的检查脚本
bash .github/skills/testing-workflow/check_resources.sh
```

若手动检查，需确认：
- **GPU 显存**: `nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader`
  - 可用显存 < 4GB 时**禁止**启动新测试
  - 正在运行其他 GPU 进程时，先确认是否冲突
- **系统内存**: `free -h` — 可用内存 < 4GB 时警告
- **CPU 负载**: `uptime` — load average > CPU 核数时警告
- **磁盘空间**: `df -h /data` — 可用空间 < 10GB 时警告

### 测试中监控
- 长时间实验（> 5 分钟）时，设置后台资源监控：
  ```bash
  # 每 30 秒记录一次 GPU 状态
  watch -n 30 nvidia-smi
  ```
- **OOM 风险识别**：
  - 当 GPU 显存使用率 > 90% 时，主动提醒用户
  - 检测到 `CUDA out of memory` 错误时，立即停止后续更大参数的测试
  - 记录 OOM 发生时的参数配置，作为该配置的上限
- **超时控制**：单次测试超过 10 分钟无输出时，视为悬挂，应终止进程

### 风险处理
- 发现资源紧张时，**立即停止测试**，不要等到 OOM
- 停止后清理 GPU 进程：确认所有测试进程已退出
- 向用户报告资源状态和建议

## 3. 性能对比与合并决策

### 基准记录
- 测试前先在 main 分支上运行相同配置，记录基准性能数据
- 或使用 `results/` 目录下已有的历史数据作为基准
- 基准数据应包含：吞吐量、延迟、显存占用

### 对比方法
- 使用**相同的硬件、参数、输入**进行对比
- 至少运行 3 次取中位数，消除波动
- 对比指标：
  - **DBO 加速比**: DBO 时间 / Serial 时间（< 1.0 才有加速效果）
  - **Overlap 率**: 通信被计算掩盖的比例
  - **显存开销**: DBO 相比 Serial 额外占用的显存

### 合并条件
满足以下**全部条件**才可合并到 main：
1. ✅ 单元测试全部通过：`pytest tests/ -v`
2. ✅ 功能测试通过：`bash scripts/run_single.sh local 2 128 --tokens 5`
3. ✅ 性能无回归：关键指标不低于 main 分支基准
4. ✅ 性能有提升（如果是优化类变更）：加速比有可测量的改善

### 性能回归处理
- 如果性能出现回归（> 5% 劣化），**禁止合并**
- 分析回归原因，记录在 commit message 或 PR 描述中
- 考虑是否需要回滚相关变更

## 4. 本项目测试命令参考

### 单元测试
```bash
source venv/bin/activate
pytest tests/ -v
```

### 单机功能测试
```bash
bash scripts/run_single.sh local 2 128 --tokens 5              # 快速 prefill DBO
bash scripts/run_single.sh local 2 128 --tokens 5 --no-dbo     # Serial baseline
bash scripts/run_single.sh local 2 128 --tokens 5 --generate   # Decode DBO
```

### 单机实验
```bash
bash scripts/run_single.sh local <batch> <seq>              # DBO 模式
bash scripts/run_single.sh local <batch> <seq> --no-dbo     # Serial 模式
bash scripts/run_single.sh local <batch> <seq> --visualize  # 自动生成 pipeline 图
bash scripts/run_single.sh local <batch> <seq> --generate --crosslayer
```

### 批量实验
```bash
bash scripts/run_experiment_matrix.sh
bash scripts/run_experiment_matrix.sh --modes serial,prefill-dbo
bash scripts/run_experiment_matrix.sh --dry-run --modes serial --batches 2 --seqs 128
```

### 多机测试
```bash
# 远程机器（FFN 节点）
bash scripts/run_node.sh ffn <master_ip> <port>

# 本地机器（Attention 节点）
bash scripts/run_node.sh attention <master_ip> <port> --prompt "Hello"
```
