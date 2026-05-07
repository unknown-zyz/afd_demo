# PPT 生成任务 Prompt（交付给下游 PPT-agent）

> 本文件是给"PPT 生成 agent"的指令文档。请按以下 meta-instructions 输出一份用于项目介绍汇报的演示文稿（PPTX 或可被 PPTX 工具消费的结构化描述）。

---

## 0. Meta-Instructions（必须先读）

### 0.1 任务
- 项目名称：**AFD-DBO** —— *Attention/FFN Disaggregated inference with Dual-Batch Overlap*
- 受众：算法 + 系统混合的内部技术评审会，约 20 分钟讲解
- 模型：**Qwen3-30B-A3B**（48 层 MoE，128 experts，8 activated）
- 主要硬件：**Ascend 910C × 16（EP7 拓扑，主目标）**；副线：**NVIDIA A800 × 2（GPU baseline）**
- 关键技术词（必须在合适位置出现，不要漏）：
  - A/F 资源分离（attention rank / ffn rank 解耦）
  - 去中心化 P2P 通信（HCCL/NCCL `isend`/`irecv`，attn 直发 ffn，CPU 不介入数据面）
  - Dual-Batch Overlap（DBO，MB0/MB1 双 micro-batch 在 4 条 stream 泳道：**Attn / A2F / FFN / F2A** 上重叠）
  - Expert Parallel（EP7 + `broadcast_reduce_overlap` dispatch + round-robin assign）
  - 可选 cross-layer pre-/post-`irecv` 流水（跨层提前 post 接收）

### 0.2 输出要求
- **严格 ≤ 15 张 slides，目标 14 张**。多于 14 张必须合并；少于 14 张允许。
- 输出形式：建议 `python-pptx` 可执行脚本 + 渲染后 `.pptx`；若仅输出 markdown，必须保留同样 14 节结构与图片引用。
- 风格：技术汇报风格，**少字多图**；每张 slide 正文不超过 ~60 中文字 / 6 bullets。
- 配色建议：深色背景（`#0E1525`）+ 强调色 `#3DDC97`（绿）/ `#FF6B6B`（红，标 baseline）/ `#FFD166`（黄，标 DBO 收益）；论文/对比表用浅灰底。
- 字体：标题 ≥ 28pt；正文 ≥ 18pt；表格内文字 ≥ 14pt；代码/路径用等宽字体（JetBrains Mono / Sarasa Mono）≥ 14pt。
- 图片：**保持原宽高比**，不要拉伸；heatmap 类大图至少占 slide 60% 宽度。
- 动画/排版提示：DBO 4 泳道图、pipeline 解析图建议加"逐元素淡入"，其他 slide 不加动画。
- 引用相对路径以 `/home/zyz/afd_demo` 为根，PPT-agent 在生成时**必须先 `ls` 验证文件存在**，不存在则在该 slide 上贴一个文字占位框 `[图待补：<原路径>]`。

### 0.3 已知图片资源（已验证存在 ✅ / 不存在 ⚠️ ）
- ✅ `results_npu_ep7/fig_prefill_speedup_heatmap.png`
- ✅ `results_npu_ep7/fig_decode_speedup_heatmap.png`
- ✅ `results_npu_ep7/fig_decode_speedup_curves.png`
- ✅ `results_npu_ep7/fig_decode_crosslayer_speedup_heatmap.png`
- ✅ `results_npu_ep7/fig_decode_dbo_vs_crosslayer_curves.png`
- ✅ `results_npu_ep7/pipeline_figs/decode_crosslayer_b{2,4,16,32,64,128,256,512}_s{128,256,512,1024}_t20.png`（典型 pipeline 时间轴图）
- ⚠️ **`results_npu_ep7/pipeline_figs/decode_dbo_*_fourlane.png` 不存在**：用于讲 DBO 4 泳道的图请由 PPT-agent 检查替换；建议临时占位 `[4-lane DBO 时间轴示意图，由 PPT agent 检查替换]`，或退而使用上面 `decode_crosslayer_b*_*_t20.png` 中的一张并标注为"含 DBO+crosslayer 实采"。

---

## 1. Slide-by-Slide 说明（共 14 张）

### Slide 1 — 封面
- **标题**：AFD-DBO：面向 MoE LLM 推理的 A/F 分离 + 双 batch 重叠
- **核心内容**：
  - 副标题：*Disaggregated Attention / FFN Inference with Dual-Batch Overlap on Ascend 910C*
  - 模型：Qwen3-30B-A3B (MoE, 48L, 128E/top-8)
  - 硬件：Ascend 910C × 16 (EP7) ｜ A800 × 2 (baseline)
  - 汇报人 / 日期占位
- **排版要求**：居中标题，副标题次行；右下放硬件 logo 占位框；纯色深背景。
- **引用图片**：无

### Slide 2 — 背景与动机
- **标题**：MoE 推理的两个根本失衡
- **核心内容**：
  - Attention 是 **memory-bound**（KV cache 主导）；FFN/MoE 是 **compute + all2all-bound**
  - 同卡同步执行：两类 kernel 互相挤占 SM/AICore 与 HBM 带宽
  - MoE 引入额外 dispatch/combine 通信，serial 执行下气泡显著
  - **目标**：在不改模型权重前提下，把 attn 与 ffn 解耦到不同 rank，并用双 batch 把通信/计算互相掩盖
- **排版要求**：左文右图（用一张"serial 时间轴示意"占位图）；红色高亮 "气泡 / bubble"。
- **引用图片**：无（PPT agent 可生成示意图占位）

### Slide 3 — 前期调研：相关论文
- **标题**：相关工作（论文）
- **核心内容**（每条 1 句点评）：
  - **DeepSeek-V3 DualPipe**：对称双向流水掩盖 all2all → 启发我们做 **dual-batch overlap**
  - **Snowflake Arctic**：MoE 训练时 expert 并行 + 通信折叠 → 借鉴 EP+overlap 思路
  - **SplitWise**（Microsoft）：prefill / decode 分离部署 → 我们做的是 **同 step 内** attn/ffn 分离，更细粒度
  - **DistServe**（OSDI'24）：goodput-aware disaggregation → 验证"分离"路线在 SLO 上的收益
- **排版要求**：4 个等大卡片网格（2×2），每卡片：论文名 + 一行点评 + 一句"对我们的启发"。
- **引用图片**：无

### Slide 4 — 前期调研：开源系统对比
- **标题**：开源推理系统对比（聚焦 MoE / 分离）
- **核心内容**：3 列对比表
  | 系统 | 是否 A/F 分离 | MoE EP 通信 | 与本工作差异 |
  |---|---|---|---|
  | **vLLM** | ❌ 同 rank | all2all (NCCL) | 单体 worker，无双 batch overlap |
  | **SGLang** | ❌ | all2all + radix | 重在调度，未做 attn/ffn 解耦 |
  | **Mooncake** | ✅ KV-centric | KV 池分离 | 分离 KV 而非 attn/ffn 计算 |
  | **本工作 AFD-DBO** | ✅ 计算分离 | **去中心化 P2P + EP7 overlap** | attn/ffn 各自独立 rank，DBO 4 泳道 |
- **排版要求**：表格占满 slide；最后一行加粗、绿色底高亮。
- **引用图片**：无

### Slide 5 — 设计思路 1：A/F 资源分离
- **标题**：Attention / FFN 分离 — 去中心化 P2P 数据面
- **核心内容**：
  - 把 48 层 Qwen3 切成 attn-rank 与 ffn-rank 两组（GPU 路径默认 layer-21 边界；NPU EP7 下按层细分）
  - 通信：**HCCL/NCCL `isend`/`irecv`** 直接 NPU↔NPU，CPU **不介入数据面**
  - 没有中心化 controller —— 路由表静态固化在每个 rank 上
  - 中心化 controller 仅作为"反向对照"baseline，用来证明去中心化的必要性
  - 控制面（step 计数、stop 信号）走轻量 broadcast，不进 hot path
- **排版要求**：左半文字 bullets，右半 A/F 拓扑图（attn ranks → ffn ranks 箭头网格）占位。
- **引用图片**：无（建议占位拓扑示意图）

### Slide 6 — 设计思路 2：DBO 双 batch 4 泳道
- **标题**：Dual-Batch Overlap：MB0/MB1 在 4 条流上交错
- **核心内容**：
  - 两个 micro-batch（MB0、MB1）按层错相位推进
  - 4 条 stream / 泳道：**Attn ｜ A2F (attn→ffn) ｜ FFN ｜ F2A (ffn→attn)**
  - 当 MB0 在 FFN 时，MB1 同步在 Attn；A2F/F2A 与计算重叠
  - 关键约束：每泳道 FIFO，跨泳道靠 event 同步，不靠 host 阻塞
  - 收益主要来源：**通信被计算吃掉**
- **排版要求**：上文字下大图（4 泳道时间轴图），高度占 slide 55%；动画建议"4 条泳道逐条出现"。
- **引用图片**：`[4-lane DBO 时间轴示意图，由 PPT agent 检查替换]`（fourlane 文件不存在；可临时使用 `results_npu_ep7/pipeline_figs/decode_crosslayer_b16_s512_t20.png` 作为带 DBO 的真实采样图并标注）

### Slide 7 — 设计思路 3：EP7 + broadcast_reduce_overlap
- **标题**：Expert Parallel：EP7 + 通信-计算折叠
- **核心内容**：
  - 910C×16 拓扑：**7 个 ffn-rank 承担 expert shard，其余作 attn-rank**
  - dispatch 策略：`broadcast_reduce_overlap` —— token broadcast 与 expert compute 同步进行
  - assignment：**round-robin** 落 expert，避免 hot expert 长尾
  - combine 阶段 reduce 与下一层 attn 的 KV load 重叠
  - 相比 all2all：少一次全量 shuffle，HBM 峰值下降
- **排版要求**：左文字右示意图；EP7 用 7 个圆 + 箭头表示。
- **引用图片**：无

### Slide 8 — 设计思路 4：Cross-layer pre/post irecv 流水
- **标题**：跨层流水：提前 post irecv，吃掉 launch 延迟
- **核心内容**：
  - 标准做法：layer L 完成后才 `irecv` layer L 的输入 → host launch 间隙暴露
  - 本方案：**在 layer L-1 的 FFN 阶段就 pre-post layer L 的 irecv**
  - 同时在 L 完成后延后 post-cleanup，使下层 attn 不等通信句柄
  - **重要事实**：在 EP7 下 cross-layer 不再增益（详见实验 slide 12）
  - 旧 2-rank 配置下 b=4 曾观察到 0.73× → 0.94× 的提升
- **排版要求**：左文字右两段时间轴对比（before/after）占位图。
- **引用图片**：无

### Slide 9 — 实验环境
- **标题**：实验配置
- **核心内容**：
  - 硬件：Ascend 910C × 16（主），A800 × 2（baseline 对照）
  - 软件：torch_npu **2.6.0** + CANN **8.x**；CUDA 12.x + PyTorch 2.4
  - 模型：Qwen3-30B-A3B，dtype=bf16（**fp8 在 910C 上完全不可用，CANN 任何 fp8 op 报 `ERR01007`**）
  - 工作负载：batch ∈ {2,4,16,32,64,128,256,512}，seqlen ∈ {128,256,512,1024(,2048)}，max_new_tokens=20
  - 指标：**TPOT = decode_loop_ms / (max_new_tokens − 1)**（batch 级 per-step 时间）
  - 入口脚本：`scripts/run_experiment_matrix*.sh`，结果在 `results_npu_ep7/`
- **排版要求**：左信息卡，右一张小代码块展示一条命令行。
- **引用图片**：无

### Slide 10 — Prefill 全矩阵 heatmap
- **标题**：Prefill 阶段 DBO 加速比（910C EP7）
- **核心内容**：
  - X 轴 batch，Y 轴 seqlen，单元格 = DBO speedup over serial
  - 大 batch × 长 seq 区域加速最显著（计算密度高，通信窗口大）
  - 小 batch 区域接近 1.0×（通信本就少，无可掩盖）
  - 没有出现 < 1.0× 的回归区
- **排版要求**：居中大图；左下角加一句结论 caption。
- **引用图片**：`results_npu_ep7/fig_prefill_speedup_heatmap.png`

### Slide 11 — Decode 全矩阵 heatmap + TPOT 曲线
- **标题**：Decode 阶段：DBO 加速比与 TPOT
- **核心内容**：
  - 左：speedup heatmap（同样 batch × seqlen 网格）
  - 右：speedup curves（按 seqlen 分组的折线）
  - decode 阶段 TPOT 改善整体 1.1–1.5×；中等 batch（16–64）收益最稳
  - 提醒：TPOT 定义见 slide 9
- **排版要求**：左右分栏 50/50，两张图各占一半。
- **引用图片**：
  - 左：`results_npu_ep7/fig_decode_speedup_heatmap.png`
  - 右：`results_npu_ep7/fig_decode_speedup_curves.png`

### Slide 12 — Cross-layer 流水的实测对比
- **标题**：Cross-layer 在 EP7 下不再增益
- **核心内容**：
  - heatmap：cross-layer over base-DBO 的 speedup，**几乎全 1.0× 或微负**
  - 折线：DBO vs DBO+crosslayer，曲线高度重合
  - 原因分析（必须念到）：
    1. base DBO 已把跨层通信掩盖
    2. cross-layer 增加 pre-posted handle，**HBM 占用变大**
    3. **NCCL/HCCL 队列竞争**，多余的 irecv 反而排队
  - 历史对照：2-rank 旧配置 b=4 时 0.73× → 0.94×（成立），但不能外推到 EP7
- **排版要求**：左 heatmap 右折线图。
- **引用图片**：
  - 左：`results_npu_ep7/fig_decode_crosslayer_speedup_heatmap.png`
  - 右：`results_npu_ep7/fig_decode_dbo_vs_crosslayer_curves.png`

### Slide 13 — 典型 pipeline 解析
- **标题**：典型样本时间轴：b=16, s=512, decode
- **核心内容**：
  - 一张实采 pipeline 图：可清楚看到 4 条泳道交错
  - 标注：MB0/MB1 错相位、A2F 与 FFN 重叠、F2A 与下层 Attn 重叠
  - 红框圈出"未被掩盖的尾巴" → 引出下一步优化
- **排版要求**：整张 slide 居中放大图，左侧留 30% 文字注释；红色框圈直接画在图上层。
- **引用图片**：`results_npu_ep7/pipeline_figs/decode_crosslayer_b16_s512_t20.png`（如缺失，PPT agent 改用同目录任一存在文件并更新标题）

### Slide 14 — 下一步计划
- **标题**：Next Steps
- **核心内容**：
  - **Token-aware dispatch**：按 token 实际命中 expert 做动态分配，替代 round-robin
  - **Fused MoE kernel**：把 gate + topk + gather 融合，减少 launch 开销
  - **FP8 推理**：当前在 910C 上被 `ERR01007` **完全阻塞**（torch_npu 2.6.0 + CANN 8.x），跟踪 CANN 后续版本
  - **中心化 controller baseline**：作为反向对照，量化"去中心化 P2P"的真实收益
  - **跨节点扩展**：从单机 16 卡到多机 EP，验证 HCCL 跨节点 P2P 行为
- **排版要求**：5 条 bullet 等距，左侧加图标占位（🚀 / 🔧 / 🧪 / ⚠️ / 🌐）。
- **引用图片**：无

---

## 2. 交付清单（PPT-agent 完成后自检）
- [ ] slide 数 ≤ 15（目标 14）
- [ ] 所有 ✅ 的图片路径已验证存在并被引用
- [ ] 所有 ⚠️ 占位（fourlane）已替换或显式占位框
- [ ] TPOT 公式、`ERR01007`、cross-layer 在 EP7 不增益、HCCL P2P 不经 CPU —— 这 4 条事实分别出现在 slide 9 / 9+14 / 12 / 5
- [ ] 字号、配色、宽高比符合 §0.2
