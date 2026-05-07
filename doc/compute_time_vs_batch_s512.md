# Attn / FFN 计算时间 vs Batch（seq=512, decode-dbo, npu-ep7, t=20）

数据源：`results_npu_ep7/decode-dbo/timing_{attention,ffn_coordinator}_*_s512_t20.json`

- `*_total_ms`：整次 decode loop 中所有 layer × 所有 MB 的累计时间
- `*_per_layer_per_mb_ms`：除以 `num_layers × num_mb` 得到的单层单 MB 平均
- `ep_experts`：FFN 内 EP local experts（routed MoE）的实际计算时间，是 `ffn_compute` 的子集
- `ffn/attn ratio`：FFN 总耗时 / Attn 总耗时

| batch | layers×mb | attn_total ms | ffn_total ms | ep_experts ms | attn/层/MB ms | ffn/层/MB ms | ep_experts/层/MB ms | ffn/attn |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 48×2 | 97.73 | 103.12 | 92.07 | 1.018 | 1.074 | 0.959 | 1.06 |
| 4 | 48×2 | 106.92 | 138.87 | 126.55 | 1.114 | 1.447 | 1.318 | 1.30 |
| 8 | 48×2 | 111.49 | 180.77 | 167.57 | 1.161 | 1.883 | 1.746 | 1.62 |
| 16 | 48×2 | 105.24 | 199.93 | 187.34 | 1.096 | 2.083 | 1.951 | 1.90 |
| 32 | 48×2 | 104.44 | 191.19 | 179.70 | 1.088 | 1.992 | 1.872 | 1.83 |
| 64 | 48×2 | 144.80 | 274.09 | 260.65 | 1.508 | 2.855 | 2.715 | 1.89 |
| 128 | 48×2 | 183.64 | 253.15 | 241.19 | 1.913 | 2.637 | 2.512 | 1.38 |
| 256 | 48×2 | 275.16 | 289.43 | 277.23 | 2.866 | 3.015 | 2.888 | 1.05 |
| 512 | 48×2 | 461.60 | 320.93 | 308.75 | 4.808 | 3.343 | 3.216 | 0.70 |

## 关键观察

- batch 从 2 → 512：attn 单层 MB 1.02 → 4.81 ms，ffn 单层 MB 1.07 → 3.34 ms。
- FFN/Attn 计算时间比在 0.70× ~ 1.90×。中 batch（8~64）时 FFN 显著占优（routed MoE 主导），DBO overlap 收益最大；极小 batch（2）和极大 batch（512）下两者接近或 attn 反超，DBO 收益相应减弱。
