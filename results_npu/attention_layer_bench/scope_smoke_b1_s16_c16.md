# Attention layer benchmark summary

| phase | case | batch | seq | cache_len | ok/fail | median ms | speedup vs HF | delta ms vs HF | max abs diff |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| decode_core | community_v2 | 1 | 1 | 16 | 0/1 | N/A | N/A | N/A | N/A |
| decode_core | community_v3 | 1 | 1 | 16 | 0/1 | N/A | N/A | N/A | N/A |
| decode_core | hf_sdpa | 1 | 1 | 16 | 1/0 | 0.0993 | 1 | 0 | N/A |
| decode_core | official_ifa | 1 | 1 | 16 | 1/0 | 0.07819 | 1.27 | -0.0211 | 0 |
| decode_full_layer | hf | 1 | 1 | 16 | 1/0 | 0.9148 | 1 | 0 | N/A |
| decode_full_layer | official | 1 | 1 | 16 | 1/0 | 0.9554 | 0.9576 | 0.04056 | 0 |
| decode_full_layer | official_fused_both | 1 | 1 | 16 | 1/0 | 0.769 | 1.19 | -0.1458 | 0.0007324 |
| decode_full_layer | official_fused_rmsnorm | 1 | 1 | 16 | 1/0 | 0.851 | 1.075 | -0.0638 | 0.0004883 |
| decode_full_layer | official_fused_rope | 1 | 1 | 16 | 1/0 | 0.8413 | 1.087 | -0.0735 | 0.0001221 |
| prefill_core | hf_sdpa | 1 | 16 | N/A | 1/0 | 0.1485 | 1 | 0 | N/A |
| prefill_core | official_pfa | 1 | 16 | N/A | 1/0 | 0.227 | 0.6545 | 0.07841 | 0 |
| prefill_full_layer | hf | 1 | 16 | N/A | 1/0 | 1.054 | 1 | 0 | N/A |
| prefill_full_layer | official | 1 | 16 | N/A | 1/0 | 1.221 | 0.8627 | 0.1677 | 0 |
| prefill_full_layer | official_fused_both | 1 | 16 | N/A | 1/0 | 1.045 | 1.008 | -0.00886 | 0.0009766 |
| prefill_full_layer | official_fused_rmsnorm | 1 | 16 | N/A | 1/0 | 1.147 | 0.9187 | 0.09319 | 0.0009766 |
| prefill_full_layer | official_fused_rope | 1 | 16 | N/A | 1/0 | 1.13 | 0.9325 | 0.07627 | 0.0009766 |
