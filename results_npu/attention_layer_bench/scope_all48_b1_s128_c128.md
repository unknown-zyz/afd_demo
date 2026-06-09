# Attention layer benchmark summary

| phase | case | batch | seq | cache_len | ok/fail | median ms | speedup vs HF | delta ms vs HF | max abs diff |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| decode_core | hf_sdpa | 1 | 1 | 128 | 48/0 | 0.0474 | 1 | 0 | N/A |
| decode_core | official_ifa | 1 | 1 | 128 | 48/0 | 0.0474 | 1 | -1.001e-06 | 0 |
| decode_full_layer | hf | 1 | 1 | 128 | 48/0 | 0.8502 | 1 | 0 | N/A |
| decode_full_layer | official | 1 | 1 | 128 | 48/0 | 0.8874 | 0.9581 | 0.03718 | 0 |
| decode_full_layer | official_fused_both | 1 | 1 | 128 | 48/0 | 0.7212 | 1.179 | -0.1291 | 0.007812 |
| decode_full_layer | official_fused_rmsnorm | 1 | 1 | 128 | 48/0 | 0.8146 | 1.044 | -0.03564 | 0.008301 |
| decode_full_layer | official_fused_rope | 1 | 1 | 128 | 48/0 | 0.7862 | 1.082 | -0.06409 | 0.00293 |
| prefill_core | hf_sdpa | 1 | 128 | N/A | 48/0 | 0.07663 | 1 | 0 | N/A |
| prefill_core | official_pfa | 1 | 128 | N/A | 48/0 | 0.1954 | 0.3921 | 0.1188 | 0 |
| prefill_full_layer | hf | 1 | 128 | N/A | 48/0 | 0.8681 | 1 | 0 | N/A |
| prefill_full_layer | official | 1 | 128 | N/A | 48/0 | 1.135 | 0.7649 | 0.2668 | 0 |
| prefill_full_layer | official_fused_both | 1 | 128 | N/A | 48/0 | 0.9898 | 0.877 | 0.1217 | 0.02783 |
| prefill_full_layer | official_fused_rmsnorm | 1 | 128 | N/A | 48/0 | 1.071 | 0.8108 | 0.2025 | 0.02393 |
| prefill_full_layer | official_fused_rope | 1 | 128 | N/A | 48/0 | 1.049 | 0.8273 | 0.1812 | 0.01562 |
