# Attention layer benchmark summary

| phase | case | batch | seq | cache_len | ok/fail | median ms | speedup vs HF | delta ms vs HF | max abs diff |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| decode_core | hf_sdpa | 1 | 1 | 128 | 1/0 | 0.1225 | 1 | 0 | N/A |
| decode_core | official_ifa | 1 | 1 | 128 | 1/0 | 0.1106 | 1.108 | -0.01191 | 0 |
| decode_full_layer | hf | 1 | 1 | 128 | 1/0 | 0.9667 | 1 | 0 | N/A |
| decode_full_layer | official | 1 | 1 | 128 | 1/0 | 1.002 | 0.9652 | 0.03486 | 0 |
| decode_full_layer | official_fused_both | 1 | 1 | 128 | 1/0 | 0.7915 | 1.221 | -0.1753 | 0.0004883 |
| decode_full_layer | official_fused_rmsnorm | 1 | 1 | 128 | 1/0 | 0.9081 | 1.065 | -0.05859 | 0.0004883 |
| decode_full_layer | official_fused_rope | 1 | 1 | 128 | 1/0 | 0.8712 | 1.11 | -0.0955 | 6.104e-05 |
| prefill_core | hf_sdpa | 1 | 128 | N/A | 1/0 | 0.1367 | 1 | 0 | N/A |
| prefill_core | official_pfa | 1 | 128 | N/A | 1/0 | 0.2656 | 0.5146 | 0.1289 | 0 |
| prefill_full_layer | hf | 1 | 128 | N/A | 1/0 | 1.437 | 1 | 0 | N/A |
| prefill_full_layer | official | 1 | 128 | N/A | 1/0 | 1.439 | 0.9991 | 0.0013 | 0 |
| prefill_full_layer | official_fused_both | 1 | 128 | N/A | 1/0 | 0.9953 | 1.444 | -0.442 | 0.001953 |
| prefill_full_layer | official_fused_rmsnorm | 1 | 128 | N/A | 1/0 | 1.097 | 1.311 | -0.3406 | 0.001953 |
| prefill_full_layer | official_fused_rope | 1 | 128 | N/A | 1/0 | 1.054 | 1.364 | -0.3833 | 0.0009766 |
