# Attention layer benchmark summary

| phase | case | batch | seq | cache_len | ok/fail | median ms | speedup vs HF | delta ms vs HF | max abs diff |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| prefill_full_layer | hf | 1 | 1024 | N/A | 3/0 | 1.143 | 1 | 0 | N/A |
| prefill_full_layer | hf | 1 | 128 | N/A | 3/0 | 1.009 | 1 | 0 | N/A |
| prefill_full_layer | hf | 128 | 1024 | N/A | 3/0 | 72.84 | 1 | 0 | N/A |
| prefill_full_layer | hf | 128 | 128 | N/A | 3/0 | 7.863 | 1 | 0 | N/A |
| prefill_full_layer | hf | 32 | 1024 | N/A | 3/0 | 18.27 | 1 | 0 | N/A |
| prefill_full_layer | hf | 32 | 128 | N/A | 3/0 | 1.983 | 1 | 0 | N/A |
| prefill_full_layer | hf_precopy | 1 | 1024 | N/A | 3/0 | 0.9887 | 1.156 | -0.1539 | 0 |
| prefill_full_layer | hf_precopy | 1 | 128 | N/A | 3/0 | 0.83 | 1.216 | -0.1789 | 0 |
| prefill_full_layer | hf_precopy | 128 | 1024 | N/A | 3/0 | 72.23 | 1.009 | -0.617 | 0 |
| prefill_full_layer | hf_precopy | 128 | 128 | N/A | 3/0 | 7.604 | 1.034 | -0.2583 | 0 |
| prefill_full_layer | hf_precopy | 32 | 1024 | N/A | 3/0 | 18.1 | 1.009 | -0.1687 | 0 |
| prefill_full_layer | hf_precopy | 32 | 128 | N/A | 3/0 | 1.763 | 1.125 | -0.22 | 0 |
| prefill_full_layer | official | 1 | 1024 | N/A | 3/0 | 1.374 | 0.8314 | 0.2318 | 0.004883 |
| prefill_full_layer | official | 1 | 128 | N/A | 3/0 | 1.254 | 0.8043 | 0.2455 | 0.001953 |
| prefill_full_layer | official | 128 | 1024 | N/A | 3/0 | 75.33 | 0.9671 | 2.481 | 0.01318 |
| prefill_full_layer | official | 128 | 128 | N/A | 3/0 | 8.311 | 0.9461 | 0.4483 | 0.007812 |
| prefill_full_layer | official | 32 | 1024 | N/A | 3/0 | 19.09 | 0.9574 | 0.8123 | 0.006836 |
| prefill_full_layer | official | 32 | 128 | N/A | 3/0 | 1.955 | 1.014 | -0.02804 | 0.003906 |
| prefill_full_layer | official_fused_both | 1 | 1024 | N/A | 3/0 | 1.253 | 0.9118 | 0.1105 | 0.004883 |
| prefill_full_layer | official_fused_both | 1 | 128 | N/A | 3/0 | 1.14 | 0.8851 | 0.131 | 0.003662 |
| prefill_full_layer | official_fused_both | 128 | 1024 | N/A | 3/0 | 59.22 | 1.23 | -13.62 | 0.009766 |
| prefill_full_layer | official_fused_both | 128 | 128 | N/A | 3/0 | 7.501 | 1.048 | -0.3618 | 0.007812 |
| prefill_full_layer | official_fused_both | 32 | 1024 | N/A | 3/0 | 15.21 | 1.202 | -3.066 | 0.007812 |
| prefill_full_layer | official_fused_both | 32 | 128 | N/A | 3/0 | 1.937 | 1.023 | -0.04552 | 0.005859 |
| prefill_full_layer | official_fused_both_precopy | 1 | 1024 | N/A | 3/0 | 1.101 | 1.038 | -0.0418 | 0.004883 |
| prefill_full_layer | official_fused_both_precopy | 1 | 128 | N/A | 3/0 | 0.9757 | 1.034 | -0.03321 | 0.003662 |
| prefill_full_layer | official_fused_both_precopy | 128 | 1024 | N/A | 3/0 | 58.67 | 1.242 | -14.18 | 0.009766 |
| prefill_full_layer | official_fused_both_precopy | 128 | 128 | N/A | 3/0 | 7.474 | 1.052 | -0.3883 | 0.007812 |
| prefill_full_layer | official_fused_both_precopy | 32 | 1024 | N/A | 3/0 | 14.92 | 1.224 | -3.349 | 0.007812 |
| prefill_full_layer | official_fused_both_precopy | 32 | 128 | N/A | 3/0 | 1.772 | 1.119 | -0.2111 | 0.005859 |
| prefill_full_layer | official_fused_rmsnorm | 1 | 1024 | N/A | 3/0 | 1.33 | 0.8589 | 0.1878 | 0.004883 |
| prefill_full_layer | official_fused_rmsnorm | 1 | 128 | N/A | 3/0 | 1.212 | 0.8323 | 0.2033 | 0.003174 |
| prefill_full_layer | official_fused_rmsnorm | 128 | 1024 | N/A | 3/0 | 69.67 | 1.046 | -3.174 | 0.01074 |
| prefill_full_layer | official_fused_rmsnorm | 128 | 128 | N/A | 3/0 | 7.973 | 0.9861 | 0.1107 | 0.007812 |
| prefill_full_layer | official_fused_rmsnorm | 32 | 1024 | N/A | 3/0 | 17.63 | 1.036 | -0.6389 | 0.007812 |
| prefill_full_layer | official_fused_rmsnorm | 32 | 128 | N/A | 3/0 | 1.898 | 1.045 | -0.08493 | 0.003906 |
| prefill_full_layer | official_fused_rope | 1 | 1024 | N/A | 3/0 | 1.305 | 0.8759 | 0.162 | 0.005859 |
| prefill_full_layer | official_fused_rope | 1 | 128 | N/A | 3/0 | 1.189 | 0.8486 | 0.18 | 0.001953 |
| prefill_full_layer | official_fused_rope | 128 | 1024 | N/A | 3/0 | 64.87 | 1.123 | -7.975 | 0.009766 |
| prefill_full_layer | official_fused_rope | 128 | 128 | N/A | 3/0 | 7.828 | 1.004 | -0.03446 | 0.007812 |
| prefill_full_layer | official_fused_rope | 32 | 1024 | N/A | 3/0 | 16.66 | 1.097 | -1.61 | 0.006836 |
| prefill_full_layer | official_fused_rope | 32 | 128 | N/A | 3/0 | 2.008 | 0.9875 | 0.02509 | 0.003906 |
