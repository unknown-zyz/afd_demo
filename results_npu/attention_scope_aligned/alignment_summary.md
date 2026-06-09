| batch | seq | tokens | num_micro_batches | mb_sizes | decode_tpot_ms | pipeline_attn_per_layer_sum_median_ms | single_layer_global_ms | single_layer_mb_sum_ms | pipeline_vs_single_global | pipeline_vs_single_mb_sum | pipeline_file |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 9 | 1 | 1 | 149.767 |  | 0.659221 | 0.625642 |  |  | results_npu/attention_scope_aligned/decode_dbo/timing_attention_aligned_decode-dbo_b1_s1024_t10_r1.json |
| 1 | 1024 | 9 | 1 | 1 | 145.796 |  | 0.659221 | 0.625642 |  |  | results_npu/attention_scope_aligned/decode_dbo/timing_attention_aligned_decode-dbo_b1_s1024_t10_r2.json |
| 1 | 1024 | 9 | 1 | 1 | 143.991 |  | 0.659221 | 0.625642 |  |  | results_npu/attention_scope_aligned/decode_dbo/timing_attention_aligned_decode-dbo_b1_s1024_t10_r3.json |
| 1 | 128 | 9 | 1 | 1 | 145.323 |  | 0.645914 | 0.627191 |  |  | results_npu/attention_scope_aligned/decode_dbo/timing_attention_aligned_decode-dbo_b1_s128_t10_r1.json |
| 1 | 128 | 9 | 1 | 1 | 147.851 |  | 0.645914 | 0.627191 |  |  | results_npu/attention_scope_aligned/decode_dbo/timing_attention_aligned_decode-dbo_b1_s128_t10_r2.json |
| 1 | 128 | 9 | 1 | 1 | 151.613 |  | 0.645914 | 0.627191 |  |  | results_npu/attention_scope_aligned/decode_dbo/timing_attention_aligned_decode-dbo_b1_s128_t10_r3.json |
| 1 | 512 | 9 | 1 | 1 | 147.646 |  | 0.643772 | 0.621449 |  |  | results_npu/attention_scope_aligned/decode_dbo/timing_attention_aligned_decode-dbo_b1_s512_t10_r1.json |
| 1 | 512 | 9 | 1 | 1 | 147.091 |  | 0.643772 | 0.621449 |  |  | results_npu/attention_scope_aligned/decode_dbo/timing_attention_aligned_decode-dbo_b1_s512_t10_r2.json |
| 1 | 512 | 9 | 1 | 1 | 147.642 |  | 0.643772 | 0.621449 |  |  | results_npu/attention_scope_aligned/decode_dbo/timing_attention_aligned_decode-dbo_b1_s512_t10_r3.json |
| 32 | 1024 | 9 | 2 | 16,16 | 700.748 | 1.90031 | 0.648791 | 1.25807 | 2.929 | 1.5105 | results_npu/attention_scope_aligned/decode_dbo/timing_attention_aligned_decode-dbo_b32_s1024_t10_r1.json |
| 32 | 1024 | 9 | 2 | 16,16 | 711.576 | 1.81146 | 0.648791 | 1.25807 | 2.79206 | 1.43988 | results_npu/attention_scope_aligned/decode_dbo/timing_attention_aligned_decode-dbo_b32_s1024_t10_r2.json |
| 32 | 1024 | 9 | 2 | 16,16 | 698.632 | 1.93489 | 0.648791 | 1.25807 | 2.98231 | 1.53799 | results_npu/attention_scope_aligned/decode_dbo/timing_attention_aligned_decode-dbo_b32_s1024_t10_r3.json |
| 32 | 128 | 9 | 2 | 16,16 | 750.708 | 1.70555 | 0.644822 | 1.25694 | 2.64499 | 1.35691 | results_npu/attention_scope_aligned/decode_dbo/timing_attention_aligned_decode-dbo_b32_s128_t10_r1.json |
| 32 | 128 | 9 | 2 | 16,16 | 736.425 | 1.92913 | 0.644822 | 1.25694 | 2.99173 | 1.53479 | results_npu/attention_scope_aligned/decode_dbo/timing_attention_aligned_decode-dbo_b32_s128_t10_r2.json |
| 32 | 128 | 9 | 2 | 16,16 | 831.394 | 1.79891 | 0.644822 | 1.25694 | 2.78978 | 1.43119 | results_npu/attention_scope_aligned/decode_dbo/timing_attention_aligned_decode-dbo_b32_s128_t10_r3.json |
| 32 | 512 | 9 | 2 | 16,16 | 755.041 | 1.90216 | 0.642161 | 1.24838 | 2.96213 | 1.52371 | results_npu/attention_scope_aligned/decode_dbo/timing_attention_aligned_decode-dbo_b32_s512_t10_r1.json |
| 32 | 512 | 9 | 2 | 16,16 | 829.075 | 1.77718 | 0.642161 | 1.24838 | 2.7675 | 1.42359 | results_npu/attention_scope_aligned/decode_dbo/timing_attention_aligned_decode-dbo_b32_s512_t10_r2.json |
| 32 | 512 | 9 | 2 | 16,16 | 708.034 | 1.79591 | 0.642161 | 1.24838 | 2.79667 | 1.4386 | results_npu/attention_scope_aligned/decode_dbo/timing_attention_aligned_decode-dbo_b32_s512_t10_r3.json |
| 8 | 1024 | 9 | 2 | 4,4 | 364.828 | 1.7423 | 0.631876 | 1.26313 | 2.75735 | 1.37935 | results_npu/attention_scope_aligned/decode_dbo/timing_attention_aligned_decode-dbo_b8_s1024_t10_r1.json |
| 8 | 1024 | 9 | 2 | 4,4 | 334.22 | 1.98577 | 0.631876 | 1.26313 | 3.14266 | 1.5721 | results_npu/attention_scope_aligned/decode_dbo/timing_attention_aligned_decode-dbo_b8_s1024_t10_r2.json |
| 8 | 1024 | 9 | 2 | 4,4 | 363.003 | 1.89872 | 0.631876 | 1.26313 | 3.0049 | 1.50319 | results_npu/attention_scope_aligned/decode_dbo/timing_attention_aligned_decode-dbo_b8_s1024_t10_r3.json |
| 8 | 128 | 9 | 2 | 4,4 | 367.607 | 1.76268 | 0.646136 | 1.26515 | 2.72804 | 1.39326 | results_npu/attention_scope_aligned/decode_dbo/timing_attention_aligned_decode-dbo_b8_s128_t10_r1.json |
| 8 | 128 | 9 | 2 | 4,4 | 316.031 | 1.79847 | 0.646136 | 1.26515 | 2.78342 | 1.42155 | results_npu/attention_scope_aligned/decode_dbo/timing_attention_aligned_decode-dbo_b8_s128_t10_r2.json |
| 8 | 128 | 9 | 2 | 4,4 | 329.983 | 1.97434 | 0.646136 | 1.26515 | 3.05561 | 1.56056 | results_npu/attention_scope_aligned/decode_dbo/timing_attention_aligned_decode-dbo_b8_s128_t10_r3.json |
| 8 | 512 | 9 | 2 | 4,4 | 322.804 | 1.84422 | 0.643742 | 1.26132 | 2.86484 | 1.46213 | results_npu/attention_scope_aligned/decode_dbo/timing_attention_aligned_decode-dbo_b8_s512_t10_r1.json |
| 8 | 512 | 9 | 2 | 4,4 | 348.26 | 1.82697 | 0.643742 | 1.26132 | 2.83805 | 1.44845 | results_npu/attention_scope_aligned/decode_dbo/timing_attention_aligned_decode-dbo_b8_s512_t10_r2.json |
| 8 | 512 | 9 | 2 | 4,4 | 325.665 | 2.12144 | 0.643742 | 1.26132 | 3.29549 | 1.68192 | results_npu/attention_scope_aligned/decode_dbo/timing_attention_aligned_decode-dbo_b8_s512_t10_r3.json |
