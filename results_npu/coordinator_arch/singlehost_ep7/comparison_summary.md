# Single-host EP7 coordinator vs static summary

- Static baseline root: `results_npu_ep7`
- Coordinator root: `results_npu/coordinator_arch/singlehost_ep7/coordinator`
- Metrics are read directly from `decode-dbo/timing_attention_*.json` and paired FFN timing JSONs.
- Throughput is computed as `1000 * batch / decode_tpot_ms`.

## Source data

- Static summary/table source: `results_npu_ep7/experiment_matrix_summary.csv`
- Static timing source: `results_npu_ep7/decode-dbo/timing_attention_*.json`
- Coordinator timing source: `results_npu/coordinator_arch/singlehost_ep7/coordinator/decode-dbo/timing_attention_*.json`
- Aggregated CSV: `results_npu/coordinator_arch/singlehost_ep7/coord_vs_static_summary.csv`

## Static reuse validation

- Representative current-code static runs within ±5%: 3/3.
- Validation CSV: `results_npu/coordinator_arch/singlehost_ep7/static_reuse_validation.csv`

| batch | seq | historical TPOT ms | current TPOT ms | delta % |
|---:|---:|---:|---:|---:|
| 8 | 128 | 295.318752 | 283.570712 | -3.978088 |
| 32 | 512 | 456.253949 | 452.296911 | -0.867289 |
| 128 | 512 | 900.349886 | 888.566609 | -1.308744 |

## Coordinator vs static TPOT / throughput

| batch | seq | static TPOT ms | coord TPOT ms | coord TPOT delta % | static tok/s | coord tok/s | coord throughput delta % |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 128 | 226.392093 | 215.970056 | -4.603534 | 8.834231 | 9.260543 | 4.825686 |
| 4 | 128 | 255.438195 | 264.324701 | 3.478926 | 15.659365 | 15.132903 | -3.361966 |
| 8 | 128 | 295.318752 | 285.887293 | -3.193654 | 27.089374 | 27.983056 | 3.299013 |
| 16 | 128 | 356.732688 | 370.513294 | 3.863006 | 44.851511 | 43.183336 | -3.719328 |
| 32 | 128 | 442.990334 | 446.496012 | 0.791367 | 72.236339 | 71.669173 | -0.785153 |
| 64 | 128 | 605.714570 | 619.246905 | 2.234111 | 105.660328 | 103.351344 | -2.185289 |
| 128 | 128 | 884.871427 | 906.979717 | 2.498475 | 144.653784 | 141.127743 | -2.437573 |
| 256 | 128 | 1334.052752 | 1358.722524 | 1.849235 | 191.896460 | 188.412274 | -1.815659 |
| 2 | 256 | 230.786524 | 214.474098 | -7.068188 | 8.666017 | 9.325135 | 7.605779 |
| 4 | 256 | 253.578399 | 251.270127 | -0.910279 | 15.774214 | 15.919123 | 0.918642 |
| 8 | 256 | 300.805435 | 294.893250 | -1.965452 | 26.595264 | 27.128461 | 2.004856 |
| 16 | 256 | 350.779972 | 333.356383 | -4.967099 | 45.612638 | 47.996681 | 5.226715 |
| 32 | 256 | 449.914766 | 421.636005 | -6.285360 | 71.124583 | 75.894847 | 6.706913 |
| 64 | 256 | 626.346062 | 575.310907 | -8.148076 | 102.179935 | 111.244197 | 8.870883 |
| 128 | 256 | 851.722009 | 825.638178 | -3.062482 | 150.283777 | 155.031591 | 3.159233 |
| 256 | 256 | 1330.536296 | 1351.852475 | 1.602074 | 192.403620 | 189.369776 | -1.576813 |
| 2 | 512 | 237.961297 | 234.565574 | -1.427006 | 8.404728 | 8.526400 | 1.447664 |
| 4 | 512 | 270.464748 | 265.139623 | -1.968879 | 14.789358 | 15.086391 | 2.008423 |
| 8 | 512 | 325.719499 | 284.108917 | -12.774974 | 24.561010 | 28.158215 | 14.645997 |
| 16 | 512 | 382.400439 | 363.380599 | -4.973802 | 41.840956 | 44.030969 | 5.234137 |
| 32 | 512 | 456.253949 | 440.600313 | -3.430904 | 70.136379 | 72.628183 | 3.552797 |
| 64 | 512 | 656.741042 | 616.010708 | -6.201887 | 97.450891 | 103.894298 | 6.611952 |
| 128 | 512 | 900.349886 | 901.613824 | 0.140383 | 142.166953 | 141.967655 | -0.140186 |
| 256 | 512 | 1452.502645 | 1407.419185 | -3.103847 | 176.247528 | 181.893215 | 3.203272 |

## Figure outputs

- Throughput ratio heatmap: `results_npu/coordinator_arch/singlehost_ep7/coord_vs_static_throughput_ratio_heatmap.png`
- TPOT delta heatmap: `results_npu/coordinator_arch/singlehost_ep7/coord_vs_static_tpot_delta_heatmap.png`
- seq=512 throughput line plot: `results_npu/coordinator_arch/singlehost_ep7/static_vs_coord_throughput_vs_batch_s512.png`
- seq=512 TPOT line plot: `results_npu/coordinator_arch/singlehost_ep7/static_vs_coord_tpot_vs_batch_s512.png`

## Notes

- Current coordinator matrix uses one-shot routing; a uniform routing table is expected to be close to static round-robin EP ownership.
- Any performance gain from load-aware routing requires a later poll/dynamic-routing experiment.
