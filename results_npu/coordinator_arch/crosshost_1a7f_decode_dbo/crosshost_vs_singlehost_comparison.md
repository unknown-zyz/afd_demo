# Cross-host 1A7F coordinator decode-dbo matrix comparison

## Summary

- Cross-host matrix status: **24/24 OK** in `results_npu/coordinator_arch/crosshost_1a7f_decode_dbo/matrix_summary_final.csv`.
- Cross-host timing JSONs and metrics are staged under `results_npu/coordinator_arch/crosshost_1a7f_decode_dbo/decode-dbo/`, with summaries `results_npu/coordinator_arch/crosshost_1a7f_decode_dbo/decode_mfu_summary.csv` and `results_npu/coordinator_arch/crosshost_1a7f_decode_dbo/ep_bandwidth_summary.csv`.
- Mean cross-host / single-host coordinator TPOT ratio: **1.018x**.
- Best relative point: b16/s512, ratio 0.963x.
- Worst relative point: b16/s256, ratio 1.109x.
- Host2 free space stayed around 7GB during the matrix; no matrix row failed due to disk pressure.

## Sources

- Cross-host matrix summary: `results_npu/coordinator_arch/crosshost_1a7f_decode_dbo/matrix_summary_final.csv`
- Cross-host MFU summary: `results_npu/coordinator_arch/crosshost_1a7f_decode_dbo/decode_mfu_summary.csv`
- Cross-host EP bandwidth summary: `results_npu/coordinator_arch/crosshost_1a7f_decode_dbo/ep_bandwidth_summary.csv`
- Single-host coordinator summary: `results_npu/coordinator_arch/singlehost_ep7/coordinator/experiment_matrix_summary.csv`
- Static vs coordinator baseline summary: `results_npu/coordinator_arch/singlehost_ep7/coord_vs_static_summary.csv`

## TPOT table

| batch | seq | cross-host TPOT ms | single-host coord TPOT ms | xhost/single ratio | static TPOT ms | xhost/static ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 128 | 227.026 | 215.970 | 1.051 | 226.392 | 1.003 |
| 2 | 256 | 224.198 | 214.474 | 1.045 | 230.787 | 0.971 |
| 2 | 512 | 230.250 | 234.566 | 0.982 | 237.961 | 0.968 |
| 4 | 128 | 259.725 | 264.325 | 0.983 | 255.438 | 1.017 |
| 4 | 256 | 253.542 | 251.270 | 1.009 | 253.578 | 1.000 |
| 4 | 512 | 261.614 | 265.140 | 0.987 | 270.465 | 0.967 |
| 8 | 128 | 293.064 | 285.887 | 1.025 | 295.319 | 0.992 |
| 8 | 256 | 298.891 | 294.893 | 1.014 | 300.805 | 0.994 |
| 8 | 512 | 297.366 | 284.109 | 1.047 | 325.719 | 0.913 |
| 16 | 128 | 363.991 | 370.513 | 0.982 | 356.733 | 1.020 |
| 16 | 256 | 369.615 | 333.356 | 1.109 | 350.780 | 1.054 |
| 16 | 512 | 349.961 | 363.381 | 0.963 | 382.400 | 0.915 |
| 32 | 128 | 457.763 | 446.496 | 1.025 | 442.990 | 1.033 |
| 32 | 256 | 447.339 | 421.636 | 1.061 | 449.915 | 0.994 |
| 32 | 512 | 460.368 | 440.600 | 1.045 | 456.254 | 1.009 |
| 64 | 128 | 629.825 | 619.247 | 1.017 | 605.715 | 1.040 |
| 64 | 256 | 611.960 | 575.311 | 1.064 | 626.346 | 0.977 |
| 64 | 512 | 627.110 | 616.011 | 1.018 | 656.741 | 0.955 |
| 128 | 128 | 886.328 | 906.980 | 0.977 | 884.871 | 1.002 |
| 128 | 256 | 846.869 | 825.638 | 1.026 | 851.722 | 0.994 |
| 128 | 512 | 913.161 | 901.614 | 1.013 | 900.350 | 1.014 |
| 256 | 128 | 1309.064 | 1358.723 | 0.963 | 1334.053 | 0.981 |
| 256 | 256 | 1369.714 | 1351.852 | 1.013 | 1330.536 | 1.029 |
| 256 | 512 | 1432.878 | 1407.419 | 1.018 | 1452.503 | 0.986 |
