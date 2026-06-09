| file | backend | tokens | hidden | top_k | ok | dispatch_ms_median | reduce_ms_median | combine_ms_median | total_ms_median | stage | error_type | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| broadcast_reduce_ep8_h2048.json | broadcast_reduce | 4 | 2048 | 8 | True | 0.2120631979778409 | 0.15185970114544034 |  | 0.3638193942606449 |  |  |  |
| broadcast_reduce_ep8_h2048.json | broadcast_reduce | 16 | 2048 | 8 | True | 0.15507625066675246 | 0.14685839996673167 |  | 0.3019298950675875 |  |  |  |
| broadcast_reduce_ep8_h2048.json | broadcast_reduce | 64 | 2048 | 8 | True | 0.14892719918861985 | 0.14548440231010318 |  | 0.2942955936305225 |  |  |  |
| official_base_ep8_h2048_fail.json | npu_moe_distribute | 4 | 2048 | 8 | False |  |  |  |  | dispatch | RuntimeError | npu_moe_distribute_dispatch:build/CMakeFiles/torch_npu.dir/compiler_depend.ts:105 NPU function error: call aclnnMoeDistributeDispatch failed, error code is 561002 |
| official_base_ep8_h7168.json | npu_moe_distribute | 4 | 7168 | 2 | True | 0.1842808909714222 |  | 0.09433130035176873 | 0.27885918971151114 |  |  |  |
| official_v2_ep8_h2048_fail.json | npu_moe_distribute_v2 | 4 | 2048 | 8 | False |  |  |  |  | dispatch | RuntimeError | npu_moe_distribute_dispatch_v2:build/CMakeFiles/torch_npu.dir/compiler_depend.ts:128 NPU function error: call aclnnMoeDistributeDispatchV2 failed, error code is 561002 |
