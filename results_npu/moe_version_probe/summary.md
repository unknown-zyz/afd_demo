# MoE distribute version probe (isolated container)

## Container

- Host: Host2 `liteserver-910c-2-00001.novalocal`
- Container: `afd-npu-version-probe-torch29-npu` (created by this task)
- Image: `quay.io/ascend/vllm-ascend:v0.18.0rc1-a3`
- CANN: `8.5.1`, inner `V100R001C25SPC002B220`
- torch: `2.9.0+cpu`
- torch_npu: `2.9.0.post1+gitee7ba04`
- Existing containers were not modified or deleted.

## Schema finding

`npu_moe_distribute_dispatch_v2` and `combine_v2` expose `comm_alg` in this environment. Current 2.6/CANN 8.5.0 containers did not expose it.

```text
torch=2.9.0+cpu
torch_npu=2.9.0.post1+gitee7ba04
=== npu_moe_distribute_dispatch_v2 ===
schema=npu::npu_moe_distribute_dispatch_v2(Tensor x, Tensor expert_ids, str group_ep, int ep_world_size, int ep_rank_id, int moe_expert_num, *, Tensor? scales=None, Tensor? x_active_mask=None, Tensor? expert_scales=None, Tensor? elastic_info=None, Tensor? performance_info=None, str group_tp="", int tp_world_size=0, int tp_rank_id=0, int expert_shard_type=0, int shared_expert_num=1, int shared_expert_rank_num=0, int quant_mode=0, int global_bs=0, int expert_token_nums_type=1, str comm_alg="", int zero_expert_num=0, int copy_expert_num=0, int const_expert_num=0) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)
=== npu_moe_distribute_combine_v2 ===
schema=npu::npu_moe_distribute_combine_v2(Tensor expand_x, Tensor expert_ids, Tensor assist_info_for_combine, Tensor ep_send_counts, Tensor expert_scales, str group_ep, int ep_world_size, int ep_rank_id, int moe_expert_num, *, Tensor? tp_send_counts=None, Tensor? x_active_mask=None, Tensor? expand_scales=None, Tensor? shared_expert_x=None, Tensor? elastic_info=None, Tensor? ori_x=None, Tensor? const_expert_alpha_1=None, Tensor? const_expert_alpha_2=None, Tensor? const_expert_v=None, Tensor? performance_info=None, str group_tp="", int tp_world_size=0, int tp_rank_id=0, int expert_shard_type=0, int shared_expert_num=1, int shared_expert_rank_num=0, int global_bs=0, int comm_quant_mode=0, str comm_alg="", int zero_expert_num=0, int copy_expert_num=0, int const_expert_num=0) -> Tensor
```

## Probe results

| Case | Result |
|---|---|
| base H=7168 top_k=2 | {"ok_count": 8, "dispatch_ms_median": 0.6469615618698299, "combine_ms_median": 0.26833696756511927, "total_ms_median": 0.9056409471668303, "max_abs_diff_max": 0.0} |
| base H=2048 top_k=8 | FAIL RuntimeError: npu_moe_distribute_dispatch:build/CMakeFiles/torch_npu.dir/compiler_depend.ts:105 NPU function error: call aclnnMoeDistributeDispatch failed, error code is 561002 |
| v2 H=2048 default comm_alg | {"ok_count": 8, "dispatch_ms_median": 2334.7647498594597, "combine_ms_median": 1.3183685950934887, "total_ms_median": 2336.128788301721, "max_abs_diff_max": 0.0} |
| v2 H=2048 comm_alg=fullmesh_v1 | {"ok_count": 8, "dispatch_ms_median": 0.884635781403631, "combine_ms_median": 0.19774853717535734, "total_ms_median": 1.0801643365994096, "max_abs_diff_max": 0.0} |
| v2 H=2048 comm_alg=fullmesh_v2 | {"ok_count": 8, "dispatch_ms_median": 0.3129368997178972, "combine_ms_median": 0.16854621935635805, "total_ms_median": 0.4817756125703454, "max_abs_diff_max": 0.0} |
| updated bench script v2 H=2048 comm_alg=fullmesh_v2 warm | [{"backend": "npu_moe_distribute_v2", "tokens": 4, "hidden": 2048, "top_k": 8, "comm_alg": "fullmesh_v2", "ok": true, "rank_count": 8, "dispatch_ms_median": 0.561873079277575, "dispatch_ms_max": 0.5853339098393917, "combine_ms_median": 0.29794208239763975, "combine_ms_max": 0.30206190422177315, "total_ms_median": 0.85977534763515, "total_ms_max": 0.8830660954117775}] |

## Interpretation

- Version pairing matters for `dispatch_v2`: CANN 8.5.1 + torch_npu 2.9 exposes `comm_alg`, and `comm_alg=fullmesh_v2` makes Qwen3-shaped H=2048 dispatch/combine pass.
- Base `npu_moe_distribute_dispatch` still rejects H=2048 and only passed the H=7168 sanity case, so base remains unsuitable for Qwen3 hidden size.
- `dispatch_v2` without explicit `comm_alg` passes in torch_npu 2.9 but is extremely slow in this probe; explicit `fullmesh_v2` is the best observed path.
- `torchair` is not installed in this image, so static-graph Dispatch_v2 tail-node validation was not run. The project path tested here is eager/torchrun.
