# 9-Task Speedup Report

| Machine | Task | vs baseline | vs frozen frontier |
|---|---|---:|---:|
| A100 | `a100_adaptive_layernorm_modulation` | 5.2149x | 1.3482x |
| A100 | `a100_bf16_int8_mixed_gemm` | 2.7666x | 1.3733x |
| A100 | `a100_fp16_transposed_dense_gemm` | 1.0160x | 0.3423x |
| A100 | `a100_indexed_row_softmax` | 2.7012x | 2.1305x |
| A100 | `a100_ragged_moe_swiglu` | 3.5001x | 1.2837x |
| A100 | `trtllm_fused_qk_norm_rope_bf16` | 4.3990x | 1.8621x |
| A100 | `vllm_block_aligned_moe_routing` | 9.3079x | 2.8596x |
| H200 | `h200_fp8_sparse_mla_prefill` | 10.1695x | 3.9450x |
| H200 | `h200_fp8_variable_m_gemm` | 4.2935x | 1.8771x |
