# 13-Task Speedup Report

| Machine | Task | vs baseline | vs frozen frontier |
|---|---|---:|---:|
| A100 | `cutlass_normalization_pr3009` | 10.3859× | 1.0913× |
| A100 | `liger_kernel_normalization_pr1225` | 8.1793× | 1.0604× |
| A100 | `linkedin_liger_kernel_activation_pr1277` | 4.5474× | 2.4109× |
| A100 | `nvidia_tensorrt_llm_normalization_pr9969` | 9.7391× | 1.3117× |
| A100 | `pytorch_pytorch_normalization_pr4693` | 3.7119× | 1.0109× |
| H200 | `liger_kernel_activation_iss1330` | 4.5616× | 1.7657× |
| H200 | `liger_kernel_normalization_pr1257` | 4.3034× | 1.8147× |
| H200 | `liger_kernel_normalization_pr1305` | 4.2976× | 1.6095× |
| H200 | `h200_bf16_silu_mul_forward` | 3.9438× | 1.5694× |
| H200 | `h200_llama_bf16_rmsnorm_forward` | 4.5845× | 1.5778× |
| B200 | `S02_layernorm` | 2.4217× | 2.0395× |
| B200 | `S03_softmax_rowmax` | 1.4318× | 1.1226× |
| B200 | `S04_swiglu` | 1.7939× | 0.5051× |
